"""
AgroCare IoT — 1D-CNN + Dual Attention Model
===============================================
Architecture: 1D Convolutional backbone with parallel Temporal Attention
and Channel Attention (SE-style), fused via concatenation for 3-class
plant health classification.

Input:  (batch, 60, 5)  — 60 timesteps × 5 sensor channels
Output: (batch, 3)      — logits for [Normal, Watering Required, Risky]

Architecture Diagram:
    ┌─────────────────────────────────────┐
    │         Input (B, 60, 5)            │
    │         Permute → (B, 5, 60)        │
    └──────────────┬──────────────────────┘
                   ▼
    ┌─────────────────────────────────────┐
    │        CNN Backbone (3 blocks)      │
    │   Conv1d → BN → GELU → Pool        │
    │   Output: (B, 128, 15)             │
    └──────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
    ┌──────────┐      ┌──────────────┐
    │ Temporal │      │   Channel    │
    │ Attention│      │  Attention   │
    │ (MHA)    │      │  (SE-Block)  │
    │ (B, 128) │      │  (B, 128)   │
    └────┬─────┘      └──────┬───────┘
         │                   │
         └────────┬──────────┘
                  ▼
    ┌─────────────────────────────────────┐
    │     Fusion: Concat → (B, 256)       │
    │     FC(256→64) → GELU → Dropout     │
    │     FC(64→3) → Logits               │
    └─────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """A single 1D convolution block: Conv → BatchNorm → GELU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CNNBackbone(nn.Module):
    """3-stage 1D-CNN feature extractor with progressive channel expansion.

    Input:  (B, 5, 60)   — 5 sensor channels, 60 timesteps
    Output: (B, 128, 15) — 128 feature channels, 15 temporal positions
    """

    def __init__(self, in_channels: int = 5, dropout: float = 0.1):
        super().__init__()

        # Block 1: (B, 5, 60) → (B, 32, 30)
        self.block1 = nn.Sequential(
            ConvBlock(in_channels, 32, kernel_size=5, padding=2),
            ConvBlock(32, 32, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # Block 2: (B, 32, 30) → (B, 64, 15)
        self.block2 = nn.Sequential(
            ConvBlock(32, 64, kernel_size=5, padding=2),
            ConvBlock(64, 64, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # Block 3: (B, 64, 15) → (B, 128, 15)
        self.block3 = nn.Sequential(
            ConvBlock(64, 128, kernel_size=3, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------


class TemporalAttention(nn.Module):
    """Multi-Head Self-Attention over the temporal dimension.

    Allows the model to learn which timesteps in the 60-step window are most
    informative for classification (e.g., recent declining moisture trend).

    Input:  (B, C, T)  — feature maps from CNN backbone
    Output: (B, C)     — temporally-attended feature vector
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Learnable aggregation query — learns "what temporal pattern matters"
        self.agg_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.agg_query)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) feature maps
        Returns:
            (B, C) attended feature vector
        """
        B = x.size(0)

        # Permute to (B, T, C) for attention over temporal positions
        x_t = x.permute(0, 2, 1)  # (B, T, C)
        x_t = self.norm(x_t)

        # Expand aggregation query for the batch
        query = self.agg_query.expand(B, -1, -1)  # (B, 1, C)

        # Cross-attention: query attends to all temporal positions
        attended, _ = self.mha(query, x_t, x_t)  # (B, 1, C)

        return attended.squeeze(1)  # (B, C)


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style attention over sensor feature channels.

    Learns which sensor-derived features (not raw sensors, but CNN features)
    are most relevant for the current input. This helps the model weight
    soil moisture features higher during watering decisions, for example.

    Input:  (B, C, T)  — feature maps from CNN backbone
    Output: (B, C)     — channel-attended feature vector
    """

    def __init__(self, num_channels: int = 128, reduction: int = 8):
        super().__init__()
        mid = max(num_channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # Global avg pool over T
        self.excitation = nn.Sequential(
            nn.Linear(num_channels, mid),
            nn.GELU(),
            nn.Linear(mid, num_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) feature maps
        Returns:
            (B, C) channel-attended feature vector
        """
        # Squeeze: (B, C, T) → (B, C, 1) → (B, C)
        s = self.squeeze(x).squeeze(-1)

        # Excitation: learn channel importance weights
        weights = self.excitation(s)  # (B, C) with values in [0, 1]

        # Scale original features and pool
        scaled = x * weights.unsqueeze(-1)  # (B, C, T) * (B, C, 1)

        # Global average pool the scaled features
        return scaled.mean(dim=-1)  # (B, C)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class AgroCareNet(nn.Module):
    """1D-CNN + Dual Attention network for plant health classification.

    Combines temporal attention (which timesteps matter) and channel attention
    (which feature channels matter) for robust multi-sensor fusion.

    Args:
        in_channels:    Number of input sensor channels (default: 5)
        num_classes:    Number of output classes (default: 3)
        cnn_dropout:    Dropout rate in CNN backbone (default: 0.1)
        attn_heads:     Number of heads in temporal attention (default: 4)
        attn_dropout:   Dropout rate in attention layers (default: 0.1)
        fc_dropout:     Dropout rate in classifier head (default: 0.3)
        feature_dim:    Feature dimension from CNN backbone (default: 128)
        se_reduction:   Reduction ratio for channel attention (default: 8)
    """

    def __init__(
        self,
        in_channels: int = 5,
        num_classes: int = 3,
        cnn_dropout: float = 0.1,
        attn_heads: int = 4,
        attn_dropout: float = 0.1,
        fc_dropout: float = 0.3,
        feature_dim: int = 128,
        se_reduction: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # --- CNN Backbone ---
        self.backbone = CNNBackbone(in_channels=in_channels, dropout=cnn_dropout)

        # --- Dual Attention ---
        self.temporal_attention = TemporalAttention(
            embed_dim=feature_dim,
            num_heads=attn_heads,
            dropout=attn_dropout,
        )
        self.channel_attention = ChannelAttention(
            num_channels=feature_dim,
            reduction=se_reduction,
        )

        # --- Classifier Head ---
        # Fusion: concat temporal + channel → 2 * feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(fc_dropout),
            nn.Linear(64, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming depending on layer type."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 60, 5) — batch of sensor windows
        Returns:
            logits: (B, 3) — class logits (apply softmax externally for probs)
        """
        # Permute to channels-first for Conv1d: (B, 60, 5) → (B, 5, 60)
        x = x.permute(0, 2, 1)

        # CNN feature extraction: (B, 5, 60) → (B, 128, 15)
        features = self.backbone(x)

        # Dual attention — parallel branches
        temporal_out = self.temporal_attention(features)  # (B, 128)
        channel_out = self.channel_attention(features)    # (B, 128)

        # Fusion via concatenation
        fused = torch.cat([temporal_out, channel_out], dim=1)  # (B, 256)

        # Classification
        logits = self.classifier(fused)  # (B, 3)

        return logits

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method for inference — returns class index and probabilities.

        Args:
            x: (B, 60, 5) or (1, 60, 5) sensor window

        Returns:
            dict with 'class_idx', 'class_probs', 'class_name'
        """
        self.eval()
        class_names = ["Normal", "Watering Required", "Risky"]

        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            class_idx = probs.argmax(dim=1)

        return {
            "class_idx": class_idx,
            "class_probs": probs,
            "class_names": [class_names[i] for i in class_idx.cpu().tolist()],
        }

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Return a formatted model summary string."""
        lines = [
            "=" * 60,
            "  AgroCareNet — 1D-CNN + Dual Attention",
            "=" * 60,
            f"  Input shape    : (B, 60, {self.in_channels})",
            f"  Output classes : {self.num_classes}",
            f"  Feature dim    : {self.feature_dim}",
            f"  Total params   : {self.count_parameters():,}",
            "=" * 60,
            "",
            "  Backbone:",
            f"    Block 1: Conv1d(5→32) × 2 + Pool → (B, 32, 30)",
            f"    Block 2: Conv1d(32→64) × 2 + Pool → (B, 64, 15)",
            f"    Block 3: Conv1d(64→128) × 2       → (B, 128, 15)",
            "",
            "  Attention:",
            f"    Temporal: MultiHeadAttention(dim=128, heads=4)",
            f"    Channel:  SE-Block(128, reduction=8)",
            "",
            "  Classifier:",
            f"    Concat(128+128) → FC(256→64) → GELU → FC(64→3)",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick sanity check (runs when this file is executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = AgroCareNet()
    print(model.summary())

    # Test forward pass with dummy data
    dummy = torch.randn(4, 60, 5)
    logits = model(dummy)
    print(f"\nForward pass test:")
    print(f"  Input shape  : {dummy.shape}")
    print(f"  Output shape : {logits.shape}")
    print(f"  Output logits: {logits[0].detach().numpy()}")

    # Test predict method
    result = model.predict(dummy)
    print(f"\nPrediction test:")
    print(f"  Classes: {result['class_names']}")
    print(f"  Probs  : {result['class_probs'][0].numpy()}")