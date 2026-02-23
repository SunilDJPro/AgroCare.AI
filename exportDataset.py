import numpy as np
import pandas as pd

data = np.load("utils/GenerateSynthData/data/agrocare_dataset.npz")

X = data["X"]
y = data["y"]

rows = []

for i in range(len(X)):
    window = X[i]
    label = y[i]

    for t in range(60):
        rows.append({
            "SampleID": i,
            "Timestep": t,
            "SoilMoisture": window[t][0],
            "SoilTemp": window[t][1],
            "AmbientTemp": window[t][2],
            "Humidity": window[t][3],
            "UV": window[t][4],
            "Label": label
        })

df = pd.DataFrame(rows)

df.to_csv("agrocare_tableau.csv", index=False)

print("CSV exported")