import pandas as pd

df = pd.read_csv("data/network_traffic_dataset.csv")
df_small = df.sample(n=3000, random_state=42)
df_small.to_csv("data/small_dataset.csv", index=False)

print("✅ Small dataset created!")