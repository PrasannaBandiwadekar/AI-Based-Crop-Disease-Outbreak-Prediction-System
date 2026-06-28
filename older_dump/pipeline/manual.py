import pandas as pd

df = pd.read_csv("early_warning_dataset.csv")

print("Total columns:", len(df.columns))
print(df.columns.tolist())