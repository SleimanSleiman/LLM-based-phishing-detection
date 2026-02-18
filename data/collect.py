import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

dfs = []
for p in glob.glob(os.path.join(RAW_DIR, "*.csv")):
    dfs.append(pd.read_csv(p))
for p in glob.glob(os.path.join(RAW_DIR, "*.json")):
    dfs.append(pd.read_json(p, lines=True))
df = pd.concat(dfs, ignore_index=True)


df = df.dropna(subset=["body"])
df["text"] = df["subject"].fillna("") + "\n\n" + df["body"].fillna("")
df = df[["id", "text", "label"]]

train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_Size=0.5, stratify=temp["label"], random_state=42)

train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
val.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

print("Saved: ", os.path.join(OUT_DIR, "train.csv"), "etc.")
