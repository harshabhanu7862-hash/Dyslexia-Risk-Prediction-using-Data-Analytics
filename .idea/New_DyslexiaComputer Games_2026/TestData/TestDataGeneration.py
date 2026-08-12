import pandas as pd
import numpy as np
import joblib

DATA_PATH = r"C:\Users\DELL\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\Dataset\Dyslexia_Dataset.csv"
FEATURE_PATH = r"C:\Users\DELL\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\ModelFiles\feature_columns.pkl"
OUTPUT_PATH = r"C:\Users\DELL\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\TestData\sample_20_records.csv"

df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8-sig")
df.columns = df.columns.str.strip()

y = df["Dyslexia"].str.strip().str.lower().map({"no": 0, "yes": 1})
X = df.drop("Dyslexia", axis=1)

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="ignore")

X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.map(str)

# 🔥 Load exact features used by model
selected_features = joblib.load(FEATURE_PATH)

# Add missing columns safely
for col in selected_features:
    if col not in X.columns:
        X[col] = 0

# Keep only required features
X_selected = X[selected_features]

final_df = pd.concat([X_selected, y], axis=1)

dyslexia_10 = final_df[final_df["Dyslexia"] == 1].sample(10, random_state=42)
non_dyslexia_10 = final_df[final_df["Dyslexia"] == 0].sample(10, random_state=42)

output_df = pd.concat([dyslexia_10, non_dyslexia_10])

output_df.to_csv(OUTPUT_PATH, index=False)

print("20 records generated successfully")