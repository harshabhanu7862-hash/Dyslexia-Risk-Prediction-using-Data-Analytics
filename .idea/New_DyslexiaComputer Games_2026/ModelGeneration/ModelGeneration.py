import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, roc_auc_score, roc_curve, f1_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE

np.random.seed(42)

BASE_DIR = r"C:\Users\DELL\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026"
REPORT_DIR = os.path.join(BASE_DIR, "Reports")
MODEL_DIR = os.path.join(BASE_DIR, "ModelFiles")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(
    r"C:\Users\DELL\PycharmProjects\UpdatedDyslexia\.venv\New_DyslexiaComputer Games_2026\Dataset\Dyslexia_Dataset.csv",
    sep=";",
    encoding="utf-8-sig"
)

df.columns = df.columns.str.strip()

y = df["Dyslexia"].map({"No": 0, "Yes": 1})
X = df.drop("Dyslexia", axis=1)

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = pd.to_numeric(X[col], errors="ignore")

X = pd.get_dummies(X, drop_first=True)
X.columns = X.columns.map(str)

feature_selector_model = ExtraTreesClassifier(n_estimators=500, random_state=42)
feature_selector_model.fit(X, y)

selector = SelectFromModel(feature_selector_model, threshold="mean", prefit=True)
mask = selector.get_support()

if mask.sum() == 0:
    X_selected_df = X.copy()
    selected_features = X.columns.tolist()
else:
    selected_features = list(np.array(X.columns)[mask])
    X_selected_df = X[selected_features]

X_selected_df.columns = X_selected_df.columns.map(str)

joblib.dump(selected_features, os.path.join(MODEL_DIR, "feature_columns.pkl"))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected_df)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, stratify=y, random_state=42
)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

THRESHOLD = 0.30

models = {
    "ExtraTrees": ExtraTreesClassifier(n_estimators=800, min_samples_leaf=2, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=600, min_samples_leaf=2, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True),
    "LogisticRegression": LogisticRegression(max_iter=2000)
}

ensemble = VotingClassifier(
    estimators=[
        ("et", models["ExtraTrees"]),
        ("rf", models["RandomForest"]),
        ("svm", models["SVM"])
    ],
    voting="soft"
)

models["Ensemble"] = ensemble

results = []
roc_data = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    auc = roc_auc_score(y_test, probs)

    results.append([name, acc, prec, rec, f1, auc])

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, auc)

metrics_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]
)

metrics_df.to_csv(os.path.join(REPORT_DIR, "model_comparison_metrics.csv"), index=False)

plt.figure(figsize=(12, 6))
metrics_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "AUC"]].plot(kind="bar")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "all_models_comparison_bar_chart.png"), dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(REPORT_DIR, "roc_curve_all_models.png"), dpi=300)
plt.close()

best_model_name = metrics_df.sort_values(by="F1", ascending=False).iloc[0]["Model"]
best_model = models[best_model_name]

joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_model, X_scaled, y, cv=skf, scoring="f1")

pd.DataFrame({"CV_F1": cv_scores}).to_csv(
    os.path.join(REPORT_DIR, "best_model_cv_scores.csv"),
    index=False
)

print("Best Model:", best_model_name)
print("Execution Completed Successfully")