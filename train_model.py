# train_model.py

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# ------------------ 1. Load Data ------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Dataset shape:", df.shape)
print(df.head())

# ------------------ 2. Data Visualization ------------------
# Class distribution
sns.countplot(x='target', data=df)
plt.title("Class Distribution (0 = Malignant, 1 = Benign)")
plt.savefig("class_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.close()

# ------------------ 3. Preprocessing ------------------
X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ 4. Train/Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ------------------ 5. Model Training ------------------
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": xgb.XGBClassifier(),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"------ {name} ------")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        results[name] = {
            "model": model,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "y_pred": y_pred
        }
        print("ROC AUC Score:", auc)
    print("\n")

# ------------------ 6. Visualize ROC Curve ------------------
plt.figure(figsize=(8, 6))
for name, res in results.items():
    plt.plot(res['fpr'], res['tpr'], label=f"{name} (AUC = {res['auc']:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png")
plt.close()

# ------------------ 7. Save the Best Model ------------------
best_model_name = "Random Forest"
best_model = results[best_model_name]["model"]
y_pred_best = results[best_model_name]["y_pred"]
y_prob_best = best_model.predict_proba(X_test)[:, 1]

joblib.dump(best_model, "breast_cancer_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# ------------------ 8. Save Evaluation Metrics ------------------
metrics = {
    "accuracy": accuracy_score(y_test, y_pred_best),
    "precision": precision_score(y_test, y_pred_best),
    "recall": recall_score(y_test, y_pred_best),
    "f1": f1_score(y_test, y_pred_best),
    "roc_auc": roc_auc_score(y_test, y_prob_best)
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Best model, scaler, and metrics saved.")
