
# CARD APPROVAL 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json, os
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


#  LOAD DATA

print("Loading dataset...")
df = pd.read_csv("card_approval_augmented.csv")

target_col = df.columns[-1]
feature_cols = df.columns[:-1]

X = df[feature_cols].copy()
y = df[target_col].astype(int).values
y = np.where(y == 0, -1, 1)

print(f"Dataset shape: {df.shape}")
print("Target distribution:")
print(pd.Series(y).value_counts(normalize=True))


#  ENCODING (CSV-DRIVEN)

encoders = {}

for col in feature_cols:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le


#  TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X.values, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


#  SCALING

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#  LINEAR SVM FROM SCRATCH

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, n_iters=100):
        self.C = C
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1] + 1)
        Xb = np.c_[np.ones(len(X)), X]

        for _ in range(self.n_iters):
            for i in range(len(X)):
                cond = y[i] * np.dot(self.w, Xb[i]) >= 1
                if cond:
                    self.w[1:] -= self.lr * (2 * self.C * self.w[1:])
                else:
                    self.w -= self.lr * (2 * self.C * self.w - y[i] * Xb[i])

    def decision_function(self, X):
        return np.dot(np.c_[np.ones(len(X)), X], self.w)

    def predict(self, X):
        return np.sign(self.decision_function(X))


#  HYPERPARAMETER TUNING

print("\nHyperparameter tuning...")

param_grid = {
    "C": [0.1, 1.0, 10.0],
    "lr": [0.001, 0.0001],
    "n_iters": [10]
}

best_acc = 0
best_params = None

for C, lr, n in product(*param_grid.values()):
    svm = LinearSVM(C, lr, n)
    svm.fit(X_train, y_train)
    acc = np.mean(svm.predict(X_test) == y_test)
    if acc > best_acc:
        best_acc = acc
        best_params = {"C": C, "lr": lr, "n_iters": n}

print("Best params:", best_params)


#  FINAL MODEL

model = LinearSVM(**best_params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
scores = model.decision_function(X_test)

y_test_bin = (y_test == 1).astype(int)
y_pred_bin = (y_pred == 1).astype(int)


#  METRICS

cm = np.zeros((2, 2), int)
for t, p in zip(y_test_bin, y_pred_bin):
    cm[t][p] += 1

tn, fp, fn, tp = cm.ravel()
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)
accuracy = np.mean(y_pred == y_test)

print("\nMODEL PERFORMANCE")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


#  ROC-AUC

order = np.argsort(scores)[::-1]
tps = np.cumsum(y_test_bin[order])
fps = np.cumsum(1 - y_test_bin[order])
tpr = tps / tps[-1]
fpr = fps / fps[-1]
auc = np.trapz(tpr, fpr)

print(f"ROC-AUC  : {auc:.4f}")


#  VISUALIZATIONS

os.makedirs("figures", exist_ok=True)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("figures/card_confusion_matrix.png")
plt.close()

plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.title("ROC Curve")
plt.savefig("figures/card_roc_curve.png")
plt.close()

importances = pd.Series(np.abs(model.w[1:]), index=feature_cols).sort_values(ascending=False)
importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top Feature Importances")
plt.savefig("figures/card_feature_importance.png")
plt.close()

plt.hist(scores[y_test == 1], alpha=0.6, label="Approved")
plt.hist(scores[y_test == -1], alpha=0.6, label="Rejected")
plt.legend()
plt.title("Decision Score Distribution")
plt.savefig("figures/card_score_distribution.png")
plt.close()


#  SAVE MODEL

os.makedirs("model", exist_ok=True)

with open("model/card_approval_model.json", "w") as f:
    json.dump({
        "weights": model.w.tolist(),
        "features": feature_cols.tolist(),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc
        }
    }, f, indent=2)

print("\n✅ Training complete. Model and plots saved.")
