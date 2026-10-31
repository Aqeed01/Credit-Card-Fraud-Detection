"""Credit card fraud detection: leak-free evaluation, PR curves, cost-based threshold, drift check.

Run:  python fraud_pipeline.py [path/to/card_transdata.csv]
Writes figures and metrics.json to results/, and the gradient-boosting model to models/.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
# Illustrative costs: a missed fraud is 20x as costly as a false alarm.
# Change these to your own numbers; the chosen threshold moves with them.
COST_MISSED_FRAUD = 100.0
COST_FALSE_ALARM = 5.0

ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)

data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "card_transdata.csv"
df = pd.read_csv(data_path)  # "NULL" strings become NaN
df = df.drop(columns=["Transaction ID"])
df["online_order"] = df["online_order"].map({"No": 0, "Yes": 1})
rows_before = len(df)
df = df.dropna().reset_index(drop=True)
FEATURES = [c for c in df.columns if c != "fraud"]
X, y = df[FEATURES], df["fraud"].astype(int)

report = {
    "rows": int(len(df)),
    "rows_dropped_for_missing": int(rows_before - len(df)),
    "fraud_rate": round(float(y.mean()), 4),
}

# 70% train / 30% test, then 20% of train held out to choose thresholds. The scaler
# lives inside each pipeline, so it only ever sees training rows.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train)

# Do any test rows also appear in the training data?
h_train = set(pd.util.hash_pandas_object(X_train, index=False))
h_test = pd.util.hash_pandas_object(X_test, index=False)
report["test_rows_duplicated_in_train"] = int(h_test.isin(h_train).sum())

models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
    "KNN (k=3)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
    "Random forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED),
    "Gradient boosting": HistGradientBoostingClassifier(random_state=SEED),
}


def cost(y_true, pred):
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return COST_MISSED_FRAUD * fn + COST_FALSE_ALARM * fp


def best_threshold(y_true, score):
    grid = np.unique(np.quantile(score, np.linspace(0, 1, 801)))
    costs = [cost(y_true, score >= t) for t in grid]
    return float(grid[int(np.argmin(costs))])


results, scores, fitted = {}, {}, {}
for name, model in models.items():
    t0 = time.time()
    model.fit(X_fit, y_fit)
    val_score = model.predict_proba(X_val)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]
    thr = best_threshold(y_val, val_score)
    at_half = test_score >= 0.5
    at_thr = test_score >= thr
    results[name] = {
        "average_precision": round(float(average_precision_score(y_test, test_score)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, test_score)), 4),
        "at_0.5": {
            "precision": round(float(precision_score(y_test, at_half)), 4),
            "recall": round(float(recall_score(y_test, at_half)), 4),
            "f1": round(float(f1_score(y_test, at_half)), 4),
            "cost": float(cost(y_test, at_half)),
        },
        "cost_threshold": round(thr, 4),
        "at_cost_threshold": {
            "precision": round(float(precision_score(y_test, at_thr)), 4),
            "recall": round(float(recall_score(y_test, at_thr)), 4),
            "f1": round(float(f1_score(y_test, at_thr)), 4),
            "cost": float(cost(y_test, at_thr)),
        },
        "seconds": round(time.time() - t0, 1),
    }
    scores[name], fitted[name] = test_score, model
    print(name, results[name], flush=True)
report["models"] = results

# 3-fold stratified cross-validation on a 200k sample (average precision).
sample = X_train.sample(200_000, random_state=SEED)
cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
report["cross_validation_average_precision"] = {}
for name in ["Logistic regression", "Gradient boosting"]:
    s = cross_val_score(models[name], sample, y_train.loc[sample.index], cv=cv, scoring="average_precision", n_jobs=-1)
    report["cross_validation_average_precision"][name] = {"mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4)}

# Figures: precision-recall curves, confusion matrices at the cost threshold, cost vs threshold.
plt.figure(figsize=(6, 5))
for name, s in scores.items():
    p, r, _ = precision_recall_curve(y_test, s)
    plt.plot(r, p, label=f"{name} (AP {results[name]['average_precision']:.3f})")
plt.axhline(y.mean(), color="grey", ls=":", label="fraud rate")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-recall curves (test set)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "pr_curves.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, len(scores), figsize=(3.2 * len(scores), 3.2))
for ax, (name, s) in zip(axes, scores.items()):
    cm = confusion_matrix(y_test, s >= results[name]["cost_threshold"])  # (true, predicted)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=8, color="black" if v < cm.max() / 2 else "white")
    ax.set_xticks([0, 1], ["pred ok", "pred fraud"], fontsize=7)
    ax.set_yticks([0, 1], ["true ok", "true fraud"], fontsize=7)
    ax.set_title(name, fontsize=8)
plt.tight_layout()
plt.savefig(OUT / "confusion_matrices.png", dpi=150)
plt.close()

best = "Gradient boosting"
grid = np.linspace(0.01, 0.99, 99)
plt.figure(figsize=(6, 4))
plt.plot(grid, [cost(y_test, scores[best] >= t) for t in grid])
plt.axvline(0.5, color="grey", ls=":", label="default 0.5")
plt.axvline(results[best]["cost_threshold"], color="C3", ls="--", label="cost-based threshold")
plt.xlabel("Decision threshold")
plt.ylabel("Total cost on test set")
plt.title(f"{best}: cost vs threshold")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "threshold_cost.png", dpi=150)
plt.close()

# Permutation importance for the gradient-boosting model (drop in average precision).
idx = X_test.sample(30_000, random_state=SEED).index
imp = permutation_importance(fitted[best], X_test.loc[idx], y_test.loc[idx], scoring="average_precision",
                             n_repeats=3, random_state=SEED, n_jobs=-1)
order = np.argsort(imp.importances_mean)[::-1]
report["permutation_importance"] = {FEATURES[i]: round(float(imp.importances_mean[i]), 4) for i in order}
plt.figure(figsize=(6, 3.5))
plt.barh([FEATURES[i] for i in order][::-1], imp.importances_mean[order][::-1])
plt.xlabel("Drop in average precision when shuffled")
plt.title("Feature importance (gradient boosting)")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=150)
plt.close()


def psi(expected, actual, bins=10):
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


# Drift check: population stability index (PSI). Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 drifted.
# The shifts are simulated (two features scaled up, labels unchanged) to show what the check does and does not catch.
report["drift"] = {"psi_train_vs_test": {f: round(psi(X_train[f].to_numpy(), X_test[f].to_numpy()), 4) for f in FEATURES}}
for label, (d, r) in {"mild": (1.5, 1.3), "strong": (3.0, 2.0)}.items():
    shifted = X_test.copy()
    shifted["distance_from_home"] *= d
    shifted["ratio_to_median_purchase_price"] *= r
    report["drift"][f"simulated_{label}_shift"] = {
        "distance_from_home_x": d,
        "ratio_to_median_x": r,
        "psi": {f: round(psi(X_train[f].to_numpy(), shifted[f].to_numpy()), 4) for f in FEATURES[:3]},
        "average_precision": round(float(average_precision_score(y_test, fitted[best].predict_proba(shifted)[:, 1])), 4),
    }

joblib.dump({"model": fitted[best], "features": FEATURES, "threshold": results[best]["cost_threshold"]},
            ROOT / "models" / "fraud_model.joblib", compress=3)
(OUT / "metrics.json").write_text(json.dumps(report, indent=2))
print("done; see results/metrics.json")
