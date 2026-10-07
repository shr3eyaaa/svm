import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("winequality-red.csv")
print("Missing values:", df.isna().sum().sum(), "| Duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
# Binary target: quality >= 6 -> "good" (1), else 0. Labels are never noised.
X = df.drop(columns="quality").values
y = (df["quality"] >= 6).astype(int).values
print("Shape after cleaning:", X.shape, "| good-wine share: %.2f" % y.mean())

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(X_tr)           # fit on clean train only
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

grid = GridSearchCV(SVC(), {"kernel": ["linear", "rbf"],
                            "C": [0.1, 1, 10, 100],
                            "gamma": ["scale", 0.01, 0.1]},
                    cv=5, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
best = grid.best_params_
print("Best params:", best, "| CV F1: %.3f" % grid.best_score_)

def metrics(model, X, y):
    p = model.predict(X)
    return dict(accuracy=accuracy_score(y, p), precision=precision_score(y, p),
                recall=recall_score(y, p), f1=f1_score(y, p))

base = metrics(SVC(**best).fit(X_tr, y_tr), X_te, y_te)
print("Clean test metrics:", {k: round(v, 3) for k, v in base.items()})

levels = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
N_REPEAT = 10
rows = []
for s in levels:
    for r in range(N_REPEAT):
        rng = np.random.default_rng(r)
        Xtr_n = X_tr + rng.normal(0, s, X_tr.shape)
        Xte_n = X_te + rng.normal(0, s, X_te.shape)
        m = SVC(**best).fit(Xtr_n, y_tr)                     # retrain at every level
        rows.append(dict(noise=s, rep=r, **metrics(m, Xte_n, y_te)))
res = pd.DataFrame(rows)
summary = res.groupby("noise")[["accuracy", "precision", "recall", "f1"]].agg(["mean", "std"])
mean = res.groupby("noise")[["accuracy", "precision", "recall", "f1"]].mean()
std = res.groupby("noise")[["accuracy", "precision", "recall", "f1"]].std()
mean.round(4).to_csv("noise_results.csv")
print("\n", mean.round(3))

acc = mean["accuracy"]
drop = acc.iloc[0] - acc
breakdown = next((s for s in levels[1:] if drop[s] > 0.02 and drop[s] > 2 * std["accuracy"][s]), None)
steps = acc.diff().dropna()
steepest = steps.idxmin()
print("\nBreakdown point (acc drop > 2 pts): noise std =", breakdown)
print("Steepest single drop ends at noise std =", steepest, "(%.3f)" % steps[steepest])

fig, ax = plt.subplots(figsize=(9, 5.5))
for c, mk in zip(["accuracy", "precision", "recall", "f1"], "osD^"):
    ax.plot(mean.index, mean[c], marker=mk, label=c.capitalize())
    ax.fill_between(mean.index, mean[c] - std[c], mean[c] + std[c], alpha=0.12)
if breakdown is not None:
    ax.axvline(breakdown, color="red", ls="--", lw=1.5)
    ax.axvspan(breakdown, max(levels), color="red", alpha=0.06)
    ax.annotate(f"Breakdown ≈ {breakdown}", (breakdown, ax.get_ylim()[0] + 0.02),
                color="red", rotation=90, va="bottom", ha="right")
ax.set_xlabel("Gaussian noise std (in units of feature std dev)")
ax.set_ylabel("Test score (mean ± std, 10 runs)")
ax.set_title("SVM on Red Wine Quality: performance vs. feature noise")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.savefig("svm_noise_plot.png", dpi=150)
