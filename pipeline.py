"""Tabular classification on an imbalanced dataset.

Preprocessing (Pandas + Scikit-Learn) -> imbalance handling (class weights,
SMOTE, undersampling) -> gradient boosting (XGBoost, CatBoost) -> evaluation
with Precision / Recall / F1 / PR-AUC and a business cost, not just accuracy.

Usage:
    python src/pipeline.py --data data/customers.csv --target churned --id-cols customer_id
"""
import argparse
import warnings
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42


# ----------------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------------
def build_preprocessor(num_cols, cat_cols):
    """Impute -> scale numerics, impute -> one-hot encode categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),  # keeps "was missing" signal
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])


def fill_cats(X, cat_cols):
    """CatBoost handles NaN numerics natively, but categoricals must be strings."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), "missing").astype(str)
    return X


# ----------------------------------------------------------------------------
# Models x imbalance strategies
# ----------------------------------------------------------------------------
def build_models(num_cols, cat_cols, pos_weight):
    """Return {name: sklearn/imblearn pipeline}. Name = '<model> | <strategy>'."""
    pre = lambda: build_preprocessor(num_cols, cat_cols)  # noqa: E731
    lr = lambda **kw: LogisticRegression(max_iter=2000, **kw)  # noqa: E731
    xgb = lambda **kw: XGBClassifier(  # noqa: E731
        n_estimators=250, learning_rate=0.05, max_depth=3, min_child_weight=5, reg_lambda=5,
        subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", random_state=SEED, n_jobs=-1, **kw)
    cb_fill = FunctionTransformer(partial(fill_cats, cat_cols=cat_cols))
    cb = lambda **kw: CatBoostClassifier(  # noqa: E731
        iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=5, cat_features=cat_cols,
        random_seed=SEED, verbose=0, **kw)

    models = {
        # --- baseline: ignores imbalance entirely
        "LogReg | none": Pipeline([("pre", pre()), ("clf", lr())]),
        # --- class weights
        "LogReg | class_weight": Pipeline([("pre", pre()), ("clf", lr(class_weight="balanced"))]),
        "XGBoost | none": Pipeline([("pre", pre()), ("clf", xgb())]),
        "XGBoost | class_weight": Pipeline([("pre", pre()), ("clf", xgb(scale_pos_weight=pos_weight))]),
        "CatBoost | none": Pipeline([("fill", cb_fill), ("clf", cb())]),
        "CatBoost | class_weight": Pipeline([("fill", cb_fill), ("clf", cb(auto_class_weights="Balanced"))]),
        # --- resampling (applied to the training fold only; imblearn pipelines guarantee that)
        "XGBoost | SMOTE": ImbPipeline([("pre", pre()), ("res", SMOTE(random_state=SEED)), ("clf", xgb())]),
        "XGBoost | undersample": ImbPipeline(
            [("pre", pre()), ("res", RandomUnderSampler(random_state=SEED)), ("clf", xgb())]),
    }
    return models


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def total_cost(y, pred, cost_fn, cost_fp):
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return fn * cost_fn + fp * cost_fp


def pick_threshold(y, p, metric, cost_fn, cost_fp):
    """Choose the decision threshold on the VALIDATION set (never on test)."""
    grid = np.linspace(0.01, 0.99, 99)
    if metric == "f1":
        scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
        return float(grid[int(np.argmax(scores))])
    costs = [total_cost(y, (p >= t).astype(int), cost_fn, cost_fp) for t in grid]
    return float(grid[int(np.argmin(costs))])


def evaluate(y, p, thr, cost_fn, cost_fp):
    pred = (p >= thr).astype(int)
    return {
        "threshold": round(thr, 2),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "business_cost": total_cost(y, pred, cost_fn, cost_fp),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/customers.csv")
    ap.add_argument("--target", default="churned")
    ap.add_argument("--id-cols", nargs="*", default=["customer_id"], help="columns to drop (identifiers)")
    ap.add_argument("--threshold-metric", choices=["f1", "cost"], default="f1")
    ap.add_argument("--cost-fn", type=float, default=100, help="cost of a missed positive (false negative)")
    ap.add_argument("--cost-fp", type=float, default=10, help="cost of a false alarm (false positive)")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- load + split (stratified so every split keeps the same class ratio)
    df = pd.read_csv(args.data)
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target] + [c for c in args.id_cols if c in df.columns])
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    pos_rate = y.mean()
    print(f"Rows: {len(df):,} | features: {X.shape[1]} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"Positive rate: {pos_rate:.2%} | missing cells: {X.isna().sum().sum():,}")
    print(f"'Always predict negative' accuracy: {1 - pos_rate:.2%}  <- why accuracy is misleading\n")

    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    models = build_models(num_cols, cat_cols, pos_weight)

    rows, fitted, test_probs = [], {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        p_val = model.predict_proba(X_val)[:, 1]
        thr = pick_threshold(y_val, p_val, args.threshold_metric, args.cost_fn, args.cost_fp)
        p_te = model.predict_proba(X_te)[:, 1]
        m_name, strategy = [s.strip() for s in name.split("|")]
        rows.append({"model": m_name, "imbalance_strategy": strategy, **evaluate(y_te, p_te, thr, args.cost_fn, args.cost_fp)})
        fitted[name], test_probs[name] = model, (p_te, thr)
        print(f"  done: {name}")

    res = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    res.to_csv(out / "results.csv", index=False)
    print(f"\nTest-set results (threshold tuned on validation by '{args.threshold_metric}'):")
    print(res.round(3).to_string(index=False))

    best = res.iloc[0]
    best_name = f"{best['model']} | {best['imbalance_strategy']}"
    print(f"\nBest by F1: {best_name}")

    # ---- plots
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (p, _) in test_probs.items():
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax.plot(rc, pr, label=f"{name} (AP={average_precision_score(y_te, p):.2f})", lw=2 if name == best_name else 1)
    ax.axhline(pos_rate, ls="--", c="gray", label=f"random ({pos_rate:.2f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (test set)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pr_curves.png", dpi=150)
    plt.close(fig)

    p_best, thr_best = test_probs[best_name]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_te, (p_best >= thr_best).astype(int), ax=ax, colorbar=False)
    ax.set_title(f"{best_name}\nthreshold={thr_best:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    clf = fitted[best_name][-1]
    if hasattr(clf, "feature_importances_"):
        names = (fitted[best_name][0].get_feature_names_out() if "pre" in fitted[best_name].named_steps
                 else clf.feature_names_)
        imp = pd.Series(clf.feature_importances_, index=names).sort_values().tail(15)
        fig, ax = plt.subplots(figsize=(6, 5))
        imp.plot.barh(ax=ax)
        ax.set_title(f"Top feature importances - {best_name}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)

    print(f"\nSaved results.csv and plots to {out}/")


if __name__ == "__main__":
    main()