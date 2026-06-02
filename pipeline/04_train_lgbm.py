"""
pipeline/04_train_lgbm.py — LGBM Tabular Expert Training (Cascade v5)

5-class LightGBM pada H4 bars dengan ATR-return labels.
Purged walk-forward CV (8 folds, purge=6 H4 bars = 1 hari).

Output:
  models/lgbm_tabular.pkl          — model final (retrain full training data)
  models/lgbm_feature_cols.json    — daftar fitur
  models/lgbm_oof_predictions.npz  — OOF probabilities (untuk 05d residual)
  models/cv_results.json           — metrik per fold

Jalankan:
  python pipeline/04_train_lgbm.py --all
"""

import argparse, json, sys, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score, log_loss

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR,
    LGBM_FEATURE_COLS, LGBM_PARAMS, LGBM_EARLY_STOPPING,
    LGBM_NUM_CLASSES, LGBM_LABEL_MAP,
    N_FOLDS, PURGE_GAP_H4,
    TRAIN_CUTOFF_DATE,
    apply_colab_settings,
    _colab_lgbm_cpu,
)

if _colab_lgbm_cpu():
    apply_colab_settings()
from pipeline.shared import build_purged_folds
from core.utils import setup_logger

logger = setup_logger("04_train_lgbm")


def load_all_coins() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Load H4 LGBM features + labels dari semua koin. Return X, y, bar timestamps."""
    all_X, all_y, all_idx = [], [], []

    for symbol in TRAINING_COINS:
        path = LABEL_DIR / f"{symbol}_h4_lgbm.parquet"
        if not path.exists():
            logger.warning(f"[{symbol}] H4 LGBM file tidak ditemukan — skip")
            continue

        df = pd.read_parquet(path)
        df = df[df.index < TRAIN_CUTOFF_DATE]

        if "label_lgbm" not in df.columns or len(df) < 50:
            logger.warning(f"[{symbol}] skip (rows={len(df)})")
            continue

        # Align features — missing cols di-fill 0
        X = np.zeros((len(df), len(LGBM_FEATURE_COLS)), dtype=np.float64)
        for idx, col in enumerate(LGBM_FEATURE_COLS):
            if col in df.columns:
                X[:, idx] = df[col].ffill().fillna(0).values

        # Map label -2..+2 ke 0..4
        raw_labels = df["label_lgbm"].values.astype(np.int64)
        y = np.array([LGBM_LABEL_MAP.get(int(l), 2) for l in raw_labels], dtype=np.int64)

        all_X.append(X)
        all_y.append(y)
        all_idx.append(df.index)
        logger.info(f"  [{symbol}] {len(df)} H4 bars loaded")

    if not all_X:
        raise RuntimeError("Tidak ada data — jalankan 03_engineer.py dulu")

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    t_all = pd.DatetimeIndex(np.concatenate([np.asarray(ix) for ix in all_idx]))
    order = np.argsort(t_all.values)
    X_all = X_all[order]
    y_all = y_all[order]
    t_all = t_all[order]

    # Distribusi label
    unique, counts = np.unique(y_all, return_counts=True)
    logger.info(f"Total: {len(X_all)} H4 bars | Label dist: " +
                " ".join(f"class{u}={c}" for u, c in zip(unique, counts)))
    return X_all, y_all, t_all


def _fit_lgbm(params: dict, X_tr, y_tr, X_val, y_val) -> lgb.Booster:
    """Native lgb.train() — tidak pakai LabelEncoder sklearn, aman untuk fold tanpa semua class."""
    p = {k: v for k, v in params.items() if k != "n_estimators"}
    num_boost_round = params.get("n_estimators", 500)
    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    try:
        return lgb.train(
            p, dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )
    except Exception as exc:
        if params.get("device_type") == "gpu":
            logger.warning(f"  LGBM GPU gagal ({exc}) — retry CPU")
            cpu = {**params, "device_type": "cpu", "n_jobs": 2}
            cpu.pop("gpu_platform_id", None)
            cpu.pop("gpu_device_id", None)
            return _fit_lgbm(cpu, X_tr, y_tr, X_val, y_val)
        raise


def train_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    fold_idx: int,
) -> tuple[lgb.Booster, float, float, float]:
    """Train satu fold, return booster + val metrics + train F1 (overfit check)."""
    booster    = _fit_lgbm(LGBM_PARAMS, X_tr, y_tr, X_val, y_val)
    prob_val   = booster.predict(X_val)                           # (n, 5)
    prob_tr    = booster.predict(X_tr)
    y_pred_val = np.argmax(prob_val, axis=1)
    y_pred_tr  = np.argmax(prob_tr,  axis=1)

    all_labels = list(range(LGBM_NUM_CLASSES))
    f1_val = f1_score(y_val, y_pred_val, average="macro", zero_division=0, labels=all_labels)
    f1_tr  = f1_score(y_tr,  y_pred_tr,  average="macro", zero_division=0, labels=all_labels)
    ll     = log_loss(y_val, prob_val, labels=all_labels)
    logger.info(f"  Fold {fold_idx}: val_F1={f1_val:.4f} train_F1={f1_tr:.4f}  "
                f"LogLoss={ll:.4f}  best_iter={booster.best_iteration}")
    return booster, f1_val, ll, f1_tr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args, _ = parser.parse_known_args()

    logger.info(f"LGBM device_type={LGBM_PARAMS.get('device_type', '?')}")
    logger.info("Loading data...")
    X, y, t = load_all_coins()

    # ── Purged Walk-Forward CV ─────────────────────────────────────────────────
    logger.info(f"Running {N_FOLDS}-fold purged CV...")
    folds = build_purged_folds(t, N_FOLDS, PURGE_GAP_H4)

    cv_results    = []
    oof_proba     = np.zeros((len(X), LGBM_NUM_CLASSES), dtype=np.float64)
    best_f1       = -1.0
    best_model    = None

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model, f1, ll, f1_tr = train_fold(X_tr, y_tr, X_val, y_val, fold_idx)
        oof_proba[val_idx] = model.predict(X_val)

        cv_results.append({
            "fold": fold_idx,
            "f1_macro": f1,
            "f1_macro_train": f1_tr,
            "f1_gap_train_minus_val": float(f1_tr - f1),
            "log_loss": ll,
            "n_train": len(tr_idx),
            "n_val": len(val_idx),
        })

        if f1 > best_f1:
            best_f1   = f1
            best_model = model

    mean_f1 = np.mean([r["f1_macro"] for r in cv_results])
    mean_ll = np.mean([r["log_loss"] for r in cv_results])
    logger.info(f"CV Mean F1={mean_f1:.4f}  Mean LogLoss={mean_ll:.4f}")

    # ── Final Retrain pada Semua Data ─────────────────────────────────────────
    logger.info("Final retrain on full training data...")
    p_final = {k: v for k, v in LGBM_PARAMS.items() if k != "n_estimators"}
    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        p_final, dtrain_full,
        num_boost_round=best_model.best_iteration or 500,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, MODEL_DIR / "lgbm_tabular.pkl")
    logger.info(f"Model saved: models/lgbm_tabular.pkl")

    with open(MODEL_DIR / "lgbm_feature_cols.json", "w") as f:
        json.dump(LGBM_FEATURE_COLS, f, indent=2)

    np.savez_compressed(
        MODEL_DIR / "lgbm_oof_predictions.npz",
        oof_proba=oof_proba,
        y_true=y,
        X=X.astype(np.float32),
    )
    logger.info("OOF predictions saved: models/lgbm_oof_predictions.npz")

    with open(MODEL_DIR / "lgbm_cv_results.json", "w") as f:
        json.dump({
            "mean_f1":    mean_f1,
            "mean_ll":    mean_ll,
            "n_features": len(LGBM_FEATURE_COLS),
            "n_classes":  LGBM_NUM_CLASSES,
            "folds":      cv_results,
        }, f, indent=2)

    # Feature importance top 20
    feat_imp = sorted(
        zip(LGBM_FEATURE_COLS, final_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    logger.info("Top 10 feature importance:")
    for name, imp in feat_imp[:10]:
        logger.info(f"  {name:<40} {imp:.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
