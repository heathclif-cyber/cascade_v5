"""
pipeline/05c_train_momentum_expert.py — Train AttentionLSTM (Cascade v5)

Melatih AttentionLSTM dengan 3 output heads:
  1. momentum_strength  (regression, MSE loss)
  2. exhaustion_score   (regression, MSE loss)
  3. residual_correction (regression, MSE loss — dari OOF residual LGBM, lihat 05d)

Training:
  - Purged walk-forward CV (8 folds, purge=24 H1 bars)
  - DirectML GPU untuk training, CPU untuk inference
  - Input: (N, seq_len=LSTM_SEQ_LEN, n_features=10) pada bar H4

Jalankan:
  python pipeline/05c_train_momentum_expert.py --all
  python pipeline/05c_train_momentum_expert.py --run-id cascade_v5_test
"""

import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, SEQ_DIR, MODEL_DIR, LSTM_SEQUENCE_COLS,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_BATCH_SIZE, LSTM_LR, LSTM_WEIGHT_DECAY,
    N_FOLDS, PURGE_GAP_BARS,
    TRAIN_CUTOFF_DATE,
)
from core.models import AttentionLSTM
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05c_train_momentum_expert")


def load_all_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load sequence data dari semua koin.
    Returns: X_seq, y_momentum, y_exhaustion, y_residual
    """
    all_X, all_mom, all_exh, all_res = [], [], [], []

    # Load OOF residuals jika ada
    oof_path = MODEL_DIR / "lgbm_oof_residuals.npz"
    oof_data = np.load(oof_path) if oof_path.exists() else None
    if oof_data is None:
        logger.warning("OOF residuals tidak ditemukan — residual head dilatih dengan zeros. "
                       "Jalankan 05d_oof_residuals.py setelah 04_train_lgbm.py selesai.")

    for symbol in TRAINING_COINS:
        seq_path = SEQ_DIR / f"{symbol}_sequences.npz"
        if not seq_path.exists():
            logger.warning(f"[{symbol}] sequences.npz tidak ditemukan — skip")
            continue

        data = np.load(seq_path, allow_pickle=True)
        X    = data["X_seq"]                      # (N, seq_len, n_features)
        mom  = data["momentum_labels"]             # (N,)
        exh  = data["exhaustion_labels"]           # (N,)

        # Residual: ambil dari OOF jika ada, otherwise zeros
        if oof_data is not None and symbol in str(oof_data.files):
            res = oof_data.get(f"{symbol}_residual", np.zeros(len(X)))
        else:
            res = np.zeros(len(X), dtype=np.float32)

        all_X.append(X)
        all_mom.append(mom)
        all_exh.append(exh)
        all_res.append(res)
        logger.info(f"  [{symbol}] {len(X)} sequences loaded")

    if not all_X:
        raise RuntimeError("Tidak ada sequence data — jalankan 05b dulu")

    X_all   = np.concatenate(all_X,   axis=0).astype(np.float32)
    mom_all = np.concatenate(all_mom, axis=0).astype(np.float32)
    exh_all = np.concatenate(all_exh, axis=0).astype(np.float32)
    res_all = np.concatenate(all_res, axis=0).astype(np.float32)

    logger.info(f"Total sequences: {len(X_all)} | shape: {X_all.shape}")
    logger.info(f"Momentum mean={mom_all.mean():.3f}  Exhaustion mean={exh_all.mean():.3f}")
    return X_all, mom_all, exh_all, res_all


class MultiHeadLoss(nn.Module):
    """Combined loss untuk 3 regression heads."""
    def __init__(self, w_mom=1.0, w_exh=1.0, w_res=0.5):
        super().__init__()
        self.w_mom = w_mom
        self.w_exh = w_exh
        self.w_res = w_res
        self.mse   = nn.MSELoss()

    def forward(self, pred_mom, pred_exh, pred_res, tgt_mom, tgt_exh, tgt_res):
        return (
            self.w_mom * self.mse(pred_mom.squeeze(-1), tgt_mom)
            + self.w_exh * self.mse(pred_exh.squeeze(-1), tgt_exh)
            + self.w_res * self.mse(pred_res.squeeze(-1), tgt_res)
        )


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, mom_b, exh_b, res_b in loader:
        X_b   = X_b.to(device)
        mom_b = mom_b.to(device)
        exh_b = exh_b.to(device)
        res_b = res_b.to(device)

        optimizer.zero_grad()
        p_mom, p_exh, p_res = model(X_b)
        loss = criterion(p_mom, p_exh, p_res, mom_b, exh_b, res_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_b, mom_b, exh_b, res_b in loader:
            X_b   = X_b.to(device)
            mom_b = mom_b.to(device)
            exh_b = exh_b.to(device)
            res_b = res_b.to(device)
            p_mom, p_exh, p_res = model(X_b)
            total_loss += criterion(p_mom, p_exh, p_res, mom_b, exh_b, res_b).item()
    return total_loss / len(loader)


def train_single(
    X_tr, mom_tr, exh_tr, res_tr,
    X_val, mom_val, exh_val, res_val,
    n_features: int,
    run_id: str,
    fold_idx: int,
    device,
) -> tuple[AttentionLSTM, float, float, float]:

    model = AttentionLSTM(n_features, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=LSTM_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = MultiHeadLoss()

    tr_ds  = TensorDataset(
        torch.tensor(X_tr),   torch.tensor(mom_tr),
        torch.tensor(exh_tr), torch.tensor(res_tr),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val),  torch.tensor(mom_val),
        torch.tensor(exh_val), torch.tensor(res_val),
    )
    tr_loader  = DataLoader(tr_ds,  batch_size=LSTM_BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False)

    best_val    = float("inf")
    best_train  = float("inf")
    patience_ct = 0
    best_state  = None

    for epoch in range(1, LSTM_EPOCHS + 1):
        tr_loss  = train_epoch(model, tr_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val    = val_loss
            best_train  = tr_loss
            patience_ct = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ct += 1
            if patience_ct >= LSTM_PATIENCE:
                logger.info(
                    f"    Fold {fold_idx} early stop epoch {epoch} | "
                    f"best_val={best_val:.4f} train={best_train:.4f}"
                )
                break

        if epoch % 10 == 0:
            logger.info(f"    Fold {fold_idx} epoch {epoch:3d} | tr={tr_loss:.4f} val={val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val, best_train, float(best_val - best_train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    action="store_true")
    parser.add_argument("--run-id", default="cascade_v5")
    args = parser.parse_args()

    logger.info("Loading sequence data...")
    X, mom, exh, res = load_all_sequences()
    n_features = X.shape[2]

    device = get_lstm_device()
    logger.info(f"Training device: {device}")

    # ── Purged CV ──────────────────────────────────────────────────────────────
    t = np.arange(len(X))
    folds = build_purged_folds(t, N_FOLDS, PURGE_GAP_BARS)

    cv_folds = []
    best_fold_loss = float("inf")
    best_fold_model = None

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, 1):
        logger.info(f"Fold {fold_idx}/{N_FOLDS} — train={len(tr_idx)} val={len(val_idx)}")
        model, val_loss, train_loss, loss_gap = train_single(
            X[tr_idx], mom[tr_idx], exh[tr_idx], res[tr_idx],
            X[val_idx], mom[val_idx], exh[val_idx], res[val_idx],
            n_features, args.run_id, fold_idx, device,
        )
        cv_folds.append({
            "fold": fold_idx,
            "val_loss": float(val_loss),
            "train_loss": float(train_loss),
            "loss_gap_val_minus_train": float(loss_gap),
            "n_train": int(len(tr_idx)),
            "n_val": int(len(val_idx)),
        })
        if val_loss < best_fold_loss:
            best_fold_loss  = val_loss
            best_fold_model = model

    cv_losses = [f["val_loss"] for f in cv_folds]
    logger.info(f"CV Mean Loss: {np.mean(cv_losses):.4f} ± {np.std(cv_losses):.4f}")

    # ── Final Retrain ──────────────────────────────────────────────────────────
    logger.info("Final retrain on full data...")
    final_model, _, _, _ = train_single(
        X, mom, exh, res,
        X[-len(X)//8:], mom[-len(X)//8:], exh[-len(X)//8:], res[-len(X)//8:],
        n_features, args.run_id, 0, device,
    )

    # Save CPU-compatible
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model_cpu = final_model.cpu()
    torch.save(final_model_cpu.state_dict(), MODEL_DIR / "lstm_momentum_expert.pt")
    logger.info("Model saved: models/lstm_momentum_expert.pt")

    # Save scaler (ambil dari sequences)
    scaler_path = list((SEQ_DIR).glob("*_seq_scaler.pkl"))
    if scaler_path:
        import shutil
        shutil.copy(scaler_path[0], MODEL_DIR / "lstm_seq_scaler.pkl")

    # Save meta
    cv_payload = {
        "run_id": args.run_id,
        "n_features": n_features,
        "seq_len": LSTM_SEQ_LEN,
        "cv_mean_loss": float(np.mean(cv_losses)),
        "cv_std_loss": float(np.std(cv_losses)),
        "folds": cv_folds,
        "feature_cols": LSTM_SEQUENCE_COLS,
    }
    with open(MODEL_DIR / "lstm_cv_results.json", "w") as f:
        json.dump(cv_payload, f, indent=2)
    with open(MODEL_DIR / "lstm_momentum_meta.json", "w") as f:
        json.dump(cv_payload, f, indent=2)
    logger.info("CV saved: models/lstm_cv_results.json")
    logger.info("Done.")


if __name__ == "__main__":
    main()
