"""
core/overfitting.py — Deteksi sinyal overfitting dari artefak training & holdout.

Tidak memakai skor performa trading sebelum data/model ada; menganalisis:
  - Stabilitas CV antar fold
  - Gap train vs validation (LGBM F1, LSTM loss, Guardian F1)
  - OOF vs in-sample log-loss (LGBM)
  - Kalibrasi confidence (menang vs kalah prediksi OOF)
  - Holdout vs proxy training (jika holdout_results.json ada)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
    from sklearn.metrics import f1_score, log_loss
except ImportError:  # pragma: no cover
    joblib = None
    f1_score = log_loss = None

from config import (
    LGBM_NUM_CLASSES,
    LGBM_LONG_CLASSES,
    LGBM_SHORT_CLASSES,
    OVERFIT_CV_F1_STD_WARN,
    OVERFIT_LGBM_F1_GAP_WARN,
    OVERFIT_LGBM_F1_GAP_FAIL,
    OVERFIT_LSTM_LOSS_GAP_WARN,
    OVERFIT_LSTM_LOSS_GAP_FAIL,
    OVERFIT_OOF_LL_GAP_WARN,
    OVERFIT_CALIB_GAP_WARN,
    OVERFIT_HOLDOUT_WR_GAP_WARN,
)


Status = str  # PASS | WARN | FAIL | SKIP


@dataclass
class CheckResult:
    name: str
    status: Status
    metric: str
    threshold: str
    detail: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class OverfittingReport:
    generated_at: str
    checks: list[CheckResult] = field(default_factory=list)
    sections: dict[str, Any] = field(default_factory=dict)

    @property
    def overall(self) -> Status:
        order = {"FAIL": 3, "WARN": 2, "PASS": 1, "SKIP": 0}
        best = "SKIP"
        for c in self.checks:
            if order.get(c.status, 0) > order.get(best, 0):
                best = c.status
        return best if self.checks else "SKIP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "overall": self.overall,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "metric": c.metric,
                    "threshold": c.threshold,
                    "detail": c.detail,
                    "values": c.values,
                }
                for c in self.checks
            ],
            "sections": self.sections,
        }


def _status_high(value: float, warn: float, fail: float) -> Status:
    if value >= fail:
        return "FAIL"
    if value >= warn:
        return "WARN"
    return "PASS"


def _cv_instability(values: list[float], warn_std: float) -> CheckResult:
    if len(values) < 2:
        return CheckResult(
            "cv_fold_stability",
            "SKIP",
            "std/mean",
            f"std >= {warn_std:.2f}",
            "Kurang dari 2 fold",
            {"values": values},
        )
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    ratio = std / mean if abs(mean) > 1e-9 else std
    st = _status_high(ratio, warn_std, warn_std * 1.5)
    return CheckResult(
        "cv_fold_stability",
        st,
        "cv_std_over_mean",
        f"WARN>={warn_std:.2f} FAIL>={warn_std * 1.5:.2f}",
        f"mean={mean:.4f} std={std:.4f} ratio={ratio:.4f}",
        {"mean": mean, "std": std, "ratio": ratio, "per_fold": values},
    )


def _train_val_gap(
    train_vals: list[float],
    val_vals: list[float],
    metric_name: str,
    warn_gap: float,
    fail_gap: float,
    higher_is_better: bool = True,
) -> CheckResult:
    if not train_vals or not val_vals or len(train_vals) != len(val_vals):
        return CheckResult(
            f"{metric_name}_train_val_gap",
            "SKIP",
            "train - val",
            f"gap >= {warn_gap}",
            "Data fold train/val tidak lengkap",
        )
    gaps = []
    for tr, va in zip(train_vals, val_vals):
        gaps.append((tr - va) if higher_is_better else (va - tr))
    gap_mean = float(np.mean(gaps))
    st = _status_high(gap_mean, warn_gap, fail_gap)
    return CheckResult(
        f"{metric_name}_train_val_gap",
        st,
        "mean_gap",
        f"WARN>={warn_gap} FAIL>={fail_gap}",
        f"mean_gap={gap_mean:.4f} (positif = overfit ke arah train)",
        {"gaps": gaps, "train": train_vals, "val": val_vals},
    )


def analyze_lgbm_cv(cv_path: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    section: dict[str, Any] = {}
    if not cv_path.exists():
        checks.append(
            CheckResult("lgbm_cv", "SKIP", "-", "-", f"Tidak ada {cv_path.name}")
        )
        return checks, section

    with open(cv_path, encoding="utf-8") as f:
        data = json.load(f)

    folds = data.get("folds", [])
    val_f1 = [float(x["f1_macro"]) for x in folds if "f1_macro" in x]
    train_f1 = [float(x["f1_macro_train"]) for x in folds if "f1_macro_train" in x]
    val_ll = [float(x["log_loss"]) for x in folds if "log_loss" in x]

    section["mean_f1"] = data.get("mean_f1")
    section["mean_ll"] = data.get("mean_ll")
    section["folds"] = folds

    checks.append(_cv_instability(val_f1, OVERFIT_CV_F1_STD_WARN))
    if train_f1:
        checks.append(
            _train_val_gap(
                train_f1,
                val_f1,
                "lgbm_f1",
                OVERFIT_LGBM_F1_GAP_WARN,
                OVERFIT_LGBM_F1_GAP_FAIL,
                higher_is_better=True,
            )
        )
    else:
        checks.append(
            CheckResult(
                "lgbm_f1_train_val_gap",
                "SKIP",
                "f1 gap",
                "-",
                "Jalankan ulang 04_train_lgbm untuk f1_macro_train per fold",
            )
        )

    return checks, section


def analyze_lgbm_oof(model_dir: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    section: dict[str, Any] = {}
    oof_path = model_dir / "lgbm_oof_predictions.npz"
    model_path = model_dir / "lgbm_tabular.pkl"

    if not oof_path.exists():
        checks.append(
            CheckResult("lgbm_oof", "SKIP", "-", "-", "lgbm_oof_predictions.npz tidak ada")
        )
        return checks, section

    oof = np.load(oof_path)
    y_true = oof["y_true"].astype(np.int64)
    oof_proba = oof["oof_proba"].astype(np.float64)

    # OOF log-loss (hanya bar yang terisi OOF)
    mask = oof_proba.sum(axis=1) > 0
    if mask.sum() < 100:
        checks.append(
            CheckResult("lgbm_oof_coverage", "WARN", "oof_rows", ">50%",
                        f"Hanya {mask.sum()} bar OOF — CV mungkin tidak lengkap",
                        {"oof_rows": int(mask.sum()), "total": len(y_true)},
                        )
        )
    else:
        oof_ll = float(log_loss(y_true[mask], oof_proba[mask], labels=list(range(LGBM_NUM_CLASSES))))
        section["oof_log_loss"] = oof_ll

    # In-sample vs OOF gap
    if model_path.exists() and joblib is not None:
        model = joblib.load(model_path)
        insample_proba = model.predict_proba(oof["X"] if "X" in oof.files else None)
        # X may not be in npz — load from cv only oof
        if "X" not in oof.files:
            checks.append(
                CheckResult(
                    "lgbm_oof_vs_insample",
                    "SKIP",
                    "log_loss gap",
                    "-",
                    "Simpan X di lgbm_oof_predictions.npz (re-run 04) untuk gap in-sample",
                )
            )
        else:
            X = oof["X"]
            insample_proba = model.predict_proba(X)
            is_ll = float(
                log_loss(y_true, insample_proba, labels=list(range(LGBM_NUM_CLASSES)))
            )
            oof_ll_full = float(
                log_loss(
                    y_true[mask],
                    oof_proba[mask],
                    labels=list(range(LGBM_NUM_CLASSES)),
                )
            )
            # Positif = OOF lebih buruk (LL lebih tinggi) dari in-sample -> overfit
            gap = oof_ll_full - is_ll
            section["insample_log_loss"] = is_ll
            section["oof_log_loss"] = oof_ll_full
            section["ll_gap_oof_minus_insample"] = gap
            st = _status_high(gap, OVERFIT_OOF_LL_GAP_WARN, OVERFIT_OOF_LL_GAP_WARN * 2)
            checks.append(
                CheckResult(
                    "lgbm_oof_vs_insample",
                    st,
                    "oof_ll - insample_ll",
                    f"WARN>={OVERFIT_OOF_LL_GAP_WARN}",
                    f"in-sample LL={is_ll:.4f} OOF LL={oof_ll_full:.4f} gap={gap:.4f}",
                    dict(section),
                )
            )

    # Calibration: confidence on correct vs wrong (directional collapse)
    long_p = oof_proba[:, LGBM_LONG_CLASSES].sum(axis=1)
    short_p = oof_proba[:, LGBM_SHORT_CLASSES].sum(axis=1)
    conf = np.maximum(long_p, short_p)
    pred_dir = np.where(long_p >= short_p, 1, 0)  # 1 long, 0 short
    true_long = np.isin(y_true, LGBM_LONG_CLASSES).astype(int)
    true_short = np.isin(y_true, LGBM_SHORT_CLASSES).astype(int)
    true_dir = np.where(true_long == 1, 1, np.where(true_short == 1, 0, -1))

    trade_mask = true_dir >= 0
    if trade_mask.sum() > 50:
        correct = pred_dir[trade_mask] == true_dir[trade_mask]
        conf_t = conf[trade_mask]
        mean_conf_win = float(conf_t[correct].mean()) if correct.any() else 0.0
        mean_conf_loss = float(conf_t[~correct].mean()) if (~correct).any() else 0.0
        calib_gap = mean_conf_loss - mean_conf_win
        section["mean_conf_correct"] = mean_conf_win
        section["mean_conf_wrong"] = mean_conf_loss
        section["calibration_gap_wrong_minus_correct"] = calib_gap
        st = _status_high(calib_gap, OVERFIT_CALIB_GAP_WARN, OVERFIT_CALIB_GAP_WARN * 2)
        checks.append(
            CheckResult(
                "lgbm_calibration",
                st,
                "conf_wrong - conf_correct",
                f"WARN>={OVERFIT_CALIB_GAP_WARN} (v4: loss lebih tinggi conf)",
                f"conf_correct={mean_conf_win:.4f} conf_wrong={mean_conf_loss:.4f} gap={calib_gap:.4f}",
                section,
            )
        )

    return checks, section


def analyze_lstm_cv(meta_path: Path, cv_path: Path | None) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    section: dict[str, Any] = {}

    data = None
    if cv_path and cv_path.exists():
        with open(cv_path, encoding="utf-8") as f:
            data = json.load(f)
    elif meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)

    if not data:
        checks.append(
            CheckResult("lstm_cv", "SKIP", "-", "-", "lstm_cv_results.json / meta tidak ada")
        )
        return checks, section

    folds = data.get("folds", [])
    val_loss = [float(x["val_loss"]) for x in folds if "val_loss" in x]
    train_loss = [float(x["train_loss"]) for x in folds if "train_loss" in x]

    section["cv_mean_loss"] = data.get("cv_mean_loss")
    section["cv_std_loss"] = data.get("cv_std_loss")
    section["folds"] = folds

    if val_loss:
        checks.append(_cv_instability(val_loss, OVERFIT_CV_F1_STD_WARN))
    if train_loss and val_loss:
        checks.append(
            _train_val_gap(
                train_loss,
                val_loss,
                "lstm_loss",
                OVERFIT_LSTM_LOSS_GAP_WARN,
                OVERFIT_LSTM_LOSS_GAP_FAIL,
                higher_is_better=False,
            )
        )
    else:
        checks.append(
            CheckResult(
                "lstm_loss_train_val_gap",
                "SKIP",
                "loss gap",
                "-",
                "Jalankan ulang 05c untuk train_loss per fold",
            )
        )

    return checks, section


def analyze_guardian_cv(cv_path: Path) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    section: dict[str, Any] = {}
    if not cv_path.exists():
        checks.append(
            CheckResult("guardian_cv", "SKIP", "-", "-", "guardian_cv_results.json tidak ada")
        )
        return checks, section

    with open(cv_path, encoding="utf-8") as f:
        data = json.load(f)

    folds = data.get("folds", [])
    val_f1 = [float(x["f1_macro"]) for x in folds if "f1_macro" in x]
    train_f1 = [float(x["f1_macro_train"]) for x in folds if "f1_macro_train" in x]
    section["mean_f1"] = data.get("mean_f1")
    section["folds"] = folds

    if val_f1:
        checks.append(_cv_instability(val_f1, OVERFIT_CV_F1_STD_WARN))
    if train_f1 and val_f1:
        checks.append(
            _train_val_gap(
                train_f1,
                val_f1,
                "guardian_f1",
                OVERFIT_LGBM_F1_GAP_WARN,
                OVERFIT_LGBM_F1_GAP_FAIL,
            )
        )

    return checks, section


def analyze_holdout(holdout_path: Path, lgbm_cv: dict[str, Any]) -> tuple[list[CheckResult], dict[str, Any]]:
    checks: list[CheckResult] = []
    section: dict[str, Any] = {}
    if not holdout_path.exists():
        checks.append(
            CheckResult("holdout_vs_train", "SKIP", "-", "-", "holdout_results.json tidak ada")
        )
        return checks, section

    with open(holdout_path, encoding="utf-8") as f:
        data = json.load(f)

    mean_wr = float(data.get("mean_wr", 0))
    mean_pf = float(data.get("mean_pf", 0))
    section["holdout"] = {
        "mean_wr": mean_wr,
        "mean_pf": mean_pf,
        "mean_sharpe": data.get("mean_sharpe"),
        "total_pnl": data.get("total_pnl"),
        "n_coins": len(data.get("coins", [])),
    }

    # Proxy: F1 macro CV tidak langsung = WR, tapi WR holdout sangat rendah vs baseline acak = red flag
    cv_f1 = float(lgbm_cv.get("mean_f1", 0) or 0)
    section["lgbm_cv_mean_f1"] = cv_f1

    if mean_wr < 0.40 and cv_f1 > 0.35:
        checks.append(
            CheckResult(
                "holdout_wr_vs_cv_f1",
                "WARN",
                "WR holdout vs CV F1",
                f"WR<{0.40:.0%} & CV F1>{0.35:.2f}",
                f"Holdout WR={mean_wr:.2%} sementara LGBM CV F1={cv_f1:.4f} — kemungkinan tidak generalisasi ke PnL",
                section,
            )
        )
    elif mean_wr < 0.35:
        checks.append(
            CheckResult(
                "holdout_wr",
                "FAIL",
                "mean_wr",
                "<35%",
                f"Holdout WR={mean_wr:.2%} — sangat buruk OOS",
                section,
            )
        )
    else:
        checks.append(
            CheckResult(
                "holdout_wr",
                "PASS",
                "mean_wr",
                f">={1 - OVERFIT_HOLDOUT_WR_GAP_WARN:.0%}",
                f"Holdout WR={mean_wr:.2%} PF={mean_pf:.2f}",
                section,
            )
        )

    return checks, section


def build_report(
    model_dir: Path,
    report_dir: Path,
    holdout_run: Path | None = None,
) -> OverfittingReport:
    report = OverfittingReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    lgbm_checks, lgbm_sec = analyze_lgbm_cv(model_dir / "lgbm_cv_results.json")
    report.checks.extend(lgbm_checks)
    report.sections["lgbm_cv"] = lgbm_sec

    oof_checks, oof_sec = analyze_lgbm_oof(model_dir)
    report.checks.extend(oof_checks)
    report.sections["lgbm_oof"] = oof_sec

    lstm_checks, lstm_sec = analyze_lstm_cv(
        model_dir / "lstm_momentum_meta.json",
        model_dir / "lstm_cv_results.json",
    )
    report.checks.extend(lstm_checks)
    report.sections["lstm"] = lstm_sec

    g_checks, g_sec = analyze_guardian_cv(model_dir / "guardian_cv_results.json")
    report.checks.extend(g_checks)
    report.sections["guardian"] = g_sec

    holdout_path = None
    if holdout_run:
        holdout_path = holdout_run / "holdout_results.json"
        if not holdout_path.exists():
            holdout_path = holdout_run
    else:
        runs = sorted((model_dir / "runs").glob("holdout_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            holdout_path = runs[0] / "holdout_results.json"

    h_checks, h_sec = analyze_holdout(holdout_path or Path("_missing"), lgbm_sec)
    report.checks.extend(h_checks)
    report.sections["holdout"] = h_sec

    return report


def render_markdown(report: OverfittingReport) -> str:
    lines = [
        "# Cascade v5 — Laporan Deteksi Overfitting",
        "",
        f"**Generated:** {report.generated_at}  ",
        f"**Overall:** `{report.overall}`",
        "",
        "> Metrik ini mendeteksi *kemungkinan* overfitting pada model ML & holdout.  ",
        "> Bukan jaminan profit. Jalankan setelah `04`–`06` (dan `07` untuk holdout).",
        "",
        "## Ringkasan",
        "",
        "| Check | Status | Metrik | Ambang | Detail |",
        "|-------|--------|--------|--------|--------|",
    ]
    for c in report.checks:
        lines.append(
            f"| {c.name} | **{c.status}** | {c.metric} | {c.threshold} | {c.detail} |"
        )

    lines.extend(
        [
            "",
            "## Interpretasi status",
            "",
            "| Status | Arti |",
            "|--------|------|",
            "| **PASS** | Gap/stabilitas dalam batas wajar |",
            "| **WARN** | Sinyal overfit lemah — pantau, kurangi fitur/kompleksitas |",
            "| **FAIL** | Gap besar atau holdout sangat buruk — jangan deploy |",
            "| **SKIP** | Artefak belum ada — jalankan pipeline training dulu |",
            "",
            "## Sinyal yang diperiksa",
            "",
            "1. **CV fold instability** — std/mean metrik val antar fold tinggi",
            "2. **Train vs val gap** — LGBM F1 train >> val; LSTM train loss << val loss",
            "3. **OOF vs in-sample** — log-loss in-sample jauh lebih baik dari OOF",
            "4. **Kalibrasi** — confidence prediksi salah > confidence prediksi benar (masalah v4)",
            "5. **Holdout trading** — WR/PF OOS vs proxy CV",
            "",
            "## Tindakan jika WARN/FAIL",
            "",
            "- Kurangi `n_estimators` / depth LGBM; naikkan `min_child_samples`",
            "- Naikkan dropout / weight decay LSTM; kurangi epoch",
            "- Perketat Smart Entry Gate; cek label leakage (`tools/benchmark_plan.py`)",
            "- Ulang CV dengan `PURGE_GAP` lebih konservatif",
            "",
            "## Artefak",
            "",
            "- `models/lgbm_cv_results.json` — fold F1 train/val",
            "- `models/lstm_cv_results.json` — fold loss train/val",
            "- `models/guardian_cv_results.json`",
            "- `models/lgbm_oof_predictions.npz` — OOF + (opsional) `X` untuk gap in-sample",
            "- `models/runs/{run_id}/holdout_results.json`",
            "",
        ]
    )
    return "\n".join(lines)