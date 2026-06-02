"""
benchmark_plan.py — Audit repo vs Cascade v5 plan (CLAUDE.md + config.py).

Jalankan:
  python tools/benchmark_plan.py
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
SKIP = "SKIP"

results: list[tuple[str, str, str]] = []


def record(cat: str, status: str, detail: str) -> None:
    results.append((cat, status, detail))


def check_files() -> None:
    required = [
        "config.py",
        "CLAUDE.md",
        "core/features.py",
        "core/models.py",
        "core/fusion.py",
        "core/evaluator.py",
        "core/utils.py",
        "pipeline/01_fetch.py",
        "pipeline/02_clean.py",
        "pipeline/03_engineer.py",
        "pipeline/04_train_lgbm.py",
        "pipeline/05a_momentum_labels.py",
        "pipeline/05b_build_sequences.py",
        "pipeline/05c_train_momentum_expert.py",
        "pipeline/05d_oof_residuals.py",
        "pipeline/06_train_guardian.py",
        "pipeline/07_holdout_backtest.py",
        "pipeline/shared.py",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    if missing:
        record("Struktur", FAIL, f"File hilang: {missing}")
    else:
        record("Struktur", PASS, f"{len(required)} file wajib ada")


def check_imports() -> None:
    mods = [
        "config",
        "core.evaluator",
        "core.fusion",
        "core.models",
        "core.features",
        "pipeline.shared",
    ]
    errs = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            errs.append(f"{m}: {e}")
    if errs:
        record("Import", FAIL, "; ".join(errs))
    else:
        record("Import", PASS, "Semua modul inti bisa di-import")


def check_config_constants() -> None:
    import config as c

    checks = [
        ("MODAL_PER_TRADE", c.MODAL_PER_TRADE, 5.0),
        ("TRAIN_CUTOFF year", c.TRAIN_CUTOFF_DATE.year, 2025),
        ("FUSION_LGBM_WEIGHT", c.FUSION_LGBM_WEIGHT, 0.55),
        ("FUSION_RESIDUAL_WEIGHT", c.FUSION_RESIDUAL_WEIGHT, 0.35),
        ("FUSION_EXHAUSTION_PENALTY", c.FUSION_EXHAUSTION_PENALTY, 0.25),
        ("ENTRY_FINAL_PROB_THR", c.ENTRY_FINAL_PROB_THR, 0.68),
        ("ENTRY_MOMENTUM_STR_THR", c.ENTRY_MOMENTUM_STR_THR, 0.55),
        ("ENTRY_EXHAUSTION_MAX", c.ENTRY_EXHAUSTION_MAX, 0.60),
        ("PURGE_GAP_BARS", c.PURGE_GAP_BARS, 24),
        ("LGBM_NUM_CLASSES", c.LGBM_NUM_CLASSES, 5),
    ]
    bad = [f"{name}={got} (harap {exp})" for name, got, exp in checks if got != exp]
    if bad:
        record("Konstanta", FAIL, ", ".join(bad))
    else:
        record("Konstanta", PASS, "Parameter inti sesuai rencana")


def check_models_code() -> None:
    src = (ROOT / "core/models.py").read_text(encoding="utf-8")
    for name in ("momentum_head", "exhaustion_head", "residual_head", "TemporalAttention"):
        if name not in src:
            record("AttentionLSTM", FAIL, f"{name} tidak ditemukan")
            return
    record("AttentionLSTM", PASS, "3 head + temporal attention ada")


def check_fusion_code() -> None:
    src = (ROOT / "core/fusion.py").read_text(encoding="utf-8")
    if "smart_entry_gate" not in src or "dynamic_fusion" not in src:
        record("Fusion", FAIL, "dynamic_fusion / smart_entry_gate hilang")
    elif "H1_CONFIRMATION_ENABLED" not in src:
        record("Fusion", WARN, "H1 confirmation gate tidak ter-wire di fusion.py")
    else:
        record("Fusion", PASS, "Dynamic fusion + smart entry gate")


def check_feature_split() -> None:
    import config as c

    n_lgbm = len(c.LGBM_FEATURE_COLS)
    n_lstm = len(c.LSTM_SEQUENCE_COLS)
    dropped = set(c.DROP_FEATURES)

    if n_lgbm == 56:
        record("Fitur LGBM", PASS, f"{n_lgbm} fitur tabular H4")
    elif 55 <= n_lgbm <= 60:
        record("Fitur LGBM", PASS, f"{n_lgbm} fitur (target 56)")
    else:
        record("Fitur LGBM", WARN, f"{n_lgbm} fitur (cek config vs CLAUDE.md)")

    if n_lstm == 10:
        record("Fitur LSTM", PASS, f"{n_lstm} fitur trajectory H4")
    else:
        record("Fitur LSTM", WARN, f"{n_lstm} fitur (target 10 per CLAUDE.md)")

    if dropped >= {"FVG_up", "FVG_down", "sell_volume"}:
        record("DROP_FEATURES", PASS, f"{len(dropped)} fitur di-drop")
    else:
        record("DROP_FEATURES", WARN, f"Set drop tidak lengkap: {dropped}")


def check_engineer_outputs() -> None:
    eng = (ROOT / "pipeline/03_engineer.py").read_text(encoding="utf-8")
    for name in ("_h4_lgbm", "_h4_lstm", "_h1_conf"):
        if name not in eng:
            record("Engineer output", FAIL, f"Output {name} tidak ada di 03_engineer")
            return
    if "_h1_lstm" in eng:
        record("Engineer output", WARN, "Masih referensi _h1_lstm di engineer")
    else:
        record("Engineer output", PASS, "h4_lgbm + h4_lstm + h1_conf")


def check_pipeline_paths() -> None:
    p06 = (ROOT / "pipeline/06_train_guardian.py").read_text(encoding="utf-8")
    p07 = (ROOT / "pipeline/07_holdout_backtest.py").read_text(encoding="utf-8")
    p05d = (ROOT / "pipeline/05d_oof_residuals.py").read_text(encoding="utf-8")

    issues = []
    if "h1_lstm" in p06:
        issues.append("06 masih pakai h1_lstm")
    if "h1_lstm" in p07:
        issues.append("07 masih pakai h1_lstm")
    if "h1_lstm" in p05d:
        issues.append("05d masih pakai h1_lstm")
    if "seq_len = 32" in p07:
        issues.append("07 hardcode seq_len=32")

    if issues:
        record("Path integrasi", FAIL, "; ".join(issues))
    else:
        record("Path integrasi", PASS, "06/07/05d path H4+H1 konsisten")

    if "load_coin_simulation_data" in p06 and "load_coin_simulation_data" in p07:
        record("Shared loader", PASS, "06 & 07 pakai load_coin_simulation_data")
    else:
        record("Shared loader", WARN, "Belum semua pipeline pakai shared loader")


def check_labels_anti_leakage() -> None:
    feat = (ROOT / "core/features.py").read_text(encoding="utf-8")
    if "shift(-horizon)" in feat and "backward" in feat.lower():
        record("Anti-leakage", PASS, "Label forward-only; fitur backward-looking")
    else:
        record("Anti-leakage", WARN, "Perlu review manual features.py")


def check_data_artifacts() -> None:
    import config as c

    n_proc = len(list(c.PROC_DIR.glob("*_h1_clean.parquet"))) if c.PROC_DIR.exists() else 0
    n_lbl = len(list(c.LABEL_DIR.glob("*_h4_lgbm.parquet"))) if c.LABEL_DIR.exists() else 0
    n_seq = len(list(c.SEQ_DIR.glob("*_sequences.npz"))) if c.SEQ_DIR.exists() else 0

    models = [
        c.MODEL_DIR / "lgbm_tabular.pkl",
        c.MODEL_DIR / "lstm_momentum_expert.pt",
        c.MODEL_DIR / "guardian_best.pkl",
    ]
    n_models = sum(1 for p in models if p.exists())

    if n_proc == 0:
        record("Data processed", SKIP, "Belum fetch/clean (0 h1_clean)")
    else:
        record("Data processed", PASS, f"{n_proc} h1_clean")

    if n_lbl == 0:
        record("Data labeled", SKIP, "Belum engineer (0 h4_lgbm)")
    else:
        record("Data labeled", PASS, f"{n_lbl} h4_lgbm")

    if n_seq == 0:
        record("Sequences", SKIP, "Belum 05a/05b")
    else:
        record("Sequences", PASS, f"{n_seq} sequences.npz")

    if n_models == 0:
        record("Model artifacts", SKIP, "Belum training (0/3 model utama)")
    elif n_models < 3:
        record("Model artifacts", WARN, f"{n_models}/3 model ada")
    else:
        record("Model artifacts", PASS, "3/3 model utama ada")


def check_claude_drift() -> None:
    claude = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    import config as c

    drift = []
    if "10 fitur trajectory" not in claude and "LSTM_SEQUENCE_COLS" not in claude:
        drift.append("CLAUDE.md tidak mendokumentasikan 10 fitur LSTM")
    if len(c.LSTM_SEQUENCE_COLS) != 10:
        drift.append(f"config LSTM count {len(c.LSTM_SEQUENCE_COLS)} != 10")
    if c.LGBM_NUM_CLASSES != 5:
        drift.append(f"LGBM_NUM_CLASSES={c.LGBM_NUM_CLASSES}")
    if c.LSTM_SEQ_LEN != 24:
        drift.append(f"LSTM_SEQ_LEN={c.LSTM_SEQ_LEN}")
    if "H4 PRIMARY" not in claude:
        drift.append("CLAUDE.md tidak menyebut H4 PRIMARY")
    if "h4_lgbm.parquet" not in claude or "h1_conf.parquet" not in claude:
        drift.append("output parquet v5 tidak didokumentasikan")
    if "55e525f4-0219-45ca-b213-2cb8ec512b5a" not in claude:
        drift.append("link Grok share utama (55e525f4) tidak ada di CLAUDE.md")
    if "8697395c-f55a-47d1-81ef-e9708a33a48a" not in claude:
        drift.append("link Grok share detail model (8697395c) tidak ada")
    if "Filosofi utama" not in claude:
        drift.append("section Filosofi utama hilang")
    if "flowchart TD" not in claude:
        drift.append("diagram mermaid alur hilang")
    if "Prinsip desain" not in claude and "anti-noise" not in claude:
        drift.append("section prinsip anti-noise hilang")
    if "Penjelasan Detail per Model" not in claude:
        drift.append("section penjelasan detail model hilang")

    if drift:
        record("Dokumen vs kode", WARN, "; ".join(drift))
    else:
        record("Dokumen vs kode", PASS, "CLAUDE.md selaras config")


def main() -> int:
    print("=" * 60)
    print("Cascade v5 Plan Benchmark")
    print(f"Root: {ROOT}")
    print("Acuan: CLAUDE.md + config.py + Grok share 55e525f4 (HTML/meta)")
    print("=" * 60)

    check_files()
    check_imports()
    check_config_constants()
    check_models_code()
    check_fusion_code()
    check_feature_split()
    check_engineer_outputs()
    check_pipeline_paths()
    check_labels_anti_leakage()
    check_data_artifacts()
    check_claude_drift()

    counts = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    for cat, status, detail in results:
        counts[status] = counts.get(status, 0) + 1
        print(f"[{status:4}] {cat:20} | {detail}")

    print("-" * 60)
    print(
        f"Ringkasan: PASS={counts[PASS]} FAIL={counts[FAIL]} "
        f"WARN={counts[WARN]} SKIP={counts[SKIP]}"
    )

    import config as c

    trained = all(
        (c.MODEL_DIR / name).exists()
        for name in (
            "lgbm_tabular.pkl",
            "lstm_momentum_expert.pt",
            "guardian_best.pkl",
        )
    )
    if not trained:
        print(
            "Training belum selesai (lgbm + lstm + guardian) — "
            "tidak ada skor performa; gunakan ringkasan PASS/FAIL di atas saja."
        )
    else:
        print(
            "Model utama ada. Metrik trading/F1: jalankan 07_holdout_backtest "
            "dan catat di EXPERIMENTS.md."
        )

    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())