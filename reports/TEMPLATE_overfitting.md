# Cascade v5 — Template Laporan Overfitting

> File ini template. Laporan aktual dihasilkan oleh:
>
> ```powershell
> python tools/overfitting_report.py
> ```

## Kapan menjalankan

| Tahap | Artefak yang harus ada |
|-------|-------------------------|
| Setelah `04_train_lgbm` | `models/lgbm_cv_results.json`, `models/lgbm_oof_predictions.npz` |
| Setelah `05c_train_momentum_expert` | `models/lstm_cv_results.json` |
| Setelah `06_train_guardian` | `models/guardian_cv_results.json` |
| Setelah `07_holdout_backtest` | `models/runs/{run_id}/holdout_results.json` |

## Checklist manual (selain tool)

- [ ] CV fold: val F1/loss tidak loncat antar fold (>25% std/mean = WARN)
- [ ] Train vs val: gap F1 LGBM < 8% (WARN), < 15% (FAIL)
- [ ] LSTM: val loss - train loss < 15% relatif (WARN)
- [ ] OOF log-loss tidak jauh lebih buruk dari in-sample (>0.15 gap = overfit)
- [ ] Confidence salah tidak lebih tinggi dari confidence benar (masalah v4)
- [ ] Holdout WR/PF masuk akal vs ekspektasi — bukan hanya CV F1 tinggi

## Output tool

| File | Format |
|------|--------|
| `reports/overfitting_*.md` | Ringkasan tabel + interpretasi |
| `reports/overfitting_*.json` | Machine-readable untuk CI/script |

## Ambang (edit di `config.py`)

- `OVERFIT_LGBM_F1_GAP_WARN` / `FAIL`
- `OVERFIT_LSTM_LOSS_GAP_WARN` / `FAIL`
- `OVERFIT_OOF_LL_GAP_WARN`
- `OVERFIT_CALIB_GAP_WARN`