# 11 — CSV market data input

## What this demonstrates

`po` accepts market data either as a single JSON file (see
[01_mvo_basic.md](01_mvo_basic.md)) or as a **two-file CSV pair**:
`assets.csv` (per-asset metadata) and `covariance.csv` (Σ as a labelled
matrix). The CLI auto-discovers `covariance.csv` next to `assets.csv`.

## Inputs

| File | Role |
|---|---|
| [../data/assets.csv](../data/assets.csv) | ticker, name, expected_return, market_cap, sector |
| [../data/covariance.csv](../data/covariance.csv) | square Σ with a `ticker` header row + column |
| [../data/params_mvo.toml](../data/params_mvo.toml) | bounds, budget, λ |

## Run it

```bash
./11_csv_inputs.sh
```

## What to notice

- The weights match [01_mvo_basic.md](01_mvo_basic.md) exactly — same data,
  different on-disk shape. Benchmark-relative metrics (Tracking Error,
  Information Ratio, Active Share, Beta) **drop out** because
  `assets.csv` / `covariance.csv` don't carry `benchmark_weights` — the
  JSON input does. If you need them via CSV, supply a separate
  `market_weights.csv` alongside the two main files.
- CSV is the path of least friction when your μ / Σ come out of a
  spreadsheet or an estimator script. JSON is preferred when you also want
  to ship `market_weights`, `benchmark_weights`, and `risk_free_rate` in a
  single file.

## Try next

- Replace the data in `assets.csv` / `covariance.csv` with your own
  estimates (keep the header rows). The same `po mvo` command will pick
  them up.
