# 10 — Returns CSV input + Ledoit-Wolf shrinkage

## What this demonstrates

The `--returns` flag tells `po` that `-d` is a time-series CSV (one column
per asset plus a leading date column) rather than a pre-computed μ/Σ file.
The CLI estimates both from the data, optionally shrinking Σ toward a
structured target via `--shrinkage`.

## Inputs

| File | Role |
|---|---|
| [../data/returns_daily.csv](../data/returns_daily.csv) | 252 trading days × 5 tickers, synthetic but reproducible |

Two runs:
1. `--shrinkage none` (plain sample covariance)
2. `--shrinkage ledoit-wolf` (exact Ledoit-Wolf shrunk Σ)

## Run it

```bash
./10_returns_shrinkage.sh
```

Outputs: `var/10_returns_shrinkage/mvo_sample.json` and
`mvo_ledoit_wolf.json`, plus a console table for the LW variant.

## Expected output (LW variant, abridged)

```
  Expected Return  : 29.95 %
  Volatility       : 16.89 %
  Sharpe Ratio     :  1.77

  AAPL  0.107
  JPM   0.893    ← concentrated because μ̂_JPM is high in this sample
```

## What to notice

- With short samples (252 days here) μ̂ is **noisy** — the optimiser may
  concentrate hard on whichever name happens to have the highest sample
  mean. Shrinkage on Σ helps a little but doesn't fix the μ problem on its
  own; in production you'd combine `--shrinkage ledoit-wolf` with a
  Black-Litterman or robust μ estimate (see
  [03_bl_basic.md](03_bl_basic.md)).
- Use `--periods-per-year 12` for monthly data, `52` for weekly, `252` for
  daily.
- Other `--shrinkage` modes: `linear` (manual δ via `--shrinkage-delta`),
  `oas` (Oracle Approximating Shrinkage).

## Try next

- Lengthen the synthetic series in
  [../data/returns_daily.csv](../data/returns_daily.csv) (or substitute a
  real returns file) and watch concentration ease.
- Pair with a params file that has tight `upper_bounds` to cap the
  concentration directly.
