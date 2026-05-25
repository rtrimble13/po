# 15 — Dollar-neutral long/short

## What this demonstrates

Setting `budget = 0` lets long weights cancel short weights — the sum is
zero, so the book is "dollar-neutral". `gross_exposure_limit = 2.0` caps the
sum of absolute weights (gross leverage). This is the building block for
130/30 and pure market-neutral mandates.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo_long_short.toml](../data/params_mvo_long_short.toml) | `budget=0`, `lower=-0.5`, `upper=0.5`, `gross_exposure_limit=2.0`, `allow_short_selling=true` |

The script also passes `--show-zero --explain` so the table includes every
asset (no near-zero suppression) and the active-constraint diagnostics are
printed.

## Run it

```bash
./15_long_short.sh
```

## Expected output (abridged)

```
  Ticker    Weight    Wt (%)
  AAPL      0.5000     50.00     ← upper bound active (long)
  MSFT      0.5000     50.00     ← upper bound active (long)
  JPM      -0.0676     -6.76
  JNJ      -0.5000    -50.00     ← lower bound active (short)
  XOM      -0.4324    -43.24

  Active Constraints
  Lower bound active: JNJ
  Upper bound active: AAPL MSFT
```

Sum of weights ≈ 0 (dollar-neutral); sum of |weights| ≈ 2.0 (the gross
exposure limit binds).

## What to notice

- Negative weights make sense in the JSON output too (`weight: -0.5`) —
  consumers should expect them when `allow_short_selling = true`.
- A 130/30 portfolio is the same idea with `budget = 1.0`,
  `gross_exposure_limit = 1.6`, and asymmetric bounds.
- Sharpe is much lower here than in [01_mvo_basic.md](01_mvo_basic.md)
  because the dollar-neutral leg is taking risk it can't fund from the rf
  rate.

## Try next

- Switch to a 130/30 setup: `budget=1.0`, `lower=-0.3`, `upper=0.3`,
  `gross_exposure_limit=1.6`.
- Lower `gross_exposure_limit` to 1.2 — the optimiser will deleverage.
