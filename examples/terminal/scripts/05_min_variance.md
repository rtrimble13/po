# 05 — Minimum-variance portfolio

## What this demonstrates

`po min-variance` solves `min w'Σw` subject to bounds and budget — μ is
ignored entirely.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | Σ (μ unused) |
| [../data/params_mvo.toml](../data/params_mvo.toml) | bounds + budget (λ ignored) |

## Run it

```bash
./05_min_variance.sh
```

## Expected output (abridged)

```
  Volatility       : 11.7081 %    ← lower than basic MVO's 16.51 %
  Sharpe Ratio     :  0.4967

  Ticker    Weight
  AAPL      0.0973
  MSFT      0.1652
  JPM       0.1642
  JNJ       0.4000   ← lowest-vol name hits its cap
  XOM       0.1733
```

## What to notice

- The portfolio diversifies across all five names because the only objective
  is variance — there is no return signal to "chase".
- Effective-N (~3.94) is much higher than the MVO result (~2.78) since
  weight is spread across more assets.

## Try next

- Lower the cap on JNJ in `upper_bounds` to force the optimiser to spread
  weight further — vol will rise but diversification rises too.
