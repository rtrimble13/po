# 07 — Target-volatility portfolio

## What this demonstrates

`po target-vol --target 0.15` returns the MVO portfolio whose realised
volatility ≈ 15 % annualised — useful when you have a risk budget rather
than a risk-aversion parameter.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo.toml](../data/params_mvo.toml) | bounds, budget |
| `--target 0.15` | annualised vol target |

## Run it

```bash
./07_target_vol.sh
```

## Expected output (abridged)

```
  Expected Return  : 15.0023 %
  Volatility       : 14.9925 %   ← ~ matches the 15% target
  Sharpe Ratio     :  0.5946
```

## What to notice

- Internally the solver runs a 1-D bisection on `risk_aversion` until the
  realised vol equals the target (within tolerance).
- If the target is **below** the min-variance vol, the solver returns the
  min-variance portfolio. If **above** the max-return vol, it returns
  the max-return corner.

## Try next

- Try `--target 0.08` (likely below min-variance vol) — the result will be
  identical to [05_min_variance.sh](05_min_variance.sh).
- Try `--target 0.25` (above the long-only frontier) — same as the
  high-λ corner of the frontier from
  [02_mvo_frontier.sh](02_mvo_frontier.sh).
