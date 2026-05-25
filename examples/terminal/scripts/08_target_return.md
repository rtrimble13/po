# 08 — Target-return portfolio

## What this demonstrates

`po target-return --target 0.10` returns the MVO portfolio whose expected
return ≈ 10 % annualised. The complement of
[07_target_vol.md](07_target_vol.md) — solve for vol given a return target.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo.toml](../data/params_mvo.toml) | bounds, budget |
| `--target 0.10` | annualised expected-return target |

## Run it

```bash
./08_target_return.sh
```

## Expected output (abridged)

```
  Expected Return  : 10.0003 %   ← ~ matches the 10% target
  Volatility       : 11.7900 %
  Sharpe Ratio     :  0.5231
```

## What to notice

- Same bisection-on-λ machinery as `target-vol`, just on a different metric.
- For target returns *outside* the achievable range, the solver clamps to
  the nearest reachable corner of the feasible set.

## Try next

- Pair this with `target-vol` to trace a few specific points on the
  frontier without sweeping the whole curve.
