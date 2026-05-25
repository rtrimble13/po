# 02 — MVO efficient frontier

## What this demonstrates

`po frontier` sweeps `risk_aversion` over `[min_risk_aversion,
max_risk_aversion]` and emits one row per point: λ, return, vol, Sharpe,
diversification, effective-N, and the full weight vector.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo.toml](../data/params_mvo.toml) | `frontier_points = 30`, λ swept 0.1 → 100 |

## Run it

```bash
./02_mvo_frontier.sh
```

Output: `var/02_mvo_frontier/frontier.csv` (30 rows + header).

## Expected output (first rows)

```
risk_aversion,expected_return,volatility,sharpe_ratio,...,AAPL,MSFT,JPM,JNJ,XOM
0.10000000,0.10171112,0.11797247,0.52309766,...,0.13560645,0.19021230,0.13444464,0.39958681,0.14014980
...
```

## What to notice

- At low λ (≈ 0.1) the solution is close to **min-variance**: high JNJ weight
  (low-vol asset). As λ grows, the optimiser shifts toward high-μ names.
- The CSV format is ready for plotting in a notebook or piping into
  another tool — each row is a self-contained portfolio.
- Tune the number of points via `frontier_points` in the params file.

## Try next

- Output JSON instead of CSV: `-o frontier.json -f json` — useful for
  programmatic consumption.
- Compare with [04_bl_frontier.md](04_bl_frontier.md) which builds a frontier
  on the Black-Litterman posterior.
