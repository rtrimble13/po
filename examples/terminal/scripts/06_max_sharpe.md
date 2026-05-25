# 06 — Maximum-Sharpe (tangency) portfolio

## What this demonstrates

`po max-sharpe` finds the portfolio that maximises `(μ'w − rf) / sqrt(w'Σw)`.
The script passes `--risk-free-rate 0.04` explicitly so the choice of rf is
visible on the command line (the params file also sets it).

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo.toml](../data/params_mvo.toml) | bounds, budget |

## Run it

```bash
./06_max_sharpe.sh
```

## Expected output (abridged)

```
  Expected Return  : 16.7227 %
  Volatility       : 14.6395 %
  Sharpe Ratio     :  0.5949   ← highest realised Sharpe among examples 01–08
```

## What to notice

- Compared to the basic MVO (40/40/20 in just three names), the tangent
  portfolio holds **all five assets** — the bounded long-only problem still
  benefits from some diversification at the tangent.
- When all bounds are slack and short-selling is allowed, the CLI hits an
  **analytical fast path** (A3 tangent reformulation) — for this example
  the bounds bite, so the iterative solver runs.

## Try next

- Drop `risk_aversion` from the params file entirely — `max-sharpe` ignores
  it.
- Change `--risk-free-rate` to 0.0 to see how the tangency point shifts.
