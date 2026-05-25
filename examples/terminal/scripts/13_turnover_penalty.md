# 13 — L2 turnover penalty (rebalancing)

## What this demonstrates

`turnover_penalty` (κ) adds `κ * ‖w − w_current‖²` to the objective —
useful when rebalancing from an existing book and you want the new weights
to stay close to the old ones unless the new μ/Σ really demand a move.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo_turnover.toml](../data/params_mvo_turnover.toml) | `current_weights = [0.2]*5`, `turnover_penalty = 1.5` |

The script runs the optimiser twice — once with the penalty in the params
file (κ = 1.5) and once with `--turnover-penalty 0.0` to show how far the
optimal weights drift when nothing penalises trading.

## Run it

```bash
./13_turnover_penalty.sh
```

## Expected output (abridged)

```
With penalty (κ = 1.5):
  Turnover (1-way) : 5.46 %
  AAPL  0.231     ← still close to the 0.20 starting weight
  MSFT  0.224
  JPM   0.191
  JNJ   0.172
  XOM   0.183

No penalty (κ = 0):
  Turnover (1-way) : 40.00 %
  AAPL  0.400     ← full move to the cap
  MSFT  0.400
  JPM   0.200
  JNJ, XOM  0
```

## What to notice

- The `Turnover (1-way)` metric makes the cost/benefit trade-off explicit —
  use it to size κ (rule of thumb: pick κ such that the realised turnover
  matches your trading-cost budget).
- `current_weights` and `turnover_penalty` both live in the params file's
  `[mvo.constraints]` block. The CLI flag `--turnover-penalty` overrides
  whatever the file says.

## Try next

- Set `current_weights` to the previous month's optimum (i.e. chain
  optimisations into a walk-forward backtest).
- Try `--turnover-penalty 0.5` for a softer rebalance.
