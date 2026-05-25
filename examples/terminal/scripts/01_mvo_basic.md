# 01 — MVO basic

## What this demonstrates

A single mean-variance-optimal portfolio: `po mvo` solving
`min w'Σw − λ μ'w` subject to long-only, per-asset cap, fully invested.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | 5-asset universe with μ, Σ, market & benchmark weights, rf |
| [../data/params_mvo.toml](../data/params_mvo.toml) | λ = 2.5, bounds 0 ≤ w_i ≤ 0.40, budget = 1.0 |

## Run it

```bash
./01_mvo_basic.sh
```

This calls `po mvo` twice — once with `--output result.json --format json` to
save a machine-readable copy, and once without `--output` so the human-readable
console table is also captured under `var/01_mvo_basic/`.

## Expected output (abridged)

```
  Expected Return  : 13.6800 %
  Volatility       : 16.5118 %
  Sharpe Ratio     :  0.5862

  Ticker    Weight    Wt (%)  RC (%vol)
  AAPL      0.4000     40.00      7.830
  MSFT      0.4000     40.00      6.904
  JPM       0.2000     20.00      1.778
```

(JNJ and XOM are zeroed; pass `--show-zero` to see them in the table.)

## What to notice

- AAPL and MSFT both hit the **40 % cap** from `upper_bounds` — try the
  `--explain` flag (see [12_group_constraints.md](12_group_constraints.md))
  to print which constraints are active at the optimum.
- The risk-contribution column (`RC (%vol)`) tells you which holdings are
  driving total portfolio volatility, not just which have the biggest weights.
- The JSON output preserves every metric and the solver diagnostics
  (`kkt_residual`, `input_hash`, `params_hash`, `library_version`) for audit.

## Try next

- Increase `risk_aversion` in [../data/params_mvo.toml](../data/params_mvo.toml)
  to 5.0 — the portfolio shrinks toward the minimum-variance allocation.
- Raise `upper_bounds` to `[0.6, 0.6, 0.6, 0.6, 0.6]` to relax the cap.
