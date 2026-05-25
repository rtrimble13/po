# 14 — Tracking-error budget

## What this demonstrates

`tracking_error_limit` in the params file constrains
`sqrt((w − w_bench)' Σ (w − w_bench)) ≤ TE_max`. Useful for benchmark-aware
mandates where the portfolio must stay within a TE budget vs an index.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ, and `benchmark_weights` |
| [../data/params_mvo_tracking.toml](../data/params_mvo_tracking.toml) | `tracking_error_limit = 0.05` (5 %) |

## Run it

```bash
./14_tracking_error.sh
```

## Expected output (abridged)

```
  Expected Return  : 13.85 %
  Volatility       : 16.84 %
  Tracking Error   :  5.00 %    ← TE constraint is binding
  Active Share     : 25.64 %

  AAPL  0.4441
  MSFT  0.3923
  JPM   0.1276
  JNJ   0.0000
  XOM   0.0360
```

Note: TE plumes right up against the 5 % limit — the constraint is binding.

## What to notice

- Without a TE budget, plain MVO (example 01) puts 40 % in AAPL and MSFT
  and ignores JNJ/XOM entirely (TE ≈ 4.6 %). The TE-constrained portfolio
  has to walk back toward the benchmark when the budget is tight.
- The `benchmark_weights` field in [../data/assets.json](../data/assets.json)
  is the reference. If the field is missing the constraint can't be
  evaluated and the CLI errors out.

## Try next

- Drop `tracking_error_limit` to `0.02` — the optimiser will hug the
  benchmark much more tightly.
- Combine with a `[[mvo.constraints.groups]]` block to add sector caps on
  top of TE.
