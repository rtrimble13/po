# 03 — Black-Litterman with views

## What this demonstrates

`po bl` blends the equilibrium prior (implied by market weights) with two
explicit views, then MVO-optimises on the posterior. `--show-model` prints
the prior, posterior, and per-view diagnostics.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ, **market_weights** (drives the prior) |
| [../data/params_bl.toml](../data/params_bl.toml) | τ = 0.05, two views, variance-confidence mode |

The two views (TOML):
1. Tech (AAPL + MSFT) outperforms JPM by 5 % (`confidence = 0.001`).
2. Energy (XOM) outperforms Healthcare (JNJ) by 2 % (`confidence = 0.002`).

## Run it

```bash
./03_bl_basic.sh
```

## Expected output (abridged)

```
"prior_returns":     [0.0645, 0.0557, 0.0364, 0.0219, 0.0310]
"posterior_returns": [0.0751, 0.0651, 0.0279, 0.0185, 0.0333]
```

Notice how the posterior μ for AAPL/MSFT lifts above the prior (consistent
with view 1) while JPM drops, and XOM lifts vs JNJ (view 2).

## What to notice

- `confidence_pct` in the diagnostics is the *relative* confidence of each
  view vs the prior — useful for sanity-checking that strong views are
  actually pulling the posterior.
- The posterior covariance Σ_BL adds the *parameter uncertainty* on top of
  Σ, so the BL portfolio tends to be slightly less aggressive than plain MVO
  with the same μ.
- `--show-model` is what you want when validating that the views are doing
  what you intended; drop it for a quiet run.

## Try next

- Switch `confidence_mode = "idzorek"` and set `confidence` to a percentage
  (e.g. 0.65) for the more interpretable Idzorek calibration.
- Tighten a view by lowering its `confidence` Ω value (smaller = stronger).
