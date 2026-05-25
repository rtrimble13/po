# 04 — Black-Litterman efficient frontier

## What this demonstrates

Same as [02_mvo_frontier.md](02_mvo_frontier.md), but the sweep is run on
the BL posterior `(μ_BL, Σ + Σ_BL)` instead of the raw sample μ, Σ.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ, market weights |
| [../data/params_bl.toml](../data/params_bl.toml) | τ, views |

## Run it

```bash
./04_bl_frontier.sh
```

Output: `var/04_bl_frontier/bl_frontier.csv`.

## What to notice

- The BL frontier sits *between* the prior-equilibrium frontier and the
  view-implied frontier — that's the whole point of BL.
- The frontier still uses the same MVO constraints (bounds, budget) from the
  `[mvo.constraints]` block in `params_bl.toml`.

## Try next

- Run [02_mvo_frontier.sh](02_mvo_frontier.sh) and this one side-by-side and
  load both CSVs into a notebook to plot the two efficient frontiers
  together.
