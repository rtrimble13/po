# 12 — Sector group constraints

## What this demonstrates

Linear group constraints `lower ≤ Gw ≤ upper` capture sector caps, regional
limits, or any other linear combination of weights. They're declared in the
parameters file under `[[mvo.constraints.groups]]`.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo_groups.toml](../data/params_mvo_groups.toml) | two group caps: Tech ≤ 50 %, Energy ≤ 15 % |

The `--explain` flag is passed so the CLI reports which bounds and
constraints are active at the optimum.

## Run it

```bash
./12_group_constraints.sh
```

## Expected output (abridged)

```
  Asset Weights
  AAPL    0.4578     45.78
  MSFT    0.0423      4.23
  JPM     0.4999     49.99

  Active Constraints
  Lower bound active: JNJ XOM
```

The Tech group constraint pushes MSFT down sharply (AAPL + MSFT just under
the 50 % cap together); JNJ and XOM bottom out at zero.

## What to notice

- The CLI treats groups as **soft** by default via an augmented-Lagrangian
  penalty (set `hard_group_constraints = true` in the params file for the
  strict KKT version — slower but exact).
- `coefficients` indexes into the asset list in the **order they appear**
  in `assets.json`. Get the order wrong and the constraint silently
  mis-targets.

## Try next

- Add a third group `Financials >= 10%` to force JPM in.
- Toggle `hard_group_constraints = true` and compare the iteration count
  (`iters=` in the output banner).
