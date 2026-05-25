# 16 — Output formats + notional dollar amounts

## What this demonstrates

The CLI writes three formats out of the box:

- **console** — pretty Unicode-bordered table (default when no `-o`).
- **JSON** — full result object with metrics, weights, and audit hashes.
- **CSV** — two CSV blocks (weights, then a key/value metrics table).

`--total-capital $X` (any subcommand that returns a portfolio) adds a
**Notional $** column to the console table — handy when the consumer
thinks in dollars rather than weights.

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_mvo.toml](../data/params_mvo.toml) | bounds, budget |

## Run it

```bash
./16_output_formats.sh
```

## Expected output (console, with `--total-capital 10000000`)

```
  Asset Weights
  Ticker    Name                     Weight   Wt (%)   Notional $
  AAPL      Apple Inc.               0.4000    40.00    4000000
  MSFT      Microsoft Corp.          0.4000    40.00    4000000
  JPM       JPMorgan Chase           0.2000    20.00    2000000
```

Generated files:

| File | Format |
|---|---|
| `var/16_output_formats/result.json` | JSON, indent=2 |
| `var/16_output_formats/result.csv`  | CSV (weights + metrics blocks) |

## What to notice

- `--format` can be set explicitly (`console|json|csv`) **or** inferred
  from the `-o` filename's extension. Useful when scripting:
  `-o result.json` is enough.
- `--json-indent` controls indentation (0 = single-line JSON, good for
  log streams).
- `--ascii` on `po` (global flag, before the subcommand) replaces Unicode
  box characters with plain ASCII — friendlier for Windows consoles or
  log capture.

## Try next

- `po --ascii mvo -d ... -p ...` — see the ASCII-only console table.
- Pipe JSON through `jq`:
  `po mvo -d ... -p ... -f json | jq '.metrics.sharpe_ratio'`.
