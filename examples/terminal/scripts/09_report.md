# 09 — Jupyter diagnostic report

## What this demonstrates

`po report` runs the bundled [diagnostic notebook](../../../notebooks/diagnostic_template.ipynb)
end-to-end and emits:

- `diagnostic_report.ipynb` — executed notebook
- `diagnostic_report.html` — HTML version (via `nbconvert --to html`)
- `mvo_result.json`, `mvo_frontier.json`, and (if BL views are provided)
  `bl_result.json`, `bl_frontier.json`, `bl_model.json`
- `manifest.json` — records the inputs/outputs for reproducibility

## External dependency

`jupyter` and `nbconvert` must be on PATH. The script prints an installation
hint and exits cleanly if they're missing — `run_all.sh` will skip this
example rather than fail the whole suite.

```bash
pip install jupyter nbconvert matplotlib pandas
```

## Inputs

| File | Role |
|---|---|
| [../data/assets.json](../data/assets.json) | μ, Σ |
| [../data/params_bl.toml](../data/params_bl.toml) | drives both MVO and BL via `--method both` |

## Run it

```bash
./09_report.sh
```

## What to notice

- This is the only example that depends on Python/Jupyter — every other
  example uses only the compiled `po` binary.
- The notebook plots the efficient frontier, weight bars, and (when BL is
  enabled) the prior-vs-posterior return comparison.

## Try next

- Pass `--method mvo` to skip BL entirely.
- Point `--notebook` at a custom template to extend the diagnostic.
