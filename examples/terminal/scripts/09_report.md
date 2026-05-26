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

`po report` runs `python -m nbconvert` to execute the notebook. The preferred
setup is the project venv created by the `python-venv` CMake target — it
already has `nbconvert`, `matplotlib`, `pandas`, and `ipykernel` plus
`portopt` itself:

```bash
cmake --build build --target python-venv
source .venv/bin/activate           # then re-run this script
```

`po report` resolves Python in this order:

1. `$PORTOPT_PYTHON` (explicit override)
2. `$VIRTUAL_ENV/bin/python` (active venv)
3. `python` on PATH

The example script auto-routes through `.venv/` if it exists but isn't
activated. If neither the venv nor a system `nbconvert` is found, the script
prints an installation hint and exits cleanly — `run_all.sh` skips this
example rather than failing the whole suite.

Manual install into an existing environment:

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
