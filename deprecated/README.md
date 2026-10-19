# Deprecated code

Old code kept for reference only. It is **not** part of the installed
`sculptor` / `sculptor_extensions` packages, is excluded from `ruff`, and is
not collected by `pytest`. Nothing outside this folder may import from it.

Files keep their original repository path below `deprecated/`, so
`deprecated/src/sculptor/dep_v1/specfit.py` used to be
`src/sculptor/dep_v1/specfit.py`. Use `git log --follow` on a file to see its
history from before the move.

## Contents

| Path | What it was |
|---|---|
| `src/sculptor/dep_v1/specfitgui.py`, `specmodelwidget.py`, `masksmodels.py`, `menu_dialogs.py`, `specfitcanvas.py`, `specmodelcanvas.py` | PyQt5 GUI of sculptor v1 |
| `src/sculptor/dep_v1/scripts/` | `run_sculptor` GUI launcher |
| `src/sculptor/dep_v1/specfit.py`, `specmodel.py` | lmfit-based v1 fitting classes (`SpecFit`, `SpecModel`) the GUI was built on; replaced by `sculptor.model.FitModel` / `sculptor.component.FitComponent` |
| `sculptor_extensions/src/sculptor_extensions/my_extension.py` | Example plugin for the v1 GUI (lmfit models, mask presets) |
| `examples/dep_v1/` | Scripts and notebooks for the v1 API and GUI |

None of this runs as-is: it needs `lmfit` (not a dependency) and `PyQt5`.
