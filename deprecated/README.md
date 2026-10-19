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
| `src/sculptor/dep_v1/specanalysis.py` | v1 analysis module. The measurement helpers still in use were ported to `sculptor.measurements`; the `SpecFit`-based `analyze_*` routines are superseded by `sculptor.analysis` |
| `src/sculptor/dep_v1/speconed.py` | Bundled copy of `SpecOneD`; replaced by the standalone `speconed` package (differences: `TODO_speconed.md`) |
| `src/sculptor/data/passbands/` | Filter curves read only by the bundled `load_passband`; `speconed` ships its own, except three Euclid NISP bands (see `TODO_speconed.md`) |
| `sculptor_extensions/src/sculptor_extensions/my_extension.py` | Example plugin for the v1 GUI (lmfit models, mask presets) |
| `examples/dep_v1/` | Scripts and notebooks for the v1 API and GUI |

The GUI and `SpecFit`/`SpecModel` do not run as-is: they need `lmfit`
(not a dependency). `specanalysis.py` uses `np.trapz`, removed in numpy 2.4.

The test-only ground-truth copies in `tests/ground_truth/` still import
`sculptor.dep_v1.speconed` / `.specanalysis`; the `deprecated_dep_v1` fixture
in `tests/conftest.py` makes this folder importable under that name for
those tests only.
