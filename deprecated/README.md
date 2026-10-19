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
| `src/sculptor/model_evaluate_parameters.py` | `FitModel.evaluate_parameters` (pairwise corner plots), removed from `model.py`: pre-1.0 chainconsumer API, broken on 1.x, unused. Not a module of the old package, just the method's code |
| `src/sculptor/colors.py` | Vendored copy of Paul Tol's colour schemes; replaced by the `tol-colors` package (identical `rainbow_PuRd`) |
| `src/sculptor/data/passbands/` | Filter curves read only by the bundled `load_passband`; `speconed` ships its own, except three Euclid NISP bands (see `TODO_speconed.md`) |
| `sculptor_extensions/src/sculptor_extensions/qso_duplicates.py` | Five model functions removed from `qso.py` as duplicates (review §3); the file's docstring lists the maintained replacement for each. Not a module of the old package, just the functions' code |
| `sculptor_extensions/src/sculptor_extensions/qso_autofit_balmer_hol.py` | `Qso_autofit.Balmer_continuum_with_hol`, replaced by `qso_advanced_models.BalmerContinuumWithHOL` (different normalisation, see the file's docstring) |
| `sculptor_extensions/src/sculptor_extensions/my_extension.py` | Example plugin for the v1 GUI (lmfit models, mask presets) |
| `examples/dep_v1/` | Scripts and notebooks for the v1 API and GUI |
| `examples/test_saving.py`, `examples/test_resolution.py` | Interactive scratch scripts (`IPython.embed` breakpoints, missing input files, a non-existent `SpecOneD.broaden_by_resolution`). Replaced by `tests/test_save_load.py`, `tests/test_broadening.py` and `benchmarks/bench_broaden_spectrum.py` |

The GUI and `SpecFit`/`SpecModel` do not run as-is: they need `lmfit`
(not a dependency). `specanalysis.py` uses `np.trapz`, removed in numpy 2.4.

The test-only ground-truth copies in `tests/ground_truth/` still import
`sculptor.dep_v1.speconed`, `.specanalysis` and `sculptor.colors`; the
`deprecated_sculptor` fixture in `tests/conftest.py` makes this folder
importable under those names for those tests only.
