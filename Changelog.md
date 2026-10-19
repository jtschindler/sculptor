# Changelog

## 2026-10-19

### Saving and loading (`sculptor.persistence`, new)
- The three copies of the pickle `save`/`load` code in `FitParameter`,
  `FitComponent` and `FitModel` are replaced by one `PickleMixin`.
- **Signature change:** all three now use `save(name, save_dir,
  prefix=None)` / `load(name, load_dir, prefix=None)` (`FitModel`'s old
  order). `FitParameter`/`FitComponent.save` used to be
  `save(save_dir, save_prefix=None)` with the name taken from `self.name`;
  `load_prefix`/`save_prefix` are now `prefix`.
- `load` raises `TypeError` if the file holds a different class.

### `sculptor.prior`
- Fixed: `sample(seed=...)` ignored the seed (an unseeded generator was
  always created first). `sample()` now lives once in the `Prior` base
  class: it uses `rng` if given, else `np.random.default_rng(seed)`;
  passing both raises `ValueError`. Subclasses only implement `_draw`.
- Removed the dead commented-out `save`/`load` block.
- `UniformPrior.logprior`/`evaluate` return floats (`0.0`) instead of `0`.
- New `tests/test_prior.py`.

### `sculptor.component`: pickling fix
- Fixed: a `FitComponent` with a `param_mapping` (every component after
  `create_params()`) could not be unpickled. The mapping is a dict
  subclass that notifies its owner on mutation; unpickling restores the
  items before the attributes, so the notification ran before `_owner`
  existed (`AttributeError`). This broke `FitModel`/`FitComponent`
  `save`/`load`, and the `pool=` options of `run_emcee` /
  `get_mcmc_model_results` hung (workers died unpickling their task and
  the pool waited forever). Introduced with the mapping cache on
  2026-08-28.

### Example scripts -> tests and benchmark
- `examples/test_saving.py` -> `tests/test_save_load.py`: save/load round
  trips for `FitParameter`, `FitComponent`, `FitModel` (incl. sampler
  chain), plus regression tests for the pickling fix (one evaluates the
  model in a spawned worker, with a timeout so a regression fails instead
  of hanging).
- `examples/test_resolution.py` -> `tests/test_broadening.py`
  (`utils.broaden_spectrum` against a plain-numpy reference, FWHMs adding
  in quadrature, flux conservation, scalar vs array resolution in
  `FitModel.eval`) and `benchmarks/bench_broaden_spectrum.py` (timing).
- Both original scripts moved to `deprecated/examples/`.

### Packaging (`pyproject.toml`)
- Static `version = "2.0.0b0"` (the intended `2.b.0`, PEP 440-normalised).
  The previous dynamic version read `sculptor.__version__`, which itself
  reads the installed metadata, so every build came out as `0.0.0`.
- Runtime data files (`data/iron_templates/*`, `data/wave_2.txt`, read by
  `sculptor_extensions`) are now packaged; before, the wheel contained no
  data files at all, so the extensions only worked from a source checkout.
  `data/example_spectra/` (~20 MB) stays repository-only.
- Classifiers: `Development Status :: 4 - Beta`, Python 3.13 / 3-only
  (was 3.10), dropped `User Interfaces` (no GUI at present). Description no
  longer says "Interactive". Removed the obsolete `examples*`/`docs*`
  package excludes.

### `speconed`
- `src/sculptor/data/passbands/` moved to `deprecated/`: only the bundled
  v1 `SpecOneD.load_passband` read it.
- New `TODO_speconed.md`: differences between the bundled v1 copy and the
  standalone package, and open upstream issues (deferred).

### Dependencies
- Removed from `sculptor`'s runtime dependencies (only used by the v1 code
  now in `deprecated/`, by the deleted Sphinx docs, or not at all):
  `pyqt5`, `corner`, `importlib-resources`, `fastparquet`, `pyyaml`,
  `jupyter`, `ipykernel`, `sphinx`, `nbsphinx`, `nbsphinx-link`.
- `h5py`, `spectres`, `tables` dropped as direct dependencies; they are
  still installed as requirements of the standalone `speconed` package.
- `linetools` moved to `sculptor-extensions` (only `qso.py` uses it).
- `ipython` moved to the dev group (only the upstream reference copy
  `tests/ground_truth/v2_dev/src/sculptor/model.py` imports it).
- `sculptor-extensions` added to the dev group (workspace source), so a
  plain `uv sync` installs it for the tests; before, nothing declared it.

### Repository layout: `deprecated/`
- New top-level `deprecated/` folder for old code, mirroring the original
  paths (not packaged, excluded from `ruff`, not collected by `pytest`; see
  `deprecated/README.md`). Moved there: the PyQt5 v1 GUI (`specfitgui`,
  `specmodelwidget`, `masksmodels`, `menu_dialogs`, `specfitcanvas`,
  `specmodelcanvas`, `scripts/run_sculptor`), the lmfit-based
  `SpecFit`/`SpecModel`, `dep_v1/specanalysis.py`, the bundled
  `dep_v1/speconed.py`, `sculptor_extensions/my_extension.py` and
  `examples/dep_v1/`. The `sculptor.dep_v1` package no longer exists.

### `sculptor.measurements` (new)
- The nine `dep_v1.specanalysis` helpers still used by `sculptor.analysis`
  and `Qso_autofit.py` (`get_integrated_flux`, `get_average_fluxden`,
  `get_peak_redshift`, `get_equivalent_width`, `get_fwhm`,
  `get_nonparametric_measurements`, `calc_lwav_from_fwav`,
  `calc_integrated_luminosity`, `calc_apparent_mag_from_fluxden`) were
  ported unchanged in name and signature.
- Fixed: `np.trapz` (removed in numpy 2.4) replaced by `np.trapezoid`.
  Previously `get_integrated_flux`, `get_equivalent_width` and
  `calc_integrated_luminosity` raised `AttributeError`, and with them
  `analysis.analyze_emission_feature` with its default measurements
  (`EW`, `flux`, `lum`).
- `calc_integrated_luminosity` now builds its rest-frame spectrum with the
  standalone `speconed.SpecOneD` (it used the bundled v1 copy, mixing two
  `SpecOneD` classes).
- New `tests/test_measurements.py`: analytic checks for every helper and a
  regression test for `analyze_emission_feature`'s default measurements.
- Tests loading the `tests/ground_truth/` reference copies use the new
  `deprecated_dep_v1` fixture (`tests/conftest.py`) to resolve their
  `sculptor.dep_v1` imports.

## 2026-08-28 (2)

### `sculptor_extensions.qso`
- Removed ~30 dead `setup_*` functions (`setup_power_law_at_2500`,
  `setup_iron_template_model`, `setup_line_model_*`, `setup_doublet_line_model_*`,
  etc.) plus `add_redshift_param` and `model_func_list`/`model_setup_list`/
  `model_func_dict`. They built `lmfit.Model`/`lmfit.Parameters` objects, but
  the `lmfit` import was commented out (and `lmfit` isn't a dependency), so
  every one of them raised `NameError` if called. Superseded by
  `Qso_autofit.py`'s `initialise_*`/`setup_iron_template_*` functions, which
  build `sculptor.component.FitComponent` objects instead. Also dropped the
  now-unused `os` import and the stray executable shebang.

### Lint cleanup (`src/sculptor`, `sculptor_extensions`; `src/sculptor/dep_v1`
  and `tests/ground_truth` intentionally excluded/out of scope, see
  `pyproject.toml`)
- Fixed all `ruff check` findings in the maintained package: unused
  variables/imports, `zip()` without `strict=` (added `strict=True`
  throughout -- all zipped pairs are same-length by construction, so this
  also hardens `FitModel.eval()` against silently evaluating with a
  wrong/truncated parameter vector), `dict.keys()` iteration, an unnecessary
  `dict()` call, a merge-able `isinstance` chain, an empty docstring,
  spurious/missing shebangs, and a stray executable bit on `model.py`.
  Three `except Exception` blocks in `Qso_autofit.py`'s per-row batch
  post-processing (log-and-continue on a per-row failure) were kept broad
  and annotated `# noqa: BLE001` with the reason, rather than narrowed,
  since the failure modes there are genuinely unpredictable.
  `Qso_autofit.py` keeps its upstream-matching filename via a `ruff`
  per-file-ignore (`N999`) rather than being renamed.
  Also removed an unused `Hbeta_disp` local in
  `setup_iron_template_Park22_OPT_Hbeta` (present in the upstream ground
  truth too; confirmed harmless dead code, not a masking bug worth fixing
  here).
- Ran `ruff format` over the same scope (quote-style/wrapping only, no
  behavior change).
- `sculptor_extensions/__init__.py`: narrowed `except BaseException` to
  `except PackageNotFoundError`, and fixed the same invalid-PEP-440
  `"unknown"` version fallback as `sculptor/__init__.py`'s earlier fix.

### Tests
- `tests/test_model_bugfixes.py`: added
  `test_eval_raises_on_params_values_length_mismatch`, verifying the new
  `strict=True` zip in `FitModel.eval()` raises on a mismatched parameter
  vector instead of silently truncating.
- `tests/test_qso_extension_port.py`: added
  `test_qso_module_has_no_leftover_lmfit_setup_functions` and
  `test_qso_module_does_not_reference_lmfit`.

## 2026-08-28

### `sculptor.model.FitModel`
- Fixed `get_params_to_sample()` not being idempotent: repeated calls (e.g.
  `eval()` then `initialize_emcee()`/`initialize_positions()`) used to
  duplicate entries in `params_constant_val`, corrupting constant-parameter
  values on subsequent `eval()` calls. Now resets its tracking collections
  at the top of the method.
- Fixed `calculate_chi_sq()` computing degrees of freedom from
  `len(self.parameters)` (fixed + variable) instead of
  `len(self.params_variable)` (only the actually-fitted parameters).
- Performance: cached each component's resolved argument-key mapping
  (`FitComponent.get_arg_keys()`) instead of recomputing it on every single
  `eval()` call — the innermost function hit on every MCMC likelihood
  evaluation.

### `sculptor.component.FitComponent`
- `param_mapping` is now a property backed by a self-invalidating dict
  (`_ObservableParamMapping`): mutating it in place (`.update()`, `.pop()`,
  etc.) or reassigning it always invalidates the `get_arg_keys()` cache.
  Protects the coupled-parameter idiom used throughout real fitting scripts
  (e.g. `comp.param_mapping.update({'redsh': 'redsh_nl'})`) from silently
  using a stale mapping if it's mutated after the component has already been
  evaluated once.

### `sculptor.analysis`
- Fixed `get_line_snr()`: the per-MCMC-sample line integration window
  (`l_min`/`l_max`, derived from that sample's FWHM) was cached back into
  the caller's `line_dict`, so every sample after the first silently reused
  sample 0's stale window instead of computing its own, and the caller's
  dict was mutated as a side effect. Now computed locally per iteration.
- Performance: reused the already-computed module-level `c_km_s` constant
  in `get_line_snr()` instead of recomputing `const.c.to('km/s').value`
  twice per line per sample.

### `sculptor.utils`
- Fixed `broaden_spectrum()` crashing on big-endian input arrays (typical of
  FITS-loaded spectra) — the `@njit(parallel=True)` kernel has no numba
  typing overload for non-native byte order. It's now a thin Python wrapper
  that casts inputs to native-order, contiguous `float64` before calling the
  jitted kernel (renamed `_broaden_spectrum_jit`); public name/signature
  unchanged, so every caller (including `FitModel.eval()`) is fixed
  transparently.

### `sculptor.__init__`
- Dynamic `__version__` fallback changed from `"unknown"` (invalid PEP 440,
  crashes `pyproject.toml`'s `attr = "sculptor.__version__"` build-time
  version lookup) to `"0.0.0"`.

### `sculptor_extensions.qso` / `sculptor_extensions.Qso_autofit`
- Brought to feature parity with the upstream `v2_dev` branch
  (github.com/jtschindler/sculptor): switched `qso.py`'s `SpecOneD` import
  from the bundled `sculptor.dep_v1.speconed` copy to the standalone
  `speconed` package; restored the FeII/MgII flux-ratio calculation in
  `Qso_autofit.get_line_prop_MgII` (previously disabled during an earlier
  port).
- Fixed a `u.kms` typo (not a real astropy unit) in the three
  `se_bhmass_civ_*` functions -> `u.km / u.s`.
- Fixed `Balmer_continuum_with_hol`'s `np.loadtxt(datadir, "wave_2.txt")`
  (passed the filename as `dtype`) -> `np.loadtxt(datadir + "wave_2.txt")`.
- Decoupled the `median_params` alternate-input mechanism from
  `initialise_power_law`, `initialise_Fe`, `initialise_balmer_without_hol`,
  `initialise_balmer`, and all ten `setup_iron_template_*` wrappers, leaving
  only the `*_dict`-based input. This also fixed a latent bug where the
  `if median_params:` / `if *_dict: ... else: raise` blocks were independent
  `if`s rather than `if/elif`, so the "provide either X or median_params"
  error fired regardless of whether `median_params` was supplied, and two of
  the four error messages named the wrong dict.
- `extract_median_params`, `get_median_params`, and
  `extract_median_params_from_summary` are unaffected by the above (they
  produce/extract a median-values dict rather than consume one as an
  alternate input).

### `sculptor_extensions.qso_advanced_models` (new module)
- New home for `out.py`'s experimental Balmer-continuum-with-higher-order-
  lines and Lyman-alpha absorption physics, kept separate from
  `Qso_autofit.py` as unvalidated/experimental. Same `median_params`
  decoupling applied to `initialise_balmer_with_hol` as above.

### Repository layout
- Consolidated the test-only ground-truth/comparison sources into
  `tests/ground_truth/` (`fgu/`, `radha/`, `v2_dev/`, `updated_modules/`,
  the latter replacing `src/sculptor/updated_modules/`), and added
  `norecursedirs = ["tests/ground_truth"]` to `pyproject.toml` so pytest
  doesn't pick up example test files bundled inside the ported reference
  copies.

### `_ground_truth/agn_fit_new.py` (external reference script)
- Reviewed real-world usage patterns against the current API. Added
  `broaden_by_resolution=False` to the two "manual analysis rewrite"
  `model.eval()` calls, matching `sculptor.analysis`'s convention for
  physical-measurement evaluations (previously relied on `eval()`'s old
  `False` default, now `True`).

### Tests
- Added `tests/test_model_bugfixes.py`, `tests/test_analysis_bugfixes.py`,
  `tests/test_upstream_parity.py`, `tests/test_qso_extension_port.py`
  (regression coverage for all fixes above, including the big-endian
  `FitModel.eval()` path, `param_mapping` cache invalidation, and the
  `median_params` removal).
- Updated `tests/update_tests/test_utils.py` (big-endian `broaden_spectrum`
  regression) and `tests/update_tests/benchmarks.py` to load the archived
  pre-optimization `utils.py` from its new `tests/ground_truth/` location.
