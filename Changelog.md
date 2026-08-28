# Changelog

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
