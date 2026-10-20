# Changelog

## 2026-10-20 (conventions)

- **Renamed module** `sculptor_extensions.Qso_autofit` ->
  `sculptor_extensions.qso_autofit` (lowercase like the other modules; it
  was kept for parity with upstream's file name). `import
  sculptor_extensions.Qso_autofit` still works with a DeprecationWarning,
  and fits saved with the old module path still load (import hook in
  `sculptor_extensions/__init__.py`; a second file differing only in case
  is not possible on case-insensitive file systems). Older Changelog
  entries keep the old name.
- **American spelling** throughout the code and docstrings. Renamed (the old
  names still work, with a DeprecationWarning; `Qso_autofit.RENAMED`):
  `initialise_power_law`, `initialise_Fe`, `initialise_Fe_fast`,
  `initialise_balmer_without_hol`, `initialise_balmer`,
  `initialise_gaussian_line_model` -> `initialize_*`;
  `qso_advanced_models.initialise_balmer_with_hol` ->
  `initialize_balmer_with_hol`; `component_setup.initialise_component` ->
  `initialize_component` (new this session, no alias).
- **`broadshifted`** for all velocity-shifted broad components:
  `setup_MgII_lines(MgII_broadshifted=)`,
  `setup_Hbeta_OIII_lines(Hbeta_broadshifted=)`, line_dict key
  `"broadshifted"`, components `<line>_broadshifted_a`, ... The old keywords
  (`MgII_broad_shifted`, `Hbeta_broad_shifted`) and a line_dict with
  `"broad_shifted"` still work, with a DeprecationWarning; components are
  named `..._broadshifted_...` either way.
- **Behavior change**: 1-sigma intervals use the 15.87 / 84.13 percentiles
  everywhere (`sculptor.utils.SIGMA_PERCENTILES`); `get_1sigma_range`,
  `get_median_params` and `plot_posterior_param_histograms` used
  15.9 / 84.1.
- **Behavior change**: `get_median_params` and `get_sampled_values` default
  to `discard=2000` like `FitModel` (it was 0), with a one-time warning when
  `discard` is not given.
- **Behavior change**: `UniformPrior.logprior` is normalized,
  -ln(upp - low) inside the interval (it was 0), like `GaussianPrior`.
  MCMC chains are unchanged (a constant offset); log-probabilities
  (`get_flat_log_prob`, the MAP sample's value) shift by that constant per
  uniform parameter and are now comparable between models with different
  prior widths.
- `setup_iron_template_VW01_CIV`: kept 1300 - 1800 Å (covers the
  `qso_contfe_CIV_Shen11` windows), to be confirmed.

## 2026-10-20 (review fixes)

### Tests no longer need the reference copies
- A fresh clone failed 31 tests: the parity tests imported
  `tests/ground_truth/`, which imports the v1 modules in the untracked
  `deprecated/`. The original outputs are now stored in
  `tests/data/reference_values.npz` (`make_reference_values.py` documents
  their origin and how to regenerate them from commit 70a5f7e) and compared
  in `test_reference_values.py`. `tests/ground_truth/` (incl. the unused
  14 MB `radha` copy), `tests/update_tests/` and the `deprecated_sculptor`
  fixture are gone; migration guards dropped; bug-fix test files renamed
  or merged by module; new tests for the untested helpers
  (`test_fit_helpers.py`).

### `FitModel`
- **Behaviour change**: `eval` (and `log_probability`) rebuild the split
  into variable and fixed parameters on every call. It was cached on the
  first call, so later edits of a fixed value or a `vary` flag were ignored.
- **Behaviour change**: `add/remove_wavelength_range_*_fit_mask` include the
  pixel nearest the upper bound (it was left out); the same for
  `get_equivalent_width(disp_range=...)`.
- emcee is seeded from `FitModel.seed` (runs are reproducible);
  `initialize_emcee(spec=...)` rejects a spectrum on another grid;
  `plot_posterior_param_histograms` defaults to `discard=2000`;
  `get_all_prefix_components` is a deprecated alias; plots label the
  highest-posterior sample "MAP sample" (it was "ML model").

### Analysis and measurements
- `analyze_emission_feature` / `analyze_continuum` default the axis and
  units to the model's spectrum (`dispersion=None` crashed);
  `analyze_continuum`'s `cosmology` is optional; `appmag` uses the
  spectrum's dispersion unit; `_mcmc_analyze` warns (deprecated).
- **Behaviour change**: `get_fwhm` (spline method) takes the peak from the
  smoothed spectrum; the raw maximum of noisy data biased the FWHM low.
- `UniformPrior` accepts arrays; `ppf` for `UniformPrior` and
  `GaussianPrior`.

### `sculptor_extensions`
- **Bug fixed**: `create_emission_line_dict` wrote the prior upper bounds
  to `.upper`, which `UniformPrior` ignores (`.upp`).
- **Behaviour change**: `get_fwhm_int_flux` sets the spline smoothing from
  the flux errors instead of a fixed `s=100` (which gave +25% FWHM for
  fluxes of order unity, and NaN in plain cgs units).
- `get_median_params`, `get_sampled_values` read the chain via
  `get_flat_chain`; `plot_burnin` has `max_walkers` (it overwrote the walker
  count with 10); `add_line_snr_stats` computes once.
- Renamed (old names forward with a DeprecationWarning,
  `qso_presets.RENAMED`): `setup_iron_template_V01_FUV` / `V01_CIV` ->
  `VW01_FUV` / `VW01_CIV`, `setup_hb_o3_lines` -> `setup_Hbeta_OIII_lines`,
  `get_civ_line_prop` -> `get_line_prop_CIV`.
- `calc_velocity_shifts` no longer uses `linetools` (dropped);
  `CIII_complex_model_func` uses `line_model_gaussian`; no `print` left
  (`FitComponent.info`, `print_variable_parameters` log).
- One definition of the speed of light and the data directory
  (`sculptor.utils`); unused palettes and module globals removed;
  `sculptor-extensions` declares its direct dependencies.

### Docs and examples
- README rewritten for v2; `examples/example_fit_setup_script.py` fixed
  (data path; three redshift parameters that varied without effect).

## 2026-10-20

### Variants built on base functions; new `sculptor_extensions.qso_presets`
- **Structure**: families of copy-pasted variants now call one base
  function. Named setups and line-property calculations moved to the new
  convenience module `sculptor_extensions.qso_presets`, with unchanged
  signatures:
  - 10 `setup_iron_template_<name>` -> `Qso_autofit.setup_iron_template(prefix,
    template_file, disp_range, fe_dict, fast, fluxden_scale)`.
  - `setup_Halpha_NII_lines`, `setup_MgII_lines`, `setup_CIV_lines`,
    `setup_hb_o3_lines`, `setup_Hgamma_lines`, `setup_OI_lines` ->
    `Qso_autofit.add_line_components(model, line_dict, line, kind, n)`.
  - `get_line_prop_MgII`, `get_civ_line_prop`, `get_line_prop_Hbeta_OIII`
    -> `Qso_autofit.get_line_properties(df, redshift, param_name_dict,
    line, cont_wav, bh_mass_relations)`.
  The old `Qso_autofit.<name>` still work, with a `DeprecationWarning`.
- Base functions for code that stays where it is (model functions keep
  their module and name, so saved fits still unpickle):
  - `component_setup.initialise_component`: the "parameter dict ->
    component" loop of every `initialise_*` (power law, iron, Balmer, HOL,
    Gaussian lines).
  - `qso.virial_bh_mass`: the power law behind all ten `se_bhmass_*`
    relations (VP06 Hbeta / CIV, Park12, GH05, DB20, VO09, S11, C17).
  - `qso_advanced_models.balmer_continuum_shape` /
    `balmer_continuum_at_edge`: shared by `fast_balmer_continuum_model`,
    both HOL models, `power_law_at_2500_plus_fractional_bc` and
    `qso.power_law_at_2500_plus_bc` (which no longer uses astropy's
    `BlackBody`; same values to ~1e-15).
  - [OIII], [NII], [SII] doublets (tied and untied) are two
    `analysis.line_model_gaussian` lines; the power-law variants call
    `analysis.power_law_at2500`.
  - `CachedBalmerContinuumWithHOL` is a subclass of
    `BalmerContinuumWithHOL` that only overrides the cached grid.
- **Bug fixed**: `initialise_power_law`, `initialise_Fe[_fast]`,
  `initialise_balmer_without_hol` and `initialise_balmer_with_hol` raised
  `RuntimeError: dictionary changed size during iteration` when an entry
  mapped a parameter onto another (`{"parameter": ...}`, e.g. a shared
  redshift).
- **Bug fixed**: `setup_iron_template_Park22_OPT_Hbeta` loaded
  `Fe_OPT_Hbeta_Park22.txt`, which does not exist; it now uses the
  packaged `Park_2022.asc` (flux density in 1e-15 erg/s/cm2/Å, used
  unscaled, as upstream's `out.py`).
- Small behaviour changes: `get_line_prop_MgII` with an unknown
  `fit_name` no longer writes NaN BH-mass/Eddington columns (it skips
  them); `get_line_prop_Hbeta_OIII` logs a failed velocity shift as a
  warning, like the others; its commented-out FeII block was dropped.
  Output column names are unchanged.
- Docstrings: BG92 range is 3685 - 7484 Å (not 7844), the FeIII template
  is non-zero over 2379 - 2456 Å, `V01_CIV` states the range it uses
  (1300 - 1800 Å).
- New `tests/test_qso_presets.py`.

### Units: astropy Quantities required in `qso` and the line-property functions
- **Behaviour change**: every scaling relation in `sculptor_extensions.qso`
  (`se_bhmass_*`, `calc_bolometric_luminosity[_db20]`,
  `calc_eddington_luminosity`, `calc_eddington_ratio`,
  `correct_CIV_fwhm_for_blueshift`) now requires astropy Quantities: a bare
  number raises `TypeError`, an incompatible unit raises
  `UnitConversionError`. Compatible units are converted (e.g. m/s, W/nm,
  nm), and the continuum wavelength is compared in Å, so `300 * u.nm`
  selects the 3000 Å relations. Before, `cont_wav.value == 3000` was read
  in whatever unit was passed, and a bare luminosity went through
  unchecked.
- Results come back in fixed units: BH masses in M_sun, luminosities in
  erg/s, the Eddington ratio dimensionless, velocity shifts in km/s. New
  module constants `qso.KM_S`, `qso.LWAV_UNIT`, `qso.LUM_UNIT`.
- `Qso_autofit.get_line_prop_MgII` passed a unitless 3000 Å luminosity to
  `calc_bolometric_luminosity`, so L_bol came out in Å instead of erg/s
  (the stored number was right); it now builds Quantities once and stores
  every column with `.to_value(<unit>)`, like `get_civ_line_prop` and
  `get_line_prop_Hbeta_OIII`. Its docstring lists `fit_name` and the
  column units, and no longer the unused `model` / `flat_samples`.
- `Qso_autofit.get_fwhm_int_flux` returned `integrated_flux.value * 1e17`,
  which is only right for spectra in erg/s/cm²/Å: for a spectrum in
  1e-17 erg/s/cm²/Å, the flux (and so the `create_emission_line_dict`
  flux priors) came out 1e17 too large. It now converts to
  1e-17 erg/s/cm² explicitly.
- `measurements.get_fwhm` returns NaN km/s instead of a bare NaN when it
  cannot measure the FWHM, so it always returns a Quantity.
- New `tests/test_qso_units.py`.

### Iron templates: `fast=True` by default
- All ten `setup_iron_template_*` functions now default to `fast=True`
  (`TemplateModel`, ~10-20x faster). `fast=False` restores
  `template_model`. The `speconed` `broaden_by_gaussian` review is still
  open (`TODO_speconed.md`).

## 2026-10-19

### `Qso_autofit.get_civ_line_prop` (restored, units fixed)
- Restored from the upstream `qso_auto_fit` branch. The fgu copy was
  disabled (`raise DeprecationWarning("DO NOT USE THIS")`) and the upstream
  version failed: it passed unitless FWHM/luminosity values to
  `se_bhmass_civ_vp06_fwhm`, giving a mass in meaningless units and a
  `UnitConversionError` in `calc_eddington_luminosity`. Now FWHM (km/s)
  and the 1350 Å luminosity (erg/s/Å) carry units, and the results are
  stored in M_sun / erg/s / km/s.
- Signature `get_civ_line_prop(df, redshift, param_name_dict,
  save_dir=None, source=None)`; upstream's unused `model` / `flat_samples`
  are gone, and a CSV (`<source>_results_CIV.csv`) is written only if
  `save_dir` is given. Output columns are named like
  `get_line_prop_MgII`'s (`CIV_BH_mass_1350_vp06`, `L_bol_1350`,
  `Eddington_ratio_1350_vp06`, `Edd_lum_CIV_1350_vp06`, `delta_v_CIV`);
  upstream's names are listed in the docstring. Messages are logged.
- Tests compare the results with direct calls of the `qso` relations.

### `Qso_autofit`: Hgamma and OI line setups
- New `setup_Hgamma_lines(spec, line_dict, narrow=False, Hgamma_broad=1,
  model=None)` and `setup_OI_lines(spec, line_dict, OI_broad=1,
  model=None)`, ported from upstream `qso_auto_fit` (`Qso_autofit_2.py`).
  Named like the other line setups (`setup_<line>_lines`,
  `<line>_broad` counts, components `<line>_narrow` / `<line>_broad_<a,b,..>`,
  `line_dict["Hgamma"]` / `line_dict["OI"]`). Test added.

### Fast iron template (`TemplateModel`, `initialise_Fe_fast`)
- New `sculptor.analysis.TemplateModel(templ_disp, templ_fluxden)`: a
  precomputed, picklable version of `template_model` for one template,
  ~10-12x faster per evaluation. Ported from the upstream `qso_auto_fit`
  branch (`Qso_autofit_2.py`), with two changes so it gives the same model
  as `template_model` (and upstream `main`'s `qso.template_model`, which
  is the same code): its normalisation now includes `speconed`'s
  1/(N dlnlambda) factor (the original was 5.6% higher for the UV
  template), and it reproduces `speconed`'s half-pixel shift for templates
  with an even number of pixels. Checked against all 11 packaged iron
  templates: agreement to ~1e-12 of the peak (odd pixel count) and
  <=1.5e-3 (even). Both `speconed` quirks are flagged for review in
  `TODO_speconed.md`. It no longer uses `SpecOneD`'s private
  `_to_log_wavelength`.
- New `Qso_autofit.initialise_Fe_fast`, and `fast=False` on all ten
  `setup_iron_template_*` functions: `fast=True` builds the component from
  `TemplateModel`, whose only parameters are `redsh`, `amp`, `fwhm` and
  `intr_fwhm`. The default stays `template_model`, as on `v2_dev`.
- The `FitComponent` signature helper (`_FakeCode` in
  `qso_advanced_models`) is now `sculptor.component.FakeCode`, shared by
  `TemplateModel` and the Balmer HOL models.
- New `tests/test_template_model.py`.

### `FitModel`: pixels masked in the spectrum are left out of the fit
- **Behaviour change**, adopted from the upstream `kat` branch (4ee90b4,
  Katharina Jurk; not merged into `v2_dev`): the fit now uses
  `gpm_fit & spec.mask`, i.e. the fit regions minus the pixels masked in
  the `SpecOneD` (`spec.mask`, True = good pixel). Before, `spec.mask` was
  ignored by the fit (only the plots used it), so a bad pixel inside a fit
  region -- e.g. a NaN flux -- entered the likelihood (making it NaN) and
  the chi-square.
- New `FitModel.get_fit_mask()` returns that combined mask; it is used by
  `log_likelihood_chi_square`, `calculate_chi_sq` (also the degrees of
  freedom), `compute_bic` (number of data points) and `plot_mcmc_result`
  (shaded fit regions and default axis limits). With the default
  all-good `spec.mask` nothing changes. A spectrum whose `mask` is `None`
  uses `gpm_fit` alone.
- Not adopted from `kat`: its degrees of freedom counted all parameters
  (ours count only the varying ones), its median-based BIC, and the other
  feature additions.

### `measurements.calc_integrated_luminosity`: observed-frame `disp_range`
- Fixed: `disp_range` was applied to the rest-frame spectrum, so an
  observed-frame range -- the convention of every other measurement and of
  `analyze_emission_feature` (upstream `v2_dev` 56832f4) -- selected
  nothing and the line luminosity came out as 0 without any error. The
  range is now converted to the rest frame before trimming. The same bug is
  in upstream `v2_dev`'s `specanalysis.calc_integrated_luminosity`.

### Sync with upstream `v2_dev` (up to a5a1749)
- New `qso.se_bhmass_db20` (Hbeta single-epoch BH mass) and
  `qso.calc_bolometric_luminosity_db20` (Hbeta bolometric correction),
  Dalla Bonta et al. 2020, ported from upstream (33b78d5, with the FWHM fix
  a5a1749). Same results as upstream for inputs in km/s and erg/s; ours
  also convert other units, log the virial factor instead of printing it,
  and raise without the `[ERROR]` prefix.
- The rest of upstream `v2_dev` since the uv-migration merge was already
  here (importlib.resources, standalone speconed, trapezoid, resolution
  broadening default, clearing old results on a new emcee run). Kept on
  purpose: the "Beta" status / `2.0.0b0` (upstream: 2.0.0, Stable), packaging
  only the runtime data files, and the `0.0.0` version fallback.
- The upstream reference copies in `tests/ground_truth/v2_dev/` were
  refreshed to `v2_dev` a5a1749 (`qso.py`, `model.py`).

### Docstrings (convention: reST)
- Every public function, method and class in `sculptor` and
  `sculptor_extensions` has a reST (`:param:` / `:type:` / `:return:`)
  docstring. Written for the 34 that had none (`FitModel` and its fit and
  plot methods, `FitComponent`, `FitParameter`, the priors, three
  `analysis` model functions, `get_line_snr`, plot styles, ...) and for
  the likelihood functions whose `:param x:` fields were empty.
- Rewrote the stale `analyze_emission_feature` / `analyze_continuum`
  docstrings (they still described v1 `SpecFit` arguments).
- Fixed `:param type: X` -> `:type <name>: X` and `:return type:` ->
  `:rtype:` (17 + 1 places), removed 10 duplicated empty
  `:param fe_dict:` lines, "FWWHM" typos, and `sculptor.speconed.SpecOneD`
  -> `speconed.speconed.SpecOneD`.
- `FitModel.add_component` / `add_parameters` lost their wrong type hints
  (`component: list`); `template_model` uses `isinstance` for its unit
  strings.
- New `tests/test_docstrings.py` keeps it that way.

### Convention: the redshift fit parameter is `redsh` (breaking)
- Model functions now all call the redshift argument `redsh`, as
  `sculptor.analysis` already did. Renamed from `z`:
  `qso_advanced_models.fast_balmer_continuum_model`,
  `power_law_at_2500_plus_fractional_bc`, `BalmerContinuumWithHOL`,
  `CachedBalmerContinuumWithHOL`; `qso.power_law_at_2500_plus_bc`,
  `line_model_gaussian_oiii_doublet`, `CIII_complex_model_func`.
- Because `FitComponent` names parameters after the function arguments,
  components built from these get `<prefix>_redsh` instead of
  `<prefix>_z`, and setup dicts need the key `"redsh"` instead of `"z"`
  (e.g. `balmer_dict` for `initialise_balmer` /
  `initialise_balmer_without_hol`, `line_dict` for [OIII]).
- Unchanged: `z_abs` (absorber redshift of the absorption models) and the
  non-model functions (`calc_velocity_shifts(z, z_sys)`, analysis and
  measurement functions' `redshift=`).
- Removed the dead commented-out `line_model_gaussian_amp` in `qso.py`.
- Tests: the reference-copy comparisons map `z` -> `redsh`; a new test
  checks the convention for every model function.

### `sculptor.model.log_probability`: prior skipped on first call
- Fixed: the prior was read from `model.params_variable` before `eval()`
  had filled it, so the first `log_probability` call on a model that had
  not been evaluated yet returned the likelihood without the prior. emcee
  runs were not affected (`initialize_emcee` fills it first), but
  `get_flat_log_prob` on a chain-only model got its first sample wrong,
  which made `test_log_prob_is_recomputed_without_sampler` fail
  intermittently (only when the missing prior term exceeded the test's
  relative tolerance). That test now compares at `rtol=1e-12`, plus a
  direct regression test.

### Logging (`sculptor.logging_config`, new)
- All progress/diagnostic `print("[INFO] ...")` / `"[WARNING]"` /
  `"[ERROR]"` calls in `sculptor` and `sculptor_extensions` now go through
  `logging.getLogger(__name__)`. `FitComponent.info()` and
  `Qso_autofit.print_variable_parameters()` still print: they exist to
  show output.
- Based on the user's `configure_logging` helper (from the
  `euclid_qso_selection` package): INFO and DEBUG to stdout, WARNING and
  above to stderr, optional log file with timestamped backup,
  `warn_once`, `all_logging_disabled`. Adapted for a library: it only
  configures the `sculptor` / `sculptor_extensions` loggers (the root
  logger and other libraries are untouched; the original cleared the
  root handlers and silenced every other logger by default), repeated
  calls replace its own handlers, handlers look up `sys.stdout`/`stderr`
  at write time (so Jupyter/pytest redirection works), and
  `reset_logging()` hands messages back to the application's root setup.
- Importing sculptor calls `configure_logging()`, so INFO messages show by
  default as before. `SCULPTOR_LOG_LEVEL` (e.g. `WARNING`) and
  `SCULPTOR_LOG_TO_FILE=1` (writes `~/.sculptor/logs/sculptor.log`) are
  read at import; `sculptor.configure_logging(...)` changes them later.
- Messages now carry the logger name and level
  (`sculptor.model - INFO: ...`) instead of `[INFO]`. Per-row failure
  notices in `Qso_autofit` and `get_fwhm`'s "cannot determine FWHM" are
  warnings; the no-prior notice in `initialize_positions` is one warning
  instead of four lines.
- Exception messages no longer start with `[ERROR]`.
- `analysis.py`: imports tidied (one `astropy.units as u`; `c_km_s` after
  the imports).
- New `tests/test_logging.py`.

### `sculptor_extensions`: doublets and Balmer + high-order lines
- `Qso_autofit.initialise_gaussian_line_model` takes `tied_doublet=True`.
  For [NII]/[SII] the default stays the tied model (one FWHM, flux ratio
  as parameter); `tied_doublet=False` selects
  `qso.line_model_gaussian_nii_doublet` / `_sii_doublet` (independent
  fluxes and FWHMs). Those two now name the redshift `redsh` (was `z`),
  like the tied models, so both modes use the same `line_dict` key.
- **Behaviour change:** `Qso_autofit.initialise_balmer` now builds its
  component from `qso_advanced_models.BalmerContinuumWithHOL` (the newer
  model from `out.py`, ~3x faster) via `initialise_balmer_with_hol`.
  The old `Balmer_continuum_with_hol` normalised and scaled differently
  (~2e17 in amplitude, up to ~40% in shape below the Balmer edge), so
  `amp_be` values and priors from older setups need rescaling. Its code
  is in `deprecated/.../qso_autofit_balmer_hol.py`. The `model` argument
  of `initialise_balmer` is unused and now optional.
- Data file `sculptor/data/wave_2.txt` renamed to
  `balmer_hol_storey_hummer1995.txt`, with a header documenting its
  origin (Storey & Hummer 1995 case-B emissivities, VizieR VI/64; Balmer
  n = 7..50, wavelengths from the Rydberg formula with R_inf) and columns.
  The data itself is unchanged. `qso_advanced_models.HOL_DATA_FILE` holds
  the name. Earlier comments calling it a synthetic placeholder were wrong.
- Tests: tied/untied doublet switch (both agree for equal widths and
  `flux_b = ratio * flux_a`); `initialise_balmer` uses the new model; the
  missing-data guard is tested without deleting the packaged file.

### `sculptor_extensions`: duplicate model functions (review §3)
- Removed from `qso.py` (code kept in
  `deprecated/sculptor_extensions/src/sculptor_extensions/qso_duplicates.py`):
  - `power_law_at_2500`, `line_model_gaussian`, `template_model`: identical
    to `sculptor.analysis.power_law_at2500` / `line_model_gaussian` /
    `template_model`, which everything already used. Note the core
    versions take `redsh` instead of `z` (and a different argument order).
  - `balmer_continuum_model`, `power_law_at_2500_plus_fractional_bc`: same
    formulas as `qso_advanced_models.fast_balmer_continuum_model` /
    `power_law_at_2500_plus_fractional_bc`, which compute the blackbody
    directly instead of through astropy's `BlackBody` (~7x / ~10x faster;
    outputs agree to ~1e-15).
- `Qso_autofit.initialise_balmer_without_hol` now builds its component from
  `qso_advanced_models.fast_balmer_continuum_model`. Same parameters and
  output, faster fits.
- Regression tests now compare the original reference copies
  (`tests/ground_truth/fgu`, `v2_dev`) against the maintained replacement.
- Not touched (not duplicates): the `qso.py` [NII]/[SII] doublets have
  independent fluxes and widths, the `Qso_autofit.py` ones tie the widths
  and fix the flux ratio. The two Balmer + high-order-lines models
  (`Qso_autofit.Balmer_continuum_with_hol`,
  `qso_advanced_models.BalmerContinuumWithHOL`) give different results
  and are left for a decision.

### `FitModel.evaluate_parameters` removed
- It saved one corner plot per parameter pair, but used the pre-1.0
  chainconsumer API and failed on the installed 1.3
  (`AttributeError: 'DataFrame' object has no attribute 'name'`); it also
  wrote to the current directory only and read the chain straight from the
  sampler. Nothing called it. Its code is kept in
  `deprecated/src/sculptor/model_evaluate_parameters.py`; use
  `plot_posterior_corner(parameters=[...])` for corner plots.

### `measurements.get_nonparametric_measurements`: `disp_range`
- Fixed: `disp_range` (observed-frame dispersion in the spectrum's own
  unit, e.g. Angstrom, like every other measurement's `disp_range`) was
  applied after the spectrum had been converted to frequency, so it
  trimmed away the whole spectrum and the function raised `ValueError`.
  This broke `analyze_emission_feature(..., emfeat_meas=[..., "nonparam"],
  disp_range=...)`. The spectrum is now trimmed first, then converted.
  The bug was already present in v1 `specanalysis`.

### Plotting (`sculptor.plot`)
- Colours now come from the `tol-colors` package (Paul Tol's own release)
  instead of the vendored 774-line `sculptor/colors.py` (moved to
  `deprecated/`; its `rainbow_PuRd` is identical to the package's).
  `tol-colors` is a new dependency of both `sculptor` and
  `sculptor-extensions`.
- The plot colour names (`dblue`, `vermillion`, ...) default to Tol's
  "bright" scheme. Set `SCULPTOR_PALETTE=okabe-ito` before importing
  sculptor to get the previous Okabe & Ito colours; an unknown value
  raises `ValueError`. The component colour map is
  `plot.component_cmap` (Tol `rainbow_PuRd`) in both palettes.
- `set_paper_defaults` / `set_presentation_defaults` share one
  `_set_common_defaults` for the tick/axis settings; the dead commented
  LaTeX-preamble lines are gone.
- Tests: the `deprecated_dep_v1` fixture is now `deprecated_sculptor`
  and covers any module under `deprecated/src/sculptor/` (the reference
  copies also import `sculptor.colors`). New `tests/test_plot.py`.

### `sculptor.analysis` / `sculptor.measurements`
- `analyze_mcmc_results` is now the working routine (the old public one
  only printed a message); it is the former private `_mcmc_analyze`, with
  its full signature and a docstring. `_mcmc_analyze` stays as an alias
  (used by `_ground_truth/agn_fit_new.py`).
- The Calzetti extinction correction, duplicated for line and continuum
  in `analyze_emission_feature`, is one helper, `_remove_extinction`.
- Fixed: `analyze_emission_feature` used `redshift` for the extinction
  correction before checking it was set (`TypeError` with `a_v`); the
  check now comes first. `analyze_continuum` gained the same check (it
  failed with `TypeError` on the default `redshift=None`).
- `measurements.get_nonparametric_measurements` works on a copy by
  default; `inplace=True` restores the v1 behaviour of converting and
  trimming the input spectrum itself.
- New `tests/test_analysis.py`, more tests in `tests/test_measurements.py`.

### `sculptor.model.FitModel`
- **One chain accessor.** New `get_flat_chain(discard)` (and
  `get_flat_log_prob(discard)`) is the only place the chain is read;
  `get_mode_model`, `get_mcmc_model_results`, `compute_bic`,
  `plot_posterior_corner` and `plot_posterior_param_histograms` use it
  instead of reading `self.sampler` or `self.flat_chain` directly. With a
  sampler it caches `flat_chain`, `flat_log_prob` and
  `flat_chain_discard`; without one it returns the stored `flat_chain`
  and raises `ValueError` if asked for a different `discard` than it was
  built with. Log-probabilities missing from a hand-assigned chain are
  recomputed, so the ML model works without a sampler. A new emcee run
  clears all cached chain-derived results. (`evaluate_parameters` is
  deliberately untouched for now.)
- `compute_bic` takes `discard` (default 2000, like the other methods)
  instead of using whatever `flat_chain` was last computed.
- **Plotting a chain-only model.** `plot_mcmc_result` no longer needs
  `nsteps`/`nwalkers` (it failed with `TypeError` on a model with only a
  `flat_chain`); it recomputes the cached posterior when `discard`
  differs from the one it was built with (`posterior_discard`).
- **Per-component curves** (`show_components=True`) go through the new
  side-effect-free `_posterior_model_bands`, i.e. the same code and
  `pool` as the full model, and no longer keep every sample of every
  component in memory.
- `show_ml_model=False` no longer plots "ML residuals" against an
  unset/stale model.
- **No spectrum:** `self.spec` is always set (`None` without spectrum).
  `initialize_emcee`, the fit-mask methods, `plot_mcmc_result`,
  `calculate_chi_sq`, `compute_bic` and `get_mode_model` raise
  `ValueError("No spectrum set, ...")` instead of `AttributeError` or
  silently doing nothing. A `spectrum` that is not a `SpecOneD` raises
  `TypeError` (was silently ignored). Adding a fit-mask range to a model
  whose `spec` was assigned later creates the mask first.
- `calculate_chi_sq` raises `ValueError` when there is no posterior model.
- `get_components_by_prefix` matches name prefixes (it matched exact names,
  or substrings when given a string); `get_all_prefix_components` calls it.
- Removed the empty `plot_posterior_histograms` stub.
- New `tests/test_fitmodel.py`; the synthetic fitted model used by the
  tests is now a shared fixture in `tests/conftest.py`.

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
