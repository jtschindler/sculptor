# Sculptor

Modeling and MCMC fitting of (electromagnetic) astronomical spectra.

##### (Version 2.0.0b0, beta)

**Sculptor** fits 1D astronomical spectra with models built from simple
components, finds the best fit with an optimizer (quick fit) and samples
the posterior with [emcee](https://emcee.readthedocs.io).
Version 2 replaces the LMFIT-based v1 API (`SpecFit`, `SpecModel` and the
Qt GUI) with a smaller core; the v1 documentation on
[Sculptor's ReadTheDocs] describes the old API. A GUI for v2 is planned.

The core package (`sculptor`):

- `sculptor.model.FitModel`: the spectrum (a `speconed.SpecOneD`), the fit
  regions, the components and parameters; `fit("quick" | "mcmc" |
  "quick+mcmc")` (`run_quick_fit`: maximum of the posterior with an
  optimizer; `run_emcee`: posterior samples, both with the same
  log-probability), posterior models, plots (`plot_quick_fit_result`,
  `plot_mcmc_result`, `plot_posterior_corner`) and save/load.
- `sculptor.autofit`: automatic model building on the quick fit. Rules
  propose components (`GaussianLineRule` for emission or absorption lines,
  `ComponentRule` for predefined components or continuum swaps); they are
  kept while they lower the BIC (or AICc, AIC, SpIC, the spectral
  information criterion of Webb et al. 2021, or a custom criterion) by more
  than a threshold, by default calibrated from simulations with the
  residuals' noise correlations (`calibrate_threshold`; the models for
  several quantiles, 95% and 99% by default, are reported); each candidate is fitted from several starting widths
  (multi-start) and the best fit counts; optional sigma clipping between
  steps; weak lines with shared kinematics are proposed together
  (`LineGroupRule`); ties (`Tie`, e.g. narrow lines sharing redshift and
  width) are hypotheses tested once the search converges and kept only if
  the data accept them; parameters at a prior bound are flagged. Quasar
  rules (lines, the CIII] / SiIII] / AlIII complex, iron templates,
  Balmer continuum, the narrow Hbeta / Halpha / [OIII] group and tie with
  the narrow limit `qso_rules.NARROW_FWHM_MAX = 1200` km/s) are in
  `sculptor_extensions.qso_rules`. See the example below.
- `sculptor.component.FitComponent`: one model function and the mapping of
  its arguments onto parameters (e.g. a redshift shared by several lines).
- `sculptor.parameter.FitParameter` and `sculptor.prior` (`UniformPrior`,
  `GaussianPrior`).
- `sculptor.analysis` / `sculptor.measurements`: model functions (power law,
  Gaussian line, broadened templates) and measurements on the posterior
  samples (fluxes, equivalent widths, FWHM, luminosities, ...).

The optional `sculptor_extensions` package adds type-I quasar models and
analysis: iron and Balmer continuum templates, emission line setups
(`sculptor_extensions.qso_presets`), single-epoch black hole masses and
bolometric luminosities (`sculptor_extensions.qso`, unit-aware with
astropy).

A minimal fit (see `examples/example_fit_setup_script.py` for a full one):

```python
from speconed import speconed as sod

from sculptor import analysis as ana
from sculptor.component import FitComponent
from sculptor.model import FitModel
from sculptor.prior import UniformPrior

spec = sod.SpecOneD()
spec.read_sdss_fits("src/sculptor/data/example_spectra/J030341.04-002321.8_0.fits")

cont = FitComponent("PL", ana.power_law_at2500)
params = cont.create_params()          # PL_redsh, PL_amp, PL_slope
params["PL_redsh"].value, params["PL_redsh"].vary = 3.227, False
params["PL_amp"].value, params["PL_amp"].prior = 10, UniformPrior("amp", 0, 100)
params["PL_slope"].value, params["PL_slope"].prior = -1.5, UniformPrior("slope", -3, 0)

line = FitComponent("CIV", ana.line_model_gaussian)
params.update(line.create_params())    # CIV_redsh, CIV_flux, CIV_cen, CIV_fwhm_km_s
params["CIV_cen"].value, params["CIV_cen"].vary = 1549.06, False
params["CIV_redsh"].value = 3.22
params["CIV_redsh"].prior = UniformPrior("redsh", 3.18, 3.26)
params["CIV_flux"].value = 1000
params["CIV_flux"].prior = UniformPrior("flux", 0, 5000)
params["CIV_fwhm_km_s"].value = 3000
params["CIV_fwhm_km_s"].prior = UniformPrior("fwhm", 500, 10000)

model = FitModel(components=[cont, line], parameters=params, spectrum=spec)
model.add_wavelength_range_to_fit_mask(5600, 5800)  # continuum
model.add_wavelength_range_to_fit_mask(6400, 6700)  # CIV

# Quick fit: the maximum of the posterior, in well under a second
result = model.fit("quick")
print(result)
model.plot_quick_fit_result()

# MCMC with the walkers started around the quick fit
model.fit("quick+mcmc", nsteps=2000, nwalkers=50)
model.plot_mcmc_result(discard=500)
```

Automatic model building, starting from a model with only the continuum:

```python
from sculptor.autofit import AutoFit, GaussianLineRule

rules = [GaussianLineRule("line", redshift=3.22, fwhm_range=(300, 15000))]
# or, for a quasar: from sculptor_extensions.qso_rules import qso_rules
# rules = qso_rules(redshift=3.22)

# BIC with a threshold calibrated on simulations (noise with the residuals'
# correlations): one model per quantile, the first one selected
auto = AutoFit(
    base_model,
    rules,
    clip={"low": 3, "grow": 2},
    calibration={"quantiles": (0.95, 0.99), "n_sims": 200},
)
result = auto.run()
print(result)                     # steps, calibration, models by quantile
strict = result.by_quantile[0.99]
# Faster, without calibration: AutoFit(base_model, rules, threshold=10)

result.model.fit("quick+mcmc", nsteps=2000, nwalkers=50)
```

![Sculptor example fit][logo]

[logo]: https://github.com/jtschindler/sculptor/blob/main/docs/images/example_fit.png "A Sculptor example fit of a quasar spectrum."

## Installation

Sculptor and its optional extensions can be installed with `pip`, `conda`, or [`uv`](https://docs.astral.sh/uv/).

### Using uv (recommended)

```bash
git clone https://github.com/jtschindler/sculptor.git
cd sculptor
uv sync
```

This installs the core `sculptor` package, `sculptor_extensions` and the
development tools (pytest, ruff, prek). To install only the core package:

```bash
uv sync --no-dev
```

Run the tests with `uv run pytest`.

### Using pip

```bash
git clone https://github.com/jtschindler/sculptor.git
cd sculptor
python -m venv .venv       # Unix systems
source .venv/bin/activate  # Unix systems, not sure about Windows!
pip install .
```

The `sculptor_extensions` module is optional and installed separately:

```bash
pip install ./sculptor_extensions
```

### Using conda

Conda is used here to manage the Python environment; the package itself is still installed with `pip`:

```bash
conda create -n sculptor python=3.13
conda activate sculptor
pip install .
# optional:
pip install ./sculptor_extensions
```

If you are interested in being involved with this project, please contact Jan-Torge Schindler via [github](https://github.com/jtschindler/sculptor).

**Disclaimer:**
Version 2 is in beta: the API may still change before 2.0.0. See
`Changelog.md` for changes, including those that break code written for
earlier versions.


[Sculptor's ReadTheDocs]: https://sculptor.readthedocs.io/en/latest/
