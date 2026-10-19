# TODO: `speconed` / `SpecOneD`

Open issues with the standalone `speconed` package (1.0.1, the one sculptor
depends on) and with the bundled v1 copy it replaced
(`deprecated/src/sculptor/dep_v1/speconed.py`). None of them affects code
paths sculptor currently uses: sculptor only calls `SpecOneD(...)`, `copy`,
`trim_dispersion`, `average_fluxden`, `peak_dispersion`,
`to_fluxden_per_unit_frequency_cgs` and `broaden_by_gaussian`, which are
identical in both copies.

## Bugs in the package

- [ ] **`np.trapz` was removed in numpy 2.4.** These methods raise
  `AttributeError` on the current numpy (2.5):
  `calculate_passband_flux`, `calculate_passband_ab_magnitude`,
  `renormalize_by_spectrum`. Replace with `np.trapezoid`.
- [ ] **Docs tools declared as runtime dependencies:** `nbsphinx`,
  `nbsphinx-link`, `notebook`, `sphinx-rtd-theme`. They pull Jupyter, Sphinx
  and `docutils` into every environment that installs sculptor. They belong
  in an optional `docs` extra.

## Differences between the bundled copy and the package

The two files differ by about 30 lines; everything else is identical.

| Area | Bundled copy | Package 1.0.1 |
|---|---|---|
| `SpecOneD.plot()` | `step_plot=True` option (step plot, wavelength at bin middle) | No `step_plot`; always a line plot |
| Passband flux | `calculate_passband_flux_density`; integrand divided by λ | Renamed `calculate_passband_flux`; no 1/λ ("written for passbands in quantum efficiency") |
| AB magnitude (`calculate_passband_ab_magnitude`) | 1/λ in both integrals; cites eq. 4 of astro-ph/0210394 | No 1/λ |
| `load_passband` | Broken: uses `pkg_resources`, whose import is commented out (`NameError`) | Works; reads `speconed`'s own `data/passbands/` |

- [ ] **Decide which photometry is correct and fix upstream.** The 1/λ
  weighting in the bundled copy matches the usual photon-counting AB
  definition; the package version gives different magnitudes whenever the
  spectrum is not flat in f_ν across the band. Check against the cited
  reference before changing anything.
- [ ] Port the `step_plot` option to the package's `plot()` if it is wanted.

## Passbands

- [ ] **Euclid NISP passbands.** `sculptor/data/passbands/` (63 files) moved to
  `deprecated/src/sculptor/data/passbands/`: only the bundled `load_passband`
  read it. The package ships the same filters except
  `Euclid_NISP-Y.dat`, `Euclid_NISP-J.dat`, `Euclid_NISP-H.dat`. Contribute
  those three upstream, then delete the deprecated folder.

## Missing functionality

- [ ] **`SpecOneD.broaden_by_resolution` doesn't exist** in either copy, but
  `examples/test_resolution.py` calls it (it was probably written against a
  local development version). Either add it to `speconed`, or rewrite the
  script to use `sculptor.utils.broaden_spectrum`, which does the same
  resolution-dependent Gaussian convolution.

## Observations

- A scaled flux density unit (e.g. `1e-17 erg/s/cm2/AA`) is stored as an
  `astropy.units.Quantity`, so `value * spec.fluxden_unit` folds the scale
  factor into the result (`2.0 * (1e-17 erg/...)` -> `2e-17 erg/...`).
  Correct, but worth documenting: comparing `.unit` fails, comparing via
  `.to(...)` works.
- `SpecOneD()` without units prints a `[WARNING]` and silently defaults to
  `erg/s/cm^2/AA` and `AA`.
