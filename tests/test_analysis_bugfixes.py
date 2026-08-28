"""Regression test for a bug fix in `sculptor.analysis.get_line_snr`.

`get_line_snr` is meant to compute, for each MCMC sample `idx`, a line
integration window `[l_min, l_max]` derived from that sample's FWHM
(`line_dict[line]['fwhm'][idx]`), unless the caller pinned a fixed window.
The old code cached the computed window back into `line_dict[line]['l_min']`
/`['l_max']`, so after the first sample those keys were no longer `None` and
the per-sample computation was skipped for every subsequent sample -- the
window silently froze at whatever the first sample produced, and the input
`line_dict` was mutated as a side effect. The fix keeps the computed bounds
local to each iteration so every sample gets its own window, and it also
reuses the already-computed module-level `c_km_s` constant instead of
recomputing `const.c.to('km/s').value` on every iteration.
"""

import numpy as np
from astropy import constants as const

from sculptor import analysis

C_KM_S = const.c.to("km/s").value


class _FakeSpec:
    def __init__(self, dispersion, fluxden, fluxden_err):
        self.dispersion = np.asarray(dispersion, dtype=float)
        self.fluxden = np.asarray(fluxden, dtype=float)
        self.fluxden_err = np.asarray(fluxden_err, dtype=float)

    def copy(self):
        return _FakeSpec(
            self.dispersion.copy(), self.fluxden.copy(), self.fluxden_err.copy()
        )

    def trim_dispersion(self, disp_range):
        lo, hi = sorted(disp_range)
        mask = (self.dispersion >= lo) & (self.dispersion <= hi)
        return _FakeSpec(
            self.dispersion[mask], self.fluxden[mask], self.fluxden_err[mask]
        )


class _FakeModel:
    def __init__(self, spec):
        self.spec = spec

    def eval(self, dispersion, params, components=None, broaden_by_resolution=False):
        return np.zeros_like(dispersion, dtype=float)


def test_get_line_snr_uses_per_sample_fwhm_window_and_does_not_mutate_input():
    cen = 5000.0
    redshift = 0.0
    sigma = 2.0

    # Sample 0 has a narrow FWHM (narrow window around `cen`); sample 1 has
    # a much wider FWHM. A spike placed just outside sample 0's window but
    # inside sample 1's window must only be picked up for sample 1 -- if the
    # window were frozen at sample 0's (the bug), sample 1 would miss it too.
    fwhm = np.array([300.0, 30000.0])
    spike_wave = 5100.0

    disp = np.linspace(4000, 6000, 20001)
    flux = np.zeros_like(disp)
    flux[np.argmin(np.abs(disp - spike_wave))] = 100.0
    fluxerr = np.ones_like(disp)

    spec = _FakeSpec(disp, flux, fluxerr)
    model = _FakeModel(spec)

    line_dict = {
        "line1": {"cen": cen, "fwhm": fwhm, "l_min": None, "l_max": None},
    }
    flat_chain = np.zeros((2, 1))

    line_snr, peak_snr = analysis.get_line_snr(
        model,
        flat_chain,
        cont_component=None,
        redshift=redshift,
        line_dict=line_dict,
        sigma=sigma,
    )

    # Sample 0's narrow window must not capture the spike.
    assert line_snr[0, 0] == 0.0
    assert peak_snr[0, 0] == 0.0

    # Sample 1's wide window must capture the spike.
    assert line_snr[0, 1] > 0.5
    assert peak_snr[0, 1] > 50.0

    # The input dict must not be mutated with a cached window.
    assert line_dict["line1"]["l_min"] is None
    assert line_dict["line1"]["l_max"] is None


def test_get_line_snr_respects_fixed_window_when_provided():
    cen = 5000.0
    redshift = 0.0
    sigma = 2.0

    # FWHM varies across samples, but l_min/l_max are pinned, so the window
    # used must be the same fixed one for every sample.
    fwhm = np.array([300.0, 30000.0])
    fixed_l_min, fixed_l_max = 5090.0, 5110.0
    spike_wave = 5100.0  # inside the fixed window regardless of sample.

    disp = np.linspace(4000, 6000, 20001)
    flux = np.zeros_like(disp)
    flux[np.argmin(np.abs(disp - spike_wave))] = 100.0
    fluxerr = np.ones_like(disp)

    spec = _FakeSpec(disp, flux, fluxerr)
    model = _FakeModel(spec)

    line_dict = {
        "line1": {
            "cen": cen,
            "fwhm": fwhm,
            "l_min": fixed_l_min,
            "l_max": fixed_l_max,
        },
    }
    flat_chain = np.zeros((2, 1))

    line_snr, peak_snr = analysis.get_line_snr(
        model,
        flat_chain,
        cont_component=None,
        redshift=redshift,
        line_dict=line_dict,
        sigma=sigma,
    )

    # Both samples use the same fixed window, so both must see the spike.
    assert line_snr[0, 0] > 0.0
    assert line_snr[0, 1] > 0.0
    assert peak_snr[0, 0] > 50.0
    assert peak_snr[0, 1] > 50.0

    assert line_dict["line1"]["l_min"] == fixed_l_min
    assert line_dict["line1"]["l_max"] == fixed_l_max
