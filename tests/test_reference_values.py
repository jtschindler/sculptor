"""The maintained models and relations reproduce the original code they were
ported from: the `fgu` reference copies (`qso.py`, `qso_autofit.py`,
`out.py`) and upstream `v2_dev` (a5a1749).

The original outputs are stored in `tests/data/reference_values.npz`; see
`tests/data/make_reference_values.py` for where they come from and how to
regenerate them. Inputs are shared with that script (`INPUTS`).
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u

import sculptor.component as sccomp
import sculptor.parameter as scpar
from sculptor import analysis as ana
from sculptor.model import FitModel
from sculptor_extensions import qso, qso_presets
from sculptor_extensions import qso_advanced_models as qam
from sculptor_extensions import qso_autofit as qa

DATA_DIR = Path(__file__).resolve().parent / "data"
KM_S = u.km / u.s
LWAV = u.erg / u.s / u.AA


def _load_generator():
    path = DATA_DIR / "make_reference_values.py"
    spec = importlib.util.spec_from_file_location("_make_reference_values", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GEN = _load_generator()
INPUTS = GEN.INPUTS
REF = np.load(DATA_DIR / "reference_values.npz")


def _close(new, key, rtol=1e-12):
    np.testing.assert_allclose(new, REF[key], rtol=rtol)


def _split_x(name):
    kwargs = dict(INPUTS[name])
    return kwargs.pop("x"), kwargs


# --- model functions -----------------------------------------------------------


def test_power_law():
    x, k = _split_x("power_law")
    _close(ana.power_law_at2500(x, **k), "power_law__y")


def test_balmer_continuum():
    x, k = _split_x("balmer")
    assert np.count_nonzero(REF["balmer__y"]) > 100  # spans the Balmer edge
    _close(qam.fast_balmer_continuum_model(x, **k), "balmer__y")


def test_gaussian_line():
    x, k = _split_x("gaussian")
    _close(ana.line_model_gaussian(x, **k), "gaussian__y")


def test_tied_doublets():
    x, k = _split_x("doublet")
    args = (x, k["redsh"], k["flux"], k["fwhm_km_s"])
    _close(qa.line_model_gaussian_NII_doublet(*args, 2.96), "nii_doublet__y")
    _close(qa.line_model_gaussian_SII_doublet(*args, 1.0), "sii_doublet__y")


def test_planck_function():
    p = INPUTS["planck"]
    _close(qam._planck_lambda_cgs(p["wavelength_aa"], Te=p["Te"]), "planck__y")


def test_power_law_plus_fractional_balmer_continuum():
    x, k = _split_x("fractional_bc")
    _close(qam.power_law_at_2500_plus_fractional_bc(x, **k), "fractional_bc__y", 1e-10)


def test_voigt_optical_depth():
    k = dict(INPUTS["voigt"])
    _close(qam._voigt_optical_depth(k.pop("wave_obs_aa"), **k), "voigt__tau", 1e-10)


def test_lya_absorption():
    x, k = _split_x("lya")
    new = qam.lya_absorption_model(x, **k)
    _close(new, "lya__transmission", 1e-10)
    assert np.all((new >= 0.0) & (new <= 1.0))


def test_balmer_continuum_with_hol():
    x, k = _split_x("hol")
    _close(qam.BalmerContinuumWithHOL()(x, **k), "hol__y", 1e-8)
    cached = qam.CachedBalmerContinuumWithHOL()
    assert isinstance(cached, qam.BalmerContinuumWithHOL)
    for _ in range(2):  # computed, then from the cache
        _close(cached(x, **k), "hol__y", 1e-8)


def test_template_model():
    t = INPUTS["template"]
    templ_disp, templ_fluxden = GEN.template_inputs()
    new = ana.template_model(
        t["x"],
        amp=t["amp"],
        redsh=t["redsh"],
        fwhm=t["fwhm"],
        intr_fwhm=t["intr_fwhm"],
        templ_disp=templ_disp,
        templ_fluxden=templ_fluxden,
    )
    _close(new, "template__y", 1e-8)


def _gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


@pytest.mark.parametrize("as_array", [False, True])
def test_eval_broadens_by_resolution_by_default(as_array):
    """`FitModel.eval` broadens by `resolution` by default, for a scalar or
    a per-pixel resolution, like upstream `v2_dev`. The values differ from
    upstream's (pixel-center convolution): the model is now convolved on a
    fine grid and averaged over the pixels (`sculptor.resolution`, checked
    against analytic results in test_broadening.py)."""
    b = INPUTS["broadening"]
    comp = sccomp.FitComponent("line", _gaussian)
    params = comp.create_params()
    for name, value in zip(("amp", "cen", "sigma"), b["values"], strict=True):
        params[f"line_{name}"] = scpar.FitParameter(f"line_{name}", value)
    resolution = b["resolution"]
    if as_array:
        resolution = np.full_like(b["x"], resolution)
    model = FitModel(components=[comp], parameters=params, resolution=resolution)

    y = model.eval(b["x"], b["values"])
    assert not np.allclose(
        y, model.eval(b["x"], b["values"], broaden_by_resolution=False)
    )
    # R = 20: the line spread function is wider than the grid; the line
    # is spread out
    raw = model.eval(b["x"], b["values"], broaden_by_resolution=False)
    assert y.max() < 0.2 * raw.max()


# --- relations -------------------------------------------------------------------


def test_velocity_shift_and_eddington():
    _close(
        qso.calc_velocity_shifts(2.005, z_sys=2.0).to_value(KM_S),
        "velocity_shift__km_s",
    )
    mass = 1e8 * u.Msun
    _close(
        qso.calc_eddington_luminosity(mass).to_value(u.erg / u.s),
        "eddington__lum_erg_s",
    )
    _close(qso.calc_eddington_ratio(1e45 * u.erg / u.s, mass).value, "eddington__ratio")


@pytest.mark.parametrize(
    "name,func,args",
    [
        ("hbeta_vp06", qso.se_bhmass_hbeta_vp06, (2000, 5100)),
        ("mgii_s11", qso.se_bhmass_mgii_s11_fwhm, (3000, 3000)),
        ("mgii_vo09", qso.se_bhmass_mgii_vo09_fwhm, (3000, 3000)),
    ],
)
def test_bh_masses(name, func, args):
    fwhm, wav = args
    mass, ref = func(fwhm * KM_S, 1e44 * LWAV, wav * u.AA)
    _close(mass.to_value(u.Msun), f"bhmass_{name}__msun")
    assert ref == str(REF[f"bhmass_{name}__ref"])


def test_bolometric_luminosity_and_civ_blueshift():
    lbol, ref = qso.calc_bolometric_luminosity(1e44 * LWAV, 1350 * u.AA)
    _close(lbol.to_value(u.erg / u.s), "lbol_1350__erg_s")
    assert ref == str(REF["lbol_1350__ref"])
    _close(
        qso.correct_CIV_fwhm_for_blueshift(5000 * KM_S, 500 * KM_S).to_value(KM_S),
        "civ_blueshift__km_s",
    )


def test_dalla_bonta_2020():
    d = INPUTS["db20"]
    masses = []
    for f in d["virial_factors"]:
        mass, ref = qso.se_bhmass_db20(
            d["fwhm"] * KM_S, d["lwav"] * u.erg / u.s, virial_factor=f
        )
        masses.append(mass.to_value(u.Msun))
        assert ref == str(REF["db20__ref"])
    _close(masses, "db20__msun", 1e-10)

    lbol, ref = qso.calc_bolometric_luminosity_db20(d["lwav"] * u.erg / u.s)
    _close(lbol.to_value(u.erg / u.s), "db20_lbol__erg_s")
    assert ref == str(REF["db20_lbol__ref"])


# --- fit helpers and setups ------------------------------------------------------


@pytest.mark.parametrize("name", INPUTS["iron_setups"])
def test_iron_setups_select_the_original_template_range(name, monkeypatch):
    captured = {}

    def _capture(prefix, disp, flux, fe_dict=None):
        captured["disp"], captured["flux"] = disp, flux
        return None, None

    monkeypatch.setattr(qa, "initialize_Fe_fast", _capture)
    getattr(qso_presets, name)("Fe")
    _close(captured["disp"], f"iron_{name}__disp")
    _close(captured["flux"], f"iron_{name}__flux")


class _AnyDict(dict):
    def __missing__(self, key):
        self[key] = _AnyDict()
        return self[key]


class _Recorder:
    def __init__(self):
        self.names = []

    def add_component(self, comp):
        self.names.append(comp)

    def add_parameters(self, pars):
        pass


def test_line_setups_add_the_original_components(monkeypatch):
    monkeypatch.setattr(
        qa, "initialize_gaussian_line_model", lambda prefix, params: (prefix, prefix)
    )
    rec = _Recorder()
    qso_presets.setup_MgII_lines(None, _AnyDict(), narrow=True, MgII_broad=2, model=rec)
    assert rec.names == list(REF["setup_mgii__components"])

    rec = _Recorder()
    qso_presets.setup_Hbeta_OIII_lines(
        None,
        _AnyDict(),
        narrow=True,
        Hbeta_broad=1,
        Hbeta_broadshifted=1,
        OIII_wind=1,
        model=rec,
    )
    # The original's "broad_shifted" components are called "broadshifted".
    expected = [
        n.replace("broad_shifted", "broadshifted")
        for n in REF["setup_hb_o3__components"]
    ]
    assert rec.names == expected
