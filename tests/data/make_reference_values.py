"""Generate `reference_values.npz` from the original (pre-port) code.

The QSO models were ported from two reference copies, which were kept in
`tests/ground_truth/` until 2026-10-20 and are still in the git history:

- `fgu`: `tests/ground_truth/fgu/{qso,Qso_autofit,out}.py`, the source of
  `sculptor_extensions` (they import the v1 `sculptor.dep_v1` modules, now
  in `deprecated/src/sculptor/`, which is not tracked);
- `v2_dev`: `tests/ground_truth/v2_dev/`, upstream jtschindler/sculptor
  `v2_dev` at a5a1749.

`tests/test_reference_values.py` compares the maintained code with the values
written here (the 1-sigma helpers are not compared: they use the 15.87 / 84.13
percentiles since 2026-10-20, the originals 15.9 / 84.1), so the tests no longer need those copies. To regenerate, check
out the copies from commit 70a5f7e (`git checkout 70a5f7e -- tests/ground_truth`),
provide the v1 modules in `deprecated/src/sculptor/`, and run

    uv run python tests/data/make_reference_values.py

Each key is `<test>__<quantity>`; the inputs are defined in `INPUTS`, which
the tests import, so inputs and stored outputs cannot drift apart.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
from astropy import units as u

REPO_ROOT = Path(__file__).resolve().parents[2]
FGU_DIR = REPO_ROOT / "tests" / "ground_truth" / "fgu"
V2_DEV_DIR = REPO_ROOT / "tests" / "ground_truth" / "v2_dev"
DEPRECATED_SCULPTOR_DIR = REPO_ROOT / "deprecated" / "src" / "sculptor"
DATA_DIR = REPO_ROOT / "src" / "sculptor" / "data"
OUT = Path(__file__).resolve().parent / "reference_values.npz"

KM_S = u.km / u.s
LWAV = u.erg / u.s / u.AA

# Inputs shared with tests/test_reference_values.py.
INPUTS = {
    "power_law": {
        "x": np.linspace(1000, 5000, 200),
        "amp": 1.5,
        "slope": -1.7,
        "redsh": 2.0,
    },
    "balmer": {
        "x": np.linspace(2000, 14000, 2000),
        "redsh": 2.0,
        "amp_be": 1.0,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
    },
    "gaussian": {
        "x": np.linspace(1500, 1600, 300),
        "redsh": 2.0,
        "flux": 10.0,
        "cen": 1549.06,
        "fwhm_km_s": 2000.0,
    },
    "doublet": {
        "x": np.linspace(6500, 6620, 300),
        "redsh": 2.0,
        "flux": 10.0,
        "fwhm_km_s": 500.0,
    },
    "planck": {"wavelength_aa": np.linspace(2000, 5000, 200), "Te": 15000.0},
    "fractional_bc": {
        "x": np.linspace(2000, 5000, 200),
        "amp": 1.0,
        "slope": -1.5,
        "redsh": 2.0,
        "f": 0.3,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
    },
    "voigt": {
        "wave_obs_aa": np.linspace(1210, 1220, 200),
        "z_abs": 2.0,
        "rest_wave_aa": 1215.67,
        "log_N": 20.0,
        "b_kms": 30.0,
        "osc_strength": 0.4164,
        "damping_gamma": 6.265e8,
    },
    "lya": {
        "x": np.linspace(1210, 1220, 200),
        "z_abs": 2.0,
        "log_N": 20.0,
        "b_kms": 30.0,
    },
    "hol": {
        "x": np.linspace(2800, 4200, 500),
        "redsh": 0.0,
        "amp_be": 1.0,
        "Te": 15000.0,
        "tau_be": 1.0,
        "lambda_be": 3646.0,
        "fwhm": 2000.0,
    },
    "broadening": {
        "x": np.linspace(4900, 5100, 400),
        "values": [10.0, 5000.0, 5.0],
        "resolution": 20.0,
    },
    "template": {
        "x": np.linspace(4200, 4800, 150),
        "amp": 2.0,
        "redsh": 0.05,
        "fwhm": 800.0,
        "intr_fwhm": 100.0,
    },
    "db20": {"fwhm": 4500.0, "lwav": 3e42, "virial_factors": [1.0, 4.3]},
    "iron_setups": [
        "setup_iron_template_VW01_UV",
        "setup_iron_template_BG92_OPT",
        "setup_iron_template_Tsuz_UV",
        "setup_iron_template_Tsuz_OPT",
    ],
}


def template_inputs():
    """Template arrays for the `template_model` comparison."""
    rng = np.random.default_rng(0)
    templ_disp = np.linspace(4000, 5000, 200)
    templ_fluxden = np.exp(-0.5 * ((templ_disp - 4500) / 50) ** 2) + 0.01 * rng.normal(
        size=templ_disp.size
    )
    return templ_disp, templ_fluxden


def _z(kwargs):
    """The reference copies call the redshift `z`."""
    return {("z" if k == "redsh" else k): v for k, v in kwargs.items()}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


def main():
    import sculptor
    import sculptor.component as sccomp
    import sculptor.parameter as scpar

    sculptor.__path__.append(str(DEPRECATED_SCULPTOR_DIR))
    legacy_hol = DATA_DIR / "wave_2.txt"  # out.py loads the HOL data under this name
    created = not legacy_hol.exists()
    if created:
        shutil.copyfile(DATA_DIR / "balmer_hol_storey_hummer1995.txt", legacy_hol)
    try:
        qso = _load("_fgu_qso", FGU_DIR / "qso.py")
        qa = _load("_fgu_qso_autofit", FGU_DIR / "Qso_autofit.py")
        out = _load("_fgu_out", FGU_DIR / "out.py")
        up_model = _load("_v2_model", V2_DEV_DIR / "src" / "sculptor" / "model.py")
        up_qso = _load("_v2_qso", V2_DEV_DIR / "sculptor_extensions" / "qso.py")
    finally:
        if created:
            legacy_hol.unlink()

    ref = {}
    i = INPUTS

    # fgu qso.py
    p = i["power_law"]
    ref["power_law__y"] = qso.power_law_at_2500(
        p["x"], p["amp"], p["slope"], p["redsh"]
    )
    b = dict(i["balmer"])
    ref["balmer__y"] = qso.balmer_continuum_model(b.pop("x"), **_z(b))
    g = dict(i["gaussian"])
    ref["gaussian__y"] = qso.line_model_gaussian(g.pop("x"), **_z(g))
    ref["velocity_shift__km_s"] = qso.calc_velocity_shifts(
        2.005, z_sys=2.0, relativistic=True
    ).to_value(KM_S)
    mass = 1e8 * u.Msun
    ref["eddington__lum_erg_s"] = qso.calc_eddington_luminosity(mass).to_value(
        u.erg / u.s
    )
    ref["eddington__ratio"] = (
        qso.calc_eddington_ratio(1e45 * u.erg / u.s, mass).decompose().value
    )
    lwav = 1e44 * LWAV
    for name, args in {
        "hbeta_vp06": (qso.se_bhmass_hbeta_vp06, (2000 * KM_S, lwav, 5100 * u.AA)),
        "mgii_s11": (qso.se_bhmass_mgii_s11_fwhm, (3000 * KM_S, lwav, 3000 * u.AA)),
        "mgii_vo09": (qso.se_bhmass_mgii_vo09_fwhm, (3000 * KM_S, lwav, 3000 * u.AA)),
    }.items():
        func, a = args
        m, r = func(*a)
        ref[f"bhmass_{name}__msun"] = m.to_value(u.Msun)
        ref[f"bhmass_{name}__ref"] = np.array(r)
    lbol, r = qso.calc_bolometric_luminosity(lwav, cont_wav=1350 * u.AA)
    ref["lbol_1350__erg_s"] = lbol.to_value(u.erg / u.s)
    ref["lbol_1350__ref"] = np.array(r)
    ref["civ_blueshift__km_s"] = qso.correct_CIV_fwhm_for_blueshift(
        5000 * KM_S, 500 * KM_S
    ).to_value(KM_S)

    # fgu Qso_autofit.py
    d = i["doublet"]
    ref["nii_doublet__y"] = qa.line_model_gaussian_NII_doublet(
        d["x"], d["redsh"], d["flux"], d["fwhm_km_s"], 2.96
    )
    ref["sii_doublet__y"] = qa.line_model_gaussian_SII_doublet(
        d["x"], d["redsh"], d["flux"], d["fwhm_km_s"], 1.0
    )

    captured = {}

    def _capture(prefix, disp, flux, *args, **kwargs):
        captured["disp"], captured["flux"] = disp, flux
        return None, None

    qa.initialize_Fe = _capture
    for name in i["iron_setups"]:
        getattr(qa, name)("Fe")
        ref[f"iron_{name}__disp"] = captured["disp"]
        ref[f"iron_{name}__flux"] = captured["flux"]

    def _stub(prefix, params):
        return prefix, prefix

    class _Recorder:
        def __init__(self):
            self.names = []

        def add_component(self, comp):
            self.names.append(comp)

        def add_parameters(self, pars):
            pass

    class _AnyDict(dict):
        def __missing__(self, key):
            self[key] = _AnyDict()
            return self[key]

    qa.initialize_gaussian_line_model = _stub
    rec = _Recorder()
    qa.setup_MgII_lines(None, _AnyDict(), narrow=True, MgII_broad=2, model=rec)
    ref["setup_mgii__components"] = np.array(rec.names)
    rec = _Recorder()
    qa.setup_hb_o3_lines(
        None,
        _AnyDict(),
        narrow=True,
        Hbeta_broad=1,
        Hbeta_broad_shifted=1,
        OIII_wind=1,
        model=rec,
    )
    ref["setup_hb_o3__components"] = np.array(rec.names)

    # fgu out.py (qso_advanced_models)
    pl = i["planck"]
    ref["planck__y"] = out._planck_lambda_cgs(pl["wavelength_aa"], Te=pl["Te"])
    fb = dict(i["fractional_bc"])
    ref["fractional_bc__y"] = out.power_law_at_2500_plus_fractional_bc(
        fb.pop("x"), **_z(fb)
    )
    v = dict(i["voigt"])
    ref["voigt__tau"] = out._voigt_optical_depth(v.pop("wave_obs_aa"), **v)
    ly = dict(i["lya"])
    ref["lya__transmission"] = out.lya_absorption_model(ly.pop("x"), **ly)
    h = dict(i["hol"])
    ref["hol__y"] = out.BalmerContinuumWithHOL()(h.pop("x"), **_z(h))

    # v2_dev model.py: FitModel.eval broadens by the resolution by default
    br = i["broadening"]
    comp = sccomp.FitComponent("line", _gaussian)
    params = comp.create_params()
    for name, value in zip(("amp", "cen", "sigma"), br["values"], strict=True):
        params[f"line_{name}"] = scpar.FitParameter(f"line_{name}", value)
    model = up_model.FitModel()
    model.add_component(comp)
    model.add_parameters(params)
    model.resolution = br["resolution"]
    model.get_params_to_sample()
    ref["broadening__y"] = model.eval(br["x"], br["values"])

    # v2_dev qso.py
    t = i["template"]
    templ_disp, templ_fluxden = template_inputs()
    ref["template__y"] = up_qso.template_model(
        t["x"],
        amp=t["amp"],
        z=t["redsh"],
        fwhm=t["fwhm"],
        intr_fwhm=t["intr_fwhm"],
        templ_disp=templ_disp,
        templ_fluxden=templ_fluxden,
    )
    db = i["db20"]
    masses = []
    for f in db["virial_factors"]:
        m, r = up_qso.se_bhmass_db20(
            db["fwhm"] * KM_S, db["lwav"] * u.erg / u.s, virial_factor=f
        )
        masses.append(m.to_value(u.Msun))
    ref["db20__msun"] = np.array(masses)
    ref["db20__ref"] = np.array(r)
    lb, r = up_qso.calc_bolometric_luminosity_db20(db["lwav"] * u.erg / u.s)
    ref["db20_lbol__erg_s"] = lb.to_value(u.erg / u.s)
    ref["db20_lbol__ref"] = np.array(r)

    np.savez_compressed(OUT, **{k: np.asarray(v) for k, v in ref.items()})
    print(f"wrote {len(ref)} reference values to {OUT}")


if __name__ == "__main__":
    main()
