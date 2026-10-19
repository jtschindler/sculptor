"""Consistency checks for changes made to bring this repo to feature parity
with the online `v2_dev` branch of https://github.com/jtschindler/sculptor.

The reference files were copied from that branch into the test-only,
repo-local ground-truth directory `tests/ground_truth/v2_dev/`, mirroring
the existing `tests/ground_truth/fgu/`-style comparison pattern already
used in this test suite.

Per the user's instructions for this parity pass: where local and upstream
had genuinely different implementations, keep the faster one; where only
default parameter values differed, upstream is ground truth. The actual
comparison work found no "two different algorithms, pick the faster" cases
and no default-value mismatches in `qso.py`/`Qso_autofit.py` -- local was
already a strict superset of upstream there (plus some bug fixes upstream
still lacks). The real behavioral changes made in this pass were:
  - `sculptor.model.FitModel.eval`: default `broaden_by_resolution` was
    `False` locally, `True` upstream (a genuine default-value difference) --
    fixed to match upstream. Upstream also fixed handling of a scalar
    `self.resolution` (broadcasting it to an array before broadening) --
    ported in too.
  - `sculptor_extensions.Qso_autofit.get_line_prop_MgII`: the FeII/MgII
    flux-ratio calculation, which upstream still has active, had been
    disabled locally during an earlier port; restored to match upstream
    (per explicit user confirmation).
  - `sculptor_extensions.qso`: switched from the bundled
    `sculptor.dep_v1.speconed` to the standalone `speconed` package for
    `SpecOneD`, matching upstream (both expose a compatible API).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import sculptor_extensions.qso as new_qso
import sculptor_extensions.Qso_autofit as new_qso_autofit
from astropy import units as u

import sculptor.component as new_component
import sculptor.model as new_model
import sculptor.parameter as new_parameter
from sculptor import analysis as sculptor_analysis

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_DEV_DIR = REPO_ROOT / "tests" / "ground_truth" / "v2_dev"


def _load_module_from_path(module_name, path):
    """Import a standalone .py file as a module, without installing it."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def upstream_model(deprecated_sculptor):
    return _load_module_from_path(
        "_v2_dev_model", V2_DEV_DIR / "src" / "sculptor" / "model.py"
    )


@pytest.fixture(scope="module")
def upstream_qso():
    return _load_module_from_path(
        "_v2_dev_qso", V2_DEV_DIR / "sculptor_extensions" / "qso.py"
    )


@pytest.fixture(scope="module")
def upstream_qso_autofit(deprecated_sculptor):
    return _load_module_from_path(
        "_v2_dev_qso_autofit", V2_DEV_DIR / "sculptor_extensions" / "Qso_autofit.py"
    )


def _gaussian(x, amp, cen, sigma):
    return amp * np.exp(-0.5 * ((x - cen) / sigma) ** 2)


def _build_model(model_module, resolution):
    """One FitModel (local or upstream), with a single Gaussian component,
    and the given `resolution` value. Uses local's FitComponent/FitParameter
    (confirmed byte-identical to upstream's) since only FitModel.eval is
    under test here.
    """
    comp = new_component.FitComponent("line", _gaussian)
    params = comp.create_params()
    params["line_amp"] = new_parameter.FitParameter("line_amp", 10.0)
    params["line_cen"] = new_parameter.FitParameter("line_cen", 5000.0)
    params["line_sigma"] = new_parameter.FitParameter("line_sigma", 5.0)

    model = model_module.FitModel()
    model.add_component(comp)
    model.add_parameters(params)
    model.resolution = resolution
    model.get_params_to_sample()
    return model


# --------------------------------------------------------------------------
# sculptor.model.FitModel.eval -- default broaden_by_resolution + scalar
# resolution handling, fixed to match upstream.
# --------------------------------------------------------------------------


def test_eval_default_broadens_by_resolution_matching_upstream(upstream_model):
    """Upstream's `eval` defaults to `broaden_by_resolution=True`; local
    used to default to `False`. Confirm local now matches upstream's
    default behavior (broadening applied without passing the kwarg).
    """
    x = np.linspace(4900, 5100, 400)
    values = [10.0, 5000.0, 5.0]

    local = _build_model(new_model, resolution=20.0)
    upstream = _build_model(upstream_model, resolution=20.0)

    y_default_local = local.eval(x, values)
    y_default_upstream = upstream.eval(x, values)
    y_explicit_off_local = local.eval(x, values, broaden_by_resolution=False)

    # Default call must actually broaden (differ from the unbroadened case).
    assert not np.allclose(y_default_local, y_explicit_off_local)
    # And must match upstream's default-broadened output.
    np.testing.assert_allclose(y_default_local, y_default_upstream, rtol=1e-10)


def test_eval_scalar_resolution_matches_upstream(upstream_model):
    """Upstream broadcasts a scalar `self.resolution` to a full array
    before broadening (local previously assumed an array and would raise
    on a scalar). Confirm local no longer raises and matches upstream.
    """
    x = np.linspace(4900, 5100, 400)
    values = [10.0, 5000.0, 5.0]

    local = _build_model(new_model, resolution=20.0)  # scalar resolution
    upstream = _build_model(upstream_model, resolution=20.0)

    y_local = local.eval(x, values)  # must not raise
    y_upstream = upstream.eval(x, values)

    assert np.all(np.isfinite(y_local))
    np.testing.assert_allclose(y_local, y_upstream, rtol=1e-10)


def test_eval_array_resolution_matches_upstream(upstream_model):
    """Same check with an already-array-valued resolution (the case that
    already worked locally before this fix), to confirm the fix didn't
    change behavior for that path.
    """
    x = np.linspace(4900, 5100, 400)
    values = [10.0, 5000.0, 5.0]
    resolution = np.full_like(x, 20.0)

    local = _build_model(new_model, resolution=resolution)
    upstream = _build_model(upstream_model, resolution=resolution)

    y_local = local.eval(x, values)
    y_upstream = upstream.eval(x, values)
    np.testing.assert_allclose(y_local, y_upstream, rtol=1e-10)


# --------------------------------------------------------------------------
# sculptor_extensions.qso -- speconed import source switch (bundled dep_v1
# copy -> standalone package, matching upstream).
# --------------------------------------------------------------------------


def test_template_model_uses_standalone_speconed_matching_upstream(upstream_qso):
    """Both local and upstream source `SpecOneD` from the standalone
    `speconed` package (not the bundled `sculptor.dep_v1` copy). Locally,
    `template_model` now lives in `sculptor.analysis` (removed from
    `qso.py` as a duplicate, review §3).
    """
    from speconed import speconed as standalone_sod

    assert sculptor_analysis.sod is standalone_sod
    assert upstream_qso.sod is standalone_sod


def test_template_model_matches_upstream_qso_template_model(upstream_qso):
    """Exercise `template_model` (which uses `sod.SpecOneD`) end-to-end and
    confirm the maintained `sculptor.analysis.template_model` matches
    upstream's `qso.template_model` (same model; local `qso.py` copy removed
    as a duplicate, review §3).
    """
    rng = np.random.default_rng(0)
    templ_disp = np.linspace(4000, 5000, 200)
    templ_fluxden = np.exp(-0.5 * ((templ_disp - 4500) / 50) ** 2) + 0.01 * rng.normal(
        size=templ_disp.size
    )
    x = np.linspace(4200, 4800, 150)

    orig = upstream_qso.template_model(
        x,
        amp=2.0,
        z=0.05,
        fwhm=800.0,
        intr_fwhm=100.0,
        templ_disp=templ_disp,
        templ_fluxden=templ_fluxden,
    )
    new = sculptor_analysis.template_model(
        x,
        amp=2.0,
        redsh=0.05,
        fwhm=800.0,
        intr_fwhm=100.0,
        templ_disp=templ_disp,
        templ_fluxden=templ_fluxden,
    )
    np.testing.assert_allclose(orig, new, rtol=1e-8)


# --------------------------------------------------------------------------
# sculptor_extensions.Qso_autofit.get_line_prop_MgII -- restored FeII/MgII
# flux-ratio calculation, matching upstream.
# --------------------------------------------------------------------------


def test_feii_over_mgii_ratio_matches_upstream(tmp_path, upstream_qso_autofit):
    """Both the restored local calculation and upstream's must produce the
    same 'FeII_over_MgII' column from the same input row.
    """
    import pandas as pd

    param_name_dict = {
        "fe_amp": "FeII_amp",
        "peak_flux": "MgII_peak_flux",
    }
    df_orig = pd.DataFrame(
        {
            "FeII_amp_median": [3.0e17],
            "MgII_peak_flux_median": [5.0],
            "FeII_amp_low": [2.9e17],
            "MgII_peak_flux_low": [4.9],
            "FeII_amp_upp": [3.1e17],
            "MgII_peak_flux_upp": [5.1],
        }
    )
    df_new = df_orig.copy()

    result_orig = upstream_qso_autofit.get_line_prop_MgII(
        df_orig,
        redshift=2.0,
        param_name_dict=param_name_dict,
        save_dir=str(tmp_path),
        source="test_source",
        fit_name="Vest",
    )
    result_new = new_qso_autofit.get_line_prop_MgII(
        df_new,
        redshift=2.0,
        param_name_dict=param_name_dict,
        save_dir=str(tmp_path),
        source="test_source",
        fit_name="Vest",
    )

    for key in ["_median", "_low", "_upp"]:
        col = "FeII_over_MgII" + key
        assert col in result_new.columns
        np.testing.assert_allclose(
            result_new[col].to_numpy(), result_orig[col].to_numpy()
        )
        # Sanity check against the formula directly (FeII_amp * 1e-17 / peak_flux).
        expected = df_orig[f"FeII_amp{key}"] * 1e-17 / df_orig[f"MgII_peak_flux{key}"]
        np.testing.assert_allclose(result_new[col].to_numpy(), expected.to_numpy())


def test_feii_over_mgii_missing_columns_skips_gracefully():
    """When fe_amp/peak_flux columns aren't in param_name_dict at all, the
    function must skip the ratio calc (not raise) -- same as before the
    restore, and same as upstream's default early-out branch.
    """
    import pandas as pd

    param_name_dict = {}  # no fe_amp / peak_flux keys at all
    df = pd.DataFrame({"a": [1.0]})

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = new_qso_autofit.get_line_prop_MgII(
            df,
            redshift=2.0,
            param_name_dict=param_name_dict,
            save_dir=tmp_dir,
            source="test_source",
            fit_name="Vest",
        )
    assert "FeII_over_MgII_median" not in result.columns


# --------------------------------------------------------------------------
# sculptor_extensions.qso -- Dalla Bonta et al. 2020 Hbeta BH mass and
# bolometric luminosity (upstream v2_dev 33b78d5 + fix a5a1749).
# --------------------------------------------------------------------------


@pytest.mark.parametrize("virial_factor", [1.0, 4.3])
def test_se_bhmass_db20_matches_upstream(upstream_qso, virial_factor):
    fwhm = 4500.0 * u.km / u.s
    lwav = 3e42 * u.erg / u.s

    orig_mass, orig_ref = upstream_qso.se_bhmass_db20(
        fwhm, lwav, virial_factor=virial_factor
    )
    new_mass, new_ref = new_qso.se_bhmass_db20(fwhm, lwav, virial_factor=virial_factor)

    assert new_ref == orig_ref == "DB20_Hb_LHb"
    np.testing.assert_allclose(new_mass.to(u.Msun).value, orig_mass.to(u.Msun).value)


def test_se_bhmass_db20_values_and_units():
    # At the pivot point (L = 1e42 erg/s, FWHM = 10^3.5 km/s) only the zero
    # point remains: log M = 7.015.
    mass, _ = new_qso.se_bhmass_db20(10**3.5 * u.km / u.s, 1e42 * u.erg / u.s)
    assert np.log10(mass.to(u.Msun).value) == pytest.approx(7.015)

    # Unit-aware: the same inputs in other units give the same mass.
    other, _ = new_qso.se_bhmass_db20(10**3.5 * 1e3 * u.m / u.s, 1e35 * u.W)
    assert other.to(u.Msun).value == pytest.approx(mass.to(u.Msun).value)

    with pytest.raises(ValueError, match="not supported"):
        new_qso.se_bhmass_db20(3000 * u.km / u.s, 1e42 * u.erg / u.s, "Ha_LHa")


def test_calc_bolometric_luminosity_db20_matches_upstream(upstream_qso):
    lwav = 3e42 * u.erg / u.s

    orig_lbol, orig_ref = upstream_qso.calc_bolometric_luminosity_db20(lwav)
    new_lbol, new_ref = new_qso.calc_bolometric_luminosity_db20(lwav)

    assert new_ref == orig_ref == "DB20_Hb"
    np.testing.assert_allclose(
        new_lbol.to(u.erg / u.s).value, orig_lbol.to(u.erg / u.s).value
    )
    # At the pivot only the zero point remains.
    pivot, _ = new_qso.calc_bolometric_luminosity_db20(10**41.746 * u.erg / u.s)
    assert np.log10(pivot.to(u.erg / u.s).value) == pytest.approx(44.717)
