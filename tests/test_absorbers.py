"""`sculptor.absorbers`: detection, identification and masking of narrow
absorbers."""

import numpy as np
import pytest
from astropy import units as u
from speconed import speconed as sod

from sculptor import absorbers as scabs
from sculptor import analysis as ana
from sculptor import component as sccomp
from sculptor import model as scmod
from sculptor import prior as scpri

Z = 2.0
ERR = 0.02
LINES = scabs.ABSORPTION_LINES


def _absorbed_model(absorbers, seed=1, n=3000):
    """Power law at z = 2 with Gaussian absorbers (observed center, FWHM
    km/s, depth), fitted with a power law."""
    x = np.geomspace(4200, 6200, n)
    rng = np.random.default_rng(seed)
    cont = ana.power_law_at2500(x, Z, 1.0, -1.5)
    flux = cont.copy()
    for cen, fwhm, depth in absorbers:
        sigma = fwhm / 2.998e5 * cen / 2.3548
        flux = flux * (1 - depth * np.exp(-0.5 * ((x - cen) / sigma) ** 2))
    spec = sod.SpecOneD(
        dispersion=x,
        fluxden=flux + rng.normal(0, ERR, n),
        fluxden_err=np.full_like(x, ERR),
        dispersion_unit=u.AA,
        fluxden_unit=u.erg / u.s / u.cm**2 / u.AA,
    )
    comp = sccomp.FitComponent("PL", ana.power_law_at2500)
    pars = comp.create_params()
    pars["PL_redsh"].value, pars["PL_redsh"].vary = Z, False
    pars["PL_amp"].value = 0.9
    pars["PL_amp"].prior = scpri.UniformPrior("PL_amp", 0, 10)
    pars["PL_slope"].value = -1.2
    pars["PL_slope"].prior = scpri.UniformPrior("PL_slope", -3, 1)
    model = scmod.FitModel(components=[comp], parameters=pars, spectrum=spec)
    model.add_wavelength_range_to_fit_mask(4250, 6150)
    model.run_quick_fit(update_values=True)
    return model


def _obs(transition, z_abs):
    species = transition.split()[0]
    return LINES[species][transition] * (1 + z_abs)


def test_identify_systems_groups_doublets_and_species():
    z1, z2 = 1.0, 2.5
    waves = [
        _obs("MgII 2796", z1),
        _obs("MgII 2803", z1),
        _obs("FeII 2383", z1),
        _obs("FeII 2600", z1),
        _obs("CIV 1548", z2),
        _obs("CIV 1551", z2),
        5555.0,  # unrelated
    ]
    per_feature, systems = scabs.identify_systems(
        np.array(waves), np.ones(len(waves)), 30.0, z_max=3.0
    )
    assert len(systems) == 2
    zs = sorted(z for z, _ in systems)
    assert zs == pytest.approx([z1, z2], abs=1e-4)
    first = dict(systems[0][1])
    assert set(first) == {"MgII 2796", "MgII 2803", "FeII 2383", "FeII 2600"}
    assert per_feature[-1] == (-1, "")
    # A single line is not a system; above z_max nothing is identified
    _, none = scabs.identify_systems(np.array(waves[3:5]), np.ones(2), 30.0, z_max=2.0)
    assert none == []


def test_identify_systems_needs_the_stronger_transitions():
    """FeII 2587 + 2600 without the stronger 2383 is a coincidence, unless
    the spectrum does not cover 2383."""
    z = 1.2
    waves = np.array([_obs("FeII 2587", z), _obs("FeII 2600", z)])
    _, systems = scabs.identify_systems(waves, np.ones(2), 30.0)
    assert systems == []
    cutoff = _obs("FeII 2383", z) + 50

    def covered(wave):
        return wave > cutoff

    _, systems = scabs.identify_systems(waves, np.ones(2), 30.0, covered=covered)
    assert len(systems) == 1
    # MgII 2796 alone with FeII 2383 + 2600: 2803 is weaker, allowed
    waves = np.array([_obs("FeII 2383", z), _obs("FeII 2600", z), _obs("MgII 2796", z)])
    _, systems = scabs.identify_systems(waves, np.ones(3), 30.0)
    assert set(systems[0][1]) == {"FeII 2383", "FeII 2600", "MgII 2796"}


def test_find_absorbers_inventory_and_mask():
    z_mg, z_c = 0.8, 1.8
    absorbers = [
        (_obs("MgII 2796", z_mg), 150.0, 0.6),
        (_obs("MgII 2803", z_mg), 150.0, 0.4),
        (_obs("CIV 1548", z_c), 150.0, 0.5),
        (_obs("CIV 1551", z_c), 150.0, 0.3),
        (5800.0, 150.0, 0.5),  # unidentified
    ]
    model = _absorbed_model(absorbers)
    inv = scabs.find_absorbers(model, redshift=Z, name="QSO", n_sims=50)

    assert inv.threshold > 0
    assert list(inv.systems["species"]) == ["MgII", "CIV"]
    assert inv.systems["z_abs"] == pytest.approx([z_mg, z_c], abs=3e-4)
    assert inv.systems["absorber_id"][0] == f"QSO z={inv.systems['z_abs'][0]:.4f}"
    assert not inv.systems["associated"][0]
    unidentified = inv.features[inv.features["system"] < 0]
    assert list(unidentified["wave"]) == pytest.approx([5800.0], abs=3)
    x = model.spec.dispersion
    for cen, _, _ in absorbers:
        assert not inv.mask[np.argmin(np.abs(x - cen))]
    assert inv.mask.mean() > 0.95
    assert "QSO z=" in str(inv)


def test_weak_doublet_found_together():
    """MgII 2803 below the single-line threshold is found as the partner of
    2796 (the pair is significant together); a line is not paired twice."""
    model = _absorbed_model(
        [(_obs("MgII 2796", 0.8), 150.0, 0.016), (_obs("MgII 2803", 0.8), 150.0, 0.014)]
    )
    inv = scabs.find_absorbers(model, n_sims=100)
    assert list(inv.systems["species"]) == ["MgII"]
    assert inv.systems["z_abs"][0] == pytest.approx(0.8, abs=3e-4)
    detected = inv.features[inv.features["found_by"] != "predicted"]
    assert len(detected) == 2
    weak = inv.features[inv.features["transition"] == "MgII 2803"][0]
    assert weak["found_by"] == "doublet"
    assert weak["significance"] < inv.threshold


def test_predicted_transitions_are_masked():
    """The other covered transitions at an identified system's redshift
    are masked, detected or not."""
    model = _absorbed_model(
        [(_obs("MgII 2796", 0.8), 150.0, 0.6), (_obs("MgII 2803", 0.8), 150.0, 0.4)]
    )
    x = model.spec.dispersion
    z = 0.8
    inv = scabs.find_absorbers(model, n_sims=30)
    predicted = inv.features[inv.features["found_by"] == "predicted"]
    names = set(predicted["transition"])
    # Covered (4250 - 6150 A at z = 0.8): FeII 2383, 2600, 2587, 2374, MgI
    assert {"FeII 2600", "MgI 2853", "FeII 2383"} <= names
    assert "FeII 2344" not in names  # 4220 A, outside the fit region
    assert set(predicted["system"]) == {0}
    mgi = np.argmin(np.abs(x - _obs("MgI 2853", z)))
    assert not inv.mask[mgi]
    assert "predicted transition(s) masked" in str(inv)

    # A MgII system predicts low-ionization lines only: no CIV / SiIV / NV
    assert not names & {"CIV 1548", "CIV 1551", "SiIV 1394", "NV 1239"}

    plain = scabs.find_absorbers(model, n_sims=30, predict=False)
    assert not (plain.features["found_by"] == "predicted").any()
    assert plain.mask[mgi]


def test_high_ionization_system_predicts_no_mgii():
    """A CIV-only system masks its own class (SiIV, NV), not MgII."""
    z = 2.6
    model = _absorbed_model(
        [(_obs("CIV 1548", z), 150.0, 0.6), (_obs("CIV 1551", z), 150.0, 0.4)]
    )
    inv = scabs.find_absorbers(model, n_sims=30)
    assert list(inv.systems["species"]) == ["CIV"]
    predicted = set(inv.features[inv.features["found_by"] == "predicted"]["transition"])
    # Covered at z = 2.6 (4250 - 6150 A): SiIV 1394 / 1403, NV out of range
    assert {"SiIV 1394", "SiIV 1403"} <= predicted
    assert not any(
        t.startswith(("MgII", "FeII", "AlIII", "CII", "SiII", "OI", "AlII"))
        for t in predicted
    )


def test_multiplet_found_together():
    """MgII + FeII lines at one redshift, too weak for the single and
    doublet passes to identify, are found together; without the multiplet
    pass the same lines are misidentified."""
    z = 1.0
    depths = {
        "MgII 2796": 1.0,
        "MgII 2803": 0.8,
        "FeII 2600": 0.9,
        "FeII 2383": 0.9,
        "FeII 2587": 0.6,
        "FeII 2344": 0.7,
    }
    model = _absorbed_model(
        [(_obs(name, z), 150.0, 0.026 * f) for name, f in depths.items()]
    )
    inv = scabs.find_absorbers(model, n_sims=50)
    assert list(inv.systems["species"]) == ["FeII, MgII"]
    assert inv.systems["z_abs"][0] == pytest.approx(z, abs=3e-4)
    assert (inv.features["found_by"] == "multiplet").any()
    assert np.isfinite(inv.multiplet_threshold)

    without = scabs.find_absorbers(model, n_sims=50, multiplets={})
    assert np.isnan(without.multiplet_threshold)
    assert not any("FeII" in species for species in without.systems["species"])


def test_find_absorbers_clean_spectrum_and_mask_in_fit():
    model = _absorbed_model([])
    inv = scabs.find_absorbers(model, n_sims=50)
    assert len(inv.features) == 0
    assert inv.mask.all()

    absorbed = _absorbed_model([(5000.0, 150.0, 0.7)])
    inv = scabs.find_absorbers(absorbed, n_sims=50)
    absorbed.absorber_mask = inv.mask
    assert not absorbed.get_fit_mask()[~inv.mask].any()
    # Masked absorbers are not counted as clipped and stay out of clipping
    absorbed.sigma_clip(low=3)
    assert absorbed.get_fit_mask().sum() <= absorbed._fit_region_mask().sum()


def test_widths_follow_the_resolution():
    model = _absorbed_model([])
    model.resolution = 2000.0
    x = model.spec.dispersion
    widths = scabs._trial_widths(model, x, None)
    assert widths[0] == pytest.approx(np.full(x.size, 2.998e5 / 2000.0), rel=1e-3)
    assert len(widths) == len(scabs.WIDTH_FACTORS)


def test_balnicity_and_absorption_index_values():
    """A box trough at 0.6 from -10000 to -5000 km/s: BI counts from 2000
    km/s into it, 3000 km/s x (1 - 0.6 / 0.9); AI all of it, 5000 x 0.4."""
    v = np.arange(-30000.0, 1.0, 10.0)
    norm = np.where((v >= -10000) & (v <= -5000), 0.6, 1.0)
    bi, bi_runs = scabs.balnicity_index(v, norm)
    ai, ai_runs = scabs.absorption_index(v, norm)
    assert bi == pytest.approx(3000 * (1 - 0.6 / 0.9), rel=0.01)
    assert ai == pytest.approx(5000 * 0.4, rel=0.01)
    assert len(bi_runs) == len(ai_runs) == 1
    # Too narrow for BI (1500 km/s), wide enough for AI
    narrow = np.where((v >= -6500) & (v <= -5000), 0.6, 1.0)
    assert scabs.balnicity_index(v, narrow)[0] == 0
    assert scabs.absorption_index(v, narrow)[0] > 0


def _civ_obs(v):
    """Observed wavelength of CIV at velocity v (km/s) at z = 2."""
    return scabs._wavelength(v, scabs.BAL_LINES["CIV"], Z)


def test_bal_trough_found_masked_and_kept_out_of_the_narrow_search():
    centre = _civ_obs(-11000.0)
    model = _absorbed_model([(centre, 6000.0, 0.4)])
    troughs = scabs.find_bal_troughs(model, model.quick_fit_result.values, Z, n_sims=30)
    assert troughs.is_bal
    assert set(troughs.lines()) == {"CIV"}
    assert troughs.balnicity["CIV"] > 0 and troughs.absorption["CIV"] > 0
    row = troughs.troughs[0]
    assert row["v_min"] < -11000 < row["v_max"]
    x = model.spec.dispersion
    assert not troughs.mask[np.argmin(np.abs(x - centre))]
    assert "BAL troughs: 1" in str(troughs)

    inv = scabs.find_absorbers(model, redshift=Z, n_sims=30)
    assert inv.bal is not None and inv.bal.is_bal
    assert not inv.mask[np.argmin(np.abs(x - centre))]
    # No narrow features inside the trough
    inside = (inv.features["wave"] > row["wave_min"]) & (
        inv.features["wave"] < row["wave_max"]
    )
    assert not inside.any()
    assert "BAL troughs" in str(inv)
    assert scabs.find_absorbers(model, redshift=Z, n_sims=10, bal=False).bal is None


def test_narrow_line_cluster_is_not_a_bal_trough():
    """Strong narrow lines 300 km/s apart look like a trough once
    smoothed, but the flux returns to the continuum between them."""
    cluster = [(_civ_obs(-12000.0 + 300.0 * k), 120.0, 0.9) for k in range(6)]
    model = _absorbed_model(cluster)
    troughs = scabs.find_bal_troughs(model, model.quick_fit_result.values, Z, n_sims=30)
    assert len(troughs.troughs) == 0
    assert not troughs.is_bal
