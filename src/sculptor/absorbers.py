"""Inventory of narrow absorption systems in the residuals of a fit, and a
mask that keeps them out of the fit.

`find_absorbers` looks for absorption features in the residuals of a model
(all pixels of the fit regions, before sigma clipping) with a matched
filter at widths tied to the instrumental resolution: unresolved and
slightly resolved absorbers, as metal absorbers are at R ~ 1000 - 3000.
A feature counts if its significance exceeds a threshold calibrated on
simulations: the model plus noise with the correlations of the residuals
(`sculptor.autofit.correlated_noise`), so that a quantile q of the largest
significance in noise-only spectra gives a chance of about 1 - q of one
false feature per spectrum.

Features are then grouped into systems: transitions of one species
(doublets, multiplets, `ABSORPTION_LINES`) matched at a common redshift,
and species at the same redshift (within `system_dv`) joined into one
system. A doublet whose lines are each below the threshold may still be
significant together: a second pass combines each pixel with its doublet
partner (MgII, CIV, SiIV, NV, AlIII, CaII; `doublet_map`), with its own
calibrated threshold, and adds both lines of a significant pair. A third
pass does the same for multiplets (`MULTIPLETS`, e.g. MgII + FeII at one
redshift; `multiplet_map`), whose lines may each be too weak for the other
passes (without its strongest lines: in a crowded spectrum, strong lines of
other systems fall on some of its transitions); there, a line counts
if its own significance exceeds a quantile of the pixel significances in
noise (the redshift is fixed). The quantile is split between the passes. A system needs at least two matched transitions, and a species
counts only if its transitions stronger than the weakest matched one are
matched too where the spectrum covers them (e.g. FeII 2587 and 2600
without 2383 are a coincidence); single features stay unidentified. Every feature is masked (`FitModel.absorber_mask`) over
+-`mask_fwhm` times its matched-filter FWHM, and so is every listed
transition of the ionization classes the system was detected in
(`ION_CLASSES`) at the redshift of an identified system where the spectrum
covers it (`predict`), detected or not: at a known redshift a line too weak
to detect may still bias e.g. an emission-line fit. A system seen only in
CIV is not evidence for MgII (which would mask e.g. the MgII emission peak
for an associated CIV absorber).

Broad absorption (BAL troughs, several 1000 km/s) of the high-ionization
lines (`BAL_LINES`: CIV, SiIV, NV) is searched for first
(`find_bal_troughs`): the spectrum over the emission model, smoothed, with
the balnicity index (Weymann et al. 1991) and the absorption index (Hall et
al. 2002) per line; significant BI troughs are masked as whole regions and
the model refitted until the troughs stop changing. AI-only troughs
(mini-BALs) are reported, not masked, and dropped if lines of an identified
narrow system explain them. The narrow absorbers are
then searched in the refitted model's residuals, outside the troughs.

`AutoFit(absorbers={...})` runs the search on its preliminary model and
masks the absorbers for the calibration and the final searches.
"""

import copy
from dataclasses import dataclass, field

import numpy as np
from astropy.table import Table

from sculptor import utils as scut

# Rest-frame vacuum wavelengths (Angstrom) of common metal absorption lines
# in quasar spectra (Morton 2003), by species, the strongest transition
# (oscillator strength) first: a species is identified only if the
# transitions stronger than its weakest matched one are matched too, where
# the spectrum covers them
ABSORPTION_LINES = {
    "MgII": {"MgII 2796": 2796.354, "MgII 2803": 2803.531},
    "FeII": {
        "FeII 2383": 2382.765,
        "FeII 2600": 2600.173,
        "FeII 2344": 2344.214,
        "FeII 2587": 2586.650,
        "FeII 2374": 2374.461,
    },
    "MgI": {"MgI 2853": 2852.964},
    "CIV": {"CIV 1548": 1548.204, "CIV 1551": 1550.781},
    "SiIV": {"SiIV 1394": 1393.760, "SiIV 1403": 1402.773},
    "NV": {"NV 1239": 1238.821, "NV 1243": 1242.804},
    "AlIII": {"AlIII 1855": 1854.718, "AlIII 1863": 1862.790},
    "AlII": {"AlII 1671": 1670.789},
    "SiII": {"SiII 1260": 1260.422, "SiII 1527": 1526.707},
    "CII": {"CII 1335": 1334.532},
    "OI": {"OI 1302": 1302.168},
    "CaII": {"CaII 3935": 3934.777, "CaII 3970": 3969.591},
}

# Species that trace the same gas: a system detected in one species of a
# class predicts the transitions of the whole class (`find_absorbers(
# predict=True)`); species in no class predict only themselves
ION_CLASSES = {
    "low": ["MgII", "FeII", "MgI", "CaII", "AlII", "SiII", "CII", "OI", "AlIII"],
    "high": ["CIV", "SiIV", "NV"],
}

# Broad absorption (`find_bal_troughs`): high-ionization lines (rest
# wavelength of the absorbing doublet: SiIV 1393.76 / 1402.77, not the
# 1399.8 A SiIV + OIV] emission blend of `qso_rules.QSO_LINES`), the
# velocity ranges, minimum trough widths and
# flux level of the balnicity index (Weymann et al. 1991) and of the
# absorption index (Hall et al. 2002), km/s
BAL_LINES = {"CIV": 1549.06, "SiIV": 1396.76, "NV": 1240.14}
BI_RANGE = (-25000.0, -3000.0)
BI_MIN_WIDTH = 2000.0
AI_RANGE = (-29000.0, 0.0)
AI_MIN_WIDTH = 1000.0
BAL_LEVEL = 0.9
# A trough must depress the flux throughout: the upper envelope of the
# unsmoothed normalized flux in it (90th percentile less 1.28 times the
# typical noise) must stay below this level. Smoothed, a cluster of strong
# narrow lines looks like a trough, but between its lines the flux returns
# to the continuum.
BAL_ENVELOPE = 0.95

# Transitions searched together at one redshift (`multiplet_map`), the
# first the reference (its strongest line): name -> transition names of
# `ABSORPTION_LINES`
MULTIPLETS = {
    "MgII+FeII": [
        "MgII 2796",
        "MgII 2803",
        "FeII 2600",
        "FeII 2383",
        "FeII 2587",
        "FeII 2344",
    ],
}
# Covered transitions a multiplet needs at a pixel, and how many of the
# strongest are left out of its joint significance (in a crowded spectrum,
# strong lines of other systems fall on some of its transitions)
MIN_MULTIPLET_LINES = 4
MULTIPLET_DROP = 2

# Matched-filter FWHMs in units of the instrumental FWHM (c / R) with a
# resolution, else in km/s
WIDTH_FACTORS = (1.0, 1.5, 2.5)
WIDTHS_KM_S = (150.0, 300.0, 600.0)


@dataclass
class AbsorberInventory:
    """Result of `find_absorbers`.

    :param features: One row per feature, from the detection: `wave`
        (observed, Angstrom), `significance`, `fwhm_km_s` and `amplitude`
        (of the best matched filter: FWHM, absorbed flux density at the
        center), `found_by` ("single" or "doublet": as part of a doublet
        significant together), `system` (index into `systems`, -1 if
        unidentified), `transition`
    :type features: astropy.table.Table
    :param systems: The inventory, one row per system: `absorber_id`
        ("<name> z=<z_abs>"), `z_abs`, `species` (identified), `transitions`
        (the matched lines), `n_lines`, `max_significance` (of its lines)
        and `associated` (within `associated_dv` of the quasar's redshift)
    :type systems: astropy.table.Table
    :param threshold: Significance threshold of single features
    :type threshold: float
    :param quantile: Quantile of the noise-only maxima (split between the
        single and the doublet pass)
    :type quantile: float
    :param null_maxima: Largest significance in each simulation
    :type null_maxima: np.ndarray
    :param mask: False on the masked pixels (`FitModel.absorber_mask`)
    :type mask: np.ndarray
    :param pair_threshold: Threshold of the joint doublet significance
    :type pair_threshold: float
    :param pair_maxima: Largest joint doublet significance in each
        simulation
    :type pair_maxima: np.ndarray
    :param multiplet_threshold: Threshold of the joint multiplet
        significance (NaN without multiplets)
    :type multiplet_threshold: float
    :param multiplet_maxima: Largest joint multiplet significance in each
        simulation
    :type multiplet_maxima: np.ndarray
    :param bal: The broad absorption troughs (`find_bal_troughs`); None if
        not searched
    :type bal: BALTroughs
    """

    features: Table
    systems: Table
    threshold: float
    quantile: float
    null_maxima: np.ndarray = field(default_factory=lambda: np.array([]))
    mask: np.ndarray | None = None
    pair_threshold: float = np.nan
    pair_maxima: np.ndarray = field(default_factory=lambda: np.array([]))
    multiplet_threshold: float = np.nan
    multiplet_maxima: np.ndarray = field(default_factory=lambda: np.array([]))
    bal: "BALTroughs | None" = None

    def __str__(self):
        detected = self.features[self.features["found_by"] != "predicted"]
        multiplets = (
            ""
            if np.isnan(self.multiplet_threshold)
            else f", multiplets above {self.multiplet_threshold:.2f}"
        )
        lines = [
            (
                f"Absorbers: {len(detected)} feature(s) (significance above "
                f"{self.threshold:.2f}, or doublets above {self.pair_threshold:.2f}"
                f"{multiplets}; {self.quantile:.0%} of noise-only maxima), "
                f"{len(self.systems)} system(s), "
                f"{len(self.features) - len(detected)} predicted transition(s) masked"
            )
        ]
        for idx in range(len(self.systems)):
            row = self.systems[idx]
            tag = " (associated)" if row["associated"] else ""
            lines.append(
                f"  {row['absorber_id']}{tag}: {row['species']} ({row['transitions']})"
            )
        if self.bal is not None:
            lines.append(f"  {self.bal}")
        unidentified = detected[detected["system"] < 0]
        if len(unidentified):
            waves = ", ".join(
                f"{w:.0f} A ({s:.1f})"
                for w, s in zip(
                    unidentified["wave"], unidentified["significance"], strict=True
                )
            )
            lines.append(f"  unidentified (wavelength, significance): {waves}")
        return "\n".join(lines)


def _trial_widths(model, x, widths):
    """Matched-filter FWHMs (km/s, per pixel): multiples of the
    instrumental FWHM with a resolution, else `WIDTHS_KM_S`."""
    if widths is not None:
        return [np.full(x.shape, w, dtype=float) for w in widths]
    resolution = getattr(model, "resolution", None)
    if resolution is None:
        return [np.full(x.shape, w, dtype=float) for w in WIDTHS_KM_S]
    res = np.broadcast_to(np.asarray(resolution, dtype=float), x.shape)
    return [factor * scut.c_km_s / res for factor in WIDTH_FACTORS]


def significance_map(model, values, flux, region, widths=None):
    """Absorption significance at each pixel, the largest over the trial
    widths, with the FWHM (km/s) and the matched-filter amplitude (the
    absorbed flux density at the center) that give it.

    :param model: The model (its spectrum's dispersion and errors)
    :type model: sculptor.model.FitModel
    :param values: Variable parameter values of the model
    :type values: np.ndarray
    :param flux: Flux density to search (e.g. the spectrum's, or a
        simulation's)
    :type flux: np.ndarray
    :param region: Pixels to search
    :type region: np.ndarray
    :param widths: FWHMs in km/s (default: see `_trial_widths`)
    :type widths: list(float)
    :return: significance, FWHM (km/s), amplitude
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
    """
    from sculptor.autofit import matched_filter

    x = model.spec.dispersion
    resid = model.eval(x, values) - flux  # absorption is positive
    weight = np.where(region, 1 / model.spec.fluxden_err**2, 0.0)
    best = np.full(x.shape, -np.inf)
    best_fwhm = np.full(x.shape, np.nan)
    best_amp = np.zeros(x.shape)
    for fwhm in _trial_widths(model, x, widths):
        signif, amp = matched_filter(x, resid, weight, fwhm)
        better = signif > best
        best[better] = signif[better]
        best_fwhm[better] = fwhm[better]
        best_amp[better] = amp[better]
    best[~region] = 0.0
    return best, best_fwhm, best_amp


def _doublets(lines):
    """Species whose two strongest transitions are a close doublet (within
    1% in wavelength, e.g. MgII, CIV): species -> (rest wavelength of the
    strongest, ratio of the second's to it).

    :rtype: dict
    """
    found = {}
    for species, transitions in lines.items():
        rests = list(transitions.values())
        if len(rests) >= 2 and abs(rests[1] / rests[0] - 1) < 0.01:
            found[species] = (rests[0], rests[1] / rests[0])
    return found


def _doublet_partners(x, region, doublets, z_max):
    """Per doublet species: the pixel of the partner of each pixel (taken as
    the strongest line) and where both are searched (and the redshift is in
    range).

    :rtype: dict(str, tuple(np.ndarray, np.ndarray))
    """
    partners = {}
    for species, (rest, ratio) in doublets.items():
        target = x * ratio
        j = np.clip(np.searchsorted(x, target), 1, x.size - 1)
        j = np.where(np.abs(x[j - 1] - target) < np.abs(x[j] - target), j - 1, j)
        z = x / rest - 1
        valid = region & region[j] & (target <= x[-1]) & (z >= 0)
        if z_max is not None:
            valid &= z <= z_max
        partners[species] = (j, valid)
    return partners


# Allowed ratio of the significances of a doublet's second line to its
# first (strongest) one: 1/2 unsaturated to 1 saturated, with room for noise
DOUBLET_RATIO = (1 / 3, 2.0)


def doublet_map(signif, partners):
    """Joint significance of the doublets, (s_1 + s_2) / sqrt(2) with the
    second line's significance within `DOUBLET_RATIO` of the first's (a
    strong single line next to noise is not a doublet), the largest over
    the species, and its species.

    :param signif: Absorption significance per pixel (`significance_map`)
    :type signif: np.ndarray
    :param partners: From `_doublet_partners`
    :type partners: dict
    :return: joint significance, species (one per pixel; "" where none)
    :rtype: (np.ndarray, np.ndarray)
    """
    best = np.zeros_like(signif)
    species = np.full(signif.shape, "", dtype=object)
    for name, (j, valid) in partners.items():
        partner = signif[j]
        low, high = DOUBLET_RATIO
        pair = (
            valid
            & (signif > 0)
            & (partner >= low * signif)
            & (partner <= high * signif)
        )
        joint = np.where(pair, (signif + partner) / np.sqrt(2), 0.0)
        better = joint > best
        best[better] = joint[better]
        species[better] = name
    return best, species


def _multiplet_partners(x, region, multiplets, lines, z_max):
    """Per multiplet: the pixel of each transition for each pixel (taken
    as the reference, the first transition), where each is searched, and
    where the multiplet can be tested (reference searched, redshift in
    range, at least `MIN_MULTIPLET_LINES` transitions searched).

    :rtype: dict(str, tuple(np.ndarray, np.ndarray, np.ndarray))
    """
    rests = {n: r for transitions in lines.values() for n, r in transitions.items()}
    partners = {}
    for name, members in multiplets.items():
        members = [m for m in members if m in rests]
        if len(members) < MIN_MULTIPLET_LINES:
            continue
        ref = rests[members[0]]
        z = x / ref - 1
        js, covered = [], []
        for member in members:
            target = x * rests[member] / ref
            j = np.clip(np.searchsorted(x, target), 1, x.size - 1)
            j = np.where(np.abs(x[j - 1] - target) < np.abs(x[j] - target), j - 1, j)
            js.append(j)
            covered.append(region[j] & (target >= x[0]) & (target <= x[-1]))
        js, covered = np.array(js), np.array(covered)
        valid = region & (z >= 0) & (covered.sum(axis=0) >= MIN_MULTIPLET_LINES)
        if z_max is not None:
            valid &= z <= z_max
        partners[name] = (js, covered, valid)
    return partners


def multiplet_map(signif, partners):
    """Joint significance of the multiplets: the sum of the significances
    of the searched transitions without the `MULTIPLET_DROP` largest, over
    sqrt(n - MULTIPLET_DROP) (strong lines of other systems on some of its
    transitions, or a strong line next to noise, are not a multiplet), with
    the reference line in absorption; the largest over the multiplets, and
    its name.

    :param signif: Absorption significance per pixel (`significance_map`)
    :type signif: np.ndarray
    :param partners: From `_multiplet_partners`
    :type partners: dict
    :return: joint significance, multiplet (one per pixel; "" where none)
    :rtype: (np.ndarray, np.ndarray)
    """
    best = np.zeros_like(signif)
    names = np.full(signif.shape, "", dtype=object)
    for name, (js, covered, valid) in partners.items():
        values = np.where(covered, signif[js], 0.0)
        n = covered.sum(axis=0)
        ranked = np.sort(np.where(covered, signif[js], -np.inf), axis=0)
        top = ranked[ranked.shape[0] - MULTIPLET_DROP :].sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            joint = (values.sum(axis=0) - top) / np.sqrt(n - MULTIPLET_DROP)
        joint = np.where(valid & (signif > 0), joint, 0.0)
        better = joint > best
        best[better] = joint[better]
        names[better] = name
    return best, names


def _peaks(x, signif, fwhm, threshold):
    """Local maxima above `threshold`, at least one FWHM apart (the
    strongest first)."""
    above = np.flatnonzero(signif >= threshold)
    order = above[np.argsort(signif[above])[::-1]]
    taken = []
    for idx in order:
        width = fwhm[idx] / scut.c_km_s * x[idx]
        if all(abs(x[idx] - x[j]) > width for j in taken):
            taken.append(idx)
    return sorted(taken)


def identify_systems(
    waves,
    significance,
    tolerance_km_s,
    z_max=None,
    lines=None,
    system_dv=200.0,
    covered=None,
):
    """Group features into absorption systems (see the module docstring).

    :param waves: Observed wavelengths of the features
    :type waves: np.ndarray
    :param significance: Their significances (to rank competing matches)
    :type significance: np.ndarray
    :param tolerance_km_s: Match tolerance, one value or one per feature
    :type tolerance_km_s: float or np.ndarray
    :param z_max: Largest absorber redshift (e.g. the quasar's plus a few
        1000 km/s); None for no limit
    :type z_max: float
    :param lines: Species -> {transition: rest wavelength} (default
        `ABSORPTION_LINES`)
    :type lines: dict
    :param system_dv: Species within this velocity form one system, km/s
    :type system_dv: float
    :param covered: Whether the spectrum covers (searched) an observed
        wavelength, callable; None: everywhere
    :type covered: callable
    :return: per feature (system index or -1, transition name), and the
        systems as (z, {transition: feature index})
    :rtype: (list(tuple(int, str)), list(tuple(float, dict)))
    """
    lines = ABSORPTION_LINES if lines is None else lines
    waves = np.asarray(waves, dtype=float)
    significance = np.asarray(significance, dtype=float)
    tol = np.broadcast_to(np.asarray(tolerance_km_s, dtype=float), waves.shape)

    def match(z, transitions):
        """Feature index per transition matched at z (each feature once)."""
        found = {}
        for name, rest in transitions.items():
            dv = (waves / (rest * (1 + z)) - 1) * scut.c_km_s
            close = np.flatnonzero(np.abs(dv) <= tol)
            close = [i for i in close if i not in found.values()]
            if close:
                found[name] = min(close, key=lambda i: abs(dv[i]))
        return found

    def consistent(z, transitions, found):
        """The transitions stronger than the weakest matched one (the
        dict's order) are matched, where covered."""
        names = list(transitions)
        weakest = max(names.index(n) for n in found)
        for name in names[:weakest]:
            if name in found:
                continue
            if covered is None or covered(transitions[name] * (1 + z)):
                return False
        return True

    # Candidates: a species at the redshift that puts one of its transitions
    # on a feature, with at least two transitions matched
    candidates = []
    for species, transitions in lines.items():
        if len(transitions) < 2:
            continue
        for wave in waves:
            for rest in transitions.values():
                z = wave / rest - 1
                if z < 0 or (z_max is not None and z > z_max):
                    continue
                found = match(z, transitions)
                if len(found) >= 2 and consistent(z, transitions, found):
                    idx = list(found.values())
                    z_fit = np.mean(
                        [waves[j] / transitions[n] - 1 for n, j in found.items()]
                    )
                    candidates.append(
                        (len(found), significance[idx].sum(), z_fit, species, found)
                    )
    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    used, accepted = set(), []
    for _, _, z, species, found in candidates:
        if used & set(found.values()):
            continue
        used |= set(found.values())
        accepted.append((z, species, found))

    # Species at the same redshift form one system
    systems = []
    for z, _species, found in sorted(accepted, key=lambda a: a[0]):
        if systems and abs(z - systems[-1][0]) / (1 + z) * scut.c_km_s <= system_dv:
            systems[-1][1].update(found)
        else:
            systems.append([z, dict(found)])

    # Single transitions (and further ones of matched species) at a system's
    # redshift
    for system in systems:
        z = system[0]
        for transitions in lines.values():
            remaining = {n: r for n, r in transitions.items() if n not in system[1]}
            for name, j in match(z, remaining).items():
                mine = {n: i for n, i in system[1].items() if n in transitions}
                if j not in used and consistent(z, transitions, {**mine, name: j}):
                    system[1][name] = j
                    used.add(j)

    per_feature = [(-1, "") for _ in waves]
    for s_idx, (_, found) in enumerate(systems):
        for name, j in found.items():
            per_feature[j] = (s_idx, name)
    return per_feature, [(z, found) for z, found in systems]


def find_absorbers(
    model,
    values=None,
    redshift=None,
    name=None,
    quantile=0.95,
    n_sims=200,
    seed=0,
    widths=None,
    mask_fwhm=1.0,
    associated_dv=5000.0,
    lines=None,
    max_iterations=5,
    multiplets=None,
    predict=True,
    bal=True,
):
    """Find, identify and mask narrow absorbers (see the module docstring).

    :param model: The fitted model (a model of the emission, e.g. AutoFit's
        preliminary one)
    :type model: sculptor.model.FitModel
    :param values: Its variable parameter values (default: its last quick
        fit)
    :type values: np.ndarray
    :param redshift: The quasar's redshift: absorbers up to 5000 km/s
        beyond it are searched, and velocities are given from it
    :type redshift: float
    :param name: Name of the quasar, for the absorber ids
    :type name: str
    :param quantile: Quantile of the noise-only maxima used as threshold
    :type quantile: float
    :param n_sims: Number of noise-only simulations
    :type n_sims: int
    :param seed: Seed of the random generator
    :type seed: int
    :param widths: Matched-filter FWHMs, km/s (default: 1, 1.5 and 2.5
        times the instrumental FWHM with a resolution, else 150, 300, 600)
    :type widths: list(float)
    :param mask_fwhm: Masked half-width around a feature, in its FWHMs
    :type mask_fwhm: float
    :param associated_dv: Systems within this velocity of `redshift` are
        associated, km/s
    :type associated_dv: float
    :param lines: Species -> {transition: rest wavelength} (default
        `ABSORPTION_LINES`)
    :type lines: dict
    :param max_iterations: Detect, mask and recompute the threshold up to
        this many times (the noise-only simulations follow the residuals
        without the absorbers found)
    :type max_iterations: int
    :param multiplets: Name -> transition names, searched together (default
        `MULTIPLETS`; {} for none)
    :type multiplets: dict
    :param predict: Also mask the transitions of `lines` of the ionization
        classes (`ION_CLASSES`) each identified system was detected in, at
        its redshift, where covered (rows with `found_by` "predicted" in the
        features)
    :type predict: bool
    :param bal: With `redshift`, first search for broad absorption troughs
        (`find_bal_troughs`), mask them and search the narrow absorbers in
        the residuals of the model refitted without them
    :type bal: bool
    :return: The inventory (with the mask)
    :rtype: AbsorberInventory
    """
    from sculptor.autofit import correlated_noise

    result = model.quick_fit_result
    if values is None:
        if result is None:
            raise ValueError("No model values; run run_quick_fit() first.")
        values = result.values
    troughs = None
    if bal and redshift is not None:
        troughs = find_bal_troughs(
            model, values, redshift, quantile=quantile, n_sims=n_sims, seed=seed
        )
        if not troughs.mask.all():
            # The narrow search in the residuals of the model without them
            model = troughs.model
            result = model.quick_fit_result
            values = result.values
    bal_mask = np.ones(model.spec.dispersion.size, dtype=bool)
    if troughs is not None:
        bal_mask = troughs.mask
    spec = model.spec
    x = spec.dispersion
    # Before sigma clipping (it removes absorbers) and earlier absorber masks
    saved = getattr(model, "absorber_mask", None)
    model.absorber_mask = None
    region = model._fit_region_mask() & bal_mask
    model.absorber_mask = saved

    signif, fwhm, amp = significance_map(model, values, spec.fluxden, region, widths)
    best = model.eval(x, values)
    z_max = None
    if redshift is not None:
        z_max = (1 + redshift) * (1 + associated_dv / scut.c_km_s) - 1
    all_lines = ABSORPTION_LINES if lines is None else lines
    partners = _doublet_partners(x, region, _doublets(all_lines), z_max)
    joint, joint_species = doublet_map(signif, partners)
    multi_partners = _multiplet_partners(
        x, region, MULTIPLETS if multiplets is None else multiplets, all_lines, z_max
    )
    if not any(valid.any() for _, _, valid in multi_partners.values()):
        multi_partners = {}
    multi, multi_names = multiplet_map(signif, multi_partners)
    # One pass each (single lines, doublets, multiplets) at an equal share
    # of the false rate
    each = 1 - (1 - quantile) / (3 if multi_partners else 2)

    def covered(wave):
        """Inside the searched pixels."""
        if wave < x[0] or wave > x[-1]:
            return False
        return bool(region[np.argmin(np.abs(x - wave))])

    def features_above(single_threshold, pair_threshold, multi_threshold):
        """Single features, plus both lines of significant doublets and the
        lines of significant multiplets above `member_threshold` (sorted
        pixel indices, and how each was found)."""
        found = {i: "single" for i in _peaks(x, signif, fwhm, single_threshold)}

        def near(k, others):
            """Pixel k within one FWHM of one of `others`."""
            width = fwhm[k] / scut.c_km_s * x[k]
            return any(abs(x[k] - x[m]) <= width for m in others)

        # Strongest pairs first; a line belongs to one pair only
        paired = []
        pair_peaks = _peaks(x, joint, fwhm, pair_threshold)
        for i in sorted(pair_peaks, key=lambda i: -joint[i]):
            members = (i, partners[joint_species[i]][0][i])
            if any(near(k, paired) for k in members):
                continue
            paired.extend(members)
            for k in members:
                if not near(k, found):
                    found[k] = "doublet"
        if multi_partners:
            for i in _peaks(x, multi, fwhm, multi_threshold):
                js, searched, _ = multi_partners[multi_names[i]]
                for k, cov in zip(js[:, i], searched[:, i], strict=True):
                    if cov and signif[k] >= member_threshold and not near(k, found):
                        found[k] = "multiplet"
        return sorted(found), found

    def mask_of(peaks):
        """False within +-mask_fwhm FWHM of the features at `peaks`."""
        mask = np.ones(x.size, dtype=bool)
        for idx in peaks:
            half = mask_fwhm * fwhm[idx] / scut.c_km_s * x[idx]
            mask &= np.abs(x - x[idx]) > half
        return mask

    # The simulated noise follows the residuals: the absorbers found so far
    # are masked from them (else they inflate it), until nothing changes
    peaks, how, mask = [], {}, np.ones(x.size, dtype=bool)
    for _ in range(max_iterations):
        model.absorber_mask = mask & bal_mask
        try:
            rng = np.random.default_rng(seed)
            maxima = np.empty(n_sims)
            pair_maxima = np.empty(n_sims)
            multi_maxima = np.zeros(n_sims)
            pixels = []
            for k in range(n_sims):
                sim = best + correlated_noise(model, result, rng)
                sim_signif = significance_map(model, values, sim, region, widths)[0]
                maxima[k] = sim_signif.max()
                pair_maxima[k] = doublet_map(sim_signif, partners)[0].max()
                if multi_partners:
                    multi_maxima[k] = multiplet_map(sim_signif, multi_partners)[0].max()
                    pixels.append(sim_signif[region])
        finally:
            model.absorber_mask = saved
        threshold = float(np.quantile(maxima, each))
        pair_threshold = float(np.quantile(pair_maxima, each))
        multi_threshold, member_threshold = np.inf, np.inf
        if multi_partners:
            multi_threshold = float(np.quantile(multi_maxima, each))
            # At the redshift of a significant multiplet a line is tested on
            # its own pixel (no search over wavelength): the quantile of the
            # pixel significances, corrected for the number of lines
            n_members = max(len(js) for js, _, _ in multi_partners.values())
            member_threshold = float(
                np.quantile(np.concatenate(pixels), 1 - (1 - each) / n_members)
            )
        new, new_how = features_above(threshold, pair_threshold, multi_threshold)
        if new == peaks:
            break
        peaks, how, mask = new, new_how, mask_of(new)

    waves = x[peaks]

    per_feature, systems = identify_systems(
        waves,
        signif[peaks],
        0.5 * fwhm[peaks],
        z_max=z_max,
        lines=lines,
        covered=covered,
    )
    if troughs is not None:
        identified = waves[[s >= 0 for s, _ in per_feature]]
        troughs.drop_explained(identified)

    features = Table(
        {
            "wave": waves,
            "significance": signif[peaks],
            "fwhm_km_s": fwhm[peaks],
            "amplitude": amp[peaks],
            "found_by": [how[i] for i in peaks],
            "system": np.array([s for s, _ in per_feature], dtype=int),
            "transition": [t for _, t in per_feature],
        }
    )
    if predict:
        # The other transitions of each system's ionization classes, at its
        # redshift, over the system's median matched-filter FWHM
        for s_idx, (z, found) in enumerate(systems):
            width = float(np.median(fwhm[peaks][list(found.values())]))
            seen = {transition.split()[0] for transition in found}
            species = set(seen)
            for members in ION_CLASSES.values():
                if seen & set(members):
                    species.update(members)
            for kind, transitions in all_lines.items():
                if kind not in species:
                    continue
                for transition, rest in transitions.items():
                    wave = rest * (1 + z)
                    if transition in found or not covered(wave):
                        continue
                    k = int(np.argmin(np.abs(x - wave)))
                    mask &= np.abs(x - wave) > mask_fwhm * width / scut.c_km_s * wave
                    features.add_row(
                        (
                            wave,
                            signif[k],
                            width,
                            amp[k],
                            "predicted",
                            s_idx,
                            transition,
                        )
                    )
    rows = []
    for z, found in systems:
        associated = redshift is not None and (
            abs((1 + z) / (1 + redshift) - 1) * scut.c_km_s < associated_dv
        )
        # Species in the order of their strongest-listed line
        species = list(dict.fromkeys(n.split()[0] for n in found))
        prefix = "" if name is None else f"{name} "
        rows.append(
            (
                f"{prefix}z={z:.4f}",
                z,
                ", ".join(species),
                ", ".join(sorted(found, key=lambda n: waves[found[n]])),
                len(found),
                float(max(signif[peaks][j] for j in found.values())),
                associated,
            )
        )
    systems_table = Table(
        rows=rows or None,
        names=(
            "absorber_id",
            "z_abs",
            "species",
            "transitions",
            "n_lines",
            "max_significance",
            "associated",
        ),
        dtype=(str, float, str, str, int, float, bool),
    )
    return AbsorberInventory(
        features,
        systems_table,
        threshold,
        quantile,
        maxima,
        mask & bal_mask,
        pair_threshold=pair_threshold,
        pair_maxima=pair_maxima,
        multiplet_threshold=multi_threshold if multi_partners else np.nan,
        multiplet_maxima=multi_maxima if multi_partners else np.array([]),
        bal=troughs,
    )


def _velocity(x, rest, redshift):
    """Velocity (km/s, relativistic Doppler) of observed wavelengths `x`
    with respect to the line `rest` at `redshift`."""
    ratio = (x / (rest * (1 + redshift))) ** 2
    return scut.c_km_s * (ratio - 1) / (ratio + 1)


def _wavelength(v, rest, redshift):
    """Inverse of `_velocity`."""
    beta = np.asarray(v) / scut.c_km_s
    return rest * (1 + redshift) * np.sqrt((1 + beta) / (1 - beta))


def _smooth(values, good, n):
    """Running mean over `n` pixels of the good pixels (NaN where none)."""
    if n <= 1:
        return np.where(good, values, np.nan)
    kernel = np.ones(n)
    num = np.convolve(np.where(good, values, 0.0), kernel, mode="same")
    den = np.convolve(good.astype(float), kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _runs(v, norm, v_range, min_width, level=BAL_LEVEL):
    """Index ranges (start, stop) of `v` (increasing) within `v_range` where
    `norm` stays below `level` over at least `min_width` km/s."""
    inside = (v >= v_range[0]) & (v <= v_range[1]) & (norm < level)
    runs, start = [], None
    for i, flag in enumerate(np.append(inside, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if v[i - 1] - v[start] >= min_width:
                runs.append((start, i))
            start = None
    return runs


def balnicity_index(v, norm):
    """Balnicity index (Weymann et al. 1991), km/s: the integral of
    1 - norm / 0.9 over -25000 to -3000 km/s, where `norm` (flux over the
    unabsorbed model) has stayed below 0.9 for more than 2000 km/s.

    :param v: Velocities, km/s, increasing
    :type v: np.ndarray
    :param norm: Normalized flux
    :type norm: np.ndarray
    :return: the index, and the troughs as (start, stop) index ranges
    :rtype: (float, list)
    """
    dv = np.gradient(v)
    total = 0.0
    runs = _runs(v, norm, BI_RANGE, BI_MIN_WIDTH)
    for start, stop in runs:
        # Counted only once the trough is 2000 km/s wide
        counted = v[start:stop] - v[start] > BI_MIN_WIDTH
        seg = slice(start, stop)
        total += float(np.sum(((1 - norm[seg] / BAL_LEVEL) * dv[seg])[counted]))
    return total, runs


def absorption_index(v, norm):
    """Absorption index (Hall et al. 2002), km/s: the integral of 1 - norm
    over -29000 to 0 km/s, in troughs where `norm` stays below 0.9 for at
    least 1000 km/s.

    :param v: Velocities, km/s, increasing
    :type v: np.ndarray
    :param norm: Normalized flux
    :type norm: np.ndarray
    :return: the index, and the troughs as (start, stop) index ranges
    :rtype: (float, list)
    """
    dv = np.gradient(v)
    runs = _runs(v, norm, AI_RANGE, AI_MIN_WIDTH)
    total = sum(float(np.sum((1 - norm[a:b]) * dv[a:b])) for a, b in runs)
    return total, runs


@dataclass
class BALTroughs:
    """Result of `find_bal_troughs`.

    :param troughs: One row per significant trough (absorption-index
        definition): `line`, `v_min`, `v_max` (km/s), `wave_min`, `wave_max`
        (observed, Angstrom), `max_depth` (1 - normalized flux), `ai` (its
        absorption index, km/s), `bal` (it contributes to the balnicity
        index) and `also_in` (other lines with flux below 0.9 at the same
        velocities, where covered)
    :type troughs: astropy.table.Table
    :param balnicity: Line -> balnicity index, km/s (NaN if its velocity
        range is mostly not covered)
    :type balnicity: dict
    :param absorption: Line -> absorption index, km/s (NaN likewise)
    :type absorption: dict
    :param threshold: Absorption index that noise alone exceeds in a
        fraction 1 - quantile of the simulations (largest over the lines)
    :type threshold: float
    :param mask: False on the masked (BI) troughs
    :type mask: np.ndarray
    :param model: Copy of the model refitted with the troughs masked
    :type model: sculptor.model.FitModel
    :param dropped: Troughs dropped as narrow-line clusters
        (`drop_explained`), "line v_min..v_max"
    :type dropped: list(str)
    """

    troughs: Table
    balnicity: dict
    absorption: dict
    threshold: float
    mask: np.ndarray
    model: object = None
    dropped: list = field(default_factory=list)

    def drop_explained(self, waves):
        """Drop the troughs that contribute only to the absorption index
        (not to the balnicity index, so not masked) and contain one of
        `waves` (lines of identified narrow systems): a cluster of strong
        narrow lines looks like a trough once smoothed. The absorption
        indices are reduced by theirs.

        :param waves: Observed wavelengths of identified narrow lines
        :type waves: np.ndarray
        """
        waves = np.asarray(waves, dtype=float)
        drop = [
            i
            for i, row in enumerate(self.troughs)
            if not row["bal"]
            and np.any((waves >= row["wave_min"]) & (waves <= row["wave_max"]))
        ]
        for i in drop:
            row = self.troughs[i]
            remaining = self.absorption[row["line"]] - row["ai"]
            self.absorption[row["line"]] = max(remaining, 0.0)
            self.dropped.append(f"{row['line']} {row['v_min']:.0f}..{row['v_max']:.0f}")
        if drop:
            self.troughs.remove_rows(drop)

    @property
    def is_bal(self):
        """Whether a line has a positive balnicity index from a significant
        trough (a BAL quasar)."""
        return bool(len(self.troughs)) and bool(np.any(self.troughs["bal"]))

    def lines(self):
        """Lines with significant troughs.

        :rtype: list(str)
        """
        return sorted(set(self.troughs["line"]))

    def __str__(self):
        def fmt(values):
            """The finite indices as "line value"."""
            return ", ".join(
                f"{k} {v:.0f}" for k, v in values.items() if np.isfinite(v)
            )

        head = (
            f"BAL troughs: {len(self.troughs)} (absorption index above "
            f"{self.threshold:.0f} km/s); BI [km/s]: {fmt(self.balnicity) or '-'}; "
            f"AI [km/s]: {fmt(self.absorption) or '-'}"
        )
        rows = [
            f"    {r['line']}: {r['v_min']:.0f} to {r['v_max']:.0f} km/s, depth "
            f"{r['max_depth']:.2f}, AI {r['ai']:.0f}"
            + (" (BI trough)" if r["bal"] else "")
            + (f", also in {r['also_in']}" if r["also_in"] else "")
            for r in self.troughs
        ]
        if self.dropped:
            rows.append(
                "    dropped as narrow-line clusters: " + ", ".join(self.dropped)
            )
        return "\n".join([head, *rows])


def _bal_measure(x, flux, err, best, region, redshift, lines, smooth_km_s):
    """Per line covered: velocities, smoothed normalized flux (good pixels
    of the absorption-index range), the range's coverage, and the unsmoothed
    normalized flux and its noise."""
    with np.errstate(invalid="ignore", divide="ignore"):
        noise = np.where(best > 0, err / best, np.nan)
    out = {}
    for name, rest in lines.items():
        v = _velocity(x, rest, redshift)
        window = (v >= AI_RANGE[0]) & (v <= AI_RANGE[1])
        if not window.any():
            continue
        covered = float(np.mean(region[window]))
        if covered == 0:
            continue
        step = float(np.median(np.abs(np.diff(v[window])))) if window.sum() > 1 else 0
        n = max(1, round(smooth_km_s / step)) if step > 0 else 1
        with np.errstate(invalid="ignore", divide="ignore"):
            norm = np.where(best > 0, flux / best, np.nan)
        good = region & np.isfinite(norm)
        smooth = _smooth(norm, good, n)
        sel = window & good
        out[name] = (v[sel], smooth[sel], covered, norm[sel], noise[sel])
    return out


def find_bal_troughs(
    model,
    values,
    redshift,
    lines=None,
    smooth_km_s=500.0,
    quantile=0.95,
    n_sims=200,
    seed=0,
    max_iterations=3,
    min_coverage=0.5,
):
    """Broad absorption troughs of the high-ionization lines (see the
    module docstring).

    The flux over the emission model, smoothed over `smooth_km_s`, gives
    per line the balnicity index (BI, Weymann et al. 1991; troughs below
    0.9 for more than 2000 km/s within -25000 to -3000 km/s) and the
    absorption index (AI, Hall et al. 2002; at least 1000 km/s, within
    -29000 to 0 km/s). A line counts if its AI exceeds the `quantile` of the
    largest AI in noise-only simulations (the model plus correlated
    noise). A trough counts only if the flux stays depressed throughout
    (`BAL_ENVELOPE`): not a cluster of narrow lines, which the narrow
    search handles; the indices are over the troughs that count. Their BI
    troughs (>= 2000 km/s) are masked, the model is refitted without them
    and the troughs
    measured again, until they stop changing: the trough biases the
    emission model it is measured against.

    :param model: The fitted model (of the emission)
    :type model: sculptor.model.FitModel
    :param values: Its variable parameter values
    :type values: np.ndarray
    :param redshift: The quasar's redshift
    :type redshift: float
    :param lines: Line -> rest wavelength (default `BAL_LINES`)
    :type lines: dict
    :param smooth_km_s: Smoothing (running mean) width, km/s
    :type smooth_km_s: float
    :param quantile: Quantile of the noise-only maxima used as threshold
    :type quantile: float
    :param n_sims: Number of noise-only simulations
    :type n_sims: int
    :param seed: Seed of the random generator
    :type seed: int
    :param max_iterations: Mask, refit and measure up to this many times
    :type max_iterations: int
    :param min_coverage: Fraction of a line's AI velocity range the fit
        region must cover for its indices (else NaN)
    :type min_coverage: float
    :return: The troughs, indices, mask and refitted model
    :rtype: BALTroughs
    """
    from sculptor.autofit import _quiet_fits, correlated_noise

    lines = BAL_LINES if lines is None else lines
    work = copy.deepcopy(model)
    work.absorber_mask = None
    region = work._fit_region_mask()
    x = work.spec.dispersion
    flux = work.spec.fluxden
    err = work.spec.fluxden_err
    result = work.quick_fit_result
    mask = np.ones(x.size, dtype=bool)

    for _ in range(max_iterations):
        best = work.eval(x, values)
        measured = _bal_measure(
            x, flux, err, best, region, redshift, lines, smooth_km_s
        )

        rng = np.random.default_rng(seed)
        maxima = np.zeros(n_sims)
        for k in range(n_sims):
            sim = best + correlated_noise(work, result, rng)
            sims = _bal_measure(x, sim, err, best, region, redshift, lines, smooth_km_s)
            maxima[k] = max(
                (absorption_index(m[0], m[1])[0] for m in sims.values()),
                default=0.0,
            )
        threshold = float(np.quantile(maxima, quantile))

        balnicity, absorption, rows = {}, {}, []
        for name, (v, norm, covered, raw, noise) in measured.items():
            ok = covered >= min_coverage
            # Troughs that depress the flux throughout (not narrow-line
            # clusters); the indices over these only
            broad = np.ones(v.size, dtype=bool)
            for a, b in absorption_index(v, norm)[1]:
                envelope = np.nanpercentile(raw[a:b], 90) - 1.28 * np.nanmedian(
                    noise[a:b]
                )
                if envelope >= BAL_ENVELOPE:
                    broad[a:b] = False
            norm = np.where(broad, norm, 1.0)
            bi, bi_runs = balnicity_index(v, norm)
            ai, ai_runs = absorption_index(v, norm)
            balnicity[name] = bi if ok else np.nan
            absorption[name] = ai if ok else np.nan
            if not ok or ai <= threshold:
                continue
            bi_spans = [(v[a], v[b - 1]) for a, b in bi_runs]
            dv = np.gradient(v)
            for a, b in ai_runs:
                v_lo, v_hi = float(v[a]), float(v[b - 1])
                others = []
                for other, (vo, no, *_) in measured.items():
                    if other == name:
                        continue
                    sel = (vo >= v_lo) & (vo <= v_hi)
                    if sel.sum() > 1 and np.nanmean(no[sel]) < BAL_LEVEL:
                        others.append(other)
                rows.append(
                    (
                        name,
                        v_lo,
                        v_hi,
                        float(_wavelength(v_lo, lines[name], redshift)),
                        float(_wavelength(v_hi, lines[name], redshift)),
                        float(np.max(1 - norm[a:b])),
                        float(np.sum((1 - norm[a:b]) * dv[a:b])),
                        any(lo <= v_hi and hi >= v_lo for lo, hi in bi_spans),
                        ", ".join(others),
                    )
                )

        new_mask = np.ones(x.size, dtype=bool)
        for row in rows:
            # BI troughs only (>= 2000 km/s); AI-only ones are left to the
            # narrow search (`find_absorbers` drops those it explains)
            if not row[7]:
                continue
            # Plus the smoothing width on each side
            pad = smooth_km_s
            lo = _wavelength(row[1] - pad, lines[row[0]], redshift)
            hi = _wavelength(row[2] + pad, lines[row[0]], redshift)
            new_mask &= ~((x >= lo) & (x <= hi))
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        work.absorber_mask = mask
        try:
            with _quiet_fits():
                result = work.run_quick_fit(update_values=True)
            values = result.values
        except (ValueError, np.linalg.LinAlgError):
            break

    troughs = Table(
        rows=rows or None,
        names=(
            "line",
            "v_min",
            "v_max",
            "wave_min",
            "wave_max",
            "max_depth",
            "ai",
            "bal",
            "also_in",
        ),
        dtype=(str, float, float, float, float, float, float, bool, str),
    )
    return BALTroughs(troughs, balnicity, absorption, threshold, mask, work)
