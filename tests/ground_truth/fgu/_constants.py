# Holds parameters for everything in the models
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from astropy import constants as const

# Ref this for ionisation potentials: https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
# Together with this that I know is a trousted source: https://gitlab.nublado.org/cloudy/cloudy/-/blob/4c5c1363b559a9d694223641e56716d252b599a6/data/ionization_potentials.dat
# To convert from cm^-1 to eV, multiply by 1.2398419843320026e-4

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

# these are just constants that I am using in other scripts
QUASAR_REDSHIFT = 8.129

C_LIGHT = (const.c.to("km/s")).value

TZ = ZoneInfo("Europe/Amsterdam")

PREFIX = Path.home() / "data" / "J1504+6252"

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

try:
    MANUAL_SPEC_MASK = np.load(
        Path.home() / "data" / "J1504+6252" / "mask_ranges_bool.npy"
    )
except FileNotFoundError:
    print(
        "[WARNING] mask_ranges_bool.npy not found! Update the script "
        "and either supply wavelength ranges that need to be masked "
        "(list of tuples, [(start, end), ...]) and call "
        "`model.remove_wavelength_range_from_fit_mask(region[0], region[1])` "
        "or generate the mask_ranges_bool.npy file by running `make_mask.py` "
        "and then re-run this script."
    )

# regions to remove from the fitting - this is now taken care by the boolean mask
#  not really sure why I went this route, it is pretty stupid...
# REGIONS_TO_REMOVE_BAD_DATA = np.loadtxt(
#     Path.home() / "data" / "J1504+6252" / "mask_ranges.csv",
#     skiprows=1,
#     delimiter=",",
# )
REGIONS_TO_REMOVE_BAD_DATA = None  # <- Left in case I need to add extra stuff

# manually added region to remove, based on whatever consideration
REGIONS_TO_REMOVE_MANUAL = np.array(
    [
        [10000.000, 10500.000],
        [13000.000, 13700.000],  # The BAL (?) feature I see, eyeBALled
        # [14300.000, 14880.000],  # Nagao+2006, 1600 A bump \pm 30 A
        [15430.000, 16700.000],  # Nagao+2006, region bluewards of CIII complex
        # [18000.000, 19500.000], # FeII there that I need to remove if I am not using VW01
        # [18500.000, 19300.000], # FeIII_2038_2113
        # [21700.000, 22400.000], # FeIII_47
    ]
)

# merge the regions together if needed
REGIONS_TO_REMOVE = REGIONS_TO_REMOVE_MANUAL

# For the previously defined regions, see github history, sometimes before 31.07.2026 at 11.50

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

# Arbitrarly defined major lines that I need to use to check for the proximate
PROXIMATE_LINES = {
    # "NV_1": 1238.821,
    # "NV_2": 1242.804,
    "SiII": 1260.4221,
    "OI": 1302.1685,
    "CII": 1334.5323,
    "CII*": 1334.5323,
    "SiIV_1": 1393.7602,
    "SiIV_2": 1402.7729,
    "CIV_1": 1548.204,
    "CIV_2": 1550.781,
    # "HeII": 1640.42,
    # "CIII]": 1908.73,
    "MgII_1": 2796.3543,
    "MgII_2": 2803.5315,
    r"H$\delta$": 4101.74,
    r"H$\gamma$": 4340.47,
    r"H$\beta$": 4861.34,
}

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #

# Claude generated these
# ------------------------------------------------------------------------------
# Vanden Berk et al. (2001, astro-ph/0105231) Table 2 -- non-iron lines
# lambda_lab values (vacuum), ID column
# ------------------------------------------------------------------------------

VANDENBERK_LINES = {
    "Ly-eps": 937.80,
    "Ly-delta": 949.74,
    "CIII/NIII": 977.02,  # blend, using CIII component
    "Ly-beta": 1025.72,
    "OVI": 1033.83,
    "ArI": 1066.66,
    "CIII*": 1175.70,
    "Ly-alpha": 1215.67,
    "NV": 1240.14,
    "SiII1263": 1262.59,
    "OI": 1304.35,
    "SiII1307": 1306.82,
    "CII": 1335.30,
    "SiIV+OIV]": 1396.76,
    "CIV": 1549.06,
    "HeII": 1640.42,
    "OIII]1663": 1663.48,
    "AlII1671": 1670.79,
    "NIV": 1718.55,
    "NIII]": 1750.26,
    "SiII1817": 1816.98,
    "NeIII1815": 1814.73,
    "AlIII": 1857.40,
    "SiIII]": 1892.03,
    "CIII]": 1908.73,
    "CII]": 2326.44,
    "NeIV]": 2423.83,
    "OII2471": 2471.03,
    "AlII]2670": 2669.95,
    "OIII2672": 2672.04,
    "MgII": 2798.75,
    "OIII3134": 3133.70,
    "HeI3189": 3188.67,
    "NeV3347": 3346.82,
    "NeV3427": 3426.84,
    "FeVII3587": 3587.34,
    "HeI3588": 3588.30,
    "OII3728": 3728.48,
    "FeVII3760": 3759.99,
    "NeIII3870": 3869.85,
    "HeI3890": 3889.74,
    "H8": 3890.15,
    "NeIII3969": 3968.58,
    "H-eps": 3971.20,
    "FeV4072": 4072.39,
    "SII4074": 4073.63,
    "H-delta": 4102.89,
    "FeII_forbid4227": 4243.97,  # [FeII] approx, listed as Opt21F blend
    "H-gamma": 4341.68,
    "OIII]4364": 4364.44,
    "HeI4473": 4472.76,
    "HeII4687": 4687.02,
    "H-beta": 4862.68,
    "OIII]4960": 4960.30,
    "OIII]5008": 5008.24,
    "FeVII5160": 5160.33,
    "FeVI5177": 5177.48,
    "NI5201": 5200.53,
    "FeVII5278": 5277.85,
    "FeXIV/CaV": 5304.34,
    "ClIII5539": 5539.43,
    "FeVII5722": 5722.30,
    "HeI5877": 5877.29,
    "FeVII6086": 6087.98,
    "OI6302": 6302.05,
    "OI6366": 6365.54,
    "FeX6376": 6376.30,
    "NII6550": 6549.85,
    "H-alpha": 6564.61,
    "NII6585": 6585.28,
    "SII6718": 6718.29,
    "SII6733": 6732.67,
    "HeI7067": 7067.20,
    "ArIII7138": 7137.80,
    "OII7321": 7321.48,
    "NiIII/FeXI": 7892.10,
}

# ------------------------------------------------------------------------------
# Fe II / Fe III lines, built directly from Table 2 -- multiplet designations
# from Netzer & Wills (1983), Grandi (1981), Phillips (1978), as cited in the
# paper. lambda_obs is used here (the composite's measured peak position),
# since several of these blends don't have a single clean lambda_lab value.
# ------------------------------------------------------------------------------

FEII_LINES = {
    "FeII UV40": 1664.74,  # blended with OIII]1663/AlII1671
    "FeII UV37": 1716.88,  # blended with NIV/AlII
    "FeII UV191": 1788.73,
    "FeII UV79": 2175.62,
    "FeII UV370/UV118/UV376": 2222.29,
    "FeII (many)": 2324.58,
    "FeII UV395": 2467.98,  # blended with [OII]
    "FeII UV1_2627": 2626.92,
    "FeII UV78": 2964.28,
    "FeII Opt82": 3127.70,  # blended with OIII3134
    "FeII Opt6/Opt7": 3191.78,
    "FeII Opt91/Opt1_3261": 3261.40,
    "FeII Opt1_3282": 3281.74,
    "FeII Opt4/Opt16": 3498.92,
    "FeII Opt15": 3785.47,
    "FeII Opt14": 3817.41,
    "FeII Opt27/Opt28_4140": 4140.50,
    "FeII Opt27/Opt28_4188": 4187.55,
    "FeII Opt32": 4318.30,  # blended with [FeII] Opt21F
    "FeII Opt37_4478": 4478.22,
    "FeII Opt37/Opt38 bump": 4564.71,  # broad feature, mainly FeII
    "FeII Opt42": 4930.75,
    "FeII Opt48/Opt49 bump": 5305.97,  # broad feature, mainly FeII
    "FeII Opt49_5278": 5277.92,
    "FeII Opt48/Opt49_5314": 5313.82,
    "FeII Opt55": 5545.16,
}

FEIII_LINES = {
    "FeIII UV1": 1117.26,
    "FeIII UV34/UV68/UV61": 1905.97,  # the famous 1900A CIII]-blended feature
    "FeIII UV50": 1991.83,
    "FeIII UV48": 2076.62,
    "FeIII UV47": 2423.46,  # blended with [NeIV]
}
