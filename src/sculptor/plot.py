"""Plot colours and matplotlib style defaults.

Colours come from Paul Tol's colour-blind-safe schemes (`tol-colors`
package) by default. Set the environment variable `SCULPTOR_PALETTE` to
`okabe-ito` before importing sculptor to use the Okabe & Ito palette the
plots used before. The colour names (`dblue`, `vermillion`, ...) are the
same in both palettes.
"""

import os

import matplotlib
import tol_colors

PALETTE_ENV_VAR = "SCULPTOR_PALETTE"

# Okabe & Ito (2008), the original sculptor palette
_OKABE_ITO = {
    "black": (0, 0, 0),
    "orange": (230 / 255.0, 159 / 255.0, 0),
    "blue": (86 / 255.0, 180 / 255.0, 233 / 255.0),
    "green": (0, 158 / 255.0, 115 / 255.0),
    "yellow": (240 / 255.0, 228 / 255.0, 66 / 255.0),
    "dblue": (0, 114 / 255.0, 178 / 255.0),
    "vermillion": (213 / 255.0, 94 / 255.0, 0),
    "purple": (181 / 255.0, 107 / 255.0, 148 / 255.0),
}

# Paul Tol's "bright" scheme (plus "vibrant" orange, which "bright" lacks),
# mapped onto the same names by hue
_TOL = {
    "black": "#000000",
    "orange": tol_colors.vibrant.orange,
    "blue": tol_colors.bright.cyan,
    "green": tol_colors.bright.green,
    "yellow": tol_colors.bright.yellow,
    "dblue": tol_colors.bright.blue,
    "vermillion": tol_colors.bright.red,
    "purple": tol_colors.bright.purple,
}

PALETTES = {"tol": _TOL, "okabe-ito": _OKABE_ITO}


def _palette_from_env():
    name = os.environ.get(PALETTE_ENV_VAR, "tol").strip().lower()
    if name not in PALETTES:
        raise ValueError(
            f"Unknown {PALETTE_ENV_VAR}={name!r}; choose one of {', '.join(PALETTES)}."
        )
    return name


palette = _palette_from_env()
_colors = PALETTES[palette]

black = _colors["black"]
orange = _colors["orange"]
blue = _colors["blue"]
green = _colors["green"]
yellow = _colors["yellow"]
dblue = _colors["dblue"]
vermillion = _colors["vermillion"]
purple = _colors["purple"]

# Colour map for model components (Tol rainbow, in both palettes)
component_cmap = tol_colors.rainbow_PuRd

# GeoDataViz Colors
gdv_div_a = [
    "#009392",
    "#39b185",
    "#9ccb86",
    "#e9e29c",
    "#eeb479",
    "#e88471",
    "#cf597e",
]
gdv_div_b = [
    "#045275",
    "#089099",
    "#7ccba2",
    "#fcde9c",
    "#f0746e",
    "#dc3977",
    "#7c1d6f",
]
gdv_div_d = [
    "#008042",
    "#6fa253",
    "#b7c370",
    "#fce498",
    "#d78287",
    "#bf5688",
    "#7c1d6f",
]


# Define default setups
def _set_common_defaults():
    # Tick and axis style shared by the paper and presentation styles
    matplotlib.rcParams["xtick.major.size"] = 4.5
    matplotlib.rcParams["xtick.minor.size"] = 3
    matplotlib.rcParams["ytick.major.size"] = 4.5
    matplotlib.rcParams["ytick.minor.size"] = 3
    matplotlib.rcParams["axes.linewidth"] = 1.2
    matplotlib.rcParams["xtick.direction"] = "in"
    matplotlib.rcParams["ytick.direction"] = "in"
    matplotlib.rcParams["xtick.minor.visible"] = True
    matplotlib.rcParams["ytick.minor.visible"] = True
    matplotlib.rcParams["xtick.top"] = True
    matplotlib.rcParams["ytick.right"] = True


def set_paper_defaults():
    # Defining the paper plotting style (needs a LaTeX installation)
    """Matplotlib style for papers: LaTeX text (needs a LaTeX installation) and inward ticks on all sides."""

    matplotlib.rc("text", usetex=True)
    _set_common_defaults()


def set_presentation_defaults():
    # Defining the presentation plotting style (uses the Roboto font)
    """Matplotlib style for presentations: Roboto font (must be installed) and inward ticks on all sides."""

    matplotlib.rc("font", weight=500)
    matplotlib.rc("font", **{"family": "sans-serif", "sans-serif": ["Roboto"]})
    matplotlib.rcParams["mathtext.fontset"] = "custom"
    matplotlib.rcParams["mathtext.it"] = "Roboto"
    matplotlib.rcParams["mathtext.rm"] = "Roboto"
    _set_common_defaults()
