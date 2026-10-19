"""Tests for `sculptor.plot`: palette switch and shared style defaults."""

import importlib

import matplotlib
import pytest
import tol_colors

from sculptor import plot as scp


@pytest.fixture
def reload_plot(monkeypatch):
    """Reload `sculptor.plot` under a given SCULPTOR_PALETTE, then restore."""

    def _reload(value):
        if value is None:
            monkeypatch.delenv(scp.PALETTE_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(scp.PALETTE_ENV_VAR, value)
        return importlib.reload(scp)

    yield _reload
    monkeypatch.delenv(scp.PALETTE_ENV_VAR, raising=False)
    importlib.reload(scp)


def test_default_palette_is_tol(reload_plot):
    module = reload_plot(None)

    assert module.palette == "tol"
    assert module.dblue == tol_colors.bright.blue
    assert module.vermillion == tol_colors.bright.red


@pytest.mark.parametrize("value", ["okabe-ito", " Okabe-Ito "])
def test_okabe_ito_via_env_var(reload_plot, value):
    module = reload_plot(value)

    assert module.palette == "okabe-ito"
    assert module.dblue == (0, 114 / 255.0, 178 / 255.0)
    assert module.vermillion == (213 / 255.0, 94 / 255.0, 0)


def test_unknown_palette_is_an_error(reload_plot):
    with pytest.raises(ValueError, match="SCULPTOR_PALETTE"):
        reload_plot("viridis")


def test_both_palettes_define_the_same_names():
    assert scp.PALETTES["tol"].keys() == scp.PALETTES["okabe-ito"].keys()


def test_component_cmap_is_tol_rainbow():
    assert scp.component_cmap is tol_colors.rainbow_PuRd


def test_style_defaults_share_the_tick_setup():
    shared = ["xtick.direction", "xtick.minor.visible", "ytick.right"]

    with matplotlib.rc_context():
        scp.set_presentation_defaults()
        presentation = {k: matplotlib.rcParams[k] for k in shared}
        assert matplotlib.rcParams["mathtext.fontset"] == "custom"

    with matplotlib.rc_context():
        scp.set_paper_defaults()
        paper = {k: matplotlib.rcParams[k] for k in shared}
        assert matplotlib.rcParams["text.usetex"] is True

    assert (
        presentation
        == paper
        == {
            "xtick.direction": "in",
            "xtick.minor.visible": True,
            "ytick.right": True,
        }
    )
