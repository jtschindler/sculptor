# Common utils
import argparse
from pathlib import Path

import matplotlib.backend_bases as mbb
import matplotlib.pyplot as plt
import numpy as np
import spectres
from _constants import (
    FEII_LINES,
    FEIII_LINES,
    QUASAR_REDSHIFT,
    VANDENBERK_LINES,
)
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button, TextBox
from sculptor import component as sccomp
from speconed import speconed as sod
from tqdm.auto import tqdm

# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def patch_resize_event_attrs():
    """
    Workaround for matplotlib widget callbacks (Button/TextBox internals)
    that access mouse/location-event attributes (inaxes, x, y, xdata,
    ydata, button, key, ...) on ANY canvas event reaching them, including
    'resize_event' -- but ResizeEvent doesn't carry these attributes,
    causing AttributeError.

    Adds safe class-level defaults so attribute lookup succeeds instead
    of raising, regardless of which specific attribute a given widget's
    internal handler happens to touch first. Values chosen to make
    "is this event relevant to me" checks resolve to "no" wherever
    possible (inaxes=None so `event.inaxes is not self.ax` is True and
    the widget returns early; x/y/xdata/ydata=None or 0 as inert
    fallbacks for any code that gets past that check).

    Call this once, before creating any Button/TextBox/Slider widgets.
    """
    defaults = {
        "inaxes": None,
        "x": 0,
        "y": 0,
        "xdata": None,
        "ydata": None,
        "button": None,
        "key": None,
        "dblclick": False,
        "step": 0,
        "guiEvent": None,
    }
    for attr, default in defaults.items():
        if not hasattr(mbb.ResizeEvent, attr):
            setattr(mbb.ResizeEvent, attr, default)


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit the spectrum of the z = 8 quasar."
    )

    parser.add_argument(
        "-p",
        "--plot_spec",
        action="store_true",
        help="Plot the spectrum and exit.",
    )

    parser.add_argument(
        "-l",
        "--plot_lines",
        action="store_true",
        help="Plot the spectrum with Vanden Berk et al. (2001) lines and exit.",
    )

    parser.add_argument(
        "-c",
        "--show-components",
        action="store_true",
        help="Show the individual components of the model.",
    )

    parser.add_argument(
        "-a",
        "--fit-sdss-analogue",
        action="store_true",
        help="Fit the SDSS local analogue to estimate the Lya properties.",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile the fit_spectrum function.",
    )

    parser.add_argument(
        "-m",
        "--make-mask",
        action="store_true",
        help="Make a mask for the spectrum.",
    )

    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="Save the model in pkl format after the fitting is complete.",
    )

    parser.add_argument(
        "-d",
        "--discard",
        type=int,
        default=0,
        help="Discard the specified number of initial steps when saving the model.",
    )

    parser.add_argument(
        "-n",
        "--nsteps",
        type=int,
        default=20000,
        help="Number of steps for the MCMC fitting.",
    )

    parser.add_argument(
        "-w",
        "--nwalkers",
        type=int,
        default=100,
        help="Number of walkers for the MCMC fitting.",
    )

    parser.add_argument(
        "-t",
        "--thin",
        type=int,
        default=10,
        help="Thinning factor for the MCMC fitting.",
    )

    return parser.parse_args()


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def inverse(array):
    """
    Calculate and return the inverse of the input array, enforcing
    positivity and setting values <= 0 to zero.  The input array should
    be a quantity expected to always be positive, like a variance or an
    inverse variance. The quantity::

        out = (array > 0.0)/(np.abs(array) + (array == 0.0))

    is returned.

    Args:
        array (`numpy.ndarray`_):
            Array to invert

    Returns:
        `numpy.ndarray`_: Result of controlled ``1/array`` calculation.
    """
    return (array > 0.0) / (np.abs(array) + (array == 0.0))


# =========================================================================== #


def convert_spec1d_flam(wave, flux, ivar, point_source=False):
    """
    Convert PypeIt/JWST spec1d fluxes to F_lambda

    F_lambda = F_nu * c / lambda^2

    :param wave: Wavelength in Angstroem
    :type wave: np.ndarray
    :param flux: Flux in MJy/sr
    :type flux: np.ndarray
    :param ivar: Inverse variance
    :type ivar: np.ndarray
    :return:
    """

    wave = wave * u.angstrom
    nu_to_flam_factor = const.c / np.square(wave)

    # Hardcoded pixel area for NIRSpec pixels
    if point_source:
        pixel_area = 1.0 * u.steradian
    else:
        pixel_area = 4.795157053786361e-13 * u.steradian

    # Conversion factor
    factor = pixel_area * nu_to_flam_factor

    # Convert flux
    flux_MJy = flux * u.MJy / u.steradian

    # Calculate sigma
    sigma_MJy = np.sqrt(inverse(ivar)) * u.MJy / u.steradian

    # Decompose units
    f_lam = (flux_MJy * factor).decompose().to(
        u.erg / u.s / u.cm**2 / u.angstrom
    ).value / 1e-17
    sigma_lam = (sigma_MJy * factor).decompose().to(
        u.erg / u.s / u.cm**2 / u.angstrom
    ).value / 1e-17

    return wave, f_lam, sigma_lam


# =========================================================================== #


def load_pypeit_spectrum(path, point_source=True):

    hdu = fits.open(path)

    if "OPT_WAVE" in hdu[1].data.columns.names:
        wave = np.array(hdu[1].data["OPT_WAVE"], dtype=np.float32)
        flux = np.array(hdu[1].data["OPT_COUNTS"], dtype=np.float32)
        ivar = np.array(hdu[1].data["OPT_COUNTS_IVAR"], dtype=np.float32)

    elif "BOX_WAVE" in hdu[1].data.columns.names:
        wave = np.array(hdu[1].data["BOX_WAVE"], dtype=np.float32)
        flux = np.array(hdu[1].data["BOX_COUNTS"], dtype=np.float32)
        ivar = np.array(hdu[1].data["BOX_COUNTS_IVAR"], dtype=np.float32)

    elif "wave" in hdu[1].data.columns.names:
        wave = np.array(hdu[1].data["WAVE_GRID_MID"], dtype=np.float32)
        flux = np.array(hdu[1].data["FLUX"], dtype=np.float32)
        ivar = np.array(hdu[1].data["IVAR"], dtype=np.float32)
        mask = np.array(hdu[1].data["MASK"], dtype=bool)

    wave, f_lam, sigma_lam = convert_spec1d_flam(
        wave,
        flux,
        ivar,
        point_source=point_source,
    )

    mask &= sigma_lam > 0

    spec = sod.SpecOneD(
        dispersion=wave.value,
        dispersion_unit=1 * u.AA,
        fluxden=f_lam,
        fluxden_unit=1e-17 * u.erg / u.s / u.cm**2 / u.AA,
        fluxden_err=sigma_lam,
        mask=mask,
    )

    return spec


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def plot_vandenberk_lines(
    ax,
    redshift,
    label_lines=True,
    color="grey",
    linestyle="--",
    alpha=0.5,
    fontsize=7,
    y_frac=0.95,
    rotation=90,
    observed_frame=True,
):
    xlim = ax.get_xlim()
    for name, rest_wave in VANDENBERK_LINES.items():
        wave = rest_wave * (1 + redshift) if observed_frame else rest_wave
        if wave < xlim[0] or wave > xlim[1]:
            continue
        ax.axvline(wave, color=color, linestyle=linestyle, alpha=alpha, lw=0.8)
        if label_lines:
            ax.text(
                wave,
                y_frac,
                name,
                transform=ax.get_xaxis_transform(),
                rotation=rotation,
                fontsize=fontsize,
                alpha=alpha,
                ha="center",
                va="top",
                color=color,
            )
    return ax


# =========================================================================== #


def plot_iron_features(
    ax,
    redshift,
    label_lines=True,
    color="firebrick",
    linestyle=":",
    alpha=0.4,
    fontsize=6,
    y_frac=0.85,
    rotation=90,
    observed_frame=True,
    species="both",
):
    """species: 'FeII', 'FeIII', or 'both'"""
    xlim = ax.get_xlim()
    lines = {}
    if species in ("FeII", "both"):
        lines.update(FEII_LINES)
    if species in ("FeIII", "both"):
        lines.update(FEIII_LINES)

    for name, rest_wave in lines.items():
        wave = rest_wave * (1 + redshift) if observed_frame else rest_wave
        if wave < xlim[0] or wave > xlim[1]:
            continue
        ax.axvline(wave, color=color, linestyle=linestyle, alpha=alpha, lw=0.8)
        if label_lines:
            ax.text(
                wave,
                y_frac,
                name,
                transform=ax.get_xaxis_transform(),
                rotation=rotation,
                fontsize=fontsize,
                alpha=alpha,
                ha="center",
                va="top",
                color=color,
            )
    return ax


# =========================================================================== #


def plot_all_lines(ax, redshift, **kwargs):
    plot_vandenberk_lines(ax, redshift, **kwargs)
    plot_iron_features(ax, redshift, **kwargs)
    return ax


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def initialise_component(prefix, model_func, param_dict, is_absorption=False):
    """
    Setup a FitComponent from a parameter dictionary.

    :param prefix: Model prefix (e.g. "CIV_br")
    :param model_func: Model function (e.g. ana.line_model_gaussian)
    :param param_dict: Dictionary of parameter configurations, keyed by
        parameter name (without prefix). Each entry can have:
            - "value": initial value
            - "vary": bool (default True)
            - "prior": prior object (optional)
            - "parameter": name of another global parameter to tie this
              one to (hard alias via param_mapping), instead of "value"/
              "vary"/"prior"
    :return: comp, params
    """
    comp = sccomp.FitComponent(prefix, model_func, is_absorption=is_absorption)
    params = comp.create_params()

    for par, config in param_dict.items():
        key = f"{prefix}_{par}"
        if key not in params:
            raise KeyError(f"Parameter '{key}' not found in component params.")

        if "parameter" in config:
            params.pop(key)
            comp.param_mapping.update({par: config["parameter"]})
            continue

        if "value" in config:
            params[key].value = config["value"]
        if "vary" in config:
            params[key].vary = config["vary"]
        if "prior" in config:
            params[key].prior = config["prior"]

    return comp, params


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def fit_spectrum(
    spec,
    redshift,
    model,
    nsteps=1000,
    discard=500,
    nwalkers=40,
    thin=1,
    save_dir=".",
    refit=False,
    resolution=None,
    show_components=False,
    norm_resid=True,
):
    """Fit the spectrum with a model consisting of a power law continuum
    and a number of emission lines.

    :param spec:
    :param redshift:
    :param narrow_line_fwhm:
    :param save_suffix:
    :param flux_factor:
    :param fit_range:
    :param dered:
    :return:
    """

    # Generate the path to the output directory
    dir_path = Path(f"{save_dir}")
    dir_path.mkdir(parents=True, exist_ok=True)

    chain_path = dir_path / "flat_chain.csv"
    result_path = dir_path / "fit_results.escv"

    if (not chain_path.exists() and not result_path.exists()) or refit:
        model, _ = model(spec, resolution=resolution)
        model.gpm_fit = model.gpm_fit & spec.mask
        model.redshift = redshift

        # Run the MCMC
        model.run_emcee(nsteps, nwalkers)

        flat_samples = model.sampler.get_chain(discard=discard, thin=1, flat=True)
        pv = list(model.params_variable.keys())

        for idx, name in tqdm(
            enumerate(pv),
            desc="[INFO] Plotting posterior distributions",
            total=len(pv),
        ):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), layout="constrained")
            ax.hist(flat_samples[:, idx], bins=50, color="k", histtype="step", lw=2)
            ax.set_xlabel(name)
            ax.set_ylabel("Posterior")

            fig.savefig(dir_path / f"{name}_posterior.pdf")
            plt.close(fig)

        model.plot_mcmc_result(
            discard=discard,
            thin=thin,
            save=True,
            save_dir=dir_path,
            xlim=[
                spec.dispersion[model.gpm_fit].min(),
                spec.dispersion[model.gpm_fit].max(),
            ],
            show_components=show_components,
            save_data=True,
            ylim=[
                -spec.fluxden[model.gpm_fit].min() * 0.05,
                spec.fluxden[model.gpm_fit].max() * 1.05,
            ],
            save_name="_fit_result.pdf",
            set_presentation_defaults=False,
            norm_resid=norm_resid,
        )

        model.plot_posterior_corner(
            discard=discard,
            save=True,
            save_dir=dir_path,
            save_name="_corner_plot.pdf",
        )

    return model, flat_samples, pv


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def add_rest_frame_toggle_button(fig, ax, redshift, button_rect=(0.4, 0.01, 0.2, 0.05)):
    # NOTE: subplots_adjust removed from here -- call it once, externally,
    # before invoking either add_*_button function (see usage below).
    state = {"frame": "observed"}
    obs_formatter = ax.xaxis.get_major_formatter()
    obs_label = ax.get_xlabel() or "Observed wavelength"

    def rest_formatter(x, pos):
        return f"{x / (1 + redshift):.0f}"

    def on_click(event):
        if state["frame"] == "observed":
            ax.xaxis.set_major_formatter(FuncFormatter(rest_formatter))
            ax.set_xlabel("Rest-frame wavelength")
            state["frame"] = "rest"
        else:
            ax.xaxis.set_major_formatter(obs_formatter)
            ax.set_xlabel(obs_label)
            state["frame"] = "observed"
        fig.canvas.draw_idle()

    button_ax = fig.add_axes(button_rect)
    button = Button(button_ax, "obs/rest")
    button.on_clicked(on_click)

    return button


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def add_velocity_resample_button(
    fig,
    ax,
    spec,
    c_kms=299792.458,
    button_rect=(0.62, 0.01, 0.12, 0.05),
    textbox_rect=(0.78, 0.01, 0.10, 0.05),
    default_dv_kms=50.0,
    flux_line=None,
    err_line=None,
):
    orig_wave = np.asarray(spec.dispersion)
    orig_flux = np.asarray(spec.fluxden)
    orig_err = (
        np.asarray(spec.fluxden_error)
        if getattr(spec, "fluxden_error", None) is not None
        else None
    )

    if flux_line is None:
        if not ax.lines:
            raise ValueError(
                "No Line2D found on `ax` to update -- pass flux_line explicitly."
            )
        flux_line = ax.lines[0]
        print(
            "[WARNING] add_velocity_resample_button: auto-selected ax.lines[0] as "
            "the flux curve. Verify this is correct, or pass flux_line explicitly."
        )

    if err_line is None:
        if orig_err is not None and len(ax.lines) > 1:
            err_line = ax.lines[1]
            print(
                "[WARNING] add_velocity_resample_button: auto-selected ax.lines[1] as "
                "the flux error curve. Verify this is correct, pass err_line explicitly, "
                "or pass err_line=False to disable error-line updating."
            )
        else:
            err_line = False

    state = {"dv_kms": default_dv_kms}

    def build_constant_velocity_grid(wave_min, wave_max, dv_kms):
        ratio = 1.0 + dv_kms / c_kms
        n = int(np.floor(np.log(wave_max / wave_min) / np.log(ratio))) + 1
        return wave_min * ratio ** np.arange(n)

    def do_resample(dv_kms):
        if dv_kms <= 0:
            print(f"[WARNING] dv must be positive, got {dv_kms} -- ignoring.")
            return

        new_wave = build_constant_velocity_grid(
            orig_wave.min(), orig_wave.max(), dv_kms
        )
        new_wave = new_wave[1:-1]

        if orig_err is not None:
            new_flux, new_err = spectres.spectres(
                new_wave,
                orig_wave,
                orig_flux,
                spec_errs=orig_err,
                fill=np.nan,
                verbose=False,
            )
        else:
            new_flux = spectres.spectres(
                new_wave, orig_wave, orig_flux, fill=np.nan, verbose=False
            )
            new_err = None

        flux_line.set_data(new_wave, new_flux)

        if err_line is not False and err_line is not None and new_err is not None:
            err_line.set_data(new_wave, new_err)

        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        fig.canvas.draw_idle()

    def on_button_click(event):
        do_resample(state["dv_kms"])

    def on_text_submit(text):
        try:
            dv = float(text)
        except ValueError:
            print(f"[WARNING] Invalid dv value '{text}' -- must be a number (km/s).")
            return
        state["dv_kms"] = dv

    # NOTE: subplots_adjust removed from here too -- see usage below.
    button_ax = fig.add_axes(button_rect)
    button = Button(button_ax, "Resample")
    button.on_clicked(on_button_click)

    textbox_ax = fig.add_axes(textbox_rect)
    textbox = TextBox(textbox_ax, "dv [km/s]", initial=str(default_dv_kms))
    textbox.on_submit(on_text_submit)

    return button, textbox


# =========================================================================== #


def prepare_model_for_save(model, discard=0):
    """
    Snapshot everything needed for post-hoc analysis into plain,
    pickle-safe attributes on the model. Defaults to discard=0 -- since
    the HDF5 backend (chain.h5) is the primary, always-full-chain source,
    this snapshot's only job is to be a self-sufficient fallback, and
    trimming it defeats that purpose. Pass a nonzero discard explicitly
    only if you're confident about the burn-in AND storage is a genuine
    constraint.
    """
    model.get_params_to_sample()
    model.param_names = list(model.params_variable.keys())

    if model.sampler is not None:
        # keep everything to not regret it later
        model.flat_chain = model.sampler.get_chain(discard=0, thin=1, flat=True)
        model.flat_log_prob = model.sampler.get_log_prob(discard=0, thin=1, flat=True)

    if model.mcmc_model_posterior is None:
        model.get_mcmc_model_results(discard=discard)

    return model


# =========================================================================== #
# =========================================================================== #
# =========================================================================== #


def load_sdss_analogue(spec, prefix):
    print("[INFO] Fitting the SDSS local analogue to estimate the Lya properties")

    # use the SDSS local analogue to estimate the Lya properties
    sdss = Table.read(prefix / "input" / "spec-4618-55600-0120.fits")
    sdss["lam"] = 10 ** sdss["loglam"]

    NW = [1230, 2000]
    SHIFT_ANA = 1400  # Taken from Julien's paper

    ZP1_QSO = 1 + QUASAR_REDSHIFT
    ZP1_ANA = 1 + 2.899
    Z_CONV = ZP1_QSO / ZP1_ANA

    NORM = np.median(
        spec.fluxden[
            (spec.dispersion > NW[0] * ZP1_QSO) & (spec.dispersion < NW[1] * ZP1_QSO)
        ]
    ) / np.median(
        sdss["flux"][(sdss["lam"] > NW[0] * ZP1_ANA) & (sdss["lam"] < NW[1] * ZP1_ANA)]
        / Z_CONV
    )

    spec.dispersion = sdss["lam"] * Z_CONV - (SHIFT_ANA / const.c.to("km/s").value) * (
        1216.57 * ZP1_QSO
    )

    spec.fluxden = sdss["flux"] / Z_CONV * NORM
    spec.fluxden_err = 1 / np.sqrt(sdss["ivar"]) / Z_CONV * NORM
    spec.fluxden_ivar = sdss["ivar"] * Z_CONV**2 / NORM**2
    spec.mask = np.ones_like(spec.fluxden, dtype=bool)

    return spec
