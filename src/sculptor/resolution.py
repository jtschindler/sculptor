"""Instrumental resolution: from a model evaluated on a fine grid to the
flux density of the pixels.

A pixel measures the model convolved with the line spread function (a
Gaussian of FWHM lambda / R) and averaged over the pixel. With a few pixels
per resolution element (e.g. JWST/NIRSpec, ~2 pixels per FWHM), the model
evaluated at the pixel centers is not a good approximation: a line narrower
than a pixel is sampled at one or two points. `resolution_matrix` builds
the sparse matrix K and the fine grid x_fine with

    pixels = K @ model(x_fine)

where row i holds the weights of

    1 / (hi_i - lo_i) * integral_lo_i^hi_i (model * LSF_i)(x) dx
        = integral model(x') [Phi((hi_i - x') / s_i) - Phi((lo_i - x') / s_i)]
          / (hi_i - lo_i) dx'

over the fine grid (`oversample` points per pixel, midpoint rule), Phi the
normal CDF and s_i the LSF sigma at pixel i. Rows are normalized to 1, so a
constant model stays constant at the edges of the spectrum. `FitModel`
caches the matrix per spectrum (`FitModel.resolution`).

`read_jwst_dispersion` reads the JWST dispersion tables
(jwst_nirspec_<grating>_disp.fits), `resolution_on_grid` maps one or several
of them (gratings stitched at given wavelengths) onto a spectrum's pixels.
"""

import hashlib
from collections import OrderedDict

import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix
from scipy.special import ndtr

# FWHM / sigma of a Gaussian
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))

_CACHE = OrderedDict()
_CACHE_SIZE = 16


def pixel_edges(dispersion):
    """Pixel edges halfway between neighboring pixel centers (the outer
    ones half a pixel beyond the first and last center).

    :param dispersion: Pixel centers, increasing
    :type dispersion: np.ndarray
    :return: N + 1 edges
    :rtype: np.ndarray
    """
    x = np.asarray(dispersion, dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two pixels.")
    mid = 0.5 * (x[1:] + x[:-1])
    return np.concatenate(([x[0] - (mid[0] - x[0])], mid, [x[-1] + (x[-1] - mid[-1])]))


def fine_grid(edges, oversample):
    """`oversample` equal sub-pixels per pixel: their centers and widths.

    :param edges: N + 1 pixel edges, increasing
    :type edges: np.ndarray
    :param oversample: Sub-pixels per pixel
    :type oversample: int
    :return: centers and widths, N * oversample each
    :rtype: (np.ndarray, np.ndarray)
    """
    edges = np.asarray(edges, dtype=float)
    width = np.diff(edges)
    frac = (np.arange(oversample) + 0.5) / oversample
    centers = (edges[:-1, None] + width[:, None] * frac[None, :]).ravel()
    widths = np.repeat(width / oversample, oversample)
    return centers, widths


def resolution_matrix(dispersion, resolution, edges=None, oversample=5, n_sigma=5.0):
    """Matrix from the model on a fine grid to the pixels (see the module
    docstring); cached.

    :param dispersion: Pixel centers, increasing
    :type dispersion: np.ndarray
    :param resolution: R = lambda / FWHM at each pixel, or one value
    :type resolution: float or np.ndarray
    :param edges: N + 1 pixel edges (default `pixel_edges(dispersion)`)
    :type edges: np.ndarray
    :param oversample: Fine-grid points per pixel
    :type oversample: int
    :param n_sigma: Kernel extent beyond the pixel, in LSF sigmas
    :type n_sigma: float
    :return: K (N x N * oversample, sparse) and the fine grid
    :rtype: (scipy.sparse.csr_matrix, np.ndarray)
    :raises ValueError: for non-positive or non-finite resolutions, or
        edges that do not match the pixels
    """
    x = np.ascontiguousarray(dispersion, dtype=float)
    res = np.broadcast_to(np.asarray(resolution, dtype=float), x.shape)
    res = np.ascontiguousarray(res)
    if not np.all(np.isfinite(res) & (res > 0)):
        raise ValueError("The resolution must be finite and positive at every pixel.")
    edges = pixel_edges(x) if edges is None else np.ascontiguousarray(edges, float)
    if edges.shape != (x.size + 1,) or np.any(np.diff(edges) <= 0):
        raise ValueError("`edges` must be N + 1 increasing values for N pixels.")

    key = hashlib.sha1(
        b"".join(
            [
                x.tobytes(),
                res.tobytes(),
                edges.tobytes(),
                repr((oversample, n_sigma)).encode(),
            ]
        )
    ).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]

    xf, wf = fine_grid(edges, oversample)
    lo, hi = edges[:-1], edges[1:]
    sigma = x / res / FWHM_PER_SIGMA
    first = np.searchsorted(xf, lo - n_sigma * sigma, side="left")
    last = np.searchsorted(xf, hi + n_sigma * sigma, side="right")
    counts = last - first
    rows = np.repeat(np.arange(x.size), counts)
    cols = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    cols += np.repeat(first, counts)
    s = sigma[rows]
    weights = (ndtr((hi[rows] - xf[cols]) / s) - ndtr((lo[rows] - xf[cols]) / s)) * (
        wf[cols] / (hi - lo)[rows]
    )
    norm = np.bincount(rows, weights=weights, minlength=x.size)
    weights /= norm[rows]
    matrix = csr_matrix((weights, (rows, cols)), shape=(x.size, xf.size))

    _CACHE[key] = (matrix, xf)
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return matrix, xf


def read_jwst_dispersion(filename):
    """A JWST dispersion table (e.g. jwst_nirspec_g395m_disp.fits).

    :param filename: The FITS file
    :type filename: str or pathlib.Path
    :return: wavelength (Angstrom) and resolution R
    :rtype: (np.ndarray, np.ndarray)
    """
    with fits.open(filename) as hdul:
        data = hdul[1].data
        wave = np.asarray(data["WAVELENGTH"], dtype=float) * 1e4  # micron -> A
        res = np.asarray(data["R"], dtype=float)
    order = np.argsort(wave)
    return wave[order], res[order]


def resolution_on_grid(dispersion, curves, boundaries=(), scale=1.0, blends=None):
    """Resolution at each pixel from one or several resolution curves (e.g.
    gratings stitched in a coadd), interpolated in wavelength.

    Where a coadd is a weighted mean of two gratings (`blends`), the line
    spread function is their weighted mix; it is replaced by the Gaussian
    with the same second moment, FWHM^2 = w FWHM_1^2 + (1 - w) FWHM_2^2.

    :param dispersion: Pixel centers (Angstrom)
    :type dispersion: np.ndarray
    :param curves: (wavelength, R) pairs, in order of wavelength
    :type curves: list(tuple(np.ndarray, np.ndarray))
    :param boundaries: Wavelengths where one curve hands over to the next
        (one fewer than the curves)
    :type boundaries: list(float)
    :param scale: Factor applied to R, e.g. > 1 for a point source in a slit
        (the JWST curves are for a uniformly illuminated slit)
    :type scale: float
    :param blends: Per boundary, None (the curves hand over at the boundary)
        or (lo, hi, w): between lo and hi the two curves are mixed, with
        weight w for the first (e.g. from `overlap_weight`)
    :type blends: list
    :return: R at each pixel
    :rtype: np.ndarray
    :raises ValueError: if a pixel is outside the curve that covers it
    """
    x = np.asarray(dispersion, dtype=float)
    if len(boundaries) != len(curves) - 1:
        raise ValueError("Need one boundary fewer than curves.")
    piece = np.searchsorted(np.asarray(boundaries, dtype=float), x, side="right")
    res = np.empty_like(x)
    for idx, (wave, values) in enumerate(curves):
        sel = piece == idx
        if not sel.any():
            continue
        if x[sel].min() < wave.min() or x[sel].max() > wave.max():
            raise ValueError(
                f"Curve {idx} covers {wave.min():.0f}-{wave.max():.0f} A, but its "
                f"pixels span {x[sel].min():.0f}-{x[sel].max():.0f} A."
            )
        res[sel] = np.interp(x[sel], wave, values)

    for idx, blend in enumerate(blends or []):
        if blend is None:
            continue
        lo, hi, weight = blend
        sel = (x >= lo) & (x <= hi)
        if not sel.any():
            continue
        fwhm2 = 0.0
        for k, w in ((idx, weight), (idx + 1, 1 - weight)):
            wave, values = curves[k]
            if x[sel].min() < wave.min() or x[sel].max() > wave.max():
                raise ValueError(
                    f"Curve {k} does not cover the blend {lo:.0f}-{hi:.0f} A."
                )
            fwhm2 = fwhm2 + w * (x[sel] / np.interp(x[sel], wave, values)) ** 2
        res[sel] = x[sel] / np.sqrt(fwhm2)
    return res * scale


def overlap_weight(dispersion, error, lo, hi, width=1000.0, mask=None):
    """Weight of the first of two gratings in an inverse-variance weighted
    mean over [lo, hi], from the errors just outside the overlap, where
    each grating is alone (median error over `width` below lo and above
    hi).

    :param dispersion: Pixel centers
    :type dispersion: np.ndarray
    :param error: Errors of the coadd
    :type error: np.ndarray
    :param lo: Start of the overlap
    :type lo: float
    :param hi: End of the overlap
    :type hi: float
    :param width: Width of the ranges outside the overlap
    :type width: float
    :param mask: Good pixels (default: all)
    :type mask: np.ndarray
    :return: Weight of the first grating (below lo)
    :rtype: float
    """
    x = np.asarray(dispersion, dtype=float)
    good = np.ones(x.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    below = good & (x >= lo - width) & (x < lo)
    above = good & (x > hi) & (x <= hi + width)
    if not below.any() or not above.any():
        raise ValueError("No good pixels next to the overlap.")
    inv1 = np.median(error[below]) ** -2.0
    inv2 = np.median(error[above]) ** -2.0
    return float(inv1 / (inv1 + inv2))
