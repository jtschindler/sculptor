"""Time `sculptor.utils.broaden_spectrum` against a plain-numpy version.

Replaces the timing part of the old `examples/test_resolution.py`, which
compared against `SpecOneD.broaden_by_resolution` (a method no `speconed`
release has, see `TODO_speconed.md`).

Run: `uv run python benchmarks/bench_broaden_spectrum.py`
"""

import argparse
import timeit

import numpy as np

from sculptor import analysis as ana
from sculptor import utils as scut

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def numpy_broaden(dispersion, fluxden, resolution, fwhm_lim=5.0):
    """Same algorithm as the numba kernel, without numba."""

    out = np.empty_like(fluxden, dtype=float)
    for i, (wave, r) in enumerate(zip(dispersion, resolution, strict=True)):
        fwhm = wave / r
        sigma = fwhm * FWHM_TO_SIGMA
        lo, hi = np.searchsorted(
            dispersion, [wave - fwhm_lim * fwhm, wave + fwhm_lim * fwhm]
        )
        weights = np.exp(-0.5 * ((dispersion[lo:hi] - wave) / sigma) ** 2)
        out[i] = np.sum(weights * fluxden[lo:hi]) / np.sum(weights)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pixels", type=int, default=4000)
    parser.add_argument("--resolution", type=float, default=350.0)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    dispersion = np.linspace(5000, 9000, args.pixels)
    fluxden = ana.power_law_at2500(dispersion, 3.227, 1.0, -1.5) + (
        ana.line_model_gaussian(dispersion, 3.227, 50.0, 1549.06, 3000.0)
    )
    resolution = np.full_like(dispersion, args.resolution)

    # First call compiles the numba kernel (cached on disk afterwards).
    start = timeit.default_timer()
    scut.broaden_spectrum(dispersion, fluxden, resolution)
    print(
        f"numba first call (incl. compile/cache load): "
        f"{timeit.default_timer() - start:.3f} s"
    )

    for name, func in [
        ("numba", scut.broaden_spectrum),
        ("numpy", numpy_broaden),
    ]:
        t = timeit.timeit(
            lambda f=func: f(dispersion, fluxden, resolution), number=args.repeat
        )
        print(
            f"{name}: {t / args.repeat * 1e3:8.3f} ms per call "
            f"({args.pixels} pixels, R={args.resolution:g})"
        )

    diff = np.max(
        np.abs(
            scut.broaden_spectrum(dispersion, fluxden, resolution)
            - numpy_broaden(dispersion, fluxden, resolution)
        )
    )
    print(f"max |numba - numpy|: {diff:.3e}")


if __name__ == "__main__":
    main()
