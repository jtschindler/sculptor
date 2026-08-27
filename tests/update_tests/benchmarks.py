import time

from sculptor import utils as scut_new
from sculptor.updated_modules import utils as scut_orig
from tests.update_tests.test_utils import make_test_spectrum

if __name__ == "__main__":
    dispersion, fluxden, resolution = make_test_spectrum(n=5000)

    t0 = time.perf_counter()
    scut_orig.broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5)
    print("original:", time.perf_counter() - t0)

    # run once to trigger JIT compilation, then time a second call
    scut_new.broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5.0)
    
    t0 = time.perf_counter()
    scut_new.broaden_spectrum(dispersion, fluxden, resolution, fwhm_lim=5.0)
    print("new:", time.perf_counter() - t0)
