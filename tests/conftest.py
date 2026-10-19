import sys
from pathlib import Path

import pytest

import sculptor

REPO_ROOT = Path(__file__).resolve().parents[1]
DEPRECATED_SCULPTOR_DIR = REPO_ROOT / "deprecated" / "src" / "sculptor"


@pytest.fixture(scope="module")
def deprecated_dep_v1():
    """Make `sculptor.dep_v1.*` (now under `deprecated/`) importable.

    The ground-truth reference copies in `tests/ground_truth/` import
    `sculptor.dep_v1.speconed` / `.specanalysis` under their old names. This
    exposes them only for the test module that requests the fixture, so the
    maintained package cannot silently depend on deprecated code.
    """
    sculptor.__path__.append(str(DEPRECATED_SCULPTOR_DIR))
    try:
        yield
    finally:
        sculptor.__path__.remove(str(DEPRECATED_SCULPTOR_DIR))
        for name in [m for m in sys.modules if m.startswith("sculptor.dep_v1")]:
            del sys.modules[name]
        if hasattr(sculptor, "dep_v1"):
            del sculptor.dep_v1
