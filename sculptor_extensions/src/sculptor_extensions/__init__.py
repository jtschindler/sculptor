import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sculptor-extensions")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Modules renamed to lowercase (old name -> new name). The old names stay
# importable, with a DeprecationWarning, so that existing scripts and fits
# saved with them (pickles refer to e.g. `sculptor_extensions.Qso_autofit`)
# keep working. A file with the old name cannot be kept next to the new one
# on case-insensitive file systems, hence the import hook.
_RENAMED_MODULES = {"Qso_autofit": "qso_autofit"}


class _RenamedModuleLoader(importlib.abc.Loader):
    """Return the renamed module for its old name."""

    def __init__(self, new_name):
        self.new_name = new_name

    def create_module(self, spec):
        """Import and return the module under its new name."""
        return importlib.import_module(self.new_name)

    def exec_module(self, module):
        """Nothing to execute: the module is already imported."""


class _RenamedModuleFinder(importlib.abc.MetaPathFinder):
    """Find the old names in `_RENAMED_MODULES`."""

    def find_spec(self, fullname, path=None, target=None):
        """Spec for an old module name, or None for any other module."""
        package, _, name = fullname.rpartition(".")
        if package != __name__ or name not in _RENAMED_MODULES:
            return None
        new_name = f"{__name__}.{_RENAMED_MODULES[name]}"
        warnings.warn(
            f"`{fullname}` was renamed to `{new_name}`",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.util.spec_from_loader(fullname, _RenamedModuleLoader(new_name))


if not any(isinstance(f, _RenamedModuleFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _RenamedModuleFinder())
