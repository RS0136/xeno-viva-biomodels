"""A BioModels SBML organism run through viva-biomodels, an organism orthogonal to it, and the measure."""

from __future__ import annotations

import ctypes
import sys
import sysconfig
from pathlib import Path

__version__ = "0.1.0"


def preload_libpython() -> bool:
    """Load the shared library of this interpreter, so that libroadrunner finds it.

    The extension of libroadrunner links against ``libpython3.x.so.1.0``. An interpreter that uv manages keeps
    that file in its own ``lib`` directory, which is not on the path of the dynamic loader, and ``import
    roadrunner`` then fails with "cannot open shared object file". Once the library is loaded here, the loader
    resolves the dependency by its name. Returns whether a library was loaded.
    """
    name = sysconfig.get_config_var("INSTSONAME")
    if not name or not sysconfig.get_config_var("Py_ENABLE_SHARED"):
        return False
    for directory in (sysconfig.get_config_var("LIBDIR"), Path(sys.base_prefix) / "lib"):
        path = Path(str(directory)) / name
        if path.exists():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            return True
    return False


preload_libpython()
