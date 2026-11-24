"""Where things are: this project, the checkout of viva-biomodels, the cache of models, the runs.

viva-biomodels is a workspace and not a wheel (its pyproject sets ``bypass-selection``), so installing it gives
its dependencies and not the package ``viva_biomodels``. The checkout is put on ``sys.path`` instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
VIVA_BIOMODELS = Path(os.environ.get("VIVA_BIOMODELS_HOME", PROJECT / "third_party" / "viva-biomodels"))
CORPUS = VIVA_BIOMODELS / "datasets" / "corpus_all_engines"
MODELS = PROJECT / "models"  # SBML files fetched from BioModels, one directory per identifier
RUNS = PROJECT / "runs"


def use_viva_biomodels() -> Path:
    """Make ``import viva_biomodels`` work, from the checkout. Returns the checkout."""
    if not (VIVA_BIOMODELS / "viva_biomodels" / "__init__.py").exists():
        raise FileNotFoundError(
            f"no checkout of viva-biomodels at {VIVA_BIOMODELS}: clone "
            "https://github.com/vivarium-collective/viva-biomodels there, or set VIVA_BIOMODELS_HOME"
        )
    text = str(VIVA_BIOMODELS)
    if text not in sys.path:
        sys.path.insert(0, text)
    return VIVA_BIOMODELS
