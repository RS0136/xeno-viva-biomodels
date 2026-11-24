"""The studies that stand behind the readings: how far they depend on what was assumed and chosen.

    sensitivity   on the assumptions of the world: dilution rate, densities, the numbers of the sender, growth law,
                  horizon; in the community the cross specificity and the dilution rate
    settings      on the choices of the measure: tolerances, Q_min, allocation, mode, noise, stored times; the
                  lemma on the runs; and on which quantities are protected at each layer
    numerics      on the numerical method: the coupling interval and the tolerances of the integrator
    channels      on where the interference enters: a sender that takes the nutrient of the receiver, or another
                  species that its model reads
    cascade       on what the layers add to one another: when the interference arrives at each, and how many
                  independent readings they are
    receivers     on the SBML models: every model of the plateau that takes up a species that it reads
    schedules     on the kind of activity: an activity that changes in time, instead of a step
    comparison    against what is used elsewhere: niche overlap, and the linear response of the steady state
    activity      on the range of the activity: D of every step, its gain, and the activity of washout
    quasistatic   on what the SBML model alone would say: its dose-response and its response coefficients,
                  beside the layers of the runs
    functional    on tolerances that follow from the function: the growth rate as its rate law has it, a function
                  of the nutrient and of the species that the uptake reads
    monolith      on the composite itself: one world as one system of equations, under two other integrators
    coexistence   on a community whose baseline is at rest: two pairs that coexist stably, and a xeno that
                  reaches its own partner through the other pair
    information   on the noise: the bits that a noisy record of the receiver carries about what the sender did,
                  against the noise and against the number of values read
    cell          on the kind of receiver: a model of a growing cell, which computes its own growth rate, with
                  the range, the model alone and the tolerances from the function read for it

Each is a module with ``run(plan, jobs, analyze_only)`` that writes under ``runs/studies/<name>/``. ``cascade``
reads what ``channels`` wrote and ``comparison`` what ``receivers`` and ``channels`` wrote; the order below takes
care of it. ``report`` sets the tables and the figures of all of them into ``runs/studies.md``.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

NAMES = (
    "sensitivity",
    "numerics",
    "channels",
    "receivers",
    "schedules",
    "settings",
    "cascade",
    "comparison",
    "activity",
    "quasistatic",
    "functional",
    "monolith",
    "coexistence",
    "information",
    "cell",
)


def run(plan: dict, names: list[str] | None = None, jobs: int = 1, analyze_only: bool = False) -> list[Path]:
    """The studies named (all of them unless given), in the order of ``NAMES``."""
    unknown = [name for name in names or [] if name not in NAMES]
    if unknown:
        raise SystemExit(f"no study {unknown}; there are {', '.join(NAMES)}")
    written: list[Path] = []
    for name in NAMES:
        if names and name not in names:
            continue
        started = time.monotonic()
        module = importlib.import_module(f"xeno_biomodels.studies.{name}")
        written += module.run(plan, jobs, analyze_only)
        print(f"study {name}: {time.monotonic() - started:.0f} s", flush=True)
    return written
