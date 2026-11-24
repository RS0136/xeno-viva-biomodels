"""A receiver that grows at a rate of its own.

The three receivers of the plan are models of a pathway. A world gives each of them a growth law (the growth rate
is the uptake over its value at rest), puts the level of its nutrient at the value of its file whatever the
dilution rate, and leaves its enzymes where the file has them. This study takes a receiver for which none of the
three is needed: the model of a growing cell of Weisse et al. (2015; BioModels MODEL1502270000, as its authors
deposited it), in which the import of a nutrient, its conversion to energy, transcription, translation and the
dilution of everything by growth give a growth rate that the model computes itself.

    the nutrient     the model holds what is outside it as a parameter (``s0``); the world lets it vary, and the
                     import reaction of the model (``r_0``) is its uptake
    growth           the growth rate of the model (``lam``), not a law of the world
    the level        in a vessel the model lives at the level of its nutrient at which it grows at the dilution
                     rate; that level is found from the model (38.5 of its units at 0.1 per hour, against 10000
                     in its file) and is far inside the range in which the nutrient limits growth
    the leak         in the units of no model, as for the receivers taken by rule: at rest the sender takes a
                     share of what the receiver takes, and cannot use it

The runs are those of the plan (the same steps of the activity, horizon and measure). They are then read as the
runs of the plan are: by the model alone (``quasistatic``), by the closed form of the vessel (``activity``), and
with tolerances that follow from the function (``functional``), which here is the growth rate of the model
itself, a function of its energy and of its translating ribosomes.

Written to ``runs/studies/cell/``: ``worlds/`` (the runs and ``summary.csv``), ``layers.csv``, ``model.json`` and
the three readings under ``quasistatic/``, ``activity/`` and ``functional/``.
"""

from __future__ import annotations

import json
from pathlib import Path

from xeno_biomodels.experiment import run_variants
from xeno_biomodels.studies import activity, copasi_course, functional, quasistatic
from xeno_biomodels.studies.common import STUDIES, changed, write_rows
from xeno_biomodels.world import World

OUT = STUDIES / "cell"
CELL = "MODEL1502270000"
#: what the plan would say of this receiver
RECEIVER = {
    "organism": "Escherichia coli",
    "what": "growth of a cell",
    "medium": ["s0"],
    "nutrient": "s0",
    "uptake_reactions": {"s0": "r_0"},
    "growth_variable": "lam",
    "time_unit_seconds": 60,  # the file gives no unit of time; the rates of the model are per minute
    "consumes": 0.9,  # alone at rest it takes nine tenths of what is fed
    "atp": "a",  # the energy of the model: one species, made from the nutrient and used by translation
    "carriers": ["a"],
    # everything that is made by transcription or translation: free ribosomes, enzymes and other proteins, the
    # messengers, and the ribosomes that are bound to them
    "proteins": [
        *("r", "et", "em", "q", "p"),
        *("mr", "mt", "mm", "mq", "mp"),
        *("rmr", "rmt", "rmm", "rmq", "rmp"),
        *("zmr", "zmt", "zmm", "zmq", "zmp"),
    ],
}
SHARES = (0.001, 0.01, 0.1)
#: the step by which a growing cell is known: the rate at which it makes ribosomes
SPECIFIC = {"complex": "{name}.rates.r_14"}


def variant(plan: dict) -> dict:
    """The plan with the growing cell as its only receiver and the leak as a share."""
    designs: dict = {"xeno": 0.0}
    for share in SHARES:
        designs[f"share-{share:g}"] = {"share": float(share)}
    specific = {"xeno": plan["protect"]["specific"].get("xeno", {}), CELL: SPECIFIC}
    return changed(
        plan,
        {
            "receivers": {CELL: RECEIVER},
            "designs": designs,
            "protect.specific": specific,
            # the density of this receiver is in units of its own: it counts as established down to a
            # thousandth of what it has at rest, which is what the plan's number is for its own receivers
            "viable": "rel:1e-3",
        },
    )


def agreement(world: World, plan: dict, minutes: float = 600.0) -> dict:
    """The file as it is, from its own initial state, under CVODE of libRoadRunner and under LSODA of COPASI: the
    largest normalised root-mean-square difference of a species between the two (the score by which the
    receivers of the plan were chosen, here between two engines), and the growth rate that the file settles at.
    The entry is not among the curated ones of BioModels, so nobody has checked it against its publication."""
    import numpy as np
    import tellurium as te

    r_tol = float(plan["time"].get("relative_tolerance", 1e-10))
    a_tol = float(plan["time"].get("absolute_tolerance", 1e-12))
    model = te.loadSBMLModel(str(world.sbml))
    model.integrator.absolute_tolerance, model.integrator.relative_tolerance = a_tol, r_tol
    species = list(model.getFloatingSpeciesIds())
    model.selections = ["time", *[f"[{name}]" for name in species]]
    count = round(minutes) + 1
    first = np.asarray(model.simulate(0.0, minutes * world.minute, count))[:, 1:]
    other = copasi_course.apart(
        Path(world.sbml).read_text(encoding="utf-8"),
        {},
        species,
        [],
        0.0,
        minutes * world.minute,
        count,
        r_tol,
        a_tol,
    )["species"]
    worst, moving = 0.0, 0
    for i, name in enumerate(species):
        a, b = first[:, i], other[name]
        scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if (
            scale > 1e-12 and float(np.ptp(a)) > 1e-9 * scale
        ):  # a species that the file leaves at zero is noise
            moving += 1
            worst = max(worst, float(np.sqrt(np.mean((a - b) ** 2))) / scale)
    model.simulate(minutes * world.minute, 2e5 * world.minute, 2)
    return {
        "largest_nrmse_between_the_two_engines": worst,
        "species_that_move": moving,
        "growth_rate_of_the_file": float(model[RECEIVER["growth_variable"]]) * world.minute,
    }


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    study = variant(plan)
    work = []
    for design, written in study["designs"].items():
        tag = {"share": written["share"] if isinstance(written, dict) else 0.0}
        work.append((tag, study, CELL, design, out / "worlds", analyze_only, True))
    rows, by_group = run_variants(work, jobs)
    world = World(study, CELL, "xeno")
    facts = {
        "biomodel": CELL,
        "level_in_the_file": 10000.0,
        "level_at_rest": world.sbml_medium[world.nutrient],
        "growth_rate_at_rest": world.own_growth_at_rest,
        "dilution_rate": world.dilution,
        "uptake_at_rest": world.uptake_at_rest,
        "density_alone_at_rest": world.biovolume,
        "feed": world.feed[world.nutrient],
        **agreement(world, study),
        "species_at_rest": world.sbml_species,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "model.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    written = [
        write_rows(out / "worlds" / "summary.csv", rows),
        write_rows(out / "layers.csv", by_group),
        out / "model.json",
    ]
    written += quasistatic.run(study, out=out / "quasistatic", runs=out / "worlds")
    written += activity.run(study, out=out / "activity", runs=out / "worlds")
    written += functional.run(study, out=out / "functional", runs=out / "worlds")
    return written
