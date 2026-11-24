"""What the SBML model alone would say.

The pathways of the receivers settle within minutes, and the vessel moves over hours. If what is inside the
organism follows the levels of its surroundings as fast as those levels move, the layers inside it are the
dose-response of the SBML model alone, read at the levels that the world produces, and a world adds only how far
those levels move, the density, and what a fast activity does. This study asks the model alone and sets the answer
beside the runs:

    dose        the steady state of the model at levels of its nutrient from far below to far above that of its
                file, followed from one level to the next; and the level below which it has no steady state
    response    the response coefficient of every protected quantity (the relative change of its steady state over
                that of the nutrient, as in metabolic control analysis), beside the same ratio read from the runs
                of the weakest control
    prediction  for every control, every admissible step and every protected quantity inside the organism: the
                steady state of the model alone, followed along the course that the species of the medium take in
                the run, gives a D; it is set beside D of the run, with how far the run is from that course at
                its furthest (in tolerances). The worlds are those of the plan (one species of the medium moves)
                and those of the study ``channels`` (two do)

Written to ``runs/studies/quasistatic/``: ``dose.csv``, ``response.csv``, ``prediction.csv`` and ``summary.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xeno_biomodels import experiment, layers
from xeno_biomodels.isolated import Isolated
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies import channels
from xeno_biomodels.studies.common import STUDIES, leak_of, read_rows, write_rows
from xeno_biomodels.world import BASELINE, World

OUT = STUDIES / "quasistatic"
BELOW = np.geomspace(1.0, 0.02, 120)[1:]  # levels of the nutrient over that of the file, going down
ABOVE = np.geomspace(1.0, 20.0, 90)[1:]  # and going up
NEAR = 0.01  # the response coefficient is read this share above and below the level of the file
INSIDE = ("particle", "molecule", "complex", "individual")  # the layers that the model alone has
STRIDE = 3  # the model is followed along a run at every third stored time
COURSE = "u=2"  # the step whose course is written time by time


def inside(plan: dict, biomodel: str) -> list[layers.Entry]:
    """The protected quantities of a receiver that its model alone has."""
    entries = layers.protected(plan, layers.sbml_organism(plan, biomodel, "B"))
    return [entry for entry in entries if entry.layer in INSIDE]


def dose_response(world: World) -> tuple[list[dict], float | None]:
    """The steady state of the model alone along the levels of its nutrient: rows (ratio, level, every column),
    in the order of the levels, and the highest ratio below 1 at which it has none (None if it has one down to
    the lowest level asked)."""
    alone, nutrient, rest = Isolated(world), world.nutrient, world.sbml_medium[world.nutrient]
    rows, lowest = [{"ratio": 1.0, "level": rest, **alone.quantities()}], None
    for ratio in BELOW:
        found = alone.steady({nutrient: ratio * rest})
        if found is None:
            lowest = float(ratio)
            break
        rows.append({"ratio": float(ratio), "level": float(ratio * rest), **found})
    alone.fresh()
    for ratio in ABOVE:
        found = alone.steady({nutrient: ratio * rest})
        if found is None:
            break
        rows.append({"ratio": float(ratio), "level": float(ratio * rest), **found})
    return sorted(rows, key=lambda row: row["ratio"]), lowest


def edge(world: World, good: float, bad: float, steps: int = 26) -> tuple[float, float]:
    """The level below which the model alone has no steady state, between a ratio at which it has one and a
    ratio at which it has none: bisection, each trial approached from the level of the file in small steps."""
    alone, nutrient, rest = Isolated(world), world.nutrient, world.sbml_medium[world.nutrient]
    for _ in range(steps):
        middle = float(np.sqrt(good * bad))
        alone.fresh()
        reached = all(
            alone.steady({nutrient: ratio * rest}) is not None for ratio in np.geomspace(1.0, good, 9)[1:]
        )
        if not reached:  # the way down is itself too steep for the model: keep what is known
            break
        if alone.steady({nutrient: middle * rest}) is not None:
            good = middle
        else:
            bad = middle
    return good, bad


def coefficients(world: World, columns: list[str]) -> dict[str, float]:
    """The response coefficient of each column: d ln(steady state) / d ln(level of the nutrient), at the file."""
    alone, nutrient, rest = Isolated(world), world.nutrient, world.sbml_medium[world.nutrient]
    low = alone.steady({nutrient: (1.0 - NEAR) * rest})
    high = alone.steady({nutrient: (1.0 + NEAR) * rest})
    out = {}
    for column in columns:
        if low is None or high is None or not (low[column] > 0.0 and high[column] > 0.0):
            continue
        out[column] = float(np.log(high[column] / low[column]) / np.log((1.0 + NEAR) / (1.0 - NEAR)))
    return out


def along(world: World, plan: dict, directory: Path, head: dict) -> list[dict]:
    """One world: for every admissible step and every protected quantity inside the organism, D of the run and D
    of the steady state of the model alone followed along the course of the medium in that run."""
    if not (directory / "measure" / "result.json").exists():
        return []
    entries = inside(plan, world.biomodel)
    result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
    delta = {row["column"]: float(row["delta"]) for row in read_rows(directory / "measure" / "deltas.csv")}
    lost = {item["activity"]: item["t"] for item in result["losses"]}
    left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
    times, base = read_series(directory / BASELINE / "protected.csv")
    alone = Isolated(world)
    at_rest = alone.steady({name: float(base[f"medium.{name}"][0]) for name in world.medium_ids})
    rows = []
    for name, activity in experiment.run_names(plan).items():
        path = directory / name / "protected.csv"
        if name in left_out or not path.exists() or at_rest is None:
            continue
        _, columns = read_series(path)
        read = np.flatnonzero(times < lost[name] if name in lost else np.ones(times.size, dtype=bool))[
            ::STRIDE
        ]
        if not read.size:
            continue
        alone.fresh()
        course = {entry.column: np.full(read.size, np.nan) for entry in entries}
        for n, at in enumerate(read):
            state = alone.steady({key: float(columns[f"medium.{key}"][at]) for key in world.medium_ids})
            if state is not None:
                for entry in entries:
                    course[entry.column][n] = state[entry.column]
        covered = ~np.isnan(course[entries[0].column])  # where the model alone has a steady state
        for entry in entries:
            tolerance, run = delta[entry.column], columns[entry.column][read]
            start = base[entry.column][read]
            row = {
                **head,
                "step": name,
                "activity": activity,
                "quantity": entry.column,
                "layer": entry.layer,
                "kind": entry.kind,
                "D_of_the_run": float(np.max(np.abs(run - start)) / tolerance),
                "D_of_the_model_alone": None,
                "furthest_from_the_model_alone": None,
                "times_read": int(read.size),
                "times_without_a_steady_state": int(np.sum(~covered)),
            }
            if np.any(covered):
                alone_course = course[entry.column]
                row["D_of_the_run"] = float(
                    np.max(np.abs(run - start)[covered]) / tolerance
                )  # on the same times
                row["D_of_the_model_alone"] = float(
                    np.nanmax(np.abs(alone_course - at_rest[entry.column])) / tolerance
                )
                row["furthest_from_the_model_alone"] = float(
                    np.nanmax(np.abs(run - alone_course)) / tolerance
                )
            rows.append(row)
    return rows


def course(world: World, plan: dict, directory: Path, step: str) -> list[dict]:
    """One run, time by time: every protected quantity inside the organism and the nutrient, over their values
    at rest, in the run and (``alone:``) as the steady state of the model alone at the medium of that time."""
    entries = inside(plan, world.biomodel)
    times, base = read_series(directory / BASELINE / "protected.csv")
    _, columns = read_series(directory / step / "protected.csv")
    alone = Isolated(world)
    at_rest = alone.steady({name: float(base[f"medium.{name}"][0]) for name in world.medium_ids})
    rows = []
    for at in range(0, times.size, STRIDE):
        state = alone.steady({key: float(columns[f"medium.{key}"][at]) for key in world.medium_ids})
        row = {"time": float(times[at])}
        nutrient = f"medium.{world.nutrient}"
        row[nutrient] = float(columns[nutrient][at] / base[nutrient][0])
        for entry in entries:
            row[entry.column] = float(columns[entry.column][at] / base[entry.column][0])
            row[f"alone:{entry.column}"] = (
                None if state is None else float(state[entry.column] / at_rest[entry.column])
            )
        rows.append(row)
    return rows


def run(
    plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT, runs: Path = experiment.OUT
) -> list[Path]:
    """``runs`` is where the worlds of ``plan`` were followed and measured. The worlds of the study ``channels``
    are read beside them only for the plan itself."""
    del jobs, analyze_only  # the model alone settles in milliseconds, and the runs are those that are there
    doses, responses, predictions, summary = [], [], [], {}
    controls = {name: leak_of(leak) for name, leak in plan["designs"].items() if leak_of(leak) > 0.0}
    weakest = min(controls, key=controls.get)
    for biomodel in plan["receivers"]:
        world = World(plan, biomodel, "xeno")
        entries = inside(plan, biomodel)
        nutrient = f"medium.{world.nutrient}"
        rows, lowest = dose_response(world)
        summary[biomodel] = {
            "level_in_the_file": world.sbml_medium[world.nutrient],
            "no_steady_state_at_ratio": lowest,
            "lowest_ratio_with_a_steady_state": rows[0]["ratio"],
            "highest_ratio_followed": rows[-1]["ratio"],
        }
        if lowest is not None:  # where exactly the steady state ends, between the two levels of the grid
            good, bad = edge(world, rows[0]["ratio"], lowest)
            summary[biomodel].update(steady_state_down_to_ratio=good, none_from_ratio=bad)
        for row in rows:
            for entry in entries:
                doses.append(
                    {
                        "biomodel": biomodel,
                        "ratio": row["ratio"],
                        "level": row["level"],
                        "quantity": entry.column,
                        "layer": entry.layer,
                        "kind": entry.kind,
                        "value": row[entry.column],
                    }
                )
        # the response coefficients, beside the same ratio in the runs of the weakest control: the relative
        # change of the quantity over that of the nutrient, each at its own peak
        found = coefficients(world, [entry.column for entry in entries])
        directory = runs / biomodel / weakest
        measured = {row["column"]: row for row in read_rows(directory / "measure" / "deltas.csv")}

        def relative(column: str, measured=measured) -> float:
            one = measured[column]
            return float(one["D"]) * float(one["delta"]) / abs(float(one["y0"]))

        for entry in entries:
            responses.append(
                {
                    "biomodel": biomodel,
                    "quantity": entry.column,
                    "layer": entry.layer,
                    "kind": entry.kind,
                    "response_coefficient": found.get(entry.column),
                    "in_the_runs_of_the_weakest_control": relative(entry.column) / relative(nutrient),
                }
            )
        for design, leak in controls.items():
            head = {"biomodel": biomodel, "world": f"plan/{design}", "leak": leak, "taken": world.nutrient}
            predictions += along(world, plan, runs / biomodel / design, head)

    # the worlds in which two species of the medium move
    if runs == experiment.OUT and (channels.OUT / "worlds.csv").exists():
        worlds: dict[str, World] = {}
        for tag, variant, biomodel, design in channels.variants(plan):
            if tag["channel"] == "none":
                continue
            if biomodel not in worlds:
                worlds[biomodel] = World(variant, biomodel, "xeno")
            head = {
                "biomodel": biomodel,
                "world": f"channels/{design}",
                "leak": tag["share"],
                "taken": tag["species"],
            }
            predictions += along(worlds[biomodel], variant, channels.OUT / "worlds" / biomodel / design, head)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    written = [
        write_rows(out / "dose.csv", doses),
        write_rows(out / "response.csv", responses),
        write_rows(out / "prediction.csv", predictions),
        out / "summary.json",
    ]
    # one run of every receiver, time by time, beside the model alone: the step to twice the activity in the
    # control in which the sender takes about a hundredth
    middle = min(controls, key=lambda name: abs(np.log10(controls[name]) + 2.0))
    for biomodel in plan["receivers"]:
        directory = runs / biomodel / middle
        if (directory / COURSE / "protected.csv").exists():
            rows = course(World(plan, biomodel, middle), plan, directory, COURSE)
            written.append(write_rows(out / f"course_{biomodel}.csv", rows))
    return written
