"""How the readings depend on the choices of the measure.

Nothing is run here: the runs of the worlds and of the community are read again.

    choices      one choice of the measure changed at a time: the tolerance of the protected quantities, Q_min of
                 the receiver and of the sender, the allocation, how the two functions are read over time, the
                 viable density, the noise of the bits, and how often a run is stored. For every world and every
                 ordered pair of the community: D, layer by layer, and whether the stages still say the same.
    lemma        the lemma on the runs. For three functions of the receiver (its growth rate, which is the
                 function of the plan; its work, density times uptake; the share of its adenylate that is ATP),
                 with a tolerance ``fn`` on the quantities that the function reads: at every stored time of every
                 run, is the function kept where every quantity is within its tolerance (the implication), and by
                 how much does the function fall below what the first-order bound of the lemma allows (what the
                 second order takes from the margin)?
    candidates   which quantities are protected. Every species and every reaction rate of a model is read alone,
                 as if it were the one protected at its layer, with the tolerance of the plan: where the
                 quantities that the plan picks stand among all that could have been picked.

Written to ``runs/studies/settings/``: ``worlds.csv``, ``worlds_layers.csv``, ``community.csv``,
``community_layers.csv``; ``lemma.csv`` (one row per function and run) and ``lemma_summary.csv``;
``candidates.csv`` (one row per world and quantity) and ``candidates_summary.csv`` (per world and layer).
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from xeno_biomodels import community, experiment, layers
from xeno_biomodels.orthogonality import (
    Function,
    MeasureError,
    measure,
    parse_protected,
    read_series,
    tolerances,
)
from xeno_biomodels.studies.common import STUDIES, changed, number, read_rows, write_rows
from xeno_biomodels.world import BASELINE, REMOVAL

OUT = STUDIES / "settings"
CHOICES = {
    "tolerance 5 %": {"protect.tolerance": "rel:0.05"},
    "tolerance 20 %": {"protect.tolerance": "rel:0.2"},
    "Q_min of the receiver a quarter": {"measure.qmin": "rel:0.25"},
    "Q_min of the receiver three quarters": {"measure.qmin": "rel:0.75"},
    "allocation all": {"measure.allocation": "all"},
    "function of the receiver read at the end": {"measure.mode": "terminal"},
    "function of the receiver read on average": {"measure.mode": "time_average"},
    "function of the sender read at every time": {"measure.sender_mode": "pointwise"},
    "function of the sender read on average": {"measure.sender_mode": "time_average"},
    "Q_min of the sender a quarter": {"measure.sender_qmin": "rel:0.25"},
    "Q_min of the sender three quarters": {"measure.sender_qmin": "rel:0.75"},
    "viable density ten times": {"viable": lambda value: 10.0 * float(value)},
    "noise a tenth": {"measure.sigma": 0.1},
    "noise ten times": {"measure.sigma": 10.0},
    "stored every 20 min": {"measure.stride": 2},
    "stored every 40 min": {"measure.stride": 4},
    "stored every 80 min": {"measure.stride": 8},
}
STAGES = ("stage1_trajectory", "stage3_tolerable", "stage4_distribution")
#: the functions on which the lemma is tried; {adp} is the species ADP of the model
FUNCTIONS = {
    "growth rate": "{name}.growth_rate",
    "work": "{name}.density * {uptake}",
    "share of ATP": "{name}.species.{atp} / ({name}.species.{atp} + {name}.species.ADP)",
}


# ---- the choices ---------------------------------------------------------------------------------------------------


def _world(item) -> tuple[list[dict], list[dict]]:
    """Every choice on one world, one after another: they share the files of its runs."""
    plan, biomodel, design, out = item
    rows, by_group = [], []
    for name, changes in CHOICES.items():
        scratch = Path(tempfile.mkdtemp(dir=out, prefix="measure-"))
        try:
            row, groups = experiment.measure_world(
                changed(plan, changes), biomodel, design, into=scratch / "measure"
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        rows.append({"choice": name, **row})
        by_group += [{"choice": name, **line} for line in groups]
    return rows, by_group


def _community(item) -> tuple[list[dict], list[dict]]:
    plan, design, out = item
    rows, by_group = [], []
    for name, changes in CHOICES.items():
        scratch = Path(tempfile.mkdtemp(dir=out, prefix="measure-"))
        try:
            found, groups = community.read_matrix(changed(plan, changes), design, into=scratch)
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
        rows += [{"choice": name, **row} for row in found]
        by_group += [{"choice": name, **line} for line in groups]
    return rows, by_group


def _cell(value) -> str:
    """A cell as a table holds it: nothing for what is not there."""
    return "" if value in (None, "", "None") else str(value)


def _lost(text) -> set[str]:
    """The runs in which the receiver was lost, from ``u=8@10(function) ...``: when is another matter."""
    return {item.split("@")[0] for item in _cell(text).split()}


def _against(rows: list[dict], reference: dict, key) -> None:
    """Beside each row what the plan reads for the same world or pair: D over it, and whether the stages, the
    runs in which the receiver is lost and the runs that are admissible are the same."""
    for row in rows:
        base = reference.get(key(row))
        if base is None or row.get("status") != "ok" or base.get("status") != "ok":
            continue
        d, d0 = number(row.get("D")), number(base.get("D"))
        if d is not None and d0 is not None:
            row["D_of_the_plan"] = d0
            row["D_over_the_plan"] = d / d0 if d0 > 0.0 else (1.0 if d == 0.0 else float("inf"))
        row["same_stages"] = all(_cell(row.get(stage)) == _cell(base.get(stage)) for stage in STAGES)
        row["same_losses"] = _lost(row.get("losses")) == _lost(base.get("losses"))
        row["same_admissible"] = _cell(row.get("excluded_by_sender_function")) == _cell(
            base.get("excluded_by_sender_function")
        )


def choices(plan: dict, jobs: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    work = [(plan, biomodel, design, out) for biomodel in plan["receivers"] for design in plan["designs"]]
    pairs = [(plan, design, out) for design in plan["community"]["designs"]]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            found, together = list(pool.map(_world, work)), list(pool.map(_community, pairs))
    else:
        found, together = [_world(item) for item in work], [_community(item) for item in pairs]
    rows = [row for part, _ in found for row in part]
    reference = {(row["biomodel"], row["design"]): row for row in read_rows(experiment.OUT / "summary.csv")}
    _against(rows, reference, lambda row: (row["biomodel"], row["design"]))
    write_rows(out / "worlds.csv", rows)
    write_rows(out / "worlds_layers.csv", [row for _, part in found for row in part])
    rows = [row for part, _ in together for row in part]
    reference = {
        (row["design"], row["sender"], row["receiver"]): row
        for row in read_rows(community.OUT / "summary.csv")
    }
    _against(rows, reference, lambda row: (row["design"], row["sender"], row["receiver"]))
    write_rows(out / "community.csv", rows)
    write_rows(out / "community_layers.csv", [row for _, part in together for row in part])


# ---- the lemma on the runs -----------------------------------------------------------------------------------------


def lemma(plan: dict, out: Path) -> None:
    """For every function, world and run: where every quantity that the function reads is within its tolerance
    ``fn``, is the function kept; and is its change within the first-order bound."""
    settings, viable = plan["measure"], float(plan["viable"])
    rows = []
    for biomodel, receiver in plan["receivers"].items():
        words = {"name": "B", "uptake": f"B.uptake.{receiver['nutrient']}", "atp": receiver["atp"]}
        for design in plan["designs"]:
            directory = experiment.OUT / biomodel / design
            times, base = read_series(directory / BASELINE / "series.csv")
            runs = [
                name
                for name in (*experiment.run_names(plan), REMOVAL)
                if (directory / name / "series.csv").exists()
            ]
            for label, pattern in FUNCTIONS.items():
                function = Function(pattern.format(**words), settings["qmin"], "pointwise")
                names = function.names()
                q0 = function.values(base)
                q_min = function.q_min(q0[0])
                margin = function.margin(q0, times, q_min)
                protected = [parse_protected(f"{name}=fn") for name in names]
                deltas, record = tolerances(protected, times, base, function, q0, margin, "all")
                slopes = np.array([item["slope_peak"] for item in record])
                for run in runs:
                    _, columns = read_series(directory / run / "series.csv")
                    alive = columns["B.density"] >= viable
                    change = np.vstack([columns[name] - base[name] for name in names])
                    within = np.max(np.abs(change) / deltas[:, None], axis=0)
                    bound = slopes @ np.abs(change)
                    q = function.values(columns)
                    moved = np.abs(q - q0)
                    kept = q >= q_min
                    slack = bound * 1e-9 + 1e-12 * abs(float(q0[0]))
                    beyond = alive & (
                        moved > bound + slack
                    )  # the change is larger than the bound, either way
                    fell = alive & (q0 - q > bound + slack)  # the function fell by more than the bound
                    inside = alive & (within <= 1.0)
                    rows.append(
                        {
                            "function": label,
                            "reads": " ".join(names),
                            "biomodel": biomodel,
                            "design": design,
                            "run": run,
                            "times": int(np.sum(alive)),
                            "times_within_tolerance": int(np.sum(inside)),
                            "of_these_function_lost": int(np.sum(inside & ~kept)),
                            "least_margin_kept_within_tolerance": float(np.min((q[inside] - q_min) / margin))
                            if inside.any()
                            else None,
                            "largest_change_over_tolerance": float(np.max(within[alive])),
                            "times_beyond_first_order_bound": int(np.sum(beyond)),
                            "largest_excess_over_margin_within_tolerance": float(
                                np.max((moved - bound)[beyond & inside]) / margin
                            )
                            if (beyond & inside).any()
                            else 0.0,
                            "times_fell_beyond_bound": int(np.sum(fell)),
                            "largest_fall_beyond_bound_over_margin": float(
                                np.max((q0 - q - bound)[fell]) / margin
                            )
                            if fell.any()
                            else 0.0,
                            "largest_fall_beyond_bound_over_margin_within_tolerance": float(
                                np.max((q0 - q - bound)[fell & inside]) / margin
                            )
                            if (fell & inside).any()
                            else 0.0,
                            "function_lost_at_some_time": bool(np.any(alive & ~kept)),
                        }
                    )
    write_rows(out / "lemma.csv", rows)
    summary = []
    for label in FUNCTIONS:
        mine = [
            row for row in rows if row["function"] == label and row["design"] != next(iter(plan["designs"]))
        ]
        least = [
            row["least_margin_kept_within_tolerance"]
            for row in mine
            if row["least_margin_kept_within_tolerance"] is not None
        ]
        summary.append(
            {
                "function": label,
                "reads": mine[0]["reads"].replace("B.", "") if mine else "",
                "runs": len(mine),
                "times": sum(row["times"] for row in mine),
                "times_within_tolerance": sum(row["times_within_tolerance"] for row in mine),
                "of_these_function_lost": sum(row["of_these_function_lost"] for row in mine),
                "least_margin_kept_within_tolerance": min(least) if least else None,
                "runs_with_the_function_lost": sum(row["function_lost_at_some_time"] for row in mine),
                "times_beyond_first_order_bound": sum(row["times_beyond_first_order_bound"] for row in mine),
                "largest_excess_over_margin_within_tolerance": max(
                    (row["largest_excess_over_margin_within_tolerance"] for row in mine), default=0.0
                ),
                "times_fell_beyond_bound": sum(row["times_fell_beyond_bound"] for row in mine),
                "largest_fall_beyond_bound_over_margin": max(
                    (row["largest_fall_beyond_bound_over_margin"] for row in mine), default=0.0
                ),
                "largest_fall_beyond_bound_over_margin_within_tolerance": max(
                    (row["largest_fall_beyond_bound_over_margin_within_tolerance"] for row in mine),
                    default=0.0,
                ),
            }
        )
    write_rows(out / "lemma_summary.csv", summary)


# ---- which quantities are protected --------------------------------------------------------------------------------


def _layer_of(column: str, carriers: set[str], proteins: set[str]) -> str:
    """The layer at which a column of a run is read: carriers at particle, the other species at molecule, and
    rates and the states of proteins at complex."""
    if column.startswith("B.rates."):
        return "complex"
    name = column[len("B.species.") :]
    return "particle" if name in carriers else "complex" if name in proteins else "molecule"


def candidates(plan: dict, out: Path) -> None:
    """Every species and every reaction rate of a model read alone, in every world of the plan."""
    settings, tolerance = plan["measure"], str(plan["protect"]["tolerance"])
    rows, summary = [], []
    for biomodel, receiver in plan["receivers"].items():
        organism = layers.sbml_organism(plan, biomodel, "B")
        picked = {entry.column: entry for entry in layers.protected(plan, organism)}
        carriers, proteins = set(receiver["carriers"]), set(receiver["proteins"])
        for design in plan["designs"]:
            directory = experiment.OUT / biomodel / design
            with (directory / BASELINE / "series.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.reader(handle)
                header, first = next(reader), next(reader)
            start = dict(zip(header, map(float, first), strict=True))
            found, left_out = {}, []
            for family in ("B.species.", "B.rates."):
                mine = [column for column in header if column.startswith(family)]
                largest = max(abs(start[column]) for column in mine)
                for column in mine:
                    if abs(start[column]) <= 1e-9 * largest:
                        left_out.append(column)  # nothing at rest to take a share of
                    else:
                        found[column] = _layer_of(column, carriers, proteins)
            runs = {
                name: directory / name / "series.csv"
                for name in experiment.run_names(plan)
                if (directory / name / "series.csv").exists()
            }
            scratch = Path(tempfile.mkdtemp(dir=out, prefix="measure-"))
            try:
                measure(
                    directory / BASELINE / "series.csv",
                    runs,
                    [f"{column}={tolerance}" for column in found],
                    receiver_density="B.density",
                    viable=float(plan["viable"]),
                    function=settings["function"].replace("{name}", "B"),
                    qmin=settings["qmin"],
                    mode=settings["mode"],
                    allocation=settings["allocation"],
                    sender_function=settings["sender_function"].replace("{name}", "A"),
                    sender_qmin=settings["sender_qmin"],
                    sender_mode=settings["sender_mode"],
                    removal=directory / REMOVAL / "series.csv"
                    if (directory / REMOVAL / "series.csv").exists()
                    else None,
                    out=scratch,
                )
                alone = {row["column"]: row for row in read_rows(scratch / "deltas.csv")}
            except MeasureError as exc:
                summary.append({"biomodel": biomodel, "design": design, "status": str(exc).splitlines()[0]})
                continue
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
            whole = {row["column"]: row for row in read_rows(directory / "measure" / "deltas.csv")}
            for column, layer in found.items():
                rows.append(
                    {
                        "biomodel": biomodel,
                        "design": design,
                        "quantity": column,
                        "layer": layer,
                        "picked_as": picked[column].kind if column in picked else "",
                        "D": float(alone[column]["D"]),
                        "D_removal": number(alone[column].get("D_removal")),
                    }
                )
            for layer in ("particle", "molecule", "complex"):
                values = sorted(
                    row["D"]
                    for row in rows
                    if (row["biomodel"], row["design"], row["layer"]) == (biomodel, design, layer)
                )
                if not values:
                    continue
                line = {
                    "biomodel": biomodel,
                    "design": design,
                    "layer": layer,
                    "status": "ok",
                    "candidates": len(values),
                    "least": values[0],
                    "median": float(np.median(values)),
                    "largest": values[-1],
                    "share_beyond_tolerance": float(np.mean(np.array(values) > 1.0)),
                    "left_out_zero_at_rest": " ".join(left_out),
                }
                for kind in (
                    layers.KINDS
                ):  # what the plan picks at this layer, and where it stands among the candidates
                    mine = [
                        entry.column
                        for entry in picked.values()
                        if entry.layer == layer and entry.kind == kind
                    ]
                    if mine:
                        d = max(float(whole[column]["D"]) for column in mine)
                        line[f"picked_{kind}"] = " ".join(mine)
                        line[f"D_picked_{kind}"] = d
                        line[f"candidates_below_picked_{kind}"] = float(np.mean(np.array(values) < d))
                summary.append(line)
    write_rows(out / "candidates.csv", rows)
    write_rows(out / "candidates_summary.csv", summary)


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    del analyze_only  # this study only reads
    choices(plan, jobs, out)
    lemma(plan, out)
    candidates(plan, out)
    (out / "choices.json").write_text(
        json.dumps(
            {
                name: {key: (value if not callable(value) else "x100") for key, value in changes.items()}
                for name, changes in CHOICES.items()
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    names = (
        "worlds",
        "worlds_layers",
        "community",
        "community_layers",
        "lemma",
        "lemma_summary",
        "candidates",
        "candidates_summary",
    )
    return [out / f"{name}.csv" for name in names]


__all__ = ["CHOICES", "FUNCTIONS", "OUT", "candidates", "choices", "lemma", "run"]
