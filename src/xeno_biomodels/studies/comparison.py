"""D beside what is used elsewhere to say how much two organisms have to do with each other.

    niche overlap    Pianka's index of the two organisms on the species of the medium: the cosine between what
                     each takes of every species, as shares of all that it takes, at the state from which the
                     runs start. It is symmetric, has no direction and no layers, and is zero exactly when the
                     two take no species in common.
    linear response  of the steady state: how far the rest state of every protected quantity moves per unit of
                     the activity (what C F^-1 G gives for the linearised world), times the size of the step.
                     Obtained here by bringing the world to rest at activities 1 % above and below the baseline.
    steady states    the rest state at the activity of every step, not linearised: what a comparison of steady
                     states, the common way to read an interaction from a model, would report.

Both of the last two are read with the tolerances of the measure, so they are numbers of the same kind as D,
whole and layer by layer. What they cannot hold is in the difference: what happens on the way (a peak that is
not there at rest, a function lost for a while), and, for the linear response, what a large step does.

For the community the niche overlap of every two organisms stands beside D and D_removal of the ordered pair,
and for the receivers taken by rule (study ``receivers``) D stands beside what the screen measured of the model
at its steady state.

Written to ``runs/studies/comparison/``: ``worlds.csv``, ``layers.csv``, ``steps.csv``, ``community.csv``,
``receivers.csv`` and ``summary.json``.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from xeno_biomodels import community, experiment, layers
from xeno_biomodels.community_report import chain
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies import receivers
from xeno_biomodels.studies.common import LINES, STUDIES, number, read_rows, write_rows
from xeno_biomodels.world import World, WorldError

OUT = STUDIES / "comparison"
NEAR = 0.01  # the activities at which the linear response is read: this share above and below the baseline


def pianka(first: dict[str, float], second: dict[str, float]) -> float:
    """Pianka's overlap of two organisms from what each takes of every species (any unit, per organism)."""
    names = sorted({*first, *second})
    a = np.array([max(first.get(name, 0.0), 0.0) for name in names])
    b = np.array([max(second.get(name, 0.0), 0.0) for name in names])
    if not (a.sum() > 0.0 and b.sum() > 0.0):
        return 0.0
    a, b = a / a.sum(), b / b.sum()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def taken_by_law(config: dict, medium: dict, activity: float) -> dict[str, float]:
    """What an organism of laws takes of every species of the medium, per cell volume."""

    def rate(name: str, share: float, half: float) -> float:
        level = max(float(medium[name]), 0.0)
        return share * activity * float(config["vmax"]) * level / (half + level)

    out = {config["substrate"]: rate(config["substrate"], 1.0, float(config["km"]))}
    for name, share in (config.get("also_takes") or {}).items():
        out[name] = rate(name, float(share), float((config.get("also_km") or {}).get(name, config["km"])))
    return out


def rank_correlation(first: list[float], second: list[float]) -> float | None:
    """Spearman's coefficient, with the mean rank for ties; None when one of the two does not vary."""

    def ranks(values: list[float]) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        order = np.argsort(values, kind="stable")
        out = np.empty(values.size)
        n = 0
        while n < values.size:
            m = n
            while m + 1 < values.size and values[order[m + 1]] == values[order[n]]:
                m += 1
            out[order[n : m + 1]] = 0.5 * (n + m)
            n = m + 1
        return out

    if len(first) < 3:
        return None
    a, b = ranks(first), ranks(second)
    if np.ptp(a) == 0.0 or np.ptp(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


# ---- the worlds of the plan ----------------------------------------------------------------------------------------


def _quantities(plan: dict, organism: layers.Organism, state: dict) -> dict[str, float]:
    """Every column of a run, and the derived ones, at one state of a world."""
    columns = {f"medium.{name}": np.array([value]) for name, value in state["medium"].items()}
    for name in ("B", "A"):
        for key, value in state[name].items():
            if isinstance(value, dict):
                columns.update(
                    {f"{name}.{key}.{inner}": np.array([amount]) for inner, amount in value.items()}
                )
            else:
                columns[f"{name}.{key}"] = np.array([value])
    columns.update(layers.derive(plan, organism, columns))
    return {name: float(values[0]) for name, values in columns.items()}


def _world(item) -> tuple[dict, list[dict], list[dict]]:
    """One world: its niche overlap, its linear response, and the steady state of every step."""
    plan, biomodel, design = item
    directory = experiment.OUT / biomodel / design
    head = {"biomodel": biomodel, "design": design}
    info = json.loads((directory / "world.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
    organism = layers.sbml_organism(plan, biomodel, "B")
    entries = layers.protected(plan, organism)
    delta = {row["column"]: float(row["delta"]) for row in read_rows(directory / "measure" / "deltas.csv")}
    lines = [line for line in LINES if any((entry.layer, entry.kind) == line for entry in entries)]
    tolerance = float(plan["time"]["rest_tolerance"])
    try:
        world = World(plan, biomodel, design)
    except WorldError as exc:
        return {**head, "status": f"WorldError: {str(exc)[:200]}"}, [], []
    start, u0 = info["rest"], world.baseline
    at_rest = _quantities(plan, organism, start)

    def over(change: dict[str, float], members: list[layers.Entry]) -> float:
        return max(abs(change[entry.column]) / delta[entry.column] for entry in members)

    # ---- the niche overlap, at the state from which the runs start
    of_receiver = {name: start["B"]["uptake"][name] for name in info["medium_species_of_receiver"]}
    of_sender = taken_by_law(info["sender"], start["medium"], u0)
    row = {**head, "status": "ok", "leak": info["leak"], "niche_overlap": pianka(of_sender, of_receiver)}

    def settle(activity: float) -> tuple[dict | None, float | None]:
        """The rest state at an activity, or None where the SBML model could not be followed to it."""
        try:
            state, moved, _ = world.rest(activity=activity, start=start)
        except RuntimeError:  # the integrator of the SBML model gave up on the way
            return None, None
        return state, moved

    # ---- the linear response of the steady state
    near, unsettled = {}, []
    for sign in (1.0, -1.0):
        state, moved = settle(u0 * (1.0 + sign * NEAR))
        if state is None:
            return {**row, "status": "the rest state beside the baseline could not be reached"}, [], []
        near[sign] = _quantities(plan, organism, state)
        if moved > 1e3 * tolerance:
            unsettled.append(f"{u0 * (1.0 + sign * NEAR):g}")
    slope = {
        entry.column: (near[1.0][entry.column] - near[-1.0][entry.column]) / (2.0 * NEAR * u0)
        for entry in entries
    }
    # the response coefficients: the relative change of each quantity over the relative change of the activity
    for entry in entries:
        if at_rest[entry.column] != 0.0:
            row[f"response_coefficient.{entry.column}"] = slope[entry.column] * u0 / at_rest[entry.column]

    # ---- the steady state of every step
    left_out = {found["activity"] for found in result["excluded_by_sender_function"]}
    _, courses = read_series(directory / "measure" / "orthogonality.csv")
    steps, linear, steady, lost_at_rest, no_rest = [], {}, {}, [], []
    for name, activity in experiment.run_names(plan).items():
        line = {**head, "step": name, "activity": activity, "admissible": name not in left_out}
        ahead = {entry.column: slope[entry.column] * (activity - u0) for entry in entries}
        line["D_linear"] = over(ahead, entries)
        if f"D[{name}]" in courses:
            course = courses[f"D[{name}]"]
            line["D_run"] = float(np.nanmax(course))
            line["D_run_end"] = float(course[-1]) if np.isfinite(course[-1]) else None
        state, moved = settle(activity)
        if state is None:  # no rest state to compare: the step is left out of the reading on steady states
            line["status"] = "the SBML model could not be followed to rest"
            steps.append(line)
            no_rest.append(name)
            if name not in left_out:
                for key in (None, *lines):
                    members = (
                        entries
                        if key is None
                        else [entry for entry in entries if (entry.layer, entry.kind) == key]
                    )
                    linear[key] = max(linear.get(key, 0.0), over(ahead, members))
            continue
        there = _quantities(plan, organism, state)
        change = {entry.column: there[entry.column] - at_rest[entry.column] for entry in entries}
        line["D_steady"] = over(change, entries)
        line["receiver_there_at_rest"] = bool(state["B"]["density"] >= float(plan["viable"]))
        line["sender_there_at_rest"] = bool(state["A"]["density"] >= float(plan["viable"]))
        line["rest_moved"] = moved
        line["status"] = "ok"
        steps.append(line)
        if moved > 1e3 * tolerance:
            unsettled.append(name)
        if name in left_out:
            continue
        if not line["receiver_there_at_rest"]:
            lost_at_rest.append(name)
        for key in (None, *lines):
            members = (
                entries if key is None else [entry for entry in entries if (entry.layer, entry.kind) == key]
            )
            linear[key] = max(linear.get(key, 0.0), over(ahead, members))
            steady[key] = max(steady.get(key, 0.0), over(change, members))
    row.update(
        D=result["D_peak"],
        D_end=result["D_end"],
        D_linear=linear.get(None, 0.0),
        D_steady=steady.get(None, 0.0),
        losses=" ".join(
            f"{found['activity']}@{found['t']:g}({found['cause']})" for found in result["losses"]
        ),
        receiver_gone_at_rest=" ".join(lost_at_rest),
        not_settled=" ".join(unsettled),
        no_rest_state=" ".join(no_rest),
    )
    measured = {
        (found["layer"], found["kind"]): found
        for found in read_rows(experiment.OUT / "layers.csv")
        if (found["biomodel"], found["design"]) == (biomodel, design)
    }
    by_line = []
    for key in lines:
        found = measured.get(key)
        by_line.append(
            {
                **head,
                "layer": key[0],
                "kind": key[1],
                "D": number(found["D"]) if found else None,
                "D_end": number(found["D_end"]) if found else None,
                "D_linear": linear.get(key, 0.0),
                "D_steady": steady.get(key, 0.0),
            }
        )
    return row, by_line, steps


# ---- the community, and the receivers taken by rule ----------------------------------------------------------------


def community_rows(plan: dict) -> list[dict]:
    """Every ordered pair of every community design: the niche overlap of the two beside D and D_removal."""
    if not (community.OUT / "summary.csv").exists():
        return []
    sender, rows = plan["sender"], []
    measured = read_rows(community.OUT / "summary.csv")
    for design in plan["community"]["designs"]:
        info = json.loads((community.OUT / design / "community.json").read_text(encoding="utf-8"))
        members, medium = info["members"], info["assembled"]["medium"]
        use = {}
        for name, body in members.items():
            if body["kind"] == "sbml":
                use[name] = dict.fromkeys(body["reads"], 1.0)  # its one nutrient
            else:
                config = {
                    "substrate": body["substrate"],
                    "vmax": sender["vmax"],
                    "km": sender["km"],
                    "also_takes": body["also_takes"],
                }
                use[name] = taken_by_law(config, medium, float(plan["activities"]["baseline"]))
        for row in measured:
            if row["design"] != design or row.get("status") != "ok":
                continue
            way = chain(members, row["sender"], row["receiver"])
            rows.append(
                {
                    "design": design,
                    "sender": row["sender"],
                    "receiver": row["receiver"],
                    "sender_kind": row["sender_kind"],
                    "receiver_kind": row["receiver_kind"],
                    "same_pair": row["same_pair"],
                    "niche_overlap": pianka(use[row["sender"]], use[row["receiver"]]),
                    "D": number(row["D"]),
                    "D_removal": number(row["D_removal"]),
                    "organisms_on_the_way": None if way is None else (len(way) - 3) // 2,
                    "way": "" if way is None else " > ".join(way),
                }
            )
    return rows


def receiver_rows() -> list[dict]:
    """The receivers taken by rule: D of each control beside what the screen measured at the steady state."""
    if not (receivers.OUT / "worlds.csv").exists():
        return []
    by_layer = {}
    for row in read_rows(receivers.OUT / "layers.csv"):
        if row["kind"] == "common" and row["layer"] != "all":
            by_layer[(row["biomodel"], row["design"], row["layer"])] = number(row["D"])
    rows = []
    for row in read_rows(receivers.OUT / "worlds.csv"):
        if row.get("status") != "ok" or float(row["share"]) == 0.0:
            continue
        line = {key: row[key] for key in ("biomodel", "biomodel_id", "name", "species", "design", "share")}
        line.update(
            D=float(row["D"]),
            D_quantity=row["D_quantity"],
            losses=row["losses"],
            not_followed=row["not_followed"],
            uptake_rises_by=float(row["uptake_rises_by"]),
            state_moves_by=float(row["state_moves_by"]),
        )
        for layer in layers.LAYERS:
            line[f"D_{layer}"] = by_layer.get((row["biomodel"], row["design"], layer))
        rows.append(line)
    return rows


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    del analyze_only  # the rest states are found again each time; nothing is kept but the tables
    work = [(plan, biomodel, design) for biomodel in plan["receivers"] for design in plan["designs"]]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(work)), mp_context=get_context("spawn")) as pool:
            found = list(pool.map(_world, work))
    else:
        found = [_world(item) for item in work]
    rows = [row for row, _, _ in found]
    write_rows(out / "worlds.csv", rows)
    write_rows(out / "layers.csv", [line for _, by_line, _ in found for line in by_line])
    write_rows(out / "steps.csv", [line for _, _, steps in found for line in steps])
    together = community_rows(plan)
    write_rows(out / "community.csv", together)
    ruled = receiver_rows()
    write_rows(out / "receivers.csv", ruled)

    summary: dict = {"community": {}, "receivers": {}}
    for design in plan["community"]["designs"]:
        mine = [row for row in together if row["design"] == design and row["D_removal"] is not None]
        if not mine:
            continue
        stepped = [row for row in mine if row["D"] is not None]
        summary["community"][design] = {
            "ordered_pairs": len(mine),
            "overlap_zero_and_moved_by_removal": sum(
                row["niche_overlap"] == 0.0 and row["D_removal"] > 0.0 for row in mine
            ),
            "overlap_zero_and_moved_by_a_step": sum(
                row["niche_overlap"] == 0.0 and row["D"] > 0.0 for row in stepped
            ),
            "overlap_above_zero_and_not_moved": sum(
                row["niche_overlap"] > 0.0 and row["D_removal"] == 0.0 for row in mine
            ),
            "rank_correlation_overlap_D_removal": rank_correlation(
                [row["niche_overlap"] for row in mine], [row["D_removal"] for row in mine]
            ),
            "rank_correlation_overlap_D": rank_correlation(
                [row["niche_overlap"] for row in stepped], [row["D"] for row in stepped]
            ),
        }
    for design in sorted({row["design"] for row in ruled}):
        mine = [row for row in ruled if row["design"] == design]
        summary["receivers"][design] = {
            "worlds": len(mine),
            "rank_correlation_D_state_moves_by": rank_correlation(
                [row["D"] for row in mine], [row["state_moves_by"] for row in mine]
            ),
            "rank_correlation_D_uptake_rises_by": rank_correlation(
                [row["D"] for row in mine], [row["uptake_rises_by"] for row in mine]
            ),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return [out / f"{name}.csv" for name in ("worlds", "layers", "steps", "community", "receivers")] + [
        out / "summary.json"
    ]
