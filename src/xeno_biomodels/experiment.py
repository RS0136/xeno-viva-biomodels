"""The runs of the plan and their measure: every receiver with every design of the sender.

For a world (one receiver, one design) the runs are

    baseline     the world at rest, the activity of the sender at its baseline
    u=<value>    the same start and the same supply, the activity stepped to <value> at time zero
    removal      the same start and the same supply, the sender taken away at time zero
    no_receiver  the same start and the same supply, the SBML organism taken away at time zero

Each leaves ``runs/worlds/<receiver>/<design>/<run>/series.csv`` (time and every stored column), and the world
leaves ``world.json``: what the calibration found and the rest state. For the measure, ``protected.csv`` is
written beside each ``series.csv``: what the organisms protect (``layers``), derived quantities among it. The
measure of the world is written to ``.../measure/`` (``orthogonality.csv``, ``deltas.csv``, ``result.json``,
``groups.csv``), ``runs/worlds/summary.csv`` has one row per world, and ``runs/worlds/layers.csv`` one row per
world and group of protected quantities: every layer, each kind, each kind at each layer.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from xeno_biomodels import layers, paths
from xeno_biomodels.orthogonality import MeasureError, measure, read_series
from xeno_biomodels.world import BASELINE, NO_RECEIVER, REMOVAL, World, WorldError, load_plan, write_series

OUT = paths.RUNS / "worlds"


def run_names(plan: dict) -> dict[str, float]:
    """Run name -> activity of the sender, for the runs with a step."""
    return {f"u={value:g}": float(value) for value in plan["activities"]["steps"]}


def simulate_world(
    plan: dict, biomodel: str, design: str, out: Path = OUT, core=None, extras: bool = True
) -> dict:
    """Calibrate a world, bring it to rest, and follow its baseline, its activities and the removal of the sender.

    With ``extras`` the rest state is also held against the steady state that COPASI finds for the SBML file, and
    the run without the receiver is followed; the studies that vary a world many times leave both out.
    """
    started = time.monotonic()
    world = World(plan, biomodel, design, core=core)
    state, moved, spent = world.rest()
    directory = out / biomodel / design
    identical = float(
        max(
            abs(state["B"]["species"][name] - value) / (abs(value) + 1e-12)
            for name, value in world.sbml_species.items()
        )
    )
    against_copasi = still = None
    if extras:
        from xeno_biomodels.verify import copasi_steady_state

        other, still = copasi_steady_state(world.sbml)  # a second engine, on the untouched SBML file
        against_copasi = float(
            max(
                abs(state["B"]["species"][name] - other[name]) / (abs(other[name]) + 1e-12)
                for name in world.sbml_species
            )
        )
    sender, receiver = world._sender_config(), world._receiver_config()
    touched_by_sender = sorted({sender["substrate"], sender["product"], *sender["also_takes"]} - {""})
    info = {
        "biomodel": biomodel,
        "design": design,
        "leak": float(sender["also_takes"].get(world.nutrient, 0.0)),
        "design_as_written": plan["designs"][design],
        "sender": sender,
        "sbml_file": str(world.sbml.relative_to(paths.PROJECT)),
        "units_of_time_per_minute": world.minute,
        "dilution": world.dilution,
        "interval": world.interval,
        "medium_species_of_receiver": world.medium_ids,
        "nutrient": world.nutrient,
        "substrate": sender["substrate"],
        "product": sender["product"],
        "sbml_value_of_medium": world.sbml_medium,
        "sbml_steady_state_after_min": world.sbml_relaxed_after,
        "feed": world.feed,
        "uptake_at_rest": world.uptake_at_rest,
        "biovolume_alone": world.biovolume,
        "medium_species_touched_by_sender": touched_by_sender,
        "medium_species_shared": sorted(set(touched_by_sender) & set(receiver["medium_species"])),
        "boundary_species_held_at_sbml_value": world.held,
        "rest_moved": moved,
        "rest_after_min": spent,
        "rest_receiver_vs_sbml_steady_state": identical,
        "rest_receiver_vs_copasi_steady_state": against_copasi,
        "copasi_still_moving": still,
        "rest": state,
    }
    runs = {BASELINE: world.document(state)}
    for name, value in run_names(plan).items():
        runs[name] = world.document(state, activity=value)
    runs[REMOVAL] = world.document(state, present=False)
    if extras:
        runs[NO_RECEIVER] = world.document(state, receiver_present=False)
    else:
        for stale in ("series.csv", "protected.csv", "partial.csv", "status.txt"):
            (directory / NO_RECEIVER / stale).unlink(missing_ok=True)
    info["not_followed"] = {}
    for name, document in runs.items():
        for stale in ("series.csv", "partial.csv", "status.txt"):
            (directory / name / stale).unlink(missing_ok=True)
        times, columns, problem = world.follow(document)
        if problem is None:
            write_series(directory / name / "series.csv", times, columns)
        else:  # what was stored until the run stopped is kept apart, and the run is left out of the measure
            write_series(directory / name / "partial.csv", times, columns)
            (directory / name / "status.txt").write_text(problem + "\n", encoding="utf-8")
            info["not_followed"][name] = {
                "problem": problem,
                "last_stored_time": float(times[-1]),
                "receiver_growth_over_dilution_then": float(columns["B.growth_rate"][-1] / world.dilution),
                "nutrient_over_rest_then": float(
                    columns[f"medium.{world.nutrient}"][-1] / state["medium"][world.nutrient]
                ),
            }
    info["seconds"] = round(time.monotonic() - started, 1)
    (directory / "world.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def measure_world(
    plan: dict, biomodel: str, design: str, out: Path = OUT, into: Path | None = None
) -> tuple[dict, list[dict]]:
    """Orthogonality of the sender to the receiver in one world, from the runs that are there.

    Returns the row of the world and its rows by group of protected quantities. The files of the measure go to
    ``into`` (``measure/`` beside the runs unless given).
    """
    directory = out / biomodel / design
    into = directory / "measure" if into is None else Path(into)
    info = json.loads((directory / "world.json").read_text(encoding="utf-8"))
    settings = plan["measure"]
    row = {"biomodel": biomodel, "design": design, "leak": info["leak"]}
    lost = info["not_followed"]
    if BASELINE in lost:
        return {**row, "status": f"the baseline could not be followed: {lost[BASELINE]['problem']}"}, []
    receiver = layers.sbml_organism(plan, biomodel, "B")
    sender = layers.xeno_organism("A", info["substrate"], info["product"])
    of_receiver, of_sender = layers.protected(plan, receiver), layers.protected(plan, sender)

    def of(name: str, key: str) -> str:
        return settings[key].replace("{name}", name)

    more = [
        of("B", "function"),
        of("B", "density"),
        of("A", "function"),
        of("A", "density"),
        of("A", "sender_function"),
    ]
    followed = [
        name
        for name in (BASELINE, *run_names(plan), REMOVAL, NO_RECEIVER)
        if name not in lost and (directory / name / "series.csv").exists()
    ]
    for name in followed:  # what the measure reads of each run
        layers.write_protected(
            directory / name / "series.csv",
            directory / name / "protected.csv",
            plan,
            [receiver, sender],
            more,
        )
    # A step in which the SBML model could not be followed to the horizon is not left out: where a model stops
    # having a solution the organism that it describes is gone, and leaving the run out would take away the
    # strongest steps. The run counts as a loss of the receiver at the first stored time after it stopped.
    collapsed = {}
    for name in run_names(plan):
        if name in lost and (directory / name / "partial.csv").exists() and BASELINE in followed:
            layers.write_collapsed(
                directory / name / "partial.csv",
                directory / BASELINE / "protected.csv",
                directory / name / "protected.csv",
                plan,
                [receiver, sender],
                more,
                gone=of("B", "density"),
            )
            followed.append(name)
            collapsed[name] = lost[name]["last_stored_time"]
    back_run = directory / NO_RECEIVER / "protected.csv"

    def viable(column: str) -> float:
        """The density below which an organism does not count as established: a number, or ``rel:x`` for a
        share of the density that it has at the start of the baseline (for worlds whose densities are in units
        of their own)."""
        written = str(plan["viable"])
        if not written.startswith("rel:"):
            return float(written)
        _, base = read_series(directory / BASELINE / "protected.csv")
        return float(written.partition(":")[2]) * float(base[column][0])

    try:
        result, _ = measure(
            directory / BASELINE / "protected.csv",
            {name: directory / name / "protected.csv" for name in run_names(plan) if name in followed},
            [entry.spec for entry in of_receiver],
            receiver_density=of("B", "density"),
            viable=viable(of("B", "density")),
            function=of("B", "function"),
            qmin=settings["qmin"],
            mode=settings["mode"],
            allocation=settings["allocation"],
            sender_function=of("A", "sender_function"),
            sender_qmin=settings["sender_qmin"],
            sender_mode=settings["sender_mode"],
            removal=directory / REMOVAL / "protected.csv" if REMOVAL in followed else None,
            groups=layers.groups(of_receiver),
            sigma=float(settings["sigma"]),
            stage_tol=float(settings["stage_tol"]),
            bits_tol=float(settings["bits_tol"]),
            stride=int(settings.get("stride", 1)),
            out=into,
        )
        # The other way round: what the sender protects, with the SBML organism taken away.
        back = {}
        if NO_RECEIVER in followed:
            back, _ = measure(
                directory / BASELINE / "protected.csv",
                [],
                [entry.spec for entry in of_sender],
                receiver_density=of("A", "density"),
                viable=viable(of("A", "density")),
                function=of("A", "function"),
                qmin=settings["qmin"],
                mode=settings["mode"],
                allocation=settings["allocation"],
                removal=back_run,
                groups=layers.groups(of_sender),
                stride=int(settings.get("stride", 1)),
                out=into.with_name(into.name + "_receiver_removed"),
            )
    except MeasureError as exc:
        return {**row, "status": str(exc).splitlines()[0]}, []
    row.update(
        status=result["status"],
        shared_species=" ".join(info["medium_species_shared"]),
        D=result["D_peak"],
        t_peak=result["t_peak"],
        D_activity=result["D_activity"],
        D_quantity=result["D_quantity"],
        D_end=result["D_end"],
        O=result["O"],
        stage1_trajectory=result["1_trajectory"],
        stage3_tolerable=result["3_tolerable"],
        stage4_distribution=result["4_distribution"],
        bits_lower=result["bits_lower"],
        bits_upper=result["bits_upper"],
        D_removal=result.get("D_removal"),
        removal_lost_at=result.get("removal_lost_at"),
        D_receiver_removed_on_sender=back.get("D_removal"),
        not_followed=" ".join(f"{name}@{item['last_stored_time']:g}" for name, item in lost.items()),
        collapsed=" ".join(f"{name}@{at:g}" for name, at in collapsed.items()),
        activities=result["activities"],
        excluded_by_sender_function=" ".join(
            item["activity"] for item in result["excluded_by_sender_function"]
        ),
        losses=" ".join(f"{item['activity']}@{item['t']:g}({item['cause']})" for item in result["losses"]),
        protected=result["protected"],
        stored_times=result["stored_times"],
        margin=result["margin"],
        bound_over_margin=result["bound_over_margin"],
        receiver_margin_least=result["receiver_margin_least"],
        rest_moved=info["rest_moved"],
        rest_receiver_vs_sbml=info["rest_receiver_vs_sbml_steady_state"],
        rest_receiver_vs_copasi=info.get("rest_receiver_vs_copasi_steady_state"),
    )
    tag = {"biomodel": biomodel, "design": design, "leak": info["leak"], "losses": row["losses"]}
    return row, [{**tag, **found} for found in layers.group_rows(result)]


def identical_columns(directory: Path, names: list[str], prefix: tuple[str, ...]) -> dict[str, bool]:
    """For each run: are its columns that begin with ``prefix`` the same numbers as the baseline's, bit for bit?"""
    _, base = read_series(directory / BASELINE / "series.csv")
    out = {}
    for name in names:
        _, columns = read_series(directory / name / "series.csv")
        out[name] = all(np.array_equal(base[key], columns[key]) for key in base if key.startswith(prefix))
    return out


def _work(item) -> tuple[dict, list[dict]]:
    plan, biomodel, design, out, analyze_only = item
    if not analyze_only:
        info = simulate_world(plan, biomodel, design, out)
        print(
            f"{biomodel}/{design}: followed in {info['seconds']} s (rest moved {info['rest_moved']:.1e})",
            flush=True,
        )
    return measure_world(plan, biomodel, design, out)


def _variant(item) -> tuple[dict, list[dict]]:
    """One world of a study: followed without the extras and measured. A world that cannot be built is a row
    that says why. The series of the world are let go once it is measured unless the study keeps them."""
    tag, plan, biomodel, design, out, analyze_only, keep = item
    row = {**tag, "biomodel": biomodel, "design": design}
    directory = out / biomodel / design
    try:
        if not analyze_only:
            simulate_world(plan, biomodel, design, out, extras=False)
        found, by_group = measure_world(plan, biomodel, design, out)
    except (WorldError, FileNotFoundError, ValueError, RuntimeError) as exc:
        return {**row, "status": f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"}, []
    row = {**row, **found}
    names = [name for name in (*run_names(plan), REMOVAL) if (directory / name / "series.csv").exists()]
    if row.get("status") == "ok" and names:
        # the columns of the receiver against the baseline, bit for bit, before the series are let go
        own = tuple(f"medium.{name}" for name in plan["receivers"][biomodel]["medium"])
        same = identical_columns(directory, names, ("B.", *own))
        row["receiver_bit_identical"] = all(same.values())
    if keep is not True and not analyze_only:
        # a study keeps the measure of its worlds, not their series; ``keep="protected"`` keeps what was read
        gone = (
            ("series.csv", "partial.csv")
            if keep == "protected"
            else ("series.csv", "protected.csv", "partial.csv")
        )
        for name in gone:
            for series in directory.glob(f"*/{name}"):
                series.unlink()
        prune(directory)
    return row, [{**tag, **found} for found in by_group]


def prune(directory: Path) -> None:
    """Take away the directories below ``directory`` that hold nothing any more."""
    for folder in sorted((path for path in directory.rglob("*") if path.is_dir()), reverse=True):
        if not any(folder.iterdir()):
            folder.rmdir()


def run_variants(work: list[tuple], jobs: int = 1) -> tuple[list[dict], list[dict]]:
    """Many worlds under many plans, for the studies. ``work`` holds (tag, plan, biomodel, design, out,
    analyze_only, keep); the tag is a mapping that is put in front of the rows of the world, and ``keep`` is
    True (every series stays), "protected" (what the measure read stays) or False. Returns the rows of the
    worlds and their rows by group of protected quantities."""
    if jobs > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            found = list(pool.map(_variant, work))
    else:
        found = [_variant(item) for item in work]
    return [row for row, _ in found], [row for _, by_group in found for row in by_group]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    plan: dict | None = None,
    out: Path = OUT,
    jobs: int = 1,
    analyze_only: bool = False,
    only: list[str] | None = None,
) -> list[dict]:
    """Every world of the plan (or those of the receivers in ``only``). Returns the rows of ``summary.csv``."""
    plan = plan or load_plan()
    receivers = [name for name in plan["receivers"] if not only or name in only]
    work = [
        (plan, biomodel, design, out, analyze_only) for biomodel in receivers for design in plan["designs"]
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            found = list(pool.map(_work, work))
    else:
        found = [_work(item) for item in work]
    rows = [row for row, _ in found]
    for row in rows:
        if row.get("status") == "ok":
            directory, sender = out / row["biomodel"] / row["design"], plan["sender"]
            names = [
                name for name in (*run_names(plan), REMOVAL) if (directory / name / "series.csv").exists()
            ]
            same = identical_columns(
                directory, names, ("B.", f"medium.{plan['receivers'][row['biomodel']]['nutrient']}")
            )
            row["receiver_bit_identical_in"] = " ".join(name for name, yes in same.items() if yes)
            if (directory / NO_RECEIVER / "series.csv").exists():
                back = identical_columns(
                    directory,
                    [NO_RECEIVER],
                    ("A.", f"medium.{sender['substrate']}", f"medium.{sender['product']}"),
                )
                row["sender_bit_identical_without_receiver"] = back[NO_RECEIVER]
    _write(out / "summary.csv", rows)
    _write(out / "layers.csv", [row for _, by_group in found for row in by_group])
    return rows


__all__ = [
    "OUT",
    "identical_columns",
    "measure_world",
    "prune",
    "run",
    "run_names",
    "run_variants",
    "simulate_world",
]
