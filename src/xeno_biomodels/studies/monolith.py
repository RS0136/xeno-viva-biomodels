"""Every control of the plan as one system of equations.

A world is a composite: three processes that each advance over a coupling interval from the same state of the
medium, and whose changes are added. Here every world of the plan with a control is written instead as a single
system of differential equations (the SBML model with the equations of the vessel, of the sender and of the
growth of the receiver added to it as rate rules), and integrated in one piece, first by CVODE of libRoadRunner
and then by LSODA of COPASI. Nothing of the composite is used but its rest state, from which the runs start. If
the composite, the exchange with the medium and the measure do what the text says, D of the single system is D
of the composite up to what the coupling interval leaves in it. A run in which the composite loses the receiver
is compared up to the loss.

Written to ``runs/studies/monolith/``: ``monolith.csv`` (one row per world, engine and step), and for one world
``courses.csv`` (D(t) of every step from the composite and from the two integrations) and ``model.xml`` (the
single system, as SBML).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xeno_biomodels import experiment, layers
from xeno_biomodels.organisms import exchange_matrix
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies import copasi_course
from xeno_biomodels.studies.common import STUDIES, read_rows, write_rows
from xeno_biomodels.world import World

OUT = STUDIES / "monolith"
WORLD = ("BIOMD0000000172", "leak-1e-2")  # the world that is written as one system


def antimony(world: World, state: dict) -> tuple[str, str]:
    """The world as one model in Antimony, starting at ``state``, and the expression of the uptake in it."""
    import tellurium as te

    text = te.sbmlToAntimony(str(world.sbml))
    body = text[: text.rindex("end")]
    sender, nutrient = world._sender_config(), world.nutrient
    model = __import__("libsbml").readSBMLFromFile(str(world.sbml)).getModel()
    reactions = [reaction.getId() for reaction in model.getListOfReactions()]
    row = exchange_matrix(str(world.sbml), [nutrient], reactions)[0]
    uptake = " + ".join(
        f"({float(-coefficient)!r})*{name}"
        for name, coefficient in zip(reactions, row, strict=True)
        if coefficient
    )
    leak = sender["also_takes"].get(nutrient, 0.0)
    numbers = {
        "xvb_k": world.dilution,
        "xvb_Sin": world.feed[nutrient],
        "xvb_Xin": world.feed[sender["substrate"]],
        "xvb_V": sender["vmax"],
        "xvb_K": sender["km"],
        "xvb_Y": sender["growth_yield"],
        "xvb_m": sender["maintenance"],
        "xvb_yW": sender["product_yield"],
        "xvb_eps": leak,
        "xvb_u": world.baseline,
        "xvb_Jstar": world.uptake_at_rest,
        "xvb_X": state["medium"][sender["substrate"]],
        "xvb_W": state["medium"][sender["product"]],
        "xvb_nA": state["A"]["density"],
        "xvb_nB": state["B"]["density"],
    }
    lines = [
        "",
        "  // the vessel, the sender and the growth of the receiver (xeno_biomodels.studies.monolith)",
        f"  xvb_J := {uptake};",
        "  xvb_qX := xvb_u*xvb_V*xvb_X/(xvb_K + xvb_X);",
        f"  xvb_qS := xvb_eps*xvb_u*xvb_V*{nutrient}/(xvb_K + {nutrient});",
        f"  {nutrient}' = xvb_k*(xvb_Sin - {nutrient}) - xvb_nB*xvb_J - xvb_nA*xvb_qS;",
        "  xvb_X' = xvb_k*(xvb_Xin - xvb_X) - xvb_nA*xvb_qX;",
        "  xvb_W' = xvb_nA*xvb_yW*xvb_qX - xvb_k*xvb_W;",
        "  xvb_nA' = (xvb_Y*(xvb_qX + xvb_qS) - xvb_m - xvb_k)*xvb_nA;",
        "  xvb_nB' = (xvb_k*xvb_J/xvb_Jstar - xvb_k)*xvb_nB;",
    ]
    lines += [f"  {name} = {float(value)!r};" for name, value in numbers.items()]
    lines += [f"  {nutrient} = {float(state['medium'][nutrient])!r};"]
    lines += [f"  {name} = {float(value)!r};" for name, value in state["B"]["species"].items()]
    return body + "\n".join(lines) + "\nend\n", "xvb_J"


def columns_of(world: World, plan: dict, species: dict, rates: dict, others: dict) -> dict[str, np.ndarray]:
    """The columns that a run of the composite stores of the receiver, from one integration of the single system."""
    nutrient = world.nutrient
    columns = {f"B.species.{name}": values for name, values in species.items()}
    columns.update({f"B.rates.{name}": values for name, values in rates.items()})
    columns[f"medium.{nutrient}"] = others[nutrient]
    columns[f"B.uptake.{nutrient}"] = others["xvb_J"]
    columns["B.growth_rate"] = world.dilution * others["xvb_J"] / world.uptake_at_rest
    columns["B.density"] = others["xvb_nB"]
    columns.update(layers.derive(plan, layers.sbml_organism(plan, world.biomodel, "B"), columns))
    return columns


def with_roadrunner(
    world: World, plan: dict, text: str, activity: float, times: np.ndarray
) -> dict[str, np.ndarray]:
    import tellurium as te

    model = te.loada(text)
    model.integrator.absolute_tolerance = float(plan["time"].get("absolute_tolerance", 1e-12))
    model.integrator.relative_tolerance = float(plan["time"].get("relative_tolerance", 1e-10))
    model.integrator.maximum_num_steps = 500000
    model["xvb_u"] = float(activity)
    species, reactions = list(model.getFloatingSpeciesIds()), list(model.getReactionIds())
    model.selections = [
        "time",
        *[f"[{name}]" for name in species],
        *reactions,
        f"[{world.nutrient}]",
        "xvb_J",
        "xvb_nB",
    ]
    found = np.asarray(model.simulate(float(times[0]), float(times[-1]), int(times.size)))
    n, r = len(species), len(reactions)
    return columns_of(
        world,
        plan,
        {name: found[:, 1 + i] for i, name in enumerate(species)},
        {name: found[:, 1 + n + i] for i, name in enumerate(reactions)},
        {world.nutrient: found[:, 1 + n + r], "xvb_J": found[:, 2 + n + r], "xvb_nB": found[:, 3 + n + r]},
    )


def with_copasi(
    world: World, plan: dict, sbml: str, activity: float, times: np.ndarray
) -> dict[str, np.ndarray]:
    """The same system under LSODA of COPASI, asked in a process of its own (see ``copasi_course``)."""
    nutrient = world.nutrient
    found = copasi_course.apart(
        sbml,
        {"xvb_u": float(activity)},
        [*world.sbml_species, nutrient],
        ["xvb_J", "xvb_nB"],
        float(times[0]),
        float(times[-1]),
        int(times.size),
        float(plan["time"].get("relative_tolerance", 1e-10)),
        float(plan["time"].get("absolute_tolerance", 1e-12)),
    )
    species = {name: found["species"][name] for name in world.sbml_species}
    others = {nutrient: found["species"][nutrient], **found["values"]}
    return columns_of(world, plan, species, found["reactions"], others)


def one_world(plan: dict, biomodel: str, design: str, out: Path | None = None) -> tuple[list[dict], dict]:
    """One world as one system, under both integrators: a row per engine and admissible step, and D(t) of every
    step (which is written, with the system itself, where ``out`` is given)."""
    import tellurium as te

    directory = experiment.OUT / biomodel / design
    info = json.loads((directory / "world.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
    delta = {row["column"]: float(row["delta"]) for row in read_rows(directory / "measure" / "deltas.csv")}
    left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
    lost = {item["activity"]: float(item["t"]) for item in result["losses"]}
    world = World(plan, biomodel, design)
    entries = layers.protected(plan, layers.sbml_organism(plan, biomodel, "B"))
    text, _ = antimony(world, info["rest"])
    sbml = te.antimonyToSBML(text)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "model.xml").write_text(sbml, encoding="utf-8")
    times, composite = read_series(directory / "measure" / "orthogonality.csv")
    steps = {name: value for name, value in experiment.run_names(plan).items() if name not in left_out}
    engines = {
        "libRoadRunner (CVODE)": lambda u, grid: with_roadrunner(world, plan, text, u, grid),
        "COPASI (LSODA)": lambda u, grid: with_copasi(world, plan, sbml, u, grid),
    }
    rows, courses = [], {"time": times}
    for engine, follow in engines.items():
        base = follow(world.baseline, times)
        still = max(
            float(np.max(np.abs(base[entry.column] - base[entry.column][0])) / delta[entry.column])
            for entry in entries
        )
        for name, activity in steps.items():
            # a run of the composite in which the receiver is lost is read until then, and so is this one
            read = int(np.sum(times < lost[name])) if name in lost else times.size
            head = {
                "biomodel": biomodel,
                "design": design,
                "leak": world.leak,
                "engine": engine,
                "step": name,
            }
            head.update(activity=activity, receiver_lost_at=lost.get(name), stored_times_read=read)
            if read < 2:
                rows.append({**head, "note": "lost at the first stored time: nothing to compare"})
                continue
            try:
                run_ = follow(activity, times[:read])
            except (RuntimeError, ValueError) as problem:  # the model stops having a solution here too
                rows.append({**head, "note": f"not followed: {type(problem).__name__}"})
                continue
            course = np.max(
                np.vstack(
                    [
                        np.abs(run_[entry.column] - base[entry.column][:read]) / delta[entry.column]
                        for entry in entries
                    ]
                ),
                axis=0,
            )
            other = composite[f"D[{name}]"][:read]
            _, stored = read_series(directory / name / "protected.csv")
            apart = max(
                float(np.max(np.abs(run_[entry.column] - stored[entry.column][:read])) / delta[entry.column])
                for entry in entries
            )
            courses[f"composite {name}"] = composite[f"D[{name}]"]
            courses[f"{engine} {name}"] = np.concatenate([course, np.full(times.size - read, np.nan)])
            rows.append(
                {
                    **head,
                    "D_of_the_composite": float(np.max(other)),
                    "D_of_the_single_system": float(np.max(course)),
                    "relative_difference_of_D": float(np.max(course) / np.max(other) - 1.0),
                    "largest_gap_along_D(t)_over_the_peak": float(
                        np.max(np.abs(course - other)) / np.max(other)
                    ),
                    "largest_gap_of_a_protected_quantity_in_tolerances": apart,
                    "baseline_moves_by_in_tolerances": still,
                    "note": "",
                }
            )
    return rows, courses


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    del jobs, analyze_only
    rows, kept = [], None
    for biomodel in plan["receivers"]:
        for design, leak in plan["designs"].items():
            if not float(leak) > 0.0:
                continue
            reference = (biomodel, design) == WORLD
            found, courses = one_world(plan, biomodel, design, out if reference else None)
            rows += found
            if reference:
                kept = courses
            print(f"monolith: {biomodel} {design}", flush=True)
    written = [write_rows(out / "monolith.csv", rows)]
    if kept is not None:
        names = list(kept)
        table = [{name: float(kept[name][i]) for name in names} for i in range(kept["time"].size)]
        written += [write_rows(out / "courses.csv", table), out / "model.xml"]
    return written
