"""A community that coexists stably and is at rest.

In the community of the plan two SBML organisms live on the same glucose, one of them is washed out, and the
xenos of the design ``partner-only`` drive each other out slowly, so every reading there is taken against a
baseline that still moves. Here the community is made so that everything in it stays, and stays stably: two pairs
whose SBML organisms live on different nutrients (glucose and ammonium), and xenos that each have a substrate of
their own. Two xenos with the same laws on one substrate would coexist only neutrally (any ratio of their
densities is a rest state, and a step of either drives the other out), so where the xenos are to reach each other
each also takes a little of the substrate of the other and grows on it: each is better at its own, and the
coexistence is stable. Three designs:

    private       each xeno on a substrate of its own, and nothing else
    overlapping   each also takes the substrate of the other xeno, at a small specificity (``OVERLAP``)
    indirect      as ``overlapping``, and each takes the nutrient of the other pair's SBML organism at a small
                  specificity, without growing on it

In ``indirect`` a xeno shares no species with its own partner, and reaches it all the same: its step changes the
substrates, with them the density of the other xeno, with that what the other xeno takes of the nutrient of the
partner. The study writes the measure for every ordered pair, the niche overlap of the two beside it, and how
far the baseline of the reading still moves.

A second table says what the reading of ``indirect`` owes to the two numbers that the design adds: the
specificity of a xeno for the nutrient of the other pair, from a tenth to ten times that of the design, and its
specificity for the substrate of the other xeno.

Written to ``runs/studies/coexistence/``: ``community.csv``, ``layers.csv``, ``baseline.csv`` and ``sweep.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xeno_biomodels import community
from xeno_biomodels.community_report import chain
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies.common import STUDIES, changed, number, write_rows
from xeno_biomodels.studies.comparison import pianka, taken_by_law

OUT = STUDIES / "coexistence"
#: SBML organisms that live on different nutrients. The yeast model is not among them: pouring two vessels together
#: halves its glucose, and below 0.52 of the level of its file the model has no steady state (study quasistatic).
PAIRS = ("BIOMD0000000217", "BIOMD0000000017")
#: minutes from the pouring to the reading. The slowest mode of the community is the density of the organism whose
#: uptake hardly reads its nutrient (about 19000 min); after five hundred residence times nothing moves any more.
ASSEMBLY = 300000.0
OVERLAP = 0.1  # the specificity of a xeno for the substrate of the other xeno
CROSS = 0.01  # the specificity of a xeno for the nutrient of the SBML organism of the other pair
DESIGNS = {
    "private": {"substrate": "own", "cross": 0.0},
    "overlapping": {"substrate": "own", "cross": 0.0, "overlap": OVERLAP},
    "indirect": {"substrate": "own", "cross": CROSS, "worth": 0.0, "overlap": OVERLAP},
}
SWEEP = (
    0.001,
    0.003,
    0.03,
    0.1,
)  # other specificities for the nutrient of the other pair, design ``indirect``
OVERLAPS = (0.03, 0.3)  # other specificities for the substrate of the other xeno, design ``indirect``


def variant(plan: dict, designs: dict | None = None) -> dict:
    """The plan of the community of this study."""
    names = {biomodel: plan["community"]["names"][biomodel] for biomodel in PAIRS}
    return changed(
        plan,
        {"community.names": names, "community.designs": designs or DESIGNS, "community.assembly": ASSEMBLY},
    )


def swept() -> dict[str, dict]:
    """The designs of the second table: ``indirect`` with one of its two numbers changed."""
    designs = {f"indirect-{cross:g}": {**DESIGNS["indirect"], "cross": cross} for cross in SWEEP}
    designs.update(
        {f"indirect-overlap-{overlap:g}": {**DESIGNS["indirect"], "overlap": overlap} for overlap in OVERLAPS}
    )
    return designs


def sweep(plan: dict, jobs: int, analyze_only: bool, out: Path) -> list[dict]:
    """The design ``indirect`` at other specificities: every xeno on every SBML organism."""
    designs = swept()
    study, rows = variant(plan, designs), []
    for design, body in [("indirect", DESIGNS["indirect"]), *designs.items()]:
        directory = out if design == "indirect" else out / "sweep"
        if design != "indirect" and not analyze_only:
            community.simulate(study, design, directory, jobs, fate=False)
        if not (directory / design / "community.json").exists():
            continue
        found, _ = community.read_matrix(variant(plan) if design == "indirect" else study, design, directory)
        for row in found:
            if row["sender_kind"] == "xeno":
                rows.append(
                    {
                        "design": design,
                        "specificity": float(body["cross"]),
                        "overlap": float(body["overlap"]),
                        "sender": row["sender"],
                        "receiver": row["receiver"],
                        "receiver_kind": row["receiver_kind"],
                        "same_pair": row["same_pair"],
                        "D": number(row.get("D")),
                        "D_quantity": row.get("D_quantity"),
                        "D_activity": row.get("D_activity"),
                        "D_removal": number(row.get("D_removal")),
                        "losses": row.get("losses"),
                    }
                )
        print(f"coexistence: {design}", flush=True)
    return sorted(rows, key=lambda row: (row["sender"], row["receiver"], row["overlap"], row["specificity"]))


def baseline(directory: Path, members: dict) -> list[dict]:
    """How far the baseline of a design moves over the reading: the density of every organism at its end over
    its start, and the largest change of what its medium holds."""
    times, columns = read_series(directory / community.BASELINE / "series.csv")
    rows = []
    for name in members:
        density = columns[f"{name}.density"]
        rows.append(
            {
                "organism": name,
                "density_at_the_end_over_the_start": float(density[-1] / density[0]),
                "largest_density_over_the_start": float(np.max(density) / density[0]),
                "least_density_over_the_start": float(np.min(density) / density[0]),
            }
        )
    medium = [key for key in columns if key.startswith("medium.")]
    moved = max(
        float(np.max(np.abs(columns[key] - columns[key][0])) / max(abs(columns[key][0]), 1e-300))
        for key in medium
    )
    for row in rows:
        row["largest_relative_change_of_a_medium_species"] = moved
        row["minutes"] = float(times[-1] - times[0])
    return rows


def steps(directory: Path, design: str, members: dict) -> list[dict]:
    """D of every admissible step of a xeno alone, on every other organism, and where the densities of the two
    xenos end in that run, over the baseline: what a step does when nobody is driven out."""
    times, base = read_series(directory / community.BASELINE / "series.csv")
    del times
    xenos = [name for name, body in members.items() if body["kind"] == "xeno"]
    rows = []
    for sender in xenos:
        for receiver in members:
            folder = directory / "measure" / f"{sender}_to_{receiver}"
            if receiver == sender or not (folder / "orthogonality.csv").exists():
                continue
            _, found = read_series(folder / "orthogonality.csv")
            for column, values in found.items():
                if not column.startswith("D[u="):
                    continue
                step = column[2:-1]
                _, moved = read_series(directory / f"{sender}.{step}" / "series.csv")
                row = {
                    "design": design,
                    "sender": sender,
                    "receiver": receiver,
                    "step": step,
                    "D": float(np.nanmax(values)),
                    "D_at_the_end": float(values[-1]),
                }
                for name in xenos:
                    row[f"density_of_{name}_at_the_end"] = float(
                        moved[f"{name}.density"][-1] / base[f"{name}.density"][-1]
                    )
                rows.append(row)
    return rows


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    study, sender = variant(plan), plan["sender"]
    rows, by_group, still, stepped = [], [], [], []
    for design in DESIGNS:
        if not analyze_only:
            community.simulate(study, design, out, jobs, fate=False)
        found, groups = community.read_matrix(study, design, out)
        info = json.loads((out / design / "community.json").read_text(encoding="utf-8"))
        members, medium = info["members"], info["assembled"]["medium"]
        use = {}
        for name, body in members.items():
            if body["kind"] == "sbml":
                use[name] = dict.fromkeys(body["reads"], 1.0)
            else:
                config = {
                    "substrate": body["substrate"],
                    "vmax": sender["vmax"],
                    "km": sender["km"],
                    "also_takes": body["also_takes"],
                }
                use[name] = taken_by_law(config, medium, float(plan["activities"]["baseline"]))
        for row in found:
            way = chain(members, row["sender"], row["receiver"])
            rows.append(
                {
                    **row,
                    "niche_overlap": pianka(use[row["sender"]], use[row["receiver"]]),
                    "organisms_on_the_way": None if way is None else (len(way) - 3) // 2,
                    "way": "" if way is None else " > ".join(way),
                    "D_number": number(row.get("D")),
                }
            )
        by_group += groups
        still += [{"design": design, **row} for row in baseline(out / design, members)]
        stepped += steps(out / design, design, members)
        print(f"coexistence: {design}", flush=True)
    return [
        write_rows(out / "community.csv", rows),
        write_rows(out / "layers.csv", by_group),
        write_rows(out / "baseline.csv", still),
        write_rows(out / "steps.csv", stepped),
        write_rows(out / "sweep.csv", sweep(plan, jobs, analyze_only, out)),
    ]
