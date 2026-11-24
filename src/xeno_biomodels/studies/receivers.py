"""Other SBML models as receivers, taken by rule.

The three receivers of the plan were taken by hand from the models that the screen lists. Here every model of
the plateau is a receiver if it passes a rule that asks nothing of what the model is a model of:

    the model settles, and the largest real part of the eigenvalues of its Jacobian there is negative, and
    not by rounding alone (below -1e-12 per unit of time);
    it holds a species constant that its reactions take up;
    with 10 % more of that species its uptake rises by ``uptake_rises_by`` or more, and a species of the model
    moves by ``state_moves_by`` or more (``runs/selection/screen.csv``).

Every (model, species) that passes gets a world in which that species is the nutrient, and nothing of the world
is in the units of a model: the density of the organism is the one at which it takes the share ``consumes`` of
what is fed, the times of the plan are turned into the unit of time of the model, and the leak of the control
is a share of what the organism takes (``World._sender_config``). What the organism protects is the quantities
that are common to every organism, at the layers that it has; where the plan would say by hand which species are
carriers, the file is read by rule (``layers.annotate``). Many of these models are not organisms at all (an
enzyme, a signalling pathway): here an organism is a model with an environment that it consumes, and no more.

Written to ``runs/studies/receivers/``: ``candidates.csv`` (what passed the rule), ``worlds.csv`` and ``layers.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

from xeno_biomodels import layers, paths
from xeno_biomodels.experiment import run_variants
from xeno_biomodels.studies.common import STUDIES, changed, read_rows, write_rows
from xeno_biomodels.world import WorldError, sbml_file, time_unit_seconds

OUT = STUDIES / "receivers"
STABLE = 1e-12


def candidates(plan: dict) -> list[dict]:
    """Every (model, species) of the screen that passes the rule."""
    settings = plan["studies"]["receivers"]
    out = []
    for row in read_rows(paths.RUNS / "selection" / "screen.csv"):
        if row.get("steady") != "True" or not row.get("sensed"):
            continue
        if not float(row["largest_real_eigenvalue"]) < -STABLE:
            continue
        for species, found in json.loads(row["sensed"]).items():
            rises, moves = found["uptake_change"] or 0.0, found["largest_species_change"]
            if (
                found["uptake"] > 0.0
                and rises >= float(settings["uptake_rises_by"])
                and moves >= float(settings["state_moves_by"])
            ):
                out.append(
                    {
                        "key": f"{row['biomodel_id']}@{species}",
                        "biomodel_id": row["biomodel_id"],
                        "name": row["name"],
                        "species": species,
                        "level_in_the_file": found["level"],
                        "uptake_rises_by": rises,
                        "state_moves_by": moves,
                        "largest_real_eigenvalue": float(row["largest_real_eigenvalue"]),
                        "is_a_receiver_of_the_plan": plan["receivers"]
                        .get(row["biomodel_id"], {})
                        .get("nutrient")
                        == species,
                    }
                )
    return out


def variant(plan: dict, found: list[dict]) -> dict:
    """The plan with the candidates as its receivers and the leak as a share."""
    settings = plan["studies"]["receivers"]
    receivers = {}
    for item in found:
        sbml = sbml_file(item["biomodel_id"])
        receivers[item["key"]] = {
            "biomodel": item["biomodel_id"],
            "organism": item["name"],
            "what": f"taking {item['species']}",
            "medium": [item["species"]],
            "nutrient": item["species"],
            "consumes": float(settings["consumes"]),
            **layers.annotate(sbml),
        }
    designs: dict = {"xeno": 0.0}
    for share in settings["shares"]:
        designs[f"share-{share:g}"] = {"share": float(share)}
    specific = {"xeno": plan["protect"]["specific"].get("xeno", {})}
    # the densities of these worlds are in units of their own: an organism counts as established down to a
    # thousandth of the density that it has at rest, which is what the number of the plan is for its receivers
    return changed(
        plan,
        {"receivers": receivers, "designs": designs, "protect.specific": specific, "viable": "rel:1e-3"},
    )


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    written = [out / "candidates.csv", out / "worlds.csv", out / "layers.csv"]
    if analyze_only:
        return written
    found = candidates(plan)
    for item in found:  # what is known of a model before its world is tried
        try:
            item["seconds_in_its_unit_of_time"] = time_unit_seconds(sbml_file(item["biomodel_id"]))
        except WorldError as exc:
            item["seconds_in_its_unit_of_time"] = str(exc)
    write_rows(out / "candidates.csv", found)
    study = variant(plan, found)
    by_key = {item["key"]: item for item in found}
    work = []
    for key in study["receivers"]:
        for design, written_as in study["designs"].items():
            item = by_key[key]
            tag = {
                "biomodel_id": item["biomodel_id"],
                "name": item["name"],
                "species": item["species"],
                "share": written_as["share"] if isinstance(written_as, dict) else 0.0,
                "uptake_rises_by": item["uptake_rises_by"],
                "state_moves_by": item["state_moves_by"],
                "is_a_receiver_of_the_plan": item["is_a_receiver_of_the_plan"],
            }
            work.append((tag, study, key, design, out / "worlds", False, False))
    rows, by_group = run_variants(work, jobs)
    write_rows(out / "worlds.csv", rows)
    write_rows(out / "layers.csv", by_group)
    return written
