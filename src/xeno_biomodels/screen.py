"""Which models of the plateau can be a receiver: an organism that takes something from a medium and senses it.

An SBML model can share an environment with another organism only where it has an environment: a species that it
holds constant (a boundary species), that its reactions take up, and whose level its kinetics read. A model whose
uptake does not depend on the level of what it takes up cannot be disturbed through it, and every organism
would be orthogonal to it for no reason. A world also needs a rest state to start from, so the model must settle.

For every model of the plateau this module finds, from the SBML file and a run of the model:

    exchanged     the boundary species that reactions consume or produce
    steady        whether the model settles, and the largest real part of the eigenvalues of its Jacobian there
    sensed        for each boundary species that is consumed: the relative change of its uptake and the largest
                  relative change of a species of the model when its level is raised by 10 %

and writes ``runs/selection/screen.csv``. It does not decide what a model is a model of: which of the models
that pass are microorganisms living on a nutrient is read from their names, by a person, and written into the
plan (``receivers``).
"""

from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import libsbml
import numpy as np

from xeno_biomodels import paths, preload_libpython
from xeno_biomodels.world import sbml_file

preload_libpython()


def structure(sbml: Path) -> dict:
    """Compartments, and the boundary species that reactions consume or produce."""
    model = libsbml.readSBMLFromFile(str(sbml)).getModel()
    ruled = {rule.getVariable() for rule in model.getListOfRules()}
    species = {item.getId(): item for item in model.getListOfSpecies()}
    boundary = {name for name, item in species.items() if item.getBoundaryCondition() or item.getConstant()}
    consumed: dict[str, list[str]] = {}
    produced: dict[str, list[str]] = {}
    for reaction in model.getListOfReactions():
        for reference in reaction.getListOfReactants():
            consumed.setdefault(reference.getSpecies(), []).append(reaction.getId())
        for reference in reaction.getListOfProducts():
            produced.setdefault(reference.getSpecies(), []).append(reaction.getId())
    return {
        "compartments": [item.getName() or item.getId() for item in model.getListOfCompartments()],
        "integrated": [name for name in species if name not in boundary and name not in ruled],
        "consumed": {name: consumed[name] for name in sorted(boundary) if name in consumed},
        "produced": {name: produced[name] for name in sorted(boundary) if name in produced},
        "events": int(model.getNumEvents()),
        "names": {name: item.getName() or name for name, item in species.items()},
    }


def dynamics(sbml: Path, consumed: dict[str, list[str]]) -> dict:
    """Whether the model settles, how fast, and what it does when a consumed boundary species is raised by 10 %."""
    import tellurium

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = tellurium.loadSBMLModel(str(sbml))
    model.integrator.absolute_tolerance = 1e-12
    model.integrator.relative_tolerance = 1e-10

    def settle() -> tuple[np.ndarray, float] | None:
        at, span, previous = model.getCurrentTime(), 100.0, None
        for _ in range(8):
            model.simulate(at, at + span, 2)
            at += span
            now = np.asarray(model.getFloatingSpeciesConcentrations(), dtype=float)
            if previous is not None and np.max(np.abs(now - previous) / (np.abs(now) + 1e-9)) < 1e-7:
                return now, at
            previous, span = now, 4.0 * span
        return None

    rest = settle()
    if rest is None:
        return {"steady": False}
    state, spent = rest
    model.conservedMoietyAnalysis = True
    jacobian = np.asarray(model.getReducedJacobian(), dtype=float)
    model.conservedMoietyAnalysis = False
    slowest = float(np.max(np.linalg.eigvals(jacobian).real)) if jacobian.size else float("nan")
    rates = dict(zip(model.getReactionIds(), model.getReactionRates(), strict=True))
    sensed = {}
    for name, reactions in consumed.items():
        level = float(model[f"[{name}]"])
        if level == 0.0:
            continue
        uptake = sum(rates[reaction] for reaction in reactions)
        model[f"[{name}]"] = 1.1 * level
        moved = settle()
        if moved is not None:
            after = dict(zip(model.getReactionIds(), model.getReactionRates(), strict=True))
            sensed[name] = {
                "level": level,
                "uptake": float(uptake),
                "uptake_change": float(sum(after[reaction] for reaction in reactions) / uptake - 1.0)
                if uptake
                else None,
                "largest_species_change": float(np.max(np.abs(moved[0] - state) / (np.abs(state) + 1e-9))),
            }
        model[f"[{name}]"] = level
        settle()
    return {"steady": True, "settled_after": spent, "largest_real_eigenvalue": slowest, "sensed": sensed}


def write(out: Path | None = None) -> Path:
    """Screen every model of ``candidates.csv`` and write ``screen.csv``."""
    out = Path(out or paths.RUNS / "selection")
    with (out / "candidates.csv").open(encoding="utf-8", newline="") as handle:
        candidates = {row["biomodel_id"]: row for row in csv.DictReader(handle)}
    rows = []
    for biomodel, row in sorted(candidates.items()):
        record = {
            "biomodel_id": biomodel,
            "name": row.get("name", ""),
            "max_variable_nrmse": row["max_variable_nrmse"],
        }
        try:
            sbml = sbml_file(biomodel)
            found = structure(sbml)
            record.update(
                compartments=";".join(found["compartments"]),
                integrated_species=len(found["integrated"]),
                consumed_boundary_species=" ".join(found["consumed"]),
                produced_boundary_species=" ".join(found["produced"]),
                events=found["events"],
            )
            if found["consumed"] and not found["events"]:
                moved = dynamics(sbml, found["consumed"])
                record.update(steady=moved["steady"])
                if moved["steady"]:
                    record.update(
                        largest_real_eigenvalue=moved["largest_real_eigenvalue"],
                        sensed=json.dumps(moved["sensed"]),
                        # a consumed species counts as sensed when 10 % more of it moves the uptake by 0.1 % or more
                        sensed_nutrients=" ".join(
                            name
                            for name, item in moved["sensed"].items()
                            if abs(item["uptake_change"] or 0.0) >= 1e-3
                        ),
                    )
        except (
            RuntimeError,
            OSError,
            ValueError,
        ) as exc:  # cannot be fetched, read or run: recorded, passed over
            record["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:120]}"
        rows.append(record)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "screen.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return out / "screen.csv"
