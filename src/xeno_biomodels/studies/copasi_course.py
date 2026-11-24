"""A time course under COPASI, in an interpreter of its own.

COPASI, libRoadRunner and Antimony each bring their own copy of libSBML, and COPASI has been seen to stop the
interpreter when it reads a model in a process that has loaded the others (and, read from a string, also
without them). The study ``monolith`` therefore asks COPASI through this module: ``apart`` starts a fresh
interpreter that loads COPASI and nothing else of the simulators, hands it the model as a file, and reads the
time course back.

    python -m xeno_biomodels.studies.copasi_course REQUEST.json     (what ``apart`` runs)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def course(
    model_file: str,
    parameters: dict[str, float],
    species: list[str],
    values: list[str],
    start: float,
    end: float,
    count: int,
    r_tol: float,
    a_tol: float,
) -> dict[str, np.ndarray]:
    """The concentrations of ``species``, every reaction flux and the global quantities ``values`` of an SBML
    file at ``count`` equally spaced times, after setting ``parameters``; all names are SBML identifiers. The
    columns come back as ``species.NAME``, ``reactions.NAME`` and ``values.NAME``."""
    import basico

    model = basico.load_model(str(model_file))
    try:
        for name, value in parameters.items():
            basico.set_parameters(name, initial_value=float(value), model=model)
        # COPASI knows species and reactions by their names; the columns are those of their SBML identifiers
        table = basico.get_species(model=model)
        shown = {row["sbml_id"]: name for name, row in table.iterrows()}
        table = basico.get_reactions(model=model)
        named = {row["sbml_id"]: name for name, row in table.iterrows()}
        reactions = list(named)
        wanted = (
            [f"[{shown[name]}]" for name in species]
            + [f"({named[name]}).Flux" for name in reactions]
            + [f"Values[{name}]" for name in values]
        )
        found = basico.run_time_course_with_output(
            output_selection=wanted,
            start_time=float(start),
            duration=float(end - start),
            intervals=int(count - 1),
            method="deterministic",
            r_tol=float(r_tol),
            a_tol=float(a_tol),
            max_steps=500000,
            model=model,
        )
    finally:
        basico.remove_datamodel(model)
    table = found.to_numpy(dtype=float)
    labels = (
        [f"species.{name}" for name in species]
        + [f"reactions.{name}" for name in reactions]
        + [f"values.{name}" for name in values]
    )
    return {label: table[:, i] for i, label in enumerate(labels)}


def apart(
    sbml: str,
    parameters: dict[str, float],
    species: list[str],
    values: list[str],
    start: float,
    end: float,
    count: int,
    r_tol: float,
    a_tol: float,
) -> dict[str, dict[str, np.ndarray]]:
    """``course`` on a model given as a string, in an interpreter of its own: ``{"species": {...}, "reactions":
    {...}, "values": {...}}``. Raises RuntimeError where that interpreter does not come back with an answer."""
    with tempfile.TemporaryDirectory() as folder:
        model = Path(folder) / "model.xml"
        # letters outside ASCII (in the notes of a BioModels file) are written as character references
        model.write_text(sbml.encode("ascii", "xmlcharrefreplace").decode("ascii"), encoding="ascii")
        request = Path(folder) / "request.json"
        request.write_text(
            json.dumps(
                {
                    "model_file": str(model),
                    "parameters": parameters,
                    "species": species,
                    "values": values,
                    "start": start,
                    "end": end,
                    "count": count,
                    "r_tol": r_tol,
                    "a_tol": a_tol,
                    "answer": str(Path(folder) / "answer.npz"),
                }
            ),
            encoding="utf-8",
        )
        done = subprocess.run(
            [sys.executable, "-m", "xeno_biomodels.studies.copasi_course", str(request)],
            capture_output=True,
            text=True,
            check=False,
        )
        answer = Path(folder) / "answer.npz"
        if done.returncode != 0 or not answer.exists():
            raise RuntimeError(f"COPASI gave no time course (exit {done.returncode}): {done.stderr[-300:]}")
        with np.load(answer) as found:
            columns = {label: np.array(found[label]) for label in found.files}
    out: dict[str, dict[str, np.ndarray]] = {"species": {}, "reactions": {}, "values": {}}
    for label, column in columns.items():
        kind, _, name = label.partition(".")
        out[kind][name] = column
    return out


if __name__ == "__main__":
    asked = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    target = asked.pop("answer")
    np.savez(target, **course(**asked))

__all__ = ["apart", "course"]
