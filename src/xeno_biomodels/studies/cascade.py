"""What the layers add to one another.

In a world of the plan everything that reaches the receiver enters through one species of the medium, and the
layers are stations of one way inward. Reading D at every layer then asks a question that this study answers
from the runs that are there (those of the plan, and those of the study ``channels``, with two ways in):

    arrival      when the interference is at each layer: the time at which D of the layer is half its peak, and
                 the time of the peak, for the step to twice the activity and over all admissible steps
    dimension    how many readings the protected quantities are. Their changes over their tolerances, at every
                 stored time of every admissible run, are the rows of a matrix. Its singular values say how many
                 independent courses there are among the rows: the participation ratio (sum of the squares,
                 squared, over the sum of the fourth powers) is 1 when every quantity follows one course, each
                 with its own gain, and the number of quantities when no two have anything in common. It is given
                 for the rows as they are and with every row brought to length 1, so that a quantity that moves
                 little counts as much as one that moves much.
    similarity   for every two protected quantities, the cosine between their rows: 1 or -1 when the two are one
                 reading
    two ways in  for a receiver of the study ``channels`` and a share: the runs in which the sender takes its
                 nutrient and those in which it takes the other species, as one matrix. If the second way in
                 moves the quantities in another proportion, the two together are more readings than either.
                 Beside it, the cosine between the two profiles (D of every layer and kind by the one way and
                 by the other).

Written to ``runs/studies/cascade/``: ``arrival.csv``, ``dimension.csv``, ``similarity.csv``, ``profiles.csv``, and
under ``courses/`` D of every layer over time for the step to twice the activity, one file per world.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xeno_biomodels import experiment, layers
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies import channels
from xeno_biomodels.studies.common import LINES, STUDIES, read_rows, write_rows
from xeno_biomodels.world import BASELINE

OUT = STUDIES / "cascade"
STEP = "u=2"  # the run whose course is shown: a step that the sender survives in every world


def worlds(plan: dict) -> list[tuple[dict, dict, str, str, Path]]:
    """(what the world is, its plan, receiver, design, where its runs are) for every world that is read."""
    out = []
    for biomodel in plan["receivers"]:
        for design in plan["designs"]:
            tag = {"study": "plan", "channel": "nutrient" if float(plan["designs"][design]) > 0.0 else "none"}
            out.append((tag, plan, biomodel, design, experiment.OUT / biomodel / design))
    if (channels.OUT / "worlds.csv").exists():
        for tag, variant, biomodel, design in channels.variants(plan):
            out.append(
                (
                    {"study": "channels", **tag},
                    variant,
                    biomodel,
                    design,
                    channels.OUT / "worlds" / biomodel / design,
                )
            )
    return out


def changes(
    plan: dict, biomodel: str, directory: Path
) -> tuple[np.ndarray, list[layers.Entry], dict[str, np.ndarray]] | None:
    """The changes of what the receiver protects over the tolerances, for every admissible run of a world:
    (times, the protected quantities, run -> array (quantity, time)), with NaN from the loss of the receiver on.
    None where the world was not measured."""
    if not (directory / "measure" / "result.json").exists():
        return None
    result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
    entries = layers.protected(plan, layers.sbml_organism(plan, biomodel, "B"))
    delta = {row["column"]: float(row["delta"]) for row in read_rows(directory / "measure" / "deltas.csv")}
    times, base = read_series(directory / BASELINE / "protected.csv")
    left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
    lost = {item["activity"]: item["t"] for item in result["losses"]}
    out = {}
    for run in experiment.run_names(plan):
        path = directory / run / "protected.csv"
        if run in left_out or not path.exists():
            continue
        _, columns = read_series(path)
        signed = np.vstack(
            [(columns[entry.column] - base[entry.column]) / delta[entry.column] for entry in entries]
        )
        if run in lost:
            signed[:, times >= lost[run]] = np.nan
        out[run] = signed
    return times, entries, out


def _half(times: np.ndarray, course: np.ndarray) -> float | None:
    """The first stored time at which a course is half its peak or more."""
    peak = float(np.nanmax(course))
    if not peak > 0.0:
        return None
    return float(times[int(np.flatnonzero(course >= 0.5 * peak)[0])])


def participation(rows: np.ndarray) -> dict:
    """How many courses the rows of a matrix are: for the rows as they are, and each brought to length 1."""
    record = {}
    unit = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    for label, matrix in (("as_they_are", rows), ("unit_length", unit)):
        squares = np.linalg.svd(matrix, compute_uv=False) ** 2
        share = np.cumsum(squares) / np.sum(squares)
        record[f"participation_{label}"] = float(np.sum(squares) ** 2 / np.sum(squares**2))
        record[f"first_course_share_{label}"] = float(share[0])
        record[f"courses_for_99_percent_{label}"] = int(np.searchsorted(share, 0.99) + 1)
    return record


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    del jobs, analyze_only  # this study only reads
    arrival, dimension, similarity, kept = [], [], [], {}
    (out / "courses").mkdir(parents=True, exist_ok=True)
    for tag, variant, biomodel, design, directory in worlds(plan):
        found = changes(variant, biomodel, directory)
        head = {**tag, "biomodel": biomodel, "design": design}
        if found is None or not found[2]:
            continue
        times, entries, signed = found
        lines = [line for line in LINES if any((entry.layer, entry.kind) == line for entry in entries)]
        rows_of = {
            line: [n for n, entry in enumerate(entries) if (entry.layer, entry.kind) == line]
            for line in lines
        }
        # ---- when the interference is at each layer
        courses = {}
        for label, runs in (
            ("every admissible step", list(signed)),
            (STEP, [STEP] if STEP in signed else []),
        ):
            if not runs:
                continue
            for line in lines:
                each = [np.max(np.abs(signed[name][rows_of[line]]), axis=0) for name in runs]
                course = np.nanmax(np.vstack([np.zeros_like(times), *each]), axis=0)
                if label == STEP:
                    courses[f"{line[0]}, {line[1]}"] = course
                peak = float(np.max(course))
                arrival.append(
                    {
                        **head,
                        "runs": label,
                        "layer": line[0],
                        "kind": line[1],
                        "D": peak,
                        "t_half": _half(times, course),
                        "t_peak": float(times[int(np.argmax(course))]) if peak > 0.0 else None,
                        "D_end_over_peak": float(course[-1] / peak) if peak > 0.0 else None,
                    }
                )
        if courses:
            write_rows(
                out / "courses" / f"{tag['study']}_{biomodel}_{design}.csv",
                [
                    {"time": float(at), **{name: float(course[n]) for name, course in courses.items()}}
                    for n, at in enumerate(times)
                ],
            )
        # ---- how many readings the quantities are
        matrix = np.hstack(list(signed.values()))
        matrix = matrix[:, np.all(np.isfinite(matrix), axis=0)]
        moving = [n for n in range(matrix.shape[0]) if float(np.max(np.abs(matrix[n]))) > 0.0]
        if len(moving) < 2:
            continue
        if tag["study"] == "channels":
            peaks = {
                line: max(float(np.nanmax(np.abs(run[rows_of[line]]))) for run in signed.values())
                for line in lines
            }
            kept[(biomodel, tag["channel"], tag["share"])] = (
                [entry.column for entry in entries],
                matrix,
                peaks,
            )
        matrix = matrix[moving]
        names = [entries[n] for n in moving]
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        dimension.append(
            {
                **head,
                "quantities": len(moving),
                "runs": len(signed),
                "samples": matrix.shape[1],
                **participation(matrix),
            }
        )
        cosine = unit @ unit.T
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                similarity.append(
                    {
                        **head,
                        "quantity_a": names[a].column,
                        "layer_a": names[a].layer,
                        "kind_a": names[a].kind,
                        "quantity_b": names[b].column,
                        "layer_b": names[b].layer,
                        "kind_b": names[b].kind,
                        "cosine": float(cosine[a, b]),
                    }
                )
    # ---- the two ways in of a receiver, together
    profiles = []
    for (biomodel, channel, share), (columns, matrix, peaks) in kept.items():
        other = kept.get((biomodel, "other", share))
        if channel != "nutrient" or other is None or other[0] != columns:
            continue
        both = np.hstack([matrix, other[1]])
        moving = [n for n in range(both.shape[0]) if float(np.max(np.abs(both[n]))) > 0.0]
        head = {
            "study": "channels",
            "channel": "both",
            "species": "",
            "share": share,
            "biomodel": biomodel,
            "design": f"nutrient and other, {share:g}",
        }
        dimension.append(
            {
                **head,
                "quantities": len(moving),
                "runs": "",
                "samples": both.shape[1],
                **participation(both[moving]),
            }
        )
        shared = [line for line in peaks if line in other[2]]
        a, b = np.array([peaks[line] for line in shared]), np.array([other[2][line] for line in shared])
        profiles.append(
            {
                "biomodel": biomodel,
                "share": share,
                "lines": len(shared),
                "cosine_of_the_two_profiles": float(a @ b / np.sqrt((a @ a) * (b @ b))),
                "cosine_of_their_logarithms": float(np.corrcoef(np.log(a), np.log(b))[0, 1]),
            }
        )
    write_rows(out / "arrival.csv", arrival)
    write_rows(out / "dimension.csv", dimension)
    write_rows(out / "similarity.csv", similarity)
    write_rows(out / "profiles.csv", profiles)
    return [out / "arrival.csv", out / "dimension.csv", out / "similarity.csv", out / "profiles.csv"]
