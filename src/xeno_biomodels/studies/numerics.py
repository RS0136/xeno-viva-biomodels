"""How much of a reading is the numerical method.

Two numbers of the method are not part of any model: the interval at which the organisms are coupled through the
medium (each is advanced over it from the same state of the medium, and what each adds to the medium is summed),
and the tolerances with which CVODE integrates the SBML model within an interval. Both are changed here, and the
worlds are run and measured again.

    interval      a quarter, a half and twice the interval of the plan, stored on the same times
    tolerances    a hundred times looser and ten times tighter than the plan

For the design that shares nothing the question is whether the zero stays a zero. For the controls it is by how
much D moves, at its peak, along its whole course, and at each layer. The coupling is of first order in the
interval, so the readings at a half and at a quarter of it also give what is left at an interval of zero
(``D_extrapolated``: twice the reading at the smaller interval less the one at the larger).

Written to ``runs/studies/numerics/``: ``numerics.csv`` and ``layers.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xeno_biomodels import paths
from xeno_biomodels.experiment import run_variants
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies.common import STUDIES, changed, number, read_rows, slug, write_rows

OUT = STUDIES / "numerics"
DESIGNS = ("xeno", "leak-1e-2", "leak-1e-1")
INTERVALS = (0.25, 0.5, 2.0)
TOLERANCES = {"looser": (1e-10, 1e-8), "tighter": (1e-13, 1e-11)}
STAGES = ("stage1_trajectory", "stage3_tolerable", "stage4_distribution")


def variants(plan: dict) -> dict[str, tuple[dict, dict]]:
    interval, every = float(plan["time"]["interval"]), int(plan["time"]["store_every"])
    out = {}
    for factor in INTERVALS:
        tag = {"what": "coupling interval", "value": interval * factor, "factor": factor}
        changes = {"time.interval": interval * factor, "time.store_every": round(every / factor)}
        out[f"interval {interval * factor:g} min"] = (tag, changed(plan, changes))
    for label, (absolute, relative) in TOLERANCES.items():
        tag = {"what": "tolerances of the integrator", "value": relative, "factor": ""}
        changes = {"time.absolute_tolerance": absolute, "time.relative_tolerance": relative}
        out[f"tolerances {label} ({relative:g})"] = (tag, changed(plan, changes))
    return out


def against(plan: dict, rows: list[dict], by_group: list[dict], out: Path = OUT) -> None:
    """Beside every world what the plan reads for it: the change of D, the gap along its course, whether the
    stages, the losses and the admissible runs are the same, and what is left at an interval of zero."""
    main = paths.RUNS / "worlds"
    reference = {(row["biomodel"], row["design"]): row for row in read_rows(main / "summary.csv")}
    layer_reference = {
        (row["biomodel"], row["design"], row["layer"], row["kind"]): row
        for row in read_rows(main / "layers.csv")
    }
    for row in rows:
        if row.get("status") != "ok":
            continue
        base = reference[(row["biomodel"], row["design"])]
        d0 = float(base["D"])
        row["D_reference"] = d0
        row["D_change"] = (float(row["D"]) / d0 - 1.0) if d0 > 0.0 else float(row["D"])
        # along the whole course: the largest gap between the two D(t), over the peak of the reference
        here = (
            out
            / "worlds"
            / slug(row["variant"])
            / row["biomodel"]
            / row["design"]
            / "measure"
            / "orthogonality.csv"
        )
        _, mine = read_series(here)
        _, theirs = read_series(main / row["biomodel"] / row["design"] / "measure" / "orthogonality.csv")
        gap = float(np.nanmax(np.abs(mine["D"] - theirs["D"])))
        row["D_course_gap"] = gap / d0 if d0 > 0.0 else gap
        row["same_verdict"] = (
            all(str(row[key]) == base[key] for key in STAGES) and (row["losses"] or "") == base["losses"]
        )
        row["same_admissible"] = (row["excluded_by_sender_function"] or "") == base[
            "excluded_by_sender_function"
        ]
        row["admissible_in_the_plan"] = base["excluded_by_sender_function"]
    # what is left at an interval of zero, from the two smallest intervals
    found = {
        (row["biomodel"], row["design"], number(row.get("factor"))): row
        for row in rows
        if row.get("status") == "ok"
    }
    for (biomodel, design, factor), row in found.items():
        quarter = found.get((biomodel, design, 0.25))
        if factor != 0.5 or quarter is None:
            continue
        limit = 2.0 * float(quarter["D"]) - float(row["D"])
        d0 = float(row["D_reference"])
        row["D_extrapolated"] = limit
        row["reference_over_extrapolated"] = (d0 / limit - 1.0) if limit > 0.0 else d0
        # the order of the coupling: the error at the interval of the plan over the error at half of it
        if limit > 0.0 and float(row["D"]) != limit:
            row["error_ratio_h_over_half"] = (d0 - limit) / (float(row["D"]) - limit)
    for row in by_group:
        base = layer_reference.get((row["biomodel"], row["design"], row["layer"], row["kind"]))
        mine = number(row.get("D"))
        if base is None or mine is None:
            continue
        d0 = float(base["D"])
        row["D_reference"] = d0
        row["D_change"] = (mine / d0 - 1.0) if d0 > 0.0 else mine


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    written = [out / "numerics.csv", out / "layers.csv"]
    if analyze_only:  # the worlds are not run again: what they left is set against the plan again
        rows, by_group = read_rows(out / "numerics.csv"), read_rows(out / "layers.csv")
    else:
        work = []
        for name, (tag, variant) in variants(plan).items():
            for biomodel in plan["receivers"]:
                for design in DESIGNS:
                    work.append(
                        (
                            {"variant": name, **tag},
                            variant,
                            biomodel,
                            design,
                            out / "worlds" / slug(name),
                            False,
                            False,
                        )
                    )
        rows, by_group = run_variants(work, jobs)
    against(plan, rows, by_group, out)
    write_rows(out / "numerics.csv", rows)
    write_rows(out / "layers.csv", by_group)
    return written
