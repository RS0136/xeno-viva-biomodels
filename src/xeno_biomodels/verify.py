"""The receivers run through viva-biomodels itself, as its corpus was: are they as accurate here?

Each receiver of the plan goes through the ``compare-simulators`` composite of viva-biomodels
(``viva_biomodels.composites.compare_simulators.run_comparison``): the loader fetches the SBML file and reads the
time course that its SED-ML file asks for, every simulator that is installed runs it (COPASI, Tellurium and
simbio; AMICI and PySCeS are extras of viva-biomodels that need a compiler), and the comparison step scores
every pair of engines with the mean nRMSE and the closeness of viva-biomodels.

Two things are added. The largest nRMSE of any single variable, which a mean over variables can hide. And the
series computed here against those that viva-biomodels committed for the same model, from all five engines:
the committed series are every tenth point of the same time course, as float32.

Written to ``runs/verify/``: ``<biomodel>/<engine>.csv`` (the time course of every engine), ``<biomodel>/
comparison.json`` (the comparison as viva-biomodels returns it) and ``summary.csv``.
"""

from __future__ import annotations

import contextlib
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from xeno_biomodels import paths
from xeno_biomodels.corpus import pair_scores
from xeno_biomodels.world import load_plan

OUT = paths.RUNS / "verify"


def _table(engine: dict) -> tuple[np.ndarray, list[str], np.ndarray]:
    return (
        np.asarray(engine["time"], dtype=float),
        list(engine["columns"]),
        np.asarray(engine["values"], dtype=float),
    )


def against_corpus(biomodel: str, engines: dict) -> dict:
    """The largest nRMSE of any variable between an engine run here and an engine of the committed corpus."""
    table = pq.read_table(
        paths.CORPUS / "corpus_timecourse_all_engines.parquet", filters=[("biomodel_id", "=", biomodel)]
    ).to_pandas()
    worst, pair, compared = 0.0, "", 0
    for job, part in table.groupby("job"):
        for corpus_engine, rows in part.groupby("engine"):
            stored = rows.pivot_table(index="time", columns="variable", values="value", aggfunc="first")
            for name, engine in engines.items():
                times, columns, values = _table(engine)
                kept = stored.index.to_numpy(dtype=float)  # the committed series keep about every tenth time
                thinned = np.clip(np.searchsorted(times, kept), 1, len(times) - 1)
                thinned = np.where(
                    np.abs(times[thinned - 1] - kept) < np.abs(times[thinned] - kept), thinned - 1, thinned
                )
                if kept[-1] > times[-1] * (1.0 + 1e-6) or np.max(np.abs(times[thinned] - kept)) > 1e-3 * (
                    times[1] - times[0]
                ):
                    continue  # not the same time course (another job of the SED-ML file)
                shared = [
                    column
                    for column in columns
                    if column in stored.columns and not stored[column].isna().any()
                ]
                if not shared:
                    continue
                here = values[thinned][:, [columns.index(column) for column in shared]]
                nrmse, _ = pair_scores(here, stored[shared].to_numpy(dtype=float))
                if np.any(np.isfinite(nrmse)):
                    compared += 1
                    at = int(np.nanargmax(nrmse))
                    if float(nrmse[at]) >= worst:
                        worst, pair = (
                            float(nrmse[at]),
                            f"{shared[at]} ({name} here, {corpus_engine} in the corpus, {job})",
                        )
    return {
        "local_vs_corpus_nrmse": worst if compared else float("nan"),
        "local_vs_corpus_worst": pair,
        "local_vs_corpus_pairs": compared,
    }


def copasi_steady_state(sbml: Path, duration: float = 5000.0) -> tuple[dict[str, float], float]:
    """Where COPASI takes an SBML model from the initial state of its file: the concentrations after ``duration``,
    by SBML identifier, and the largest relative change of any of them over the last tenth of that time.

    A second engine on the untouched file: what the rest state of a world, found with Tellurium inside the
    composite, is held against.
    """
    import basico

    basico.load_model(str(sbml))
    identifiers = dict(zip(basico.get_species().index, basico.get_species()["sbml_id"], strict=True))
    course = basico.run_time_course(
        duration=duration, intervals=10, use_initial_values=True, a_tol=1e-12, r_tol=1e-10
    )
    last, before = course.iloc[-1], course.iloc[-2]
    state = {identifiers[name]: float(last[name]) for name in course.columns if name in identifiers}
    moving = max(
        abs(float(last[name]) - float(before[name])) / (abs(float(last[name])) + 1e-12)
        for name in course.columns
        if name in identifiers
    )
    return state, float(moving)


def run(plan_path: str | Path | None = None, out: Path = OUT) -> list[dict]:
    plan = load_plan(plan_path)
    paths.use_viva_biomodels()
    from viva_biomodels.composites.compare_simulators import run_comparison

    paths.MODELS.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(paths.PROJECT):  # the loader of viva-biomodels keeps the SBML files under ./models
        results = run_comparison(list(plan["receivers"]), simulators="all")
    scores = {}
    selection = paths.RUNS / "selection" / "corpus_scores.csv"
    if selection.exists():
        with selection.open(encoding="utf-8", newline="") as handle:
            scores = {row["biomodel_id"]: row for row in csv.DictReader(handle)}

    rows = []
    for biomodel, result in results.items():
        directory = out / biomodel
        directory.mkdir(parents=True, exist_ok=True)
        if result["error"]:
            rows.append({"biomodel": biomodel, "status": result["error"]})
            continue
        comparison, engines = result["comparison"], result["engines"]
        (directory / "comparison.json").write_text(json.dumps(comparison, indent=1) + "\n", encoding="utf-8")
        for name, engine in engines.items():
            times, columns, values = _table(engine)
            with (directory / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", *columns])
                for at, line in zip(times, values, strict=True):
                    writer.writerow([repr(float(at)), *(repr(float(value)) for value in line)])
        worst_variable, worst_name = 0.0, ""
        for first, second in itertools.combinations(sorted(engines), 2):
            _, columns_a, values_a = _table(engines[first])
            _, columns_b, values_b = _table(engines[second])
            shared = [column for column in columns_a if column in columns_b]
            nrmse, _ = pair_scores(
                values_a[:, [columns_a.index(c) for c in shared]],
                values_b[:, [columns_b.index(c) for c in shared]],
            )
            at = int(np.nanargmax(nrmse))
            if float(nrmse[at]) >= worst_variable:
                worst_variable, worst_name = float(nrmse[at]), f"{shared[at]} ({first}, {second})"
        times, _, _ = _table(next(iter(engines.values())))
        corpus_row = scores.get(biomodel, {})
        rows.append(
            {
                "biomodel": biomodel,
                "status": "ok",
                "engines": ",".join(comparison["engines"]),
                "points": len(times),
                "t_end": float(times[-1]),
                "max_pair_mean_nrmse": comparison["max_nrmse"],
                "worst_pair": "__".join(comparison["worst_pair"]),
                "bucket": comparison["bucket"],
                "max_closeness": comparison["max_score"],
                "closeness_bucket": comparison["closeness_bucket"],
                "max_variable_nrmse": worst_variable,
                "worst_variable": worst_name,
                **against_corpus(biomodel, engines),
                "corpus_engines": corpus_row.get("engines", ""),
                "corpus_max_pair_mean_nrmse": corpus_row.get("max_pair_mean_nrmse", ""),
                "corpus_max_variable_nrmse": corpus_row.get("max_variable_nrmse", ""),
                "corpus_max_closeness": corpus_row.get("max_closeness", ""),
            }
        )
    out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows
