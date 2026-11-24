"""How well the simulators agree on each model of the viva-biomodels corpus, and which models to take.

viva-biomodels commits the time course of every BioModel under up to five independent ODE engines (amici,
copasi, pysces, simbio, tellurium) in ``datasets/corpus_all_engines``. Its investigation states that this
dataset establishes coverage and that agreement is "not yet quantified corpus-wide". This module quantifies
it, with the two measures that viva-biomodels itself uses for a pair of engines (``viva_biomodels.comparison``):

    nRMSE of a variable    RMSE of the two series over the largest absolute value in either
    mean nRMSE of a pair   the mean over the variables that both engines report; "good" at 0.01 or less
    closeness of a pair    max over variables and times of |a - b| / (atol + rtol |b|), with
                           atol = max(1e-3, 1e-5 max|a|, 1e-5 max|b|) and rtol = 1e-4; "close" at 1 or less

A model is scored by its worst pair. Two more numbers are kept, because a mean over variables hides a single
variable that disagrees and counts constants (compartments, fixed parameters) as agreement: the largest nRMSE
of any variable in any pair, and the number of variables that move.

What the score is: agreement between independent implementations on the same SBML file. It is not agreement
with an experiment. The committed series are float32 and thinned to 100 points, so nothing below about 1e-7
is resolved.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from xeno_biomodels import paths

ENGINES = ("amici", "copasi", "pysces", "simbio", "tellurium")
GOOD = 0.01  # the threshold of the bucket "good" in viva_biomodels.comparison
#: A variable moves when its range is more than this share of its largest absolute value.
MOVES = 1e-6


def pair_scores(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """nRMSE and closeness of every variable, for two arrays (time, variable) on the same times.

    The nRMSE of a variable that is zero in both engines is NaN, as in viva-biomodels, which leaves such a
    variable out of the mean.
    """
    difference = a - b
    rmse = np.sqrt(np.mean(difference**2, axis=0))
    largest_a, largest_b = np.max(np.abs(a), axis=0), np.max(np.abs(b), axis=0)
    scale = np.maximum(largest_a, largest_b)
    with np.errstate(invalid="ignore", divide="ignore"):
        nrmse = np.where(scale > 0.0, rmse / scale, np.nan)
    atol = np.maximum(1e-3, np.maximum(1e-5 * largest_a, 1e-5 * largest_b))
    closeness = np.max(np.abs(difference) / (atol + 1e-4 * np.abs(b)), axis=0)
    return nrmse, closeness


def score_job(frame: pd.DataFrame) -> dict | None:
    """The agreement of the engines on one job of one model. ``frame`` has engine, variable, time, value."""
    series: dict[str, pd.DataFrame] = {}
    for engine, part in frame.groupby("engine", observed=True):
        table = part.pivot_table(
            index="time", columns="variable", values="value", aggfunc="first", observed=True
        )
        table = table.dropna(axis=1, how="any")
        if table.shape[0] >= 2 and table.shape[1] >= 1 and np.all(np.isfinite(table.to_numpy())):
            series[str(engine)] = table
    if len(series) < 2:
        return None
    worst_mean, worst_variable, worst_closeness = 0.0, 0.0, 0.0
    worst_pair, worst_name, least_shared, least_moving = "", "", None, None
    for first, second in itertools.combinations(sorted(series), 2):
        a, b = series[first], series[second]
        shared = [name for name in a.columns if name in b.columns]
        if not shared or len(a) != len(b) or not np.allclose(a.index, b.index, rtol=1e-5, atol=1e-9):
            return None  # the engines were not stored on the same times: not comparable point by point
        x, y = a[shared].to_numpy(dtype=np.float64), b[shared].to_numpy(dtype=np.float64)
        nrmse, closeness = pair_scores(x, y)
        if not np.any(np.isfinite(nrmse)):
            return None
        scale = np.maximum(np.max(np.abs(x), axis=0), 1e-300)
        moving = int(np.sum((np.max(x, axis=0) - np.min(x, axis=0)) / scale > MOVES))
        mean = float(np.nanmean(nrmse))
        if mean >= worst_mean:
            worst_mean, worst_pair = mean, f"{first}__{second}"
        at = int(np.nanargmax(nrmse))
        if float(nrmse[at]) >= worst_variable:
            worst_variable, worst_name = float(nrmse[at]), f"{shared[at]} ({first}, {second})"
        worst_closeness = max(worst_closeness, float(np.max(closeness)))
        least_shared = len(shared) if least_shared is None else min(least_shared, len(shared))
        least_moving = moving if least_moving is None else min(least_moving, moving)
    lengths = {len(table) for table in series.values()}
    return {
        "engines": ",".join(sorted(series)),
        "n_engines": len(series),
        "n_points": min(lengths),
        "t_end": float(max(table.index.max() for table in series.values())),
        "shared_min": least_shared,
        "moving_min": least_moving,
        "max_pair_mean_nrmse": worst_mean,
        "worst_pair": worst_pair,
        "max_variable_nrmse": worst_variable,
        "worst_variable": worst_name,
        "max_closeness": worst_closeness,
    }


def score_corpus(directory: Path | None = None) -> pd.DataFrame:
    """One row per model and job of the corpus that two or more engines ran, with the worst pair of each."""
    directory = Path(directory or paths.CORPUS)
    table = pq.read_table(
        directory / "corpus_timecourse_all_engines.parquet",
        read_dictionary=["biomodel_id", "job", "engine", "variable"],
    )
    frame = table.to_pandas()
    rows = []
    for (model, job), part in frame.groupby(["biomodel_id", "job"], observed=True, sort=True):
        row = score_job(part)
        if row is not None:
            rows.append({"biomodel_id": str(model), "job": str(job), **row})
    scores = pd.DataFrame(rows)
    scores["good"] = scores["max_pair_mean_nrmse"] <= GOOD
    scores["close"] = scores["max_closeness"] <= 1.0
    return scores.sort_values(["max_variable_nrmse", "biomodel_id"]).reset_index(drop=True)


def candidates(
    scores: pd.DataFrame, *, engines: int = len(ENGINES), least_moving: int = 5, variable_nrmse: float = 1e-5
) -> pd.DataFrame:
    """The plateau: the models on which ``engines`` engines agree to the tolerance of their solvers. Every pair is
    good and close, and no variable of any pair is off by more than ``variable_nrmse`` of its size, on at least
    ``least_moving`` variables that move."""
    keep = (
        (scores["n_engines"] >= engines)
        & scores["good"]
        & scores["close"]
        & (scores["max_variable_nrmse"] <= variable_nrmse)
        & (scores["moving_min"] >= least_moving)
    )
    return scores[keep].reset_index(drop=True)


def names_of(identifiers: list[str], cache: Path) -> dict[str, dict]:
    """Name and publication of each BioModel, from the REST interface of BioModels; kept in ``cache``."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    known = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}

    def fetch(identifier: str) -> tuple[str, dict | None]:
        request = urllib.request.Request(
            f"https://www.ebi.ac.uk/biomodels/{identifier}?format=json",
            headers={"Accept": "application/json", "User-Agent": "python-urllib"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                record = json.load(response)
        except (OSError, ValueError):  # no network, or the service is down: the name is left out
            return identifier, None
        publication = record.get("publication") or {}
        fields = ("title", "link", "year", "journal")
        return identifier, {"name": record.get("name"), **{key: publication.get(key) for key in fields}}

    missing = [identifier for identifier in identifiers if identifier not in known]
    if missing:
        with ThreadPoolExecutor(6) as pool:
            for identifier, record in pool.map(fetch, missing):
                if record is not None:
                    known[identifier] = record
        cache.write_text(json.dumps(known, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return known


def write(out: Path | None = None, plan_path: Path | None = None) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Score the corpus and write ``corpus_scores.csv``, ``candidates.csv`` (the plateau) and ``summary.json``."""
    from xeno_biomodels.world import load_plan

    selection = load_plan(plan_path)["selection"]
    out = Path(out or paths.RUNS / "selection")
    out.mkdir(parents=True, exist_ok=True)
    scores = score_corpus()
    scores.to_csv(out / "corpus_scores.csv", index=False)
    chosen = candidates(
        scores,
        engines=int(selection["engines"]),
        least_moving=int(selection["moving_variables"]),
        variable_nrmse=float(selection["variable_nrmse"]),
    )
    names = names_of(sorted(set(chosen["biomodel_id"])), out / "names.json")
    chosen = chosen.assign(
        name=[(names.get(identifier) or {}).get("name") or "" for identifier in chosen["biomodel_id"]]
    )
    chosen.to_csv(out / "candidates.csv", index=False)
    index = json.loads((paths.CORPUS / "index.json").read_text(encoding="utf-8"))
    five = scores["n_engines"] == len(ENGINES)
    summary = {
        "corpus_models": int(index["n_models"]),
        "models_scored": int(scores["biomodel_id"].nunique()),
        "jobs_scored": len(scores),
        "jobs_run_by_five_engines": int(five.sum()),
        "of_these_good_and_close_in_every_pair": int((five & scores["good"] & scores["close"]).sum()),
        "plateau_jobs": len(chosen),
        "plateau_models": int(chosen["biomodel_id"].nunique()),
        "plateau_is": selection,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return out, scores, chosen
