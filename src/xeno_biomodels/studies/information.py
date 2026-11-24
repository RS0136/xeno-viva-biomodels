"""What the receiver says of the sender, in bits, when it is read with noise.

The third criterion of the definition calls a sender orthogonal when the protected quantities of the receiver,
read with noise, say next to nothing about which activity the sender chose. The measure gives two bounds of that
information. Here it is computed itself, and the noise is varied.

A run is a point x_u in the space of all stored values of all protected quantities, in tolerances (the baseline
is the origin). With independent Gaussian noise of standard deviation sigma on every value and a uniform choice
among the n admissible runs, the information is

    I = - (1/n) sum_i  E_z  log2 [ (1/n) sum_j exp( -(|x_i - x_j|^2 + 2 sigma z.(x_i - x_j)) / (2 sigma^2) ) ]

and depends on the runs only through their distances: z.(x_i - x_j) is Gaussian with the covariance
(x_i - x_j).(x_i - x_k), so the expectation is taken by sampling n numbers, not n times the length of a run.

    bits        every world of the plan: the information at the noise of the plan, between its two bounds, and
                the noise at which half of what can be said (log2 n bits) is still said
    noise       the information against the noise, from a hundredth of a tolerance to a thousand tolerances
    readings    the same with fewer stored times: every 2nd, 4th, ... is read, and the noise is the same

A world in which a run loses the receiver has no bits (the definition gives none): its row says so.

Written to ``runs/studies/information/``: ``bits.csv``, ``noise.csv`` and ``readings.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xeno_biomodels import experiment
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.studies.common import STUDIES, leak_of, read_rows, write_rows
from xeno_biomodels.world import BASELINE

OUT = STUDIES / "information"
SAMPLES = 40000  # draws of the noise for every run
NOISES = np.geomspace(0.01, 1000.0, 51)  # standard deviations of the noise, in tolerances
THINNED = (1, 2, 4, 8, 16, 32, 64)  # every how many stored times is read


def information(points: np.ndarray, sigma: float, samples: int = SAMPLES, seed: int = 0) -> float:
    """The information, in bits, that an observation of one of the rows of ``points`` (chosen uniformly) with
    Gaussian noise ``sigma`` on every component carries about which row it was."""
    n = points.shape[0]
    gram = points @ points.T
    random = np.random.default_rng(seed)
    total = 0.0
    for i in range(n):
        # the differences x_i - x_j: their squared lengths and their covariance under the noise
        cross = gram[i, i] - gram[i][:, None] - gram[i][None, :] + gram
        cross = 0.5 * (cross + cross.T)
        squared = np.clip(np.diag(cross), 0.0, None)
        values, vectors = np.linalg.eigh(cross)
        root = vectors * np.sqrt(np.clip(values, 0.0, None))
        along = random.standard_normal((samples, n)) @ root.T  # z.(x_i - x_j) for every j, per draw
        exponent = -(squared[None, :] + 2.0 * sigma * along) / (2.0 * sigma**2)
        top = np.max(exponent, axis=1, keepdims=True)
        mixed = top[:, 0] + np.log(np.mean(np.exp(exponent - top), axis=1))
        total += -float(np.mean(mixed)) / np.log(2.0)
    return total / n


def points_of(directory: Path, plan: dict, every: int = 1) -> tuple[np.ndarray | None, list[str], str]:
    """The admissible runs of a world as points in tolerances (the baseline first), the names of the runs, and
    why there are none where there are none."""
    result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
    if result["losses"]:
        return None, [], "a run loses the receiver"
    left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
    delta = {row["column"]: float(row["delta"]) for row in read_rows(directory / "measure" / "deltas.csv")}
    _, base = read_series(directory / BASELINE / "protected.csv")
    names, rows = [BASELINE], [np.zeros(sum(base[key][::every].size for key in delta))]
    for name in experiment.run_names(plan):
        path = directory / name / "protected.csv"
        if name in left_out or not path.exists():
            continue
        _, columns = read_series(path)
        rows.append(np.concatenate([(columns[key] - base[key])[::every] / delta[key] for key in delta]))
        names.append(name)
    return np.vstack(rows), names, ""


def half(points: np.ndarray) -> float | None:
    """The noise at which half of log2 n bits are still said (bisection on a coarse sample); None where nothing
    is said at any noise."""
    most = np.log2(points.shape[0])
    if not np.any(points):
        return None
    low, high = 1e-6, 1e9
    for _ in range(60):
        middle = float(np.sqrt(low * high))
        if information(points, middle, samples=4000) > 0.5 * most:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def run(
    plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT, runs: Path = experiment.OUT
) -> list[Path]:
    del jobs, analyze_only  # the runs are there
    sigma = float(plan["measure"]["sigma"])
    summary = {(row["biomodel"], row["design"]): row for row in read_rows(runs / "summary.csv")}
    bits, noise, readings = [], [], []
    for biomodel in plan["receivers"]:
        for design, written in plan["designs"].items():
            directory = runs / biomodel / design
            if not (directory / "measure" / "result.json").exists():
                continue
            head = {"biomodel": biomodel, "design": design, "leak": leak_of(written)}
            points, names, why = points_of(directory, plan)
            found = summary[(biomodel, design)]
            if points is None:
                bits.append({**head, "runs": 0, "note": why})
                continue
            row = {
                **head,
                "runs": len(names),
                "values_read": points.shape[1],
                "most_that_can_be_said": float(np.log2(len(names))),
                "bits": information(points, sigma),
                "bits_lower_bound": float(found["bits_lower"]),
                "bits_upper_bound": float(found["bits_upper"]),
                "noise_at_half": half(points),
                "longest_run_in_tolerances": float(np.max(np.linalg.norm(points, axis=1))),
                "D": float(found["D"]),
                "note": "",
            }
            bits.append(row)
            for level in NOISES:
                noise.append({**head, "sigma": float(level), "bits": information(points, float(level), 8000)})
            for every in THINNED:
                thinned, _, _ = points_of(directory, plan, every)
                readings.append(
                    {
                        **head,
                        "every": every,
                        "values_read": thinned.shape[1],
                        "bits": information(thinned, sigma, 8000),
                    }
                )
    return [
        write_rows(out / "bits.csv", bits),
        write_rows(out / "noise.csv", noise),
        write_rows(out / "readings.csv", readings),
    ]
