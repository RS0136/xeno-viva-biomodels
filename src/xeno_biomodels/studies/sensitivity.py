"""How the readings depend on what the world assumes.

Everything around an SBML model is assumed: the dilution rate, the density of the organism, every number of the
sender, the law by which the organism grows, how long a run is followed, and, in the community, how much of the
other pairs' nutrients a xeno takes. Each is changed here, one at a time, and the worlds are run and measured
again, layer by layer.

    worlds       every receiver with the design that shares nothing (xeno) and with a control (leak-1e-2):
                 each assumed number at a quarter, a half, twice and four times its value, two other growth
                 laws, and the horizon at twice and four times its length
    community    the design partner-only with the cross specificity from 0.001 to 0.1, and all three designs at
                 half and at twice the dilution rate

Written to ``runs/studies/sensitivity/``: ``worlds.csv`` and ``community.csv``, one row per world or per ordered
pair of organisms with the variant that it belongs to, and ``worlds_layers.csv`` and ``community_layers.csv``,
one row per group of protected quantities.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from xeno_biomodels import community
from xeno_biomodels.experiment import prune, run_variants
from xeno_biomodels.studies.common import STUDIES, changed, slug, write_rows
from xeno_biomodels.world import WorldError

OUT = STUDIES / "sensitivity"
DESIGNS = ("xeno", "leak-1e-2")
FACTORS = (0.25, 0.5, 2.0, 4.0)
PARAMETERS = {
    "dilution rate": "supply.dilution",
    "density of the receiver": "receivers.*.biovolume",
    "sender: substrate in the feed": "sender.feed",
    "sender: capacity V": "sender.vmax",
    "sender: half-saturation K": "sender.km",
    "sender: yield": "sender.growth_yield",
    "sender: maintenance": "sender.maintenance",
}
HORIZONS = (2.0, 4.0)
CROSS = (0.001, 0.003, 0.01, 0.03, 0.1)
COMMUNITY_DILUTION = (0.5, 2.0)


def variants(plan: dict) -> dict[str, tuple[dict, dict]]:
    """Name of a variant -> (what it changes, its plan)."""
    out = {"reference": ({"parameter": "reference", "factor": 1.0}, plan)}
    for label, path in PARAMETERS.items():
        for factor in FACTORS:
            tag = {"parameter": label, "factor": factor}
            out[f"{label} x{factor:g}"] = (tag, changed(plan, {path: lambda value, f=factor: value * f}))
    dilution = float(plan["supply"]["dilution"])
    out["growth with maintenance"] = (
        {"parameter": "growth law: a maintenance equal to the dilution rate", "factor": ""},
        changed(plan, {"receivers.*.maintenance": dilution}),
    )
    out["density held"] = (
        {"parameter": "growth law: none, the density of the receiver held", "factor": ""},
        changed(plan, {"receivers.*.growth": False}),
    )
    for factor in HORIZONS:
        tag = {"parameter": "horizon of a run", "factor": factor}
        out[f"horizon x{factor:g}"] = (
            tag,
            changed(plan, {"time.horizon": lambda value, f=factor: value * f}),
        )
    return out


def worlds(plan: dict, jobs: int = 1, out: Path = OUT) -> list[dict]:
    work = []
    for name, (tag, variant) in variants(plan).items():
        for biomodel in plan["receivers"]:
            for design in DESIGNS:
                tagged = {"variant": name, **tag}
                work.append((tagged, variant, biomodel, design, out / "worlds" / slug(name), False, False))
    rows, by_group = run_variants(work, jobs)
    write_rows(out / "worlds.csv", rows)
    write_rows(out / "worlds_layers.csv", by_group)
    return rows


def communities(plan: dict, jobs: int = 1, out: Path = OUT) -> list[dict]:
    """The community under other assumptions: the cross specificity swept, the dilution rate halved and doubled."""
    cases = []
    for value in CROSS:
        designs = {"partner-only": {"substrate": "common", "cross": value}}
        tag = {"parameter": "cross specificity", "value": value}
        cases.append(
            (f"cross={value:g}", tag, changed(plan, {"community.designs": designs}), ["partner-only"])
        )
    for factor in COMMUNITY_DILUTION:
        variant = changed(plan, {"supply.dilution": lambda value, f=factor: value * f})
        tag = {"parameter": "dilution rate", "value": factor}
        cases.append((f"dilution x{factor:g}", tag, variant, list(plan["community"]["designs"])))
    rows, by_group = [], []
    for name, tag, variant, designs in cases:
        for design in designs:
            directory = out / "community" / slug(name)
            tagged = {"variant": name, **tag}
            try:
                community.simulate(variant, design, directory, jobs, fate=False)
                found, groups = community.read_matrix(variant, design, directory)
            except (WorldError, FileNotFoundError, ValueError, RuntimeError) as exc:
                rows.append({**tagged, "design": design, "status": f"{type(exc).__name__}: {str(exc)[:200]}"})
                continue
            finally:
                # the measure of every ordered pair is kept in the tables; the series are not
                for kind in ("series.csv", "protected.csv", "partial.csv"):
                    for series in (directory / design).glob(f"*/{kind}"):
                        series.unlink()
                shutil.rmtree(directory / design / "measure", ignore_errors=True)
                if (directory / design).exists():
                    prune(directory / design)
            rows += [{**tagged, **row} for row in found]
            by_group += [{**tagged, **row} for row in groups]
            print(f"sensitivity: community {name}, {design}", flush=True)
            write_rows(out / "community.csv", rows)
            write_rows(out / "community_layers.csv", by_group)
    write_rows(out / "community.csv", rows)
    write_rows(out / "community_layers.csv", by_group)
    return rows


def run(plan: dict, jobs: int = 1, analyze_only: bool = False) -> list[Path]:
    written = [
        OUT / "worlds.csv",
        OUT / "worlds_layers.csv",
        OUT / "community.csv",
        OUT / "community_layers.csv",
    ]
    if analyze_only:  # the series of these worlds are not kept: there is nothing to measure again
        return written
    worlds(plan, jobs)
    communities(plan, jobs)
    return written
