"""Where the interference enters, and what the layers then say.

In every world of the plan the sender can reach the receiver through one species only, its nutrient, so the
interference always enters at the medium and runs inward by the same way. Here each receiver has a second way
in: another species of the medium that its model reads (``studies.channels.other`` in the plan). The world lets
that species vary as it lets the nutrient vary, the receiver protects its level as a quantity of its own, and
the sender takes a share of the nutrient, or of the other species, or of neither:

    xeno             the sender takes nothing of the receiver's
    nutrient-<f>     at rest it takes the share f of what the receiver takes of its nutrient
    other-<f>        at rest it takes the share f of what the receiver takes, or makes, of the other species

The leak is written in the units of no model (``World._sender_config``): the transporter is half-saturated at the
level of the SBML file, and the species is worth nothing to the sender. With the same share on both ways in, what
differs between the two is where the interference enters, and the layers are read for that.

Written to ``runs/studies/channels/``: ``worlds.csv``, ``layers.csv``, and under ``worlds/`` what the measure read
of every run (``protected.csv``), which the study ``cascade`` reads.
"""

from __future__ import annotations

from pathlib import Path

from xeno_biomodels.experiment import run_variants
from xeno_biomodels.studies.common import STUDIES, changed, write_rows

OUT = STUDIES / "channels"


def variants(plan: dict) -> list[tuple[dict, dict, str, str]]:
    """(what the world is, its plan, receiver, design) for every world of the study."""
    settings = plan["studies"]["channels"]
    out = []
    for biomodel, other in settings["other"].items():
        nutrient, second = plan["receivers"][biomodel]["nutrient"], other["species"]
        designs: dict = {"xeno": 0.0}
        tags = {"xeno": {"channel": "none", "species": "", "share": 0.0}}
        for share in settings["shares"]:
            designs[f"nutrient-{share:g}"] = {"share": float(share)}
            tags[f"nutrient-{share:g}"] = {"channel": "nutrient", "species": nutrient, "share": float(share)}
            designs[f"other-{share:g}"] = {"share": float(share), "of": second}
            tags[f"other-{share:g}"] = {"channel": "other", "species": second, "share": float(share)}
        variant = changed(
            plan,
            {
                f"receivers.{biomodel}.medium": [nutrient, second],
                "designs": designs,
                f"protect.specific.{biomodel}.medium": f"medium.{second}",
            },
        )
        out += [(tags[design], variant, biomodel, design) for design in designs]
    return out


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    written = [out / "worlds.csv", out / "layers.csv"]
    work = [
        (tag, variant, biomodel, design, out / "worlds", analyze_only, "protected")
        for tag, variant, biomodel, design in variants(plan)
    ]
    if analyze_only:  # the series are not kept: what is there is what was measured
        return written
    rows, by_group = run_variants(work, jobs)
    write_rows(out / "worlds.csv", rows)
    write_rows(out / "layers.csv", by_group)
    return written
