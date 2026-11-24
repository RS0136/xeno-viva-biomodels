"""What the studies share: plans that differ from the plan in one thing, and tables."""

from __future__ import annotations

import copy
import csv
import re
from pathlib import Path

from xeno_biomodels import paths

STUDIES = paths.RUNS / "studies"
#: the groups of protected quantities that are one kind at one layer, in the order in which they are shown
LINES = (
    ("medium", "common"),
    ("medium", "specific"),
    ("particle", "common"),
    ("molecule", "common"),
    ("molecule", "specific"),
    ("complex", "common"),
    ("complex", "specific"),
    ("individual", "common"),
    ("population", "common"),
)


def changed(plan: dict, changes: dict) -> dict:
    """A copy of the plan with values replaced.

    A key of ``changes`` is a path with dots (``supply.dilution``); ``*`` stands for every key at its level
    (``receivers.*.biovolume``). A value that can be called is applied to the value that is there.
    """
    out = copy.deepcopy(plan)
    for path, value in changes.items():
        _set(out, path.split("."), value)
    return out


def _set(node: dict, keys: list[str], value) -> None:
    key, rest = keys[0], keys[1:]
    for target in list(node) if key == "*" else [key]:
        if rest:
            _set(node.setdefault(target, {}), rest, value)
        else:
            node[target] = value(node.get(target)) if callable(value) else value


def slug(text: str) -> str:
    """The name of a variant as the name of a directory."""
    return re.sub(r"[^A-Za-z0-9.=+-]+", "_", text).strip("_")


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_rows(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def leak_of(written) -> float:
    """The leak of a design as one number: the specificity of the transporter of the sender for the nutrient, or,
    for a leak that is written as a share of what the receiver takes, that share."""
    return float(written["share"]) if isinstance(written, dict) else float(written)


def number(text) -> float | None:
    """A cell of a table as a number, or None where it holds none."""
    if text in (None, "", "None", "nan"):
        return None
    return float(text)


def truth(text) -> bool:
    """A cell of a table as yes or no."""
    return str(text) == "True"


__all__ = ["LINES", "STUDIES", "changed", "number", "read_rows", "slug", "truth", "write_rows"]
