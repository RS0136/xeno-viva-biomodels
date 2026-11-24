"""What an organism protects: quantities of two kinds, layer by layer.

Two kinds of quantity are protected (``protect`` in the plan):

    common      the same quantity for every organism, one at each layer, so that organisms can be set side by side
    specific    what is characteristic of one organism: the pool and the step by which its metabolism is known

Each belongs to a layer of the organism. The names are those of the levels of a cell; what an SBML model of
metabolism holds at each is on the right:

    medium       what the organism lives on, in the vessel
    particle     energy conversion: the ATP that it makes
    molecule     its metabolites
    complex      its enzymes at work, and the states of its proteins where the model has them
    individual   its growth
    population   its density

Such a model has nothing at the levels of the atom, the supramolecule (a genome), the condensate or the
organelle, and nothing can be read there. An organism of laws has no carriers and no pools of metabolites, so it
has no ``particle`` and no ``molecule`` either, and an SBML model in which the plan names no ATP has no
``particle``.

Two of the common quantities are not columns of a run and are derived from them here, by one rule for every SBML
model:

    atp_made    ATP made per cell volume and time: over the reactions of the model, the net coefficient of ATP
                times the rate, where that is positive
    pool        the sum of the pools of the metabolites that are neither carriers nor states of a protein

The measure is then read once for all that an organism protects and, with the same tolerances, for every layer,
for each kind, and for each kind at each layer (``orthogonality.measure`` with groups).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import libsbml
import numpy as np

from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.world import WorldError, sbml_file, write_series

LAYERS = ("medium", "particle", "molecule", "complex", "individual", "population")
KINDS = ("common", "specific")
#: what an SBML model of metabolism holds at each layer
WHAT = {
    "medium": "what the organism lives on, in the vessel",
    "particle": "energy conversion: the ATP that it makes",
    "molecule": "its metabolites",
    "complex": "its enzymes at work, and the states of its proteins where the model has them",
    "individual": "its growth",
    "population": "its density",
}


@dataclass(frozen=True)
class Organism:
    """What the reading has to know of an organism."""

    name: str  # the beginning of its columns
    kind: str  # sbml or xeno
    key: str  # where its specific quantities are in the plan: its BioModel, or "xeno"
    words: dict[str, str] = field(compare=False)  # what a pattern of the plan may name


def sbml_organism(plan: dict, biomodel: str, name: str, food: str | None = None) -> Organism:
    """An SBML organism under ``name``; ``food`` is the name of its nutrient in the medium, where that is not the
    name of the species in the model."""
    nutrient = plan["receivers"][biomodel]["nutrient"]
    words = {"name": name, "food": food or nutrient, "uptake": f"{name}.uptake.{nutrient}"}
    return Organism(name, "sbml", biomodel, words)


def xeno_organism(name: str, substrate: str, product: str) -> Organism:
    """An organism of laws under ``name``."""
    words = {"name": name, "food": substrate, "uptake": f"{name}.uptake", "product": product}
    return Organism(name, "xeno", "xeno", words)


@dataclass(frozen=True)
class Entry:
    """One protected quantity: its column, how its tolerance is written, its layer and its kind."""

    column: str
    tolerance: str
    layer: str
    kind: str

    @property
    def spec(self) -> str:
        return f"{self.column}={self.tolerance}"


def layers_of(plan: dict, organism: Organism) -> tuple[str, ...]:
    """The layers that an organism has: an organism of laws those that the plan gives it, an SBML organism all
    of them, but for ``particle`` where the plan names no ATP in its model."""
    if organism.kind != "sbml":
        return tuple(plan["protect"]["xeno_layers"])
    atp = plan["receivers"][organism.key].get("atp")
    return tuple(layer for layer in LAYERS if layer != "particle" or atp)


def protected(plan: dict, organism: Organism) -> list[Entry]:
    """What an organism protects, from ``protect`` of the plan: the common quantities of the layers that it has,
    then those that are specific to it."""
    settings = plan["protect"]
    has = layers_of(plan, organism)
    out: list[Entry] = []
    tables = (
        ("common", settings["common"]),
        ("specific", (settings.get("specific") or {}).get(organism.key) or {}),
    )
    for kind, table in tables:
        for layer, patterns in table.items():
            if layer not in LAYERS:
                raise WorldError(f"protect.{kind}: no layer {layer}; the layers are {', '.join(LAYERS)}")
            if layer not in has:
                if kind == "specific":
                    raise WorldError(f"protect.specific.{organism.key}: this organism has no layer {layer}")
                continue  # an organism protects the common quantity of a layer only if it has the layer
            for pattern in [patterns] if isinstance(patterns, str) else patterns:
                try:
                    column = pattern.format(**organism.words)
                except KeyError as exc:
                    raise WorldError(
                        f"protect.{kind}.{layer}: {pattern!r} names {exc}, which {organism.name} has not"
                    ) from None
                tolerance = "fn" if pattern in settings["from_function"] else str(settings["tolerance"])
                out.append(Entry(column, tolerance, layer, kind))
    columns = [entry.column for entry in out]
    twice = sorted({column for column in columns if columns.count(column) > 1})
    if twice:
        raise WorldError(f"{organism.name} protects {twice} twice")
    return out


def groups(entries: list[Entry]) -> dict[str, list[str]]:
    """The groups in which the measure is read besides the whole: every layer, each kind, each kind at each layer."""
    out: dict[str, list[str]] = {}
    for layer in LAYERS:
        for kind in KINDS:
            found = [entry.column for entry in entries if entry.layer == layer and entry.kind == kind]
            if found:
                out.setdefault(f"layer:{layer}", []).extend(found)
                out[f"{layer}/{kind}"] = found
    for kind in KINDS:
        found = [entry.column for entry in entries if entry.kind == kind]
        if found:
            out[f"kind:{kind}"] = found
    return out


def split(label: str) -> tuple[str, str]:
    """(layer, kind) of a group; ``all`` where the group is every layer or both kinds."""
    if label.startswith("layer:"):
        return label[6:], "all"
    if label.startswith("kind:"):
        return "all", label[5:]
    layer, _, kind = label.partition("/")
    return layer, kind


# ---- the quantities that are derived from the columns of a run -----------------------------------------------------


@cache
def atp_coefficients(sbml: Path, atp: str) -> dict[str, float]:
    """Reaction -> net coefficient of ATP (what the reaction forms less what it uses), for those that have one."""
    model = libsbml.readSBMLFromFile(str(sbml)).getModel()
    if model.getSpecies(atp) is None:
        raise WorldError(f"{sbml}: no species {atp}")
    out = {}
    for reaction in model.getListOfReactions():
        net = sum(ref.getStoichiometry() for ref in reaction.getListOfProducts() if ref.getSpecies() == atp)
        net -= sum(ref.getStoichiometry() for ref in reaction.getListOfReactants() if ref.getSpecies() == atp)
        if net != 0.0:
            out[reaction.getId()] = float(net)
    if not any(value > 0.0 for value in out.values()):
        raise WorldError(f"{sbml}: no reaction forms {atp}")
    return out


#: names by which a carrier of energy or of electrons is known in an SBML file
CARRIERS = ("ATP", "ADP", "AMP", "NAD", "NADH", "NADP", "NADPH")


def annotate(sbml: Path) -> dict:
    """What the plan says by hand of a receiver, read from the SBML file by rule instead: ``atp``, the species
    whose identifier or name is ATP, if the model integrates it and a reaction forms it, and ``carriers``, the
    integrated species whose identifier or name is one of ``CARRIERS``. No species counts as a state of a protein:
    a file does not say which are."""
    model = libsbml.readSBMLFromFile(str(sbml)).getModel()
    ruled = {rule.getVariable() for rule in model.getListOfRules()}
    known = {}
    for item in model.getListOfSpecies():
        if item.getBoundaryCondition() or item.getConstant() or item.getId() in ruled:
            continue
        for label in (item.getId(), item.getName()):
            if label.upper() in CARRIERS:
                known[item.getId()] = label.upper()
                break
    atp = next((name for name, label in known.items() if label == "ATP"), None)
    if atp is not None:
        try:
            atp_coefficients(sbml, atp)
        except WorldError:
            atp = None
    return {"atp": atp, "carriers": sorted(known), "proteins": []}


def derive(plan: dict, organism: Organism, columns: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The derived quantities of an organism along a run: ``atp_made`` and ``pool`` of an SBML organism."""
    if organism.kind != "sbml":
        return {}
    receiver, name = plan["receivers"][organism.key], organism.name
    out = {}
    if receiver.get("atp"):
        made = None
        sbml = sbml_file(receiver.get("biomodel", organism.key))
        for reaction, coefficient in atp_coefficients(sbml, receiver["atp"]).items():
            formed = np.maximum(coefficient * columns[f"{name}.rates.{reaction}"], 0.0)
            made = formed if made is None else made + formed
        out[f"{name}.atp_made"] = made
    apart = {*(receiver.get("carriers") or []), *(receiver.get("proteins") or [])}
    start = f"{name}.species."
    metabolites = [
        column for column in columns if column.startswith(start) and column[len(start) :] not in apart
    ]
    if not metabolites:
        raise WorldError(
            f"{organism.key}: no metabolite is left for the pool once carriers and proteins are set apart"
        )
    out[f"{name}.pool"] = np.sum([columns[column] for column in metabolites], axis=0)
    return out


def write_protected(
    series: Path, target: Path, plan: dict, organisms: list[Organism], more: list[str]
) -> Path:
    """From the stored columns of a run, the file that the measure reads: what the organisms protect, derived
    quantities among it, and the columns in ``more`` (those of the functions and of the densities)."""
    times, columns = read_series(series)
    for organism in organisms:
        columns.update(derive(plan, organism, columns))
    wanted = [entry.column for organism in organisms for entry in protected(plan, organism)] + list(more)
    wanted = list(dict.fromkeys(wanted))
    missing = [column for column in wanted if column not in columns]
    if missing:
        raise WorldError(f"{series}: no column {missing}")
    return write_series(target, times, {column: columns[column] for column in wanted})


def write_collapsed(
    partial: Path,
    baseline: Path,
    target: Path,
    plan: dict,
    organisms: list[Organism],
    more: list[str],
    gone: str,
) -> Path:
    """The file that the measure reads for a run that stopped before the horizon: what the organisms protect up
    to the last stored time of the run, and from the next stored time of the baseline on the last values with
    the column ``gone`` (the density of the receiver) at zero, so that the measure finds the receiver lost there
    and reads nothing after it."""
    import csv

    with Path(partial).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    header, data = rows[0], np.array(rows[1:], dtype=np.float64)
    columns = {name: data[:, i].copy() for i, name in enumerate(header) if name != "time"}
    for organism in organisms:
        columns.update(derive(plan, organism, columns))
    wanted = [entry.column for organism in organisms for entry in protected(plan, organism)] + list(more)
    wanted = list(dict.fromkeys(wanted))
    times, _ = read_series(baseline)
    stored = data.shape[0]
    if stored >= times.size:
        raise WorldError(f"{partial}: a run that was stored to the horizon did not stop")
    out = {}
    for name in wanted:
        if name not in columns:
            raise WorldError(f"{partial}: no column {name}")
        out[name] = np.concatenate([columns[name], np.full(times.size - stored, columns[name][-1])])
    out[gone][stored:] = 0.0
    return write_series(target, times, out)


def group_rows(result: dict) -> list[dict]:
    """One row per group of a measure: its layer, its kind, and its numbers."""
    rows = []
    for label, numbers in (result.get("groups") or {}).items():
        layer, kind = split(label)
        rows.append(
            {
                "layer": layer,
                "kind": kind,
                "quantities": numbers["protected"],
                "D": numbers["D_peak"],
                "D_end": numbers["D_end"],
                "t_peak": numbers["t_peak"],
                "D_quantity": numbers["D_quantity"],
                "D_activity": numbers["D_activity"],
                "stage1_trajectory": numbers["1_trajectory"],
                "stage3_tolerable": numbers["3_tolerable"],
                "stage4_distribution": numbers["4_distribution"],
                "bits_lower": numbers["bits_lower"],
                "bits_upper": numbers["bits_upper"],
                "D_removal": numbers.get("D_removal"),
            }
        )
    return rows


__all__ = [
    "CARRIERS",
    "KINDS",
    "LAYERS",
    "WHAT",
    "Entry",
    "Organism",
    "annotate",
    "atp_coefficients",
    "derive",
    "group_rows",
    "groups",
    "layers_of",
    "protected",
    "sbml_organism",
    "split",
    "write_collapsed",
    "write_protected",
    "xeno_organism",
]
