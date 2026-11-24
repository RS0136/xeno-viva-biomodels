"""Several pairs in one vessel: what the SBML organisms and their xenos do to each other.

A pair is an SBML organism of the plan and its xeno. Each pair is first brought to rest in a vessel of its own,
exactly as in ``world`` (design ``xeno``). The vessels are then poured together in equal volumes. That is a
physical operation and fixes everything about the community: every density and every concentration becomes the
mean over the vessels, the feed the mean of the feeds, and the dilution rate stays. Species of different SBML
models that are one chemical become one species of the medium (``same_chemical`` in the plan).

    supply      one node: the dilution rate and the merged feed
    medium      one store, shared by every organism of every pair
    Y, E, L     the SBML organisms (the names are those of the plan)
    xY, xE, xL  their xenos

Three things are done with a community, for each design of the plan (how the xenos stand to the other pairs):

    fate            the poured community is followed for a long time with nothing stepped: who stays
    assembly        the same start followed until the fast changes are over; its end is where the reading starts
    interference    from that state: the baseline; every xeno with its activity stepped; every organism removed.
                    Orthogonality is then read for every ordered pair of organisms with the measure of
                    ``orthogonality``: D from the steps of a xeno, D_removal from taking an organism away. What
                    an organism protects is what it protects alone (``layers``): quantities common to all and
                    quantities of its own, layer by layer.

Written under ``runs/community/<design>/``: ``community.json``, a ``series.csv`` and a ``protected.csv`` per run,
the measure of every ordered pair under ``measure/``, ``matrix.csv`` (one row per ordered pair) and ``layers.csv``
(one row per ordered pair and group of protected quantities); ``runs/community/summary.csv`` and ``layers.csv``
hold those of all designs.
"""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from xeno_biomodels import layers, paths
from xeno_biomodels.orthogonality import MeasureError, measure, read_series
from xeno_biomodels.world import World, WorldError, follow_document, load_plan, make_core, write_series

OUT = paths.RUNS / "community"
BASELINE, FATE, ASSEMBLY = "baseline", "fate", "assembly"


@dataclass
class Pair:
    """An SBML organism, its xeno, and the vessel in which the two are alone."""

    biomodel: str
    organism: str
    xeno: str
    world: World
    keys: dict[str, str]  # species of the SBML model -> its name in the common medium
    per_unit: dict[str, float]  # species of the SBML model -> units of it per unit of the medium

    @property
    def nutrient_key(self) -> str:
        return self.keys[self.world.nutrient]


class Community:
    """The pairs of the plan in one vessel, with the xenos of one design."""

    def __init__(self, plan: dict, design: str, core=None):
        settings = plan["community"]
        if design not in settings["designs"]:
            raise WorldError(f"no community design {design}; the plan has {', '.join(settings['designs'])}")
        self.plan, self.design, self.settings = plan, design, settings
        self.own = settings["designs"][design]["substrate"] == "own"
        self.cross = float(settings["designs"][design]["cross"])
        self.core = core if core is not None else make_core()
        self.dilution = float(plan["supply"]["dilution"])
        self.interval = float(plan["time"]["interval"])
        self.baseline = float(plan["activities"]["baseline"])
        chemical = {}
        for key, members in (settings.get("same_chemical") or {}).items():
            for biomodel, item in members.items():
                chemical[(biomodel, item["species"])] = (key, float(item["per_mM"]))
        self.pairs: list[Pair] = []
        for biomodel, names in settings["names"].items():
            world = World(plan, biomodel, "xeno", core=self.core)
            if world.minute != 1.0:
                raise WorldError(
                    f"{biomodel}: the models of a community must have the unit of time of the plan"
                )
            found = {name: chemical.get((biomodel, name), (name, 1.0)) for name in world.medium_ids}
            self.pairs.append(
                Pair(
                    biomodel,
                    names["organism"],
                    names["xeno"],
                    world,
                    {name: key for name, (key, _) in found.items()},
                    {name: per for name, (_, per) in found.items()},
                )
            )
        self.kinds = {pair.organism: "sbml" for pair in self.pairs} | {
            pair.xeno: "xeno" for pair in self.pairs
        }
        self.feed: dict[str, float] = {}

    # ---- the organisms ---------------------------------------------------------------------------------------------

    def xeno_config(self, pair: Pair) -> dict:
        """The xeno of a pair: on a substrate of its own or on the common one, and what else its transporter takes.

        It never takes a species that its own partner has in the medium. With ``cross`` above zero it takes the
        nutrients of the SBML organisms of the other pairs at that specificity. With ``overlap`` above zero (and
        substrates of their own) it also takes the substrates of the other xenos at that specificity and grows on
        them as on its own: two xenos that are each better at their own substrate coexist stably, where two that
        share one substrate with the same laws coexist only neutrally.
        """
        config = pair.world._sender_config()
        base = config["substrate"]
        if self.own:
            config["substrate"] = f"{base}_{pair.xeno}"
            config["product"] = f"{config['product']}_{pair.xeno}" if config["product"] else ""
        mine = set(pair.keys.values())
        others = sorted({other.nutrient_key for other in self.pairs if other is not pair} - mine)
        config["also_takes"] = dict.fromkeys(others, self.cross) if self.cross > 0.0 else {}
        worth = self.settings["designs"][self.design].get("worth")
        if worth is not None and config["also_takes"]:
            # what a unit of a nutrient of another pair is worth to the xeno for growth, relative to its own
            # substrate (as its own substrate unless the design says so; 0: it takes what it cannot use)
            config["also_worth"] = dict.fromkeys(others, float(worth))
        overlap = float(self.settings["designs"][self.design].get("overlap", 0.0))
        if overlap > 0.0:
            if not self.own:
                raise WorldError(f"{self.design}: an overlap needs substrates of their own to overlap on")
            for other in self.pairs:
                if other is not pair:
                    config["also_takes"][f"{base}_{other.xeno}"] = overlap
        return config

    def members(self) -> dict[str, dict]:
        """Every organism: its kind, its pair, and what it has to do with the species of the medium.

        ``reads`` are the species whose level enters its kinetics, ``writes`` those whose level it changes, and
        ``touches`` both together. An SBML organism reads and writes the species that the world lets vary for
        it; a xeno reads what it takes and writes that and its product, which nothing reads.
        """
        out = {}
        for pair in self.pairs:
            config = self.xeno_config(pair)
            mine = sorted(pair.keys.values())
            out[pair.organism] = {
                "kind": "sbml",
                "biomodel": pair.biomodel,
                "partner": pair.xeno,
                "nutrient": pair.nutrient_key,
                "reads": mine,
                "writes": mine,
                "touches": mine,
            }
            taken = sorted({config["substrate"], *config["also_takes"]})
            out[pair.xeno] = {
                "kind": "xeno",
                "partner": pair.organism,
                "substrate": config["substrate"],
                "product": config["product"],
                "also_takes": config["also_takes"],
                "reads": taken,
                "writes": sorted({*taken, config["product"]} - {""}),
                "touches": sorted({*taken, config["product"]} - {""}),
            }
        return out

    # ---- the pouring -----------------------------------------------------------------------------------------------

    def pour(self) -> dict:
        """The vessels of the pairs, each at rest, poured together in equal volumes.

        Sets the feed of the community and returns its state at the moment of pouring.
        """
        share = 1.0 / len(self.pairs)
        medium: dict[str, float] = {}
        feed: dict[str, float] = {}
        organisms: dict[str, dict] = {}
        self.vessels = {}
        for pair in self.pairs:
            world, sender = pair.world, pair.world.sender
            state, moved, _ = world.rest()
            config = self.xeno_config(pair)
            # the vessel of this pair, in the names and the units of the common medium
            levels = {
                pair.keys[name]: state["medium"][name] / pair.per_unit[name] for name in world.medium_ids
            }
            supplied = {pair.keys[name]: world.feed[name] / pair.per_unit[name] for name in world.medium_ids}
            levels[config["substrate"]] = state["medium"][sender["substrate"]]
            supplied[config["substrate"]] = world.feed[sender["substrate"]]
            if config["product"]:
                levels[config["product"]] = state["medium"][sender["product"]]
                supplied[config["product"]] = 0.0
            for key, value in levels.items():
                medium[key] = medium.get(key, 0.0) + share * value
            for key, value in supplied.items():
                feed[key] = feed.get(key, 0.0) + share * value
            organisms[pair.organism] = {**state["B"], "density": share * state["B"]["density"]}
            organisms[pair.xeno] = {
                "density": share * state["A"]["density"],
                "uptake": state["A"]["uptake"],
                "growth_rate": state["A"]["growth_rate"],
                "work": share * state["A"]["work"],
            }
            self.vessels[pair.organism] = {"medium": levels, "feed": supplied, "rest_moved": moved}
        self.feed = feed
        return {"medium": medium, "organisms": organisms}

    # ---- the composite ---------------------------------------------------------------------------------------------

    def document(
        self,
        state: dict,
        activities: dict[str, float] | None = None,
        absent: tuple[str, ...] = (),
        store_every: int | None = None,
    ) -> dict:
        """The document of the community from ``state``, with the activities given (the baseline for the others)
        and the organisms in ``absent`` taken away at time zero."""
        activities = activities or {}
        tree: dict = {
            "supply": {"dilution": self.dilution, "feed": dict(self.feed)},
            "medium": dict(state["medium"]),
        }
        emit: dict[str, str] = {"time": "float", "medium": "map[float]"}
        wires: dict[str, list[str]] = {"time": ["global_time"], "medium": ["medium"]}
        shared = {
            "medium": ["..", "medium"],
            "density": ["density"],
            "dilution": ["..", "supply", "dilution"],
        }
        for pair in self.pairs:
            body = state["organisms"][pair.organism]
            overrides = {f"[{name}]": float(value) for name, value in body["species"].items()}
            for name in pair.world.medium_ids:
                overrides[f"[{name}]"] = pair.per_unit[name] * float(state["medium"][pair.keys[name]])
            config = pair.world._receiver_config(overrides)
            config.update(medium_keys=dict(pair.keys), medium_per_unit=dict(pair.per_unit))
            tree[pair.organism] = {
                "density": 0.0 if pair.organism in absent else float(body["density"]),
                "growth_rate": float(body["growth_rate"]),
                "species": dict(body["species"]),
                "rates": dict(body["rates"]),
                "uptake": dict(body["uptake"]),
                "organism": {
                    "_type": "process",
                    "address": "local:SBMLOrganism",
                    "config": config,
                    "interval": self.interval,
                    "inputs": dict(shared),
                    "outputs": {
                        "medium": ["..", "medium"],
                        "density": ["density"],
                        "species": ["species"],
                        "rates": ["rates"],
                        "uptake": ["uptake"],
                        "growth_rate": ["growth_rate"],
                    },
                },
            }
            for store, kind in (
                ("density", "float"),
                ("growth_rate", "float"),
                ("uptake", "map[float]"),
                ("species", "map[float]"),
                ("rates", "map[float]"),
            ):
                emit[f"{pair.organism}.{store}"] = kind
                wires[f"{pair.organism}.{store}"] = [pair.organism, store]

            body, gone = state["organisms"][pair.xeno], pair.xeno in absent
            tree[pair.xeno] = {
                "activity": float(activities.get(pair.xeno, self.baseline)),
                "density": 0.0 if gone else float(body["density"]),
                "uptake": 0.0 if gone else float(body["uptake"]),
                "growth_rate": 0.0 if gone else float(body["growth_rate"]),
                "work": 0.0 if gone else float(body["work"]),
                "organism": {
                    "_type": "process",
                    "address": "local:LawOrganism",
                    "config": self.xeno_config(pair),
                    "interval": self.interval,
                    "inputs": {**shared, "activity": ["activity"]},
                    "outputs": {
                        "medium": ["..", "medium"],
                        "density": ["density"],
                        "uptake": ["uptake"],
                        "growth_rate": ["growth_rate"],
                        "work": ["work"],
                    },
                },
            }
            for store in ("activity", "density", "uptake", "growth_rate", "work"):
                emit[f"{pair.xeno}.{store}"] = "float"
                wires[f"{pair.xeno}.{store}"] = [pair.xeno, store]
        tree["chemostat"] = {
            "_type": "process",
            "address": "local:ChemostatSupply",
            "config": {},
            "interval": self.interval,
            "inputs": {"medium": ["medium"], "feed": ["supply", "feed"], "dilution": ["supply", "dilution"]},
            "outputs": {"medium": ["medium"]},
        }
        tree["emitter"] = {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": emit, "subsample": int(store_every or self.plan["time"]["store_every"])},
            "inputs": wires,
        }
        return {"state": tree, "skip_process_state": True}

    def snapshot(self, tree: dict) -> dict:
        """The state of a running composite, in the form that ``document`` takes."""
        organisms = {}
        for pair in self.pairs:
            body = tree[pair.organism]
            organisms[pair.organism] = {
                "density": float(body["density"]),
                "growth_rate": float(body["growth_rate"]),
                "species": {name: float(value) for name, value in body["species"].items()},
                "rates": {name: float(value) for name, value in body["rates"].items()},
                "uptake": {name: float(value) for name, value in body["uptake"].items()},
            }
            organisms[pair.xeno] = {
                key: float(tree[pair.xeno][key]) for key in ("density", "uptake", "growth_rate", "work")
            }
        return {
            "medium": {name: float(value) for name, value in tree["medium"].items()},
            "organisms": organisms,
        }

    def assemble(self, state: dict) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
        """Follow the poured community until the fast changes are over. Returns the state then, and the run."""
        from process_bigraph import Composite, gather_emitter_results

        from xeno_biomodels.world import columns_of

        simulation = Composite(self.document(state), core=self.core)
        simulation.run(float(self.settings["assembly"]))
        times, columns = columns_of(gather_emitter_results(simulation)[("emitter",)])
        return self.snapshot(simulation.state), times, columns


# ---- the runs ------------------------------------------------------------------------------------------------------

_CORE = None


def _follow(item: tuple[dict, float]):
    """One run, in a worker that keeps its core between runs."""
    global _CORE
    if _CORE is None:
        _CORE = make_core()
    document, horizon = item
    return follow_document(_CORE, document, horizon)


def run_names(plan: dict, community: Community) -> dict[str, dict]:
    """Run name -> what is stepped or taken away, for the runs that start from the assembled community."""
    runs: dict[str, dict] = {BASELINE: {}}
    for pair in community.pairs:
        for value in plan["activities"]["steps"]:
            runs[f"{pair.xeno}.u={value:g}"] = {"activities": {pair.xeno: float(value)}, "sender": pair.xeno}
    for name in community.kinds:
        runs[f"no_{name}"] = {"absent": (name,), "sender": name}
    return runs


def simulate(plan: dict, design: str, out: Path = OUT, jobs: int = 1, fate: bool = True) -> dict:
    """Pour the community of one design, follow its fate, assemble it, and follow every step and every removal.

    ``fate=False`` leaves the long run with nothing stepped out (the studies that vary a community do).
    """
    started = time.monotonic()
    community = Community(plan, design)
    directory = out / design
    poured = community.pour()
    assembled, times, columns = community.assemble(poured)
    write_series(directory / ASSEMBLY / "series.csv", times, columns)
    runs = run_names(plan, community)
    work = [
        (
            community.document(assembled, spec.get("activities"), spec.get("absent", ())),
            float(plan["time"]["horizon"]),
        )
        for spec in runs.values()
    ]
    followed = list(runs)
    if fate:
        long_run = community.document(poured, store_every=int(community.settings["fate_store_every"]))
        work.append((long_run, float(community.settings["fate"])))
        followed.append(FATE)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn")) as pool:
            results = list(pool.map(_follow, work))
    else:
        results = [_follow(item) for item in work]
    not_followed = {}
    for name, (times, columns, problem) in zip(followed, results, strict=True):
        for stale in ("series.csv", "partial.csv", "status.txt"):
            (directory / name / stale).unlink(missing_ok=True)
        if problem is None:
            write_series(directory / name / "series.csv", times, columns)
        else:
            write_series(directory / name / "partial.csv", times, columns)
            (directory / name / "status.txt").write_text(problem + "\n", encoding="utf-8")
            not_followed[name] = {"problem": problem, "last_stored_time": float(times[-1])}
    info = {
        "design": design,
        "xenos_on": "a substrate of their own" if community.own else "one common substrate",
        "cross": community.cross,
        "dilution": community.dilution,
        "interval": community.interval,
        "members": community.members(),
        "vessels": community.vessels,
        "feed": community.feed,
        "poured": poured,
        "assembled": assembled,
        "assembly_min": float(community.settings["assembly"]),
        "fate_min": float(community.settings["fate"]),
        "runs": {name: spec for name, spec in runs.items()},
        "not_followed": not_followed,
        "seconds": round(time.monotonic() - started, 1),
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "community.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


# ---- the reading ---------------------------------------------------------------------------------------------------


def organisms_of(plan: dict, members: dict) -> dict[str, layers.Organism]:
    """Every organism of a community as the reading knows it, from ``members`` of ``community.json``."""
    out = {}
    for name, body in members.items():
        if body["kind"] == "sbml":
            out[name] = layers.sbml_organism(plan, body["biomodel"], name, food=body["nutrient"])
        else:
            out[name] = layers.xeno_organism(name, body["substrate"], body["product"])
    return out


def read_matrix(
    plan: dict, design: str, out: Path = OUT, into: Path | None = None
) -> tuple[list[dict], list[dict]]:
    """Orthogonality of every organism to every other, from the runs of one design that are there.

    Returns one row per ordered pair, and one per ordered pair and group of protected quantities. The files of
    the measure, ``matrix.csv`` and ``layers.csv`` go to ``into`` (the directory of the design unless given).
    """
    directory = out / design
    into = directory if into is None else Path(into)
    info = json.loads((directory / "community.json").read_text(encoding="utf-8"))
    settings, members, lost = plan["measure"], info["members"], info["not_followed"]
    if BASELINE in lost:
        raise WorldError(f"{design}: the baseline could not be followed: {lost[BASELINE]['problem']}")
    organisms = organisms_of(plan, members)
    entries = {name: layers.protected(plan, organism) for name, organism in organisms.items()}

    def of(name: str, key: str) -> str:
        return settings[key].replace("{name}", name)

    more = [of(name, key) for name in members for key in ("function", "density")]
    more += [of(name, "sender_function") for name, body in members.items() if body["kind"] == "xeno"]
    followed = [name for name in info["runs"] if name not in lost]
    for name in followed:  # what the measure reads of each run, for every organism at once
        layers.write_protected(
            directory / name / "series.csv",
            directory / name / "protected.csv",
            plan,
            list(organisms.values()),
            more,
        )

    base = directory / BASELINE / "series.csv"
    with base.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    _, base_columns = read_series(base)
    cache: dict[str, dict[str, np.ndarray]] = {}

    def columns_of_run(name: str) -> dict[str, np.ndarray]:
        if name not in cache:
            cache[name] = read_series(directory / name / "series.csv")[1]
        return cache[name]

    def owned(name: str) -> list[str]:
        """The columns that are an organism's own: its stores, and the species of the medium that it touches."""
        mine = [column for column in header if column.startswith(f"{name}.")]
        return mine + [f"medium.{key}" for key in members[name]["touches"]]

    rows, by_group = [], []
    for sender, giving in members.items():
        steps = [
            name
            for name, spec in info["runs"].items()
            if spec.get("sender") == sender and "activities" in spec
        ]
        stepped = [name for name in steps if name not in lost]
        removal = f"no_{sender}"
        for receiver, taking in members.items():
            if receiver == sender:
                continue
            row = {
                "design": design,
                "sender": sender,
                "sender_kind": giving["kind"],
                "receiver": receiver,
                "receiver_kind": taking["kind"],
                "same_pair": giving["partner"] == receiver,
                "species_shared": " ".join(sorted(set(giving["touches"]) & set(taking["touches"]))),
            }
            xeno = giving["kind"] == "xeno"
            try:
                result, _ = measure(
                    directory / BASELINE / "protected.csv",
                    {name.split(".", 1)[1]: directory / name / "protected.csv" for name in stepped},
                    [entry.spec for entry in entries[receiver]],
                    receiver_density=of(receiver, "density"),
                    viable=float(plan["viable"]),
                    function=of(receiver, "function"),
                    qmin=settings["qmin"],
                    mode=settings["mode"],
                    allocation=settings["allocation"],
                    sender_function=of(sender, "sender_function") if xeno else None,
                    sender_qmin=settings["sender_qmin"] if xeno else None,
                    sender_mode=settings["sender_mode"],
                    removal=None if removal in lost else directory / removal / "protected.csv",
                    groups=layers.groups(entries[receiver]),
                    sigma=float(settings["sigma"]),
                    stage_tol=float(settings["stage_tol"]),
                    bits_tol=float(settings["bits_tol"]),
                    stride=int(settings.get("stride", 1)),
                    out=into / "measure" / f"{sender}_to_{receiver}",
                )
            except MeasureError as exc:
                rows.append({**row, "status": str(exc).splitlines()[0]})
                continue
            mine = owned(receiver)
            read = [name for name in (*stepped, removal) if name not in lost]
            same = [
                name
                for name in read
                if all(np.array_equal(columns_of_run(name)[column], base_columns[column]) for column in mine)
            ]
            losses = " ".join(
                f"{item['activity']}@{item['t']:g}({item['cause']})" for item in result["losses"]
            )
            row.update(
                status="ok",
                D=result["D_peak"]
                if steps
                else None,  # an organism without a knob has no D, only its removal
                D_end=result["D_end"] if steps else None,
                D_quantity=result["D_quantity"],
                D_activity=result["D_activity"],
                O=result["O"] if steps else None,
                stage1_trajectory=result["1_trajectory"] if steps else None,
                stage3_tolerable=result["3_tolerable"] if steps else None,
                stage4_distribution=result["4_distribution"] if steps else None,
                bits_upper=result["bits_upper"] if steps else None,
                activities=result["activities"],
                excluded_by_sender_function=" ".join(
                    item["activity"] for item in result["excluded_by_sender_function"]
                ),
                losses=losses,
                D_removal=result.get("D_removal"),
                removal_lost_at=result.get("removal_lost_at"),
                removal_loss_cause=result.get("removal_loss_cause"),
                not_followed=" ".join(
                    f"{name}@{lost[name]['last_stored_time']:g}" for name in (*steps, removal) if name in lost
                ),
                receiver_bit_identical_in=" ".join(same),
                receiver_bit_identical=len(same) == len(read),
            )
            rows.append(row)
            tag = {
                key: row[key]
                for key in ("design", "sender", "sender_kind", "receiver", "receiver_kind", "same_pair")
            }
            for found in layers.group_rows(result):
                if not steps:  # D of a group, too, is that of the steps of a knob
                    found.update(
                        D=None,
                        D_end=None,
                        t_peak=None,
                        D_quantity=None,
                        D_activity=None,
                        stage1_trajectory=None,
                        stage3_tolerable=None,
                        stage4_distribution=None,
                        bits_lower=None,
                        bits_upper=None,
                    )
                by_group.append({**tag, "losses": losses, **found})
    _write(into / "matrix.csv", rows)
    _write(into / "layers.csv", by_group)
    return rows, by_group


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(
    plan: dict | None = None,
    out: Path = OUT,
    jobs: int = 1,
    analyze_only: bool = False,
    only: list[str] | None = None,
) -> list[dict]:
    """Every community design of the plan (or those in ``only``). Returns the rows of ``summary.csv``."""
    plan = plan or load_plan()
    rows, by_group = [], []
    for design in plan["community"]["designs"]:
        if only and design not in only:
            continue
        if not analyze_only:
            info = simulate(plan, design, out, jobs)
            print(
                f"community/{design}: {len(info['runs']) + 2} runs followed in {info['seconds']} s",
                flush=True,
            )
        pairs, groups = read_matrix(plan, design, out)
        rows += pairs
        by_group += groups
    _write(out / "summary.csv", rows)
    _write(out / "layers.csv", by_group)
    return rows


__all__ = ["OUT", "Community", "Pair", "organisms_of", "read_matrix", "run", "run_names", "simulate"]
