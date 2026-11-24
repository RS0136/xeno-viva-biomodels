"""A world: a chemostat, an SBML organism (the receiver B) and an organism of laws (the sender A), as one composite.

    supply      the dilution rate and the feed: one node, read by the chemostat and by both organisms
    medium      the concentrations in the vessel, shared by everything
    B           the SBML model, run by the Tellurium process of viva-tellurium with the interface of an organism
    A           the organism of laws, with its activity
    chemostat   the process of the supply
    emitter     what is stored

A world is calibrated to its SBML model and adds no free number to it beyond the dilution rate and the density of
the organism. The model is first brought to its steady state as the SBML file has it, boundary species held. The
feed concentration of each medium species is then the one that holds the vessel at the value of the SBML file
when the organism is alone, and the yield of the organism is the one at which it grows at the dilution rate there.
The steady state of the SBML model is therefore the rest state of the organism in the world.
"""

from __future__ import annotations

import contextlib
import csv
from pathlib import Path

import libsbml
import numpy as np
import yaml

from xeno_biomodels import paths
from xeno_biomodels.organisms import SBMLOrganism, register

BASELINE, REMOVAL, NO_RECEIVER = "baseline", "removal", "no_receiver"
TINY = 1e-300


class WorldError(RuntimeError):
    """The world cannot be built or brought to rest as the plan asks."""


def load_plan(path: str | Path | None = None) -> dict:
    return yaml.safe_load(
        Path(path or paths.PROJECT / "configs" / "experiment.yaml").read_text(encoding="utf-8")
    )


def sbml_file(biomodel: str) -> Path:
    """The SBML file of a BioModel, fetched with the loader of viva-biomodels when it is not there yet."""
    directory = paths.MODELS / biomodel
    found = sorted(directory.glob("*.xml")) if directory.exists() else []
    if not found:
        paths.use_viva_biomodels()
        import biomodels
        from viva_biomodels.run_biomodels import load_biomodel

        paths.PROJECT.mkdir(parents=True, exist_ok=True)
        with contextlib.chdir(paths.PROJECT):  # the loader writes models/<id>/ under the working directory
            load_biomodel(biomodel, biomodels.get_metadata(biomodel))
        found = sorted(directory.glob("*.xml"))
    if not found:
        raise WorldError(f"no SBML file for {biomodel} under {directory}")
    return found[0]


def make_core():
    """The core of viva-biomodels (its simulators, its steps, its types) with the three processes of a world."""
    paths.use_viva_biomodels()
    from viva_biomodels.core import build_core

    return register(build_core())


def time_unit_seconds(sbml: Path) -> float:
    """Seconds in one unit of time of an SBML model (1 when the model does not say)."""
    model = libsbml.readSBMLFromFile(str(sbml)).getModel()
    definition = model.getUnitDefinition(model.getTimeUnits() or "time")
    if definition is None or definition.getNumUnits() != 1:
        return 1.0
    unit = definition.getUnit(0)
    if unit.getKind() != libsbml.UNIT_KIND_SECOND or unit.getExponent() != 1:
        raise WorldError(f"{sbml}: the unit of time is not a multiple of the second")
    return float(unit.getMultiplier() * 10.0 ** unit.getScale())


class World:
    """One receiver with one design of the sender, under the supply of the plan."""

    def __init__(self, plan: dict, biomodel: str, design: str, core=None):
        if biomodel not in plan["receivers"]:
            raise WorldError(f"no receiver {biomodel}; the plan has {', '.join(plan['receivers'])}")
        if design not in plan["designs"]:
            raise WorldError(f"no design {design}; the plan has {', '.join(plan['designs'])}")
        self.plan, self.biomodel, self.design = plan, biomodel, design
        self.receiver = plan["receivers"][biomodel]
        self.sender = plan["sender"]
        # A design is the specificity of the transporter of the sender for the nutrient of the receiver, or
        # {share: f, of: species}: a leak in the units of no model, on any medium species (see _sender_config).
        self.leak_as = plan["designs"][design]
        self.leak = 0.0 if isinstance(self.leak_as, dict) else float(self.leak_as)
        self.sbml = sbml_file(self.receiver.get("biomodel", biomodel))  # a receiver may name its model
        # The plan is written in minutes and a model in its own unit of time: what follows is in that of the model.
        # units of time of the model in one minute (a receiver says its unit where its file does not)
        self.minute = 60.0 / float(self.receiver.get("time_unit_seconds") or time_unit_seconds(self.sbml))
        self.dilution = float(plan["supply"]["dilution"]) / self.minute
        self.interval = float(plan["time"]["interval"]) * self.minute
        self.baseline = float(plan["activities"]["baseline"])
        self.medium_ids = list(self.receiver["medium"])
        self.nutrient = self.receiver["nutrient"]
        # the density of the organism alone at rest: given, or the one at which it takes a share of what is fed
        self.biovolume = float(self.receiver["biovolume"]) if "biovolume" in self.receiver else None
        for name in (self.sender["substrate"], self.sender["product"]):
            if name in self.medium_ids:
                raise WorldError(f"the sender's {name} is a medium species of the receiver")
        self.core = core if core is not None else make_core()
        self._calibrate()

    # ---- the receiver ----------------------------------------------------------------------------------------------

    def _receiver_config(self, overrides: dict | None = None, growth: bool = True) -> dict:
        paths.use_viva_biomodels()
        from viva_biomodels.simulators import process_config

        config = dict(
            process_config("tellurium", str(self.sbml))
        )  # as biomodel-process of viva-biomodels has it
        times = self.plan["time"]
        growth = growth and self.receiver.get(
            "growth", True
        )  # growth: false holds the density of the organism
        config.update(
            absolute_tolerance=float(times.get("absolute_tolerance", 1e-12)),
            relative_tolerance=float(times.get("relative_tolerance", 1e-10)),
            medium_species=list(self.medium_ids),
            growth_nutrient=self.nutrient if growth else "",
            uptake_at_rest=float(self.uptake_at_rest) if growth else 0.0,
            maintenance=float(self.receiver.get("maintenance", 0.0)) / self.minute if growth else 0.0,
            species_overrides=dict(overrides or {}),
            uptake_reactions=dict(self.receiver.get("uptake_reactions") or {}),
            growth_variable=str(self.receiver.get("growth_variable") or "") if growth else "",
        )
        return config

    def _calibrate(self) -> None:
        """The steady state of the SBML model as the file has it, and the feed and the yield that follow from it."""
        own = str(self.receiver.get("growth_variable") or "")
        config = self._receiver_config(growth=False)
        config["growth_variable"] = own  # read here, also where the density is held
        organism = SBMLOrganism(config=config, core=self.core)
        organism._build()
        model = organism._rr
        at = 0.0

        def settle() -> None:
            """Follow the model at the levels that it has until nothing in it moves."""
            nonlocal at
            span, previous, moved = 50.0 * self.minute, None, np.inf
            for _ in range(18 if own else 14):
                model.simulate(at, at + span, 2)
                at += span
                now = np.asarray(model.getFloatingSpeciesConcentrations(), dtype=float)
                if previous is not None:
                    moved = float(np.max(np.abs(now - previous) / (np.abs(now) + 1e-12)))
                    if moved < 1e-11:
                        return
                previous, span = now, 2.0 * span
            raise WorldError(
                f"{self.biomodel}: no steady state within {at / self.minute:g} min (still moving by {moved:.2g})"
            )

        settle()
        if own:
            # A model that grows at a rate of its own lives, in a vessel, at the level of its nutrient at which
            # that rate is the dilution rate: the level is found here and not taken from the file.
            high = organism.level(self.nutrient)
            if not organism.own_growth() > self.dilution:
                raise WorldError(
                    f"{self.biomodel}: at the level of its file the model grows at {organism.own_growth():.3g}, "
                    f"which is not above the dilution rate {self.dilution:.3g}"
                )
            low, top = high, high
            while organism.own_growth() > self.dilution:  # downward in small steps: each from the one before
                high, low = low, 0.7 * low
                if low < 1e-9 * top:
                    raise WorldError(f"{self.biomodel}: the model grows faster than the vessel dilutes it")
                organism.hold(self.nutrient, low)
                settle()
            for _ in range(60):
                middle = float(np.sqrt(low * high))
                organism.hold(self.nutrient, middle)
                settle()
                if organism.own_growth() > self.dilution:
                    high = middle
                else:
                    low = middle
                if high / low - 1.0 < 1e-13:
                    break
            organism.hold(self.nutrient, float(np.sqrt(low * high)))
            settle()
            self.own_growth_at_rest = organism.own_growth()
        production = organism.production()
        read = organism.read()
        self.sbml_species, self.sbml_rates = read["species"], read["rates"]
        self.sbml_medium = {name: organism.level(name) for name in self.medium_ids}
        # how the model is told each level: as a concentration, or as the value of a parameter
        self.held_as = dict(zip(organism._medium_ids, organism._held, strict=True))
        # what else the model holds constant: these stay at the values of the SBML file, in every run
        self.held = sorted(name for name in model.getBoundarySpeciesIds() if name not in self.medium_ids)
        self.sbml_production = {name: float(production[i]) for i, name in enumerate(self.medium_ids)}
        self.sbml_relaxed_after = at / self.minute
        self.uptake_at_rest = -self.sbml_production[self.nutrient]
        if not self.uptake_at_rest > 0.0:
            raise WorldError(
                f"{self.biomodel}: at its steady state the model does not take up {self.nutrient}"
            )
        if self.biovolume is None:
            # consumes: the share of the nutrient in the feed that the organism takes when it is alone at rest
            share = float(self.receiver["consumes"])
            if not 0.0 < share < 1.0:
                raise WorldError(f"{self.biomodel}: consumes must be a share between 0 and 1")
            level = self.sbml_medium[self.nutrient]
            if not level > 0.0:
                raise WorldError(f"{self.biomodel}: the SBML file holds {self.nutrient} at zero")
            self.biovolume = share / (1.0 - share) * level * self.dilution / self.uptake_at_rest
        self.feed = {}
        for name in self.medium_ids:
            level = self.sbml_medium[name]
            self.feed[name] = level - self.biovolume * self.sbml_production[name] / self.dilution
            if self.feed[name] < 0.0:
                raise WorldError(
                    f"{self.biomodel}: {name} cannot be held at {level:g} mM with a feed that is not negative; "
                    "lower the biovolume or raise the dilution rate"
                )
            turnover = self.interval * self.biovolume * abs(self.sbml_production[name]) / max(level, TINY)
            if turnover > 0.05:
                raise WorldError(
                    f"{self.biomodel}: the organism exchanges {100 * turnover:.0f} % of {name} per interval; "
                    "shorten the interval or lower the biovolume"
                )
        self.feed[self.sender["substrate"]] = float(self.sender["feed"])
        if self.sender["product"]:
            self.feed[self.sender["product"]] = 0.0

    # ---- the sender ------------------------------------------------------------------------------------------------

    def _sender_config(self) -> dict:
        sender = self.sender
        config = {
            "substrate": sender["substrate"],
            "product": sender["product"],
            "vmax": float(sender["vmax"]) / self.minute,
            "km": float(sender["km"]),
            "growth_yield": float(sender["growth_yield"]),
            "maintenance": float(sender["maintenance"]) / self.minute,
            "product_yield": float(sender["product_yield"]),
            "also_takes": {self.nutrient: self.leak} if self.leak > 0.0 else {},
        }
        if isinstance(self.leak_as, dict) and float(self.leak_as.get("share", 0.0)) > 0.0:
            # A leak in the units of no model, on the nutrient or on the medium species that the design names
            # (``of``). The transporter is half-saturated for the species at the level of the SBML file; its
            # specificity is the one at which the sender, at the density that it has alone, takes at that level
            # the share of what the receiver takes of the species alone (or of what it makes of it, for a
            # species that it excretes); and the species is worth nothing to the sender unless the design says
            # what (it takes what it cannot use).
            species = self.leak_as.get("of", self.nutrient)
            if species not in self.medium_ids:
                raise WorldError(f"{self.design}: {species} is not a medium species of {self.biomodel}")
            k, u, level = self.dilution, self.baseline, self.sbml_medium[species]
            own = (k + config["maintenance"]) / config["growth_yield"]
            alone = k * (float(sender["feed"]) - config["km"] * own / (u * config["vmax"] - own)) / own
            wanted = float(self.leak_as["share"]) * self.biovolume * abs(self.sbml_production[species])
            config["also_takes"] = {species: 2.0 * wanted / (alone * u * config["vmax"])}
            config["also_km"] = {species: level}
            config["also_worth"] = {species: float(self.leak_as.get("worth", 0.0))}
        return config

    def guess(self) -> dict:
        """The state at rest in closed form: exact when the sender takes nothing of the receiver's, and where the
        search for the rest state starts otherwise."""
        config, k, u = self._sender_config(), self.dilution, self.baseline
        needed = (k + config["maintenance"]) / config[
            "growth_yield"
        ]  # the uptake at which growth is dilution
        taken, gained = (
            {},
            0.0,
        )  # what it takes of the medium species of the receiver, and what that is worth to it
        for name, share in config["also_takes"].items():
            level = self.sbml_medium[name]
            half = config.get("also_km", {}).get(name, config["km"])
            taken[name] = share * u * config["vmax"] * level / (half + level)
            gained += config.get("also_worth", {}).get(name, 1.0) * taken[name]
        other = taken.get(self.nutrient, 0.0)  # of the nutrient: the receiver pays for it with its density
        own = needed - gained
        if not 0.0 < own < u * config["vmax"]:
            raise WorldError(
                f"{self.design}: the sender cannot settle on {config['substrate']} at activity {u:g} "
                f"(it would need to take {own:g} mM/min of it, of at most {u * config['vmax']:g})"
            )
        substrate = config["km"] * own / (u * config["vmax"] - own)
        density_a = k * (self.feed[config["substrate"]] - substrate) / own
        if not density_a > 0.0:
            raise WorldError(
                f"{self.design}: the feed of {config['substrate']} is below what the sender needs"
            )
        density_b = self.biovolume - density_a * other / self.uptake_at_rest
        if not self.receiver.get("growth", True):
            density_b = self.biovolume  # a density that is held: the nutrient settles instead, in ``rest``
        if not density_b > 0.0:
            raise WorldError(
                f"{self.design}: the sender takes all of {self.nutrient}; the receiver has no rest state"
            )
        medium = dict(self.sbml_medium)
        medium[config["substrate"]] = substrate
        if config["product"]:
            medium[config["product"]] = density_a * config["product_yield"] * own / k
        return {
            "medium": medium,
            "B": {
                "density": density_b,
                "growth_rate": k,
                "species": dict(self.sbml_species),
                "rates": dict(self.sbml_rates),
                "uptake": {name: -self.sbml_production[name] for name in self.medium_ids},
            },
            "A": {"density": density_a, "uptake": own, "growth_rate": k, "work": density_a * own},
        }

    # ---- the composite ---------------------------------------------------------------------------------------------

    def document(
        self,
        state: dict,
        activity: float | None = None,
        present: bool = True,
        receiver_present: bool = True,
        store: bool = True,
        schedule: tuple[list[float], list[float]] | None = None,
    ) -> dict:
        """The document of the composite that starts at ``state`` with the activity of the sender at ``activity``.

        ``present=False`` takes the sender away at time zero: its density is zero, and everything else is as in
        ``state``. ``receiver_present=False`` does the same to the SBML organism. ``schedule`` is (times in
        minutes, values): an activity that changes in time, each value holding from its time on.
        """
        activity = self.baseline if activity is None else float(activity)
        if schedule is not None:
            activity = float(schedule[1][0])
        overrides = {f"[{name}]": float(value) for name, value in state["B"]["species"].items()}
        overrides.update({self.held_as[name]: float(state["medium"][name]) for name in self.medium_ids})
        sender = dict(state["A"])
        if not present:
            sender.update(density=0.0, uptake=0.0, growth_rate=0.0, work=0.0)
        tree = {
            "supply": {"dilution": self.dilution, "feed": dict(self.feed)},
            "medium": dict(state["medium"]),
            "B": {
                "density": float(state["B"]["density"]) if receiver_present else 0.0,
                "growth_rate": float(state["B"]["growth_rate"]),
                "species": dict(state["B"]["species"]),
                "rates": dict(state["B"]["rates"]),
                "uptake": dict(state["B"]["uptake"]),
                "organism": {
                    "_type": "process",
                    "address": "local:SBMLOrganism",
                    "config": self._receiver_config(overrides),
                    "interval": self.interval,
                    "inputs": {
                        "medium": ["..", "medium"],
                        "density": ["density"],
                        "dilution": ["..", "supply", "dilution"],
                    },
                    "outputs": {
                        "medium": ["..", "medium"],
                        "density": ["density"],
                        "species": ["species"],
                        "rates": ["rates"],
                        "uptake": ["uptake"],
                        "growth_rate": ["growth_rate"],
                    },
                },
            },
            "A": {
                "activity": activity,
                "density": float(sender["density"]),
                "uptake": float(sender["uptake"]),
                "growth_rate": float(sender["growth_rate"]),
                "work": float(sender["work"]),
                "organism": {
                    "_type": "process",
                    "address": "local:LawOrganism",
                    "config": self._sender_config(),
                    "interval": self.interval,
                    "inputs": {
                        "medium": ["..", "medium"],
                        "density": ["density"],
                        "activity": ["activity"],
                        "dilution": ["..", "supply", "dilution"],
                    },
                    "outputs": {
                        "medium": ["..", "medium"],
                        "density": ["density"],
                        "uptake": ["uptake"],
                        "growth_rate": ["growth_rate"],
                        "work": ["work"],
                    },
                },
            },
            "chemostat": {
                "_type": "process",
                "address": "local:ChemostatSupply",
                "config": {},
                "interval": self.interval,
                "inputs": {
                    "medium": ["medium"],
                    "feed": ["supply", "feed"],
                    "dilution": ["supply", "dilution"],
                },
                "outputs": {"medium": ["medium"]},
            },
        }
        if schedule is not None:
            tree["A"]["schedule"] = {
                "_type": "process",
                "address": "local:ActivitySchedule",
                "config": {
                    "times": [float(at) * self.minute for at in schedule[0]],
                    "values": [float(value) for value in schedule[1]],
                },
                "interval": self.interval,
                "inputs": {},
                "outputs": {"activity": ["activity"]},
            }
        if store:
            tree["emitter"] = {
                "_type": "step",
                "address": "local:RAMEmitter",
                "config": {
                    "emit": {
                        "time": "float",
                        "medium": "map[float]",
                        "B.density": "float",
                        "B.growth_rate": "float",
                        "B.uptake": "map[float]",
                        "B.species": "map[float]",
                        "B.rates": "map[float]",
                        "A.activity": "float",
                        "A.density": "float",
                        "A.uptake": "float",
                        "A.growth_rate": "float",
                        "A.work": "float",
                    },
                    "subsample": int(self.plan["time"]["store_every"]),
                },
                "inputs": {
                    "time": ["global_time"],
                    "medium": ["medium"],
                    "B.density": ["B", "density"],
                    "B.growth_rate": ["B", "growth_rate"],
                    "B.uptake": ["B", "uptake"],
                    "B.species": ["B", "species"],
                    "B.rates": ["B", "rates"],
                    "A.activity": ["A", "activity"],
                    "A.density": ["A", "density"],
                    "A.uptake": ["A", "uptake"],
                    "A.growth_rate": ["A", "growth_rate"],
                    "A.work": ["A", "work"],
                },
            }
        return {"state": tree, "skip_process_state": True}

    @staticmethod
    def _snapshot(tree: dict) -> dict:
        """The state of a running composite, in the form that ``document`` takes."""
        return {
            "medium": {name: float(value) for name, value in tree["medium"].items()},
            "B": {
                "density": float(tree["B"]["density"]),
                "growth_rate": float(tree["B"]["growth_rate"]),
                "species": {name: float(value) for name, value in tree["B"]["species"].items()},
                "rates": {name: float(value) for name, value in tree["B"]["rates"].items()},
                "uptake": {name: float(value) for name, value in tree["B"]["uptake"].items()},
            },
            "A": {key: float(tree["A"][key]) for key in ("density", "uptake", "growth_rate", "work")},
        }

    @staticmethod
    def _vector(state: dict) -> np.ndarray:
        """What must stand still at rest: the medium, the two densities and the species of the SBML model."""
        values = [
            *state["medium"].values(),
            state["B"]["density"],
            state["A"]["density"],
            *state["B"]["species"].values(),
        ]
        return np.array(values, dtype=float)

    def _ahead(self, state: dict, last: list[np.ndarray]) -> dict | None:
        """Where a world that creeps to rest is going: the limit of its last three states, one residence time
        apart, if what is left of the approach is one slow mode (each step the same share of the one before).

        None when the steps do not say so, or when the limit would not be a state (a level below zero).
        """
        first, second = last[1] - last[0], last[2] - last[1]
        size = np.abs(last[2]) + 1e-12
        a, b = first / size, second / size
        if not float(a @ a) > 0.0:
            return None
        ratio = float(a @ b) / float(a @ a)
        if not 0.0 < ratio < 1.0 or float(np.linalg.norm(b - ratio * a)) > 0.02 * float(np.linalg.norm(b)):
            return None
        limit = last[2] + second * ratio / (1.0 - ratio)
        if np.any(limit < 0.0) or not np.all(np.isfinite(limit)):
            return None
        medium, species = list(state["medium"]), list(state["B"]["species"])
        ahead = {
            "medium": dict(zip(medium, map(float, limit[: len(medium)]), strict=True)),
            "B": dict(state["B"]),
            "A": dict(state["A"]),
        }
        ahead["B"]["density"], ahead["A"]["density"] = (
            float(limit[len(medium)]),
            float(limit[len(medium) + 1]),
        )
        ahead["B"]["species"] = dict(zip(species, map(float, limit[len(medium) + 2 :]), strict=True))
        return ahead

    def rest(self, activity: float | None = None, start: dict | None = None) -> tuple[dict, float, float]:
        """The rest state of the world at the baseline activity: (state, what still moves, minutes it took).

        With ``activity`` it is the rest state at that activity, and with ``start`` the search begins at that
        state instead of the closed form.

        The composite is followed from the closed form, one residence time after another, until nothing moves by
        more than ``rest_tolerance`` of its size over a residence time. Where the closed form is not the rest
        state and the world creeps towards it (a nutrient that the model hardly reads is held only weakly), the
        search goes ahead to where the last steps point (``_ahead``) and is followed again from there.
        """
        from process_bigraph import Composite

        times = self.plan["time"]
        residence = round(1.0 / self.dilution / self.interval) * self.interval
        begin = self.guess() if start is None else start
        simulation = Composite(self.document(begin, activity=activity, store=False), core=self.core)
        previous = self._vector(self._snapshot(simulation.state))
        moved, spent, last = np.inf, 0.0, [previous]
        for _ in range(int(times["rest_residence_times"])):
            simulation.run(residence)
            spent += residence
            state = self._snapshot(simulation.state)
            now = self._vector(state)
            moved = float(np.max(np.abs(now - previous) / (np.abs(now) + 1e-12)))
            if moved <= float(times["rest_tolerance"]):
                break
            previous = now
            last.append(now)
            if len(last) >= 7:  # six residence times of creeping: look ahead
                ahead = self._ahead(state, last[-3:])
                if ahead is not None:
                    simulation = Composite(
                        self.document(ahead, activity=activity, store=False), core=self.core
                    )
                    previous = self._vector(ahead)
                last = [previous]
        return state, moved, spent / self.minute

    def follow(self, document: dict) -> tuple[np.ndarray, dict[str, np.ndarray], str | None]:
        """Run a document over the horizon of the plan (see ``follow_document``)."""
        return follow_document(self.core, document, float(self.plan["time"]["horizon"]), self.minute)


def follow_document(
    core, document: dict, horizon: float, minute: float = 1.0
) -> tuple[np.ndarray, dict[str, np.ndarray], str | None]:
    """Run a document that has an emitter over ``horizon`` minutes; ``minute`` is how many units of time of the
    document a minute is.

    Returns the stored times in minutes, every stored column, and None, or, when the run could not be followed to
    the horizon (the integrator of an SBML model gave up), what was stored until then and the reason.
    """
    from process_bigraph import Composite, gather_emitter_results

    simulation = Composite(document, core=core)
    problem = None
    try:
        simulation.run(float(horizon) * minute)
    except RuntimeError as exc:  # an SBML model left the range in which it can be integrated
        reason = " ".join(str(exc).split())
        problem = f"stopped at t = {float(simulation.state['global_time']) / minute:g} min: {reason[:300]}"
    times, columns = columns_of(gather_emitter_results(simulation)[("emitter",)])
    return times / minute, columns, problem


def columns_of(rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The rows of the emitter as columns: ``medium.GLCo``, ``B.species.ATP``, ``B.rates.HXT``, ``A.density``, ..."""
    names: list[str] = []
    for key, value in rows[0].items():
        if key == "time":
            continue
        if isinstance(value, dict):
            names += [f"{key}.{inner}" for inner in value]
        else:
            names.append(key)
    columns = {name: np.empty(len(rows)) for name in names}
    times = np.empty(len(rows))
    for k, row in enumerate(rows):
        times[k] = row["time"]
        for key, value in row.items():
            if key == "time":
                continue
            if isinstance(value, dict):
                for inner, number in value.items():
                    columns[f"{key}.{inner}"][k] = number
            else:
                columns[key][k] = value
    return times, columns


def write_series(path: Path, times: np.ndarray, columns: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *columns])
        for k, at in enumerate(times):
            writer.writerow([repr(float(at)), *(repr(float(values[k])) for values in columns.values())])
    return path


__all__ = [
    "BASELINE",
    "NO_RECEIVER",
    "REMOVAL",
    "World",
    "WorldError",
    "columns_of",
    "follow_document",
    "load_plan",
    "make_core",
    "sbml_file",
    "write_series",
]
