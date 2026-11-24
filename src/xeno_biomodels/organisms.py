"""The processes of a world: the supply of a chemostat, an organism that is an SBML model, an organism of laws.

All three are processes of Vivarium 2.0 (process-bigraph). They share one store, ``medium``, the concentrations
in the vessel, and each returns what it adds to that store over an interval; the composite adds the three up.
Every exchange is a rate at the start of the interval times the interval (for the SBML organism, the mean of
the rates at its two ends), so a state in which the rates balance stays where it is whatever the interval: the
steady state of the SBML model is a steady state of the composite, exactly.

The laws that the world adds to an SBML model are written here and nowhere else, and they are assumptions of
this world, not parts of the curated model:

    supply      dS/dt = k (S_feed - S) for every species of the medium                     (chemostat)
    exchange    a species of the medium changes by  density * (net production per cell volume)
    growth      mu = (k + m) * J / J_rest - m, with J the uptake of the limiting nutrient: a constant yield
                and a maintenance m (zero unless given), the yield fixed by mu = k at the curated steady state;
                d(density)/dt = (mu - k) density
"""

from __future__ import annotations

from itertools import pairwise
from typing import ClassVar

import libsbml
import numpy as np
from process_bigraph import Process

from xeno_biomodels import preload_libpython

preload_libpython()

from viva_tellurium.processes import TelluriumProcess  # libroadrunner needs the preload


class ChemostatSupply(Process):
    """The supply: medium flows in at the feed concentrations and out at those of the vessel.

    The feed and the dilution rate are stores, read by this process and (the dilution rate) by every organism,
    so that one node of the bigraph is the supply that all of them are under.
    """

    config_schema: ClassVar[dict] = {}

    def inputs(self):
        return {"medium": "map[float]", "feed": "map[float]", "dilution": "float"}

    def outputs(self):
        return {"medium": "map[float]"}

    def update(self, state, interval):
        k, feed = state["dilution"], state["feed"]
        return {
            "medium": {
                name: k * (feed.get(name, 0.0) - value) * interval for name, value in state["medium"].items()
            }
        }


def cell_volume(sbml_file: str) -> float:
    """The volume of the cell of an SBML model: the compartments that hold a species that the model integrates."""
    model = libsbml.readSBMLFromFile(str(sbml_file)).getModel()
    if model is None:
        raise ValueError(f"{sbml_file}: not an SBML model")
    ruled = {rule.getVariable() for rule in model.getListOfRules()}
    cell = set()
    for item in model.getListOfSpecies():
        if not item.getBoundaryCondition() and not item.getConstant() and item.getId() not in ruled:
            cell.add(item.getCompartment())
    volume = sum(model.getCompartment(name).getSize() for name in cell)
    if not volume > 0.0:
        raise ValueError(f"{sbml_file}: the compartments of the integrated species have no volume")
    return float(volume)


def exchange_matrix(sbml_file: str, species: list[str], reactions: list[str]) -> np.ndarray:
    """Net production of each of ``species`` per unit of each reaction rate, per cell volume: (species, reaction).

    The cell is the compartments that hold a species that the model integrates. A reaction rate of SBML is an
    amount per time, so it is divided by the volume of the cell to give a change of concentration in the cell.
    """
    volume = cell_volume(sbml_file)
    model = libsbml.readSBMLFromFile(str(sbml_file)).getModel()
    matrix = np.zeros((len(species), len(reactions)))
    for j, name in enumerate(reactions):
        reaction = model.getReaction(name)
        for sign, references in ((-1.0, reaction.getListOfReactants()), (1.0, reaction.getListOfProducts())):
            for reference in references:
                if reference.getSpecies() in species:
                    if reference.isSetStoichiometryMath():
                        raise ValueError(f"{sbml_file}: the stoichiometry of {name} is a formula")
                    matrix[species.index(reference.getSpecies()), j] += sign * reference.getStoichiometry()
    return matrix / volume


class SBMLOrganism(TelluriumProcess):
    """An SBML model as an organism in a medium.

    It is the Tellurium process of viva-tellurium, which viva-biomodels uses for its ``biomodel-process``
    composite: the same loading of the model, the same integrator (CVODE of libroadrunner), the same stepping.
    What is added is the interface of an organism. Boundary species of the model, which the SBML file holds
    constant, are read from the medium before each interval (``medium_species``), and what the reactions of the
    model take from them or give to them is returned as a change of the medium, times the density of the
    organism. The model itself, its species, its rate laws and its parameters, is not touched.

    A species of the model need not have the name or the unit of its chemical in the vessel: ``medium_keys`` gives
    the name in the medium of a species of the model, and ``medium_per_unit`` how many units of the species of the
    model one unit of the medium is (a model that counts glucose in halves has 2). Both default to the species
    itself, one to one.

    Two things let a model of a growing cell be an organism as it is. A model may hold what is outside it as a
    constant parameter and not as a boundary species; a medium species may then be that parameter, and since no
    reaction lists a parameter as a reactant, ``uptake_reactions`` names the reactions whose rate is its uptake.
    And a model may have a growth rate of its own: ``growth_variable`` names it, and the organism then grows at
    that rate, whatever ``growth_nutrient`` says.

    Ports
        medium       in: concentrations in the vessel.  out: what the organism adds to them (additive)
        density      in, out (additive): volume of cells per volume of medium
        dilution     in: the dilution rate of the supply
        species      out: concentrations of the species that the model integrates
        rates        out: the reaction rates of the model
        uptake       out: what the organism takes of each medium species, per cell volume, over the interval
        growth_rate  out: mu over the interval
    """

    config_schema: ClassVar[dict] = {
        **TelluriumProcess.config_schema,
        "medium_species": {"_type": "list[string]", "_default": []},
        "medium_keys": {"_type": "map[string]", "_default": {}},
        "medium_per_unit": {"_type": "map[float]", "_default": {}},
        "growth_nutrient": {"_type": "string", "_default": ""},
        "uptake_at_rest": {"_type": "float", "_default": 0.0},
        "maintenance": {"_type": "float", "_default": 0.0},
        "uptake_reactions": {"_type": "map[string]", "_default": {}},
        "growth_variable": {"_type": "string", "_default": ""},
    }

    def inputs(self):
        return {"medium": "map[float]", "density": "float", "dilution": "float"}

    def outputs(self):
        return {
            "medium": "map[float]",
            "density": "float",
            "species": "overwrite[map[float]]",
            "rates": "overwrite[map[float]]",
            "uptake": "overwrite[map[float]]",
            "growth_rate": "overwrite[float]",
        }

    def initial_state(self):
        return {}  # the document carries the whole state at rest

    def _build(self):
        if self._rr is not None:
            return
        super()._build()
        config = self.config
        self._medium_ids = list(config["medium_species"])
        boundary = set(self._rr.getBoundarySpeciesIds())
        constant = set(self._rr.getGlobalParameterIds())
        unknown = [name for name in self._medium_ids if name not in boundary and name not in constant]
        if unknown:
            raise ValueError(
                f"{unknown} are neither boundary species nor parameters of the model; its boundary species "
                f"are {sorted(boundary)}"
            )
        # how the model is told the level of each medium species: a concentration, or the value of a parameter
        self._held = [f"[{name}]" if name in boundary else name for name in self._medium_ids]
        self._exchange = exchange_matrix(config["model_file"], self._medium_ids, self._reaction_ids)
        for name, listed in config["uptake_reactions"].items():
            if name not in self._medium_ids:
                raise ValueError(f"uptake_reactions names {name}, which is not a medium species")
            volume = cell_volume(config["model_file"])
            for reaction in listed.split():
                if reaction not in self._reaction_ids:
                    raise ValueError(f"the model has no reaction {reaction}")
                self._exchange[self._medium_ids.index(name), self._reaction_ids.index(reaction)] -= (
                    1.0 / volume
                )
        self._keys = [config["medium_keys"].get(name, name) for name in self._medium_ids]
        self._per_unit = [float(config["medium_per_unit"].get(name, 1.0)) for name in self._medium_ids]
        if len(set(self._keys)) != len(self._keys) or not all(factor > 0.0 for factor in self._per_unit):
            raise ValueError(
                "every medium species of the model needs its own name in the medium and a positive unit"
            )
        self._nutrient = None
        if config["growth_nutrient"]:
            if config["growth_nutrient"] not in self._medium_ids:
                raise ValueError(f"the growth nutrient {config['growth_nutrient']} is not a medium species")
            if not config["uptake_at_rest"] > 0.0:
                raise ValueError("growth needs the uptake of the nutrient at rest, and it must be positive")
            self._nutrient = self._medium_ids.index(config["growth_nutrient"])

    def production(self) -> np.ndarray:
        """Net production of each medium species per cell volume, at the state the model is in."""
        return self._exchange @ np.asarray(self._rr.getReactionRates(), dtype=float)

    def level(self, name: str) -> float:
        """The level at which the model holds a medium species."""
        return float(self._rr[self._held[self._medium_ids.index(name)]])

    def hold(self, name: str, value: float) -> None:
        """Tell the model the level of a medium species."""
        self._rr[self._held[self._medium_ids.index(name)]] = max(float(value), 0.0)

    def own_growth(self) -> float | None:
        """The growth rate that the model itself has at the state it is in; None for a model without one."""
        return float(self._rr[self.config["growth_variable"]]) if self.config["growth_variable"] else None

    def read(self) -> dict:
        """The species that the model integrates and its reaction rates, at the state the model is in."""
        self._build()
        concentrations = np.asarray(self._rr.getFloatingSpeciesConcentrations(), dtype=float)
        rates = np.asarray(self._rr.getReactionRates(), dtype=float)
        return {
            "species": {name: float(concentrations[i]) for i, name in enumerate(self._species_ids)},
            "rates": {name: float(rates[i]) for i, name in enumerate(self._reaction_ids)},
        }

    def update(self, state, interval):
        self._build()
        model = self._rr
        for name, key, per_unit in zip(self._medium_ids, self._keys, self._per_unit, strict=True):
            self.hold(name, per_unit * float(state["medium"][key]))
        before, grew = self.production(), self.own_growth()
        start = model.getCurrentTime()
        model.simulate(start, start + interval, 2)
        produced = 0.5 * (before + self.production())  # per cell volume, mean over the interval

        density, dilution = float(state["density"]), float(state["dilution"])
        growth = dilution
        if grew is not None:  # the model grows at its own rate: the mean over the interval, as for the uptake
            growth = 0.5 * (grew + self.own_growth())
        elif self._nutrient is not None:
            growth = dilution * (-produced[self._nutrient]) / self.config["uptake_at_rest"]
            upkeep = self.config["maintenance"]
            if upkeep:  # a yield and a maintenance: the same growth at rest, a steeper fall below it
                growth = (dilution + upkeep) * (-produced[self._nutrient]) / self.config[
                    "uptake_at_rest"
                ] - upkeep
        return {
            "medium": {
                key: density * produced[i] / self._per_unit[i] * interval for i, key in enumerate(self._keys)
            },
            "density": density * (growth - dilution) * interval,
            "uptake": {name: float(-produced[i]) for i, name in enumerate(self._medium_ids)},
            "growth_rate": float(growth),
            **self.read(),
        }


class LawOrganism(Process):
    """An organism made of three laws, living on one substrate of the medium.

        uptake        q = u V S / (K + S)                     (Michaelis-Menten; u is its activity)
        growth        mu = Y (q + what else it takes) - m      (Pirt: a constant yield and a maintenance)
        excretion     one product, ``product_yield`` per substrate taken

    ``also_takes`` lists other species of the medium that its transporter takes, each with a specificity relative
    to its own substrate: it takes s at  e u V s / (K_s + s)  and grows on it with the same yield. K_s is K unless
    ``also_km`` gives another half-saturation for the species, and ``also_worth`` gives what a unit of the species
    is worth for growth, relative to a unit of its own substrate (1 unless given; 0 for a species that it takes
    and cannot use). With nothing listed, the organism touches its substrate and its product and nothing else.

    Ports
        medium       in: concentrations in the vessel.  out: what the organism adds to them (additive)
        density      in, out (additive): volume of cells per volume of medium
        activity     in: u, the knob that is stepped at time zero
        dilution     in: the dilution rate of the supply
        uptake       out: q at the start of the interval
        growth_rate  out: mu at the start of the interval
        work         out: density * q, the substrate that the population converts per volume of medium and time
    """

    config_schema: ClassVar[dict] = {
        "substrate": "string",
        "product": {"_type": "string", "_default": ""},
        "vmax": "float",
        "km": "float",
        "growth_yield": "float",
        "maintenance": "float",
        "product_yield": {"_type": "float", "_default": 1.0},
        "also_takes": {"_type": "map[float]", "_default": {}},
        "also_km": {"_type": "map[float]", "_default": {}},
        "also_worth": {"_type": "map[float]", "_default": {}},
    }

    def inputs(self):
        return {"medium": "map[float]", "density": "float", "activity": "float", "dilution": "float"}

    def outputs(self):
        return {
            "medium": "map[float]",
            "density": "float",
            "uptake": "overwrite[float]",
            "growth_rate": "overwrite[float]",
            "work": "overwrite[float]",
        }

    def initial_state(self):
        return {}

    def rates(self, medium: dict, activity: float) -> tuple[float, dict, float]:
        """(uptake of its own substrate, uptake of every other species it takes, growth rate), per cell volume."""
        config = self.config

        def taken(name: str, share: float, half: float) -> float:
            level = max(float(medium[name]), 0.0)
            return share * activity * config["vmax"] * level / (half + level)

        own = taken(config["substrate"], 1.0, config["km"])
        other = {
            name: taken(name, share, config["also_km"].get(name, config["km"]))
            for name, share in config["also_takes"].items()
        }
        if config["also_worth"]:
            gained = own + sum(config["also_worth"].get(name, 1.0) * rate for name, rate in other.items())
        else:
            gained = own + sum(other.values())
        growth = config["growth_yield"] * gained - config["maintenance"]
        return own, other, growth

    def update(self, state, interval):
        config = self.config
        density, dilution = float(state["density"]), float(state["dilution"])
        own, other, growth = self.rates(state["medium"], float(state["activity"]))
        change = {config["substrate"]: -density * own * interval}
        for name, rate in other.items():
            change[name] = -density * rate * interval
        if config["product"]:
            change[config["product"]] = density * config["product_yield"] * own * interval
        return {
            "medium": change,
            "density": density * (growth - dilution) * interval,
            "uptake": own,
            "growth_rate": growth,
            "work": density * own,
        }


class ActivitySchedule(Process):
    """The activity of an organism as a function of time, piecewise constant.

    ``values[i]`` holds from ``times[i]`` until ``times[i + 1]``; the first time is zero and the last value holds
    to the end. The process keeps its own clock and, over each interval, writes the value that holds at the end
    of it, so that the organism reads at the start of every interval the value of that moment. The value at time
    zero is the one that the document gives the store.
    """

    config_schema: ClassVar[dict] = {"times": "list[float]", "values": "list[float]"}

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._now = 0.0
        times, values = list(self.config["times"]), list(self.config["values"])
        ordered = all(later > earlier for earlier, later in pairwise(times))
        if len(times) != len(values) or not times or times[0] != 0.0 or not ordered:
            raise ValueError(
                "a schedule has as many values as times, and times that start at zero and increase"
            )

    def inputs(self):
        return {}

    def outputs(self):
        return {"activity": "overwrite[float]"}

    def initial_state(self):
        return {}

    def value(self, at: float) -> float:
        """The activity that holds at a time."""
        index = int(np.searchsorted(self.config["times"], at, side="right")) - 1
        return float(self.config["values"][max(index, 0)])

    def update(self, state, interval):
        self._now += interval
        return {"activity": self.value(self._now + 1e-9 * interval)}


def register(core):
    """Make the processes of a world known to a core under their class names."""
    for cls in (ChemostatSupply, SBMLOrganism, LawOrganism, ActivitySchedule):
        core.register_link(cls.__name__, cls)
    return core
