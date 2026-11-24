"""The SBML model of a receiver on its own, as its file has it, with its surroundings held where one puts them.

A world reads an SBML model through a medium that moves. To say what of a reading is the world's and what is the
model's, the model is also asked alone: its steady state at a given level of what it lives on (its dose-response),
and what its reactions take up at a given state without any integration (the rate law of its uptake as a function
of the state). Both use the process that the worlds use, with no growth and no medium.
"""

from __future__ import annotations

import numpy as np

from xeno_biomodels import layers
from xeno_biomodels.organisms import SBMLOrganism
from xeno_biomodels.world import World


class Isolated:
    """One receiver's SBML model alone. ``levels`` are those of the species that the world lets vary."""

    def __init__(self, world: World, name: str = "B"):
        self.world, self.name = world, name
        self.described = layers.sbml_organism(world.plan, world.biomodel, name)
        self.fresh()

    def fresh(self) -> None:
        """The model at the steady state of its file."""
        config = self.world._receiver_config(growth=False)
        config["growth_variable"] = str(
            self.world.receiver.get("growth_variable") or ""
        )  # read, not followed
        self.organism = SBMLOrganism(config=config, core=self.world.core)
        self.organism._build()
        self.model = self.organism._rr
        self.put(self.world.sbml_species, self.world.sbml_medium)

    def put(self, species: dict[str, float] | None = None, levels: dict[str, float] | None = None) -> None:
        """Set species of the model, and levels of its surroundings, without integrating."""
        for name, value in (levels or {}).items():
            self.organism.hold(name, value)
        for name, value in (species or {}).items():
            self.model[f"[{name}]"] = float(value)

    def uptake(
        self, species: dict[str, float] | None = None, levels: dict[str, float] | None = None
    ) -> float:
        """What the reactions of the model take of the nutrient per cell volume, at a state that is put in."""
        self.put(species, levels)
        index = self.organism._medium_ids.index(self.world.nutrient)
        return float(-self.organism.production()[index])

    def growth(
        self, species: dict[str, float] | None = None, levels: dict[str, float] | None = None
    ) -> float:
        """The growth rate at a state that is put in: that of the model where it has one of its own, and
        otherwise what the growth law of the world makes of its uptake."""
        self.put(species, levels)
        own = self.organism.own_growth()
        if own is not None:
            return own
        index = self.organism._medium_ids.index(self.world.nutrient)
        taken = float(-self.organism.production()[index])
        return self.world.dilution * taken / self.world.uptake_at_rest

    def relax(self, tolerance: float = 1e-10, doublings: int = 16) -> bool:
        """Follow the model at the levels that it has until nothing moves; False where it does not settle."""
        model, previous = self.model, None
        at, span = model.getCurrentTime(), 50.0 * self.world.minute
        try:
            for _ in range(doublings):
                model.simulate(at, at + span, 2)
                at += span
                now = np.asarray(model.getFloatingSpeciesConcentrations(), dtype=float)
                if not np.all(np.isfinite(now)) or np.any(now < -1e-9):
                    return False
                moved = (
                    np.inf
                    if previous is None
                    else float(np.max(np.abs(now - previous) / (np.abs(now) + 1e-12)))
                )
                if moved < tolerance:
                    return True
                previous, span = now, 2.0 * span
        except RuntimeError:
            return False
        return False

    def quantities(self) -> dict[str, float]:
        """Every column that a run stores of the receiver, and the derived ones, at the state the model is in."""
        read, name = self.organism.read(), self.name
        columns = {f"{name}.species.{key}": np.array([value]) for key, value in read["species"].items()}
        columns.update({f"{name}.rates.{key}": np.array([value]) for key, value in read["rates"].items()})
        production = self.organism.production()
        for i, key in enumerate(self.organism._medium_ids):
            columns[f"{name}.uptake.{key}"] = np.array([float(-production[i])])
            columns[f"medium.{key}"] = np.array([self.organism.level(key)])
        columns[f"{name}.growth_rate"] = np.array([self.growth()])
        columns.update(layers.derive(self.world.plan, self.described, columns))
        return {key: float(values[0]) for key, values in columns.items()}

    def steady(self, levels: dict[str, float]) -> dict[str, float] | None:
        """The steady state at given levels, reached from the state the model is in; None where there is none
        (the model is then put back to the steady state of its file)."""
        self.put(None, levels)
        if self.relax():
            return self.quantities()
        self.fresh()
        return None


__all__ = ["Isolated"]
