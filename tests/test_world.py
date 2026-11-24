"""The processes of a world against their laws, and a world against what it promises.

The tests of a whole world need the checkout of viva-biomodels and the SBML file of BIOMD0000000172 under
``models/`` (``xvb select`` or ``xvb verify`` fetches it); they are skipped where either is missing.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from xeno_biomodels import paths
from xeno_biomodels.world import columns_of

YEAST = "BIOMD0000000172"
HAVE_WORLD = (paths.VIVA_BIOMODELS / "viva_biomodels" / "__init__.py").exists() and any(
    (paths.MODELS / YEAST).glob("*.xml")
)
needs_world = pytest.mark.skipif(
    not HAVE_WORLD, reason="needs third_party/viva-biomodels and models/BIOMD0000000172"
)

TINY = """
model tiny
  compartment cell = 2; compartment vessel = 1;
  species $N in vessel, S in cell, $P in vessel;
  J1: $N -> S; cell * k1 * N;
  J2: 2 S -> $P; cell * k2 * S;
  N = 3; S = 1; P = 0; k1 = 0.5; k2 = 0.1;
end
"""


@pytest.fixture(scope="module")
def core():
    from process_bigraph import allocate_core

    from xeno_biomodels.organisms import register

    return register(allocate_core())


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import tellurium

    path = tmp_path_factory.mktemp("sbml") / "tiny.xml"
    path.write_text(tellurium.antimonyToSBML(TINY), encoding="utf-8")
    return path


# ---- the laws ------------------------------------------------------------------------------------------------------


def test_supply_is_inflow_at_the_feed_and_outflow_at_the_vessel(core):
    from xeno_biomodels.organisms import ChemostatSupply

    supply = ChemostatSupply(config={}, core=core)
    change = supply.update({"medium": {"N": 2.0, "W": 5.0}, "feed": {"N": 10.0}, "dilution": 0.1}, 0.5)
    assert change["medium"]["N"] == pytest.approx(0.1 * (10.0 - 2.0) * 0.5)
    assert change["medium"]["W"] == pytest.approx(0.1 * (0.0 - 5.0) * 0.5)  # not in the feed: only washed out


def law_organism(core, **more):
    from xeno_biomodels.organisms import LawOrganism

    config = {
        "substrate": "X",
        "product": "W",
        "vmax": 20.0,
        "km": 1.0,
        "growth_yield": 2.5e-4,
        "maintenance": 2.0e-4,
        "product_yield": 1.0,
        **more,
    }
    return LawOrganism(config=config, core=core)


def test_law_organism_follows_its_three_laws(core):
    organism = law_organism(core)
    state = {"medium": {"X": 3.0, "W": 0.0, "N": 2.0}, "density": 0.004, "activity": 2.0, "dilution": 0.001}
    change = organism.update(state, 0.25)
    uptake = 2.0 * 20.0 * 3.0 / (1.0 + 3.0)  # u V S / (K + S)
    growth = 2.5e-4 * uptake - 2.0e-4  # Y q - m
    assert change["uptake"] == pytest.approx(uptake) and change["growth_rate"] == pytest.approx(growth)
    assert change["medium"] == pytest.approx({"X": -0.004 * uptake * 0.25, "W": 0.004 * uptake * 0.25})
    assert change["density"] == pytest.approx(0.004 * (growth - 0.001) * 0.25)
    assert change["work"] == pytest.approx(0.004 * uptake)
    assert "N" not in change["medium"]  # it touches its substrate and its product, and nothing else


def test_a_transporter_that_is_not_specific_takes_the_other_species_and_grows_on_it(core):
    organism = law_organism(core, also_takes={"N": 0.1})
    state = {"medium": {"X": 3.0, "W": 0.0, "N": 2.0}, "density": 0.004, "activity": 1.0, "dilution": 0.001}
    change = organism.update(state, 0.25)
    own, other = 20.0 * 3.0 / 4.0, 0.1 * 20.0 * 2.0 / 3.0
    assert change["medium"]["N"] == pytest.approx(-0.004 * other * 0.25)
    assert change["growth_rate"] == pytest.approx(2.5e-4 * (own + other) - 2.0e-4)
    assert change["uptake"] == pytest.approx(own)


def test_exchange_matrix_is_per_volume_of_the_cell(tiny):
    from xeno_biomodels.organisms import exchange_matrix

    matrix = exchange_matrix(str(tiny), ["N", "P"], ["J1", "J2"])
    assert matrix == pytest.approx(np.array([[-0.5, 0.0], [0.0, 0.5]]))  # the cell has volume 2


def test_sbml_organism_reads_the_medium_and_returns_what_its_reactions_took(core, tiny):
    from xeno_biomodels.organisms import SBMLOrganism

    organism = SBMLOrganism(
        config={"model_file": str(tiny), "bulk_species": False, "medium_species": ["N"]}, core=core
    )
    change = organism.update({"medium": {"N": 4.0, "X": 9.0}, "density": 0.01, "dilution": 0.002}, 0.5)
    # J1 = cell k1 N = 2 * 0.5 * 4, per cell volume 2: the model takes k1 N = 2 of N per cell volume and time
    assert change["uptake"]["N"] == pytest.approx(2.0)
    assert change["medium"] == pytest.approx({"N": -0.01 * 2.0 * 0.5})  # and only N: it does not know X
    assert change["growth_rate"] == pytest.approx(0.002) and change["density"] == pytest.approx(
        0.0
    )  # no growth law asked
    assert set(change["species"]) == {"S"} and set(change["rates"]) == {"J1", "J2"}
    # dS/dt = (2 k1 N - 2 * 2 k2 S) / 2 = k1 N - 2 k2 S, from S = 1 with N = 4, over 0.5
    expected = 10.0 + (1.0 - 10.0) * np.exp(-0.2 * 0.5)
    assert change["species"]["S"] == pytest.approx(expected, rel=1e-6)


def test_sbml_organism_grows_with_the_uptake_of_its_nutrient(core, tiny):
    from xeno_biomodels.organisms import SBMLOrganism

    config = {
        "model_file": str(tiny),
        "bulk_species": False,
        "medium_species": ["N"],
        "growth_nutrient": "N",
        "uptake_at_rest": 1.5,
    }
    organism = SBMLOrganism(config=config, core=core)
    change = organism.update({"medium": {"N": 4.0}, "density": 0.01, "dilution": 0.002}, 0.5)
    assert change["growth_rate"] == pytest.approx(0.002 * 2.0 / 1.5)  # mu = k J / J_rest
    assert change["density"] == pytest.approx(0.01 * (0.002 * 2.0 / 1.5 - 0.002) * 0.5)
    with pytest.raises(ValueError, match="neither boundary species nor parameters"):
        SBMLOrganism(
            config={"model_file": str(tiny), "bulk_species": False, "medium_species": ["S"]}, core=core
        ).update({"medium": {"S": 1.0}, "density": 0.01, "dilution": 0.002}, 0.5)


GROWING = """
model growing
  compartment cell = 2;
  species S in cell;
  J1: -> S; cell * k1 * s0;
  J2: S -> ; cell * k2 * S;
  lam := g * S;
  s0 = 3; S = 1; k1 = 0.5; k2 = 0.2; g = 0.01;
end
"""


def test_a_model_that_holds_its_nutrient_as_a_parameter_and_grows_at_a_rate_of_its_own(core, tmp_path):
    import tellurium

    from xeno_biomodels.organisms import SBMLOrganism

    path = tmp_path / "growing.xml"
    path.write_text(tellurium.antimonyToSBML(GROWING), encoding="utf-8")
    config = {
        "model_file": str(path),
        "bulk_species": False,
        "medium_species": ["s0"],  # a parameter of the model, not a species
        "uptake_reactions": {"s0": "J1"},  # the reaction whose rate is what the cell takes of it
        "growth_variable": "lam",
    }
    organism = SBMLOrganism(config=config, core=core)
    change = organism.update({"medium": {"s0": 4.0}, "density": 0.01, "dilution": 0.002}, 0.5)
    assert organism.level("s0") == 4.0  # the model was told the level of the medium
    # J1 = cell k1 s0 = 2 * 0.5 * 4: per cell volume the model takes k1 s0 = 2, and the medium loses it
    assert change["uptake"]["s0"] == pytest.approx(2.0)
    assert change["medium"] == pytest.approx({"s0": -0.01 * 2.0 * 0.5})
    # dS/dt = k1 s0 - k2 S, from S = 1 towards 10; the organism grows at the mean of g S over the interval
    end = 10.0 + (1.0 - 10.0) * np.exp(-0.2 * 0.5)
    assert change["species"]["S"] == pytest.approx(end, rel=1e-6)
    assert change["growth_rate"] == pytest.approx(0.5 * 0.01 * (1.0 + end), rel=1e-6)
    assert change["density"] == pytest.approx(0.01 * (change["growth_rate"] - 0.002) * 0.5)
    assert organism.own_growth() == pytest.approx(0.01 * end, rel=1e-6)
    with pytest.raises(ValueError, match="no reaction"):
        SBMLOrganism(config={**config, "uptake_reactions": {"s0": "J9"}}, core=core).update(
            {"medium": {"s0": 4.0}, "density": 0.01, "dilution": 0.002}, 0.5
        )


def test_growth_with_a_maintenance_is_the_same_at_rest_and_falls_faster_below(core, tiny):
    from xeno_biomodels.organisms import SBMLOrganism

    config = {
        "model_file": str(tiny),
        "bulk_species": False,
        "medium_species": ["N"],
        "growth_nutrient": "N",
        "uptake_at_rest": 2.0,  # what the model takes at N = 4
        "maintenance": 0.002,
    }
    organism = SBMLOrganism(config=config, core=core)
    at_rest = organism.update({"medium": {"N": 4.0}, "density": 0.01, "dilution": 0.002}, 0.5)
    assert at_rest["growth_rate"] == pytest.approx(0.002)  # (k + m) J / J_rest - m = k where J = J_rest
    below = SBMLOrganism(config=config, core=core).update(
        {"medium": {"N": 2.0}, "density": 0.01, "dilution": 0.002}, 0.5
    )
    assert below["growth_rate"] == pytest.approx(0.004 * 0.5 - 0.002)  # no growth at half the uptake


def test_a_leak_can_have_its_own_half_saturation_and_be_worth_nothing(core):
    organism = law_organism(core, also_takes={"N": 0.1}, also_km={"N": 2.0}, also_worth={"N": 0.0})
    state = {"medium": {"X": 3.0, "W": 0.0, "N": 2.0}, "density": 0.004, "activity": 1.0, "dilution": 0.001}
    change = organism.update(state, 0.25)
    own, other = 20.0 * 3.0 / 4.0, 0.1 * 20.0 * 2.0 / (2.0 + 2.0)  # half-saturated at N = 2
    assert change["medium"]["N"] == pytest.approx(-0.004 * other * 0.25)  # it takes it
    assert change["growth_rate"] == pytest.approx(2.5e-4 * own - 2.0e-4)  # and does not grow on it


def test_an_activity_schedule_holds_each_value_from_its_time_on(core):
    from xeno_biomodels.organisms import ActivitySchedule

    schedule = ActivitySchedule(config={"times": [0.0, 1.0, 2.5], "values": [2.0, 0.5, 4.0]}, core=core)
    assert [schedule.value(at) for at in (0.0, 0.99, 1.0, 2.49, 2.5, 100.0)] == [2.0, 2.0, 0.5, 0.5, 4.0, 4.0]
    # over an interval it writes what holds at the end of it, for the organism to read at the start of the next
    written = [schedule.update({}, 0.5)["activity"] for _ in range(6)]
    assert written == [2.0, 0.5, 0.5, 0.5, 4.0, 4.0]  # at t = 0.5, 1, 1.5, 2, 2.5, 3
    for times, values in (([1.0, 2.0], [1.0, 1.0]), ([0.0, 2.0, 2.0], [1.0, 1.0, 1.0]), ([0.0, 1.0], [1.0])):
        with pytest.raises(ValueError, match="schedule"):
            ActivitySchedule(config={"times": times, "values": values}, core=core)


def test_a_creeping_search_looks_ahead_to_the_limit_of_its_steps():
    from xeno_biomodels.world import World

    limit = np.array([2.0, 0.5, 1e-3, 4e-3, 3.0, 7.0])  # medium (2), the two densities, species (2)
    mode = np.array([0.3, -0.1, 2e-4, 0.0, -0.5, 0.1])
    last = [limit + mode * 0.9**n for n in (5, 6, 7)]  # one slow mode: each step 0.9 of the one before
    state = {
        "medium": {"N": 0.0, "X": 0.0},
        "B": {"density": 0.0, "growth_rate": 0.1, "species": {"a": 0.0, "b": 0.0}, "rates": {}, "uptake": {}},
        "A": {"density": 0.0, "uptake": 1.0, "growth_rate": 0.1, "work": 0.0},
    }
    ahead = World._ahead(None, state, last)
    assert list(ahead["medium"].values()) == pytest.approx([2.0, 0.5], rel=1e-9)
    assert (ahead["B"]["density"], ahead["A"]["density"]) == pytest.approx((1e-3, 4e-3), rel=1e-9)
    assert list(ahead["B"]["species"].values()) == pytest.approx([3.0, 7.0], rel=1e-9)
    assert ahead["A"]["uptake"] == 1.0  # what is not a state is left as it was
    two_modes = [limit + mode * 0.9**n + mode[::-1] * 0.5**n for n in (0, 1, 2)]
    assert World._ahead(None, state, two_modes) is None  # the steps do not say where it goes
    assert World._ahead(None, state, [limit, limit, limit]) is None  # nothing moves: nothing to look ahead to
    below = [
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) - np.array([0, 0, 0, 0, 0, 0.5]) * (2.0 - 0.9**n)
        for n in (0, 1, 2)
    ]
    assert World._ahead(None, state, below) is not None
    falling = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.3 * 0.9**n - 0.2]) for n in (0, 1, 2)]
    assert World._ahead(None, state, falling) is None  # the limit would be a level below zero


# ---- the files -----------------------------------------------------------------------------------------------------


def test_columns_of_flattens_the_rows_of_the_emitter():
    rows = [
        {"time": 0.0, "medium": {"N": 2.0, "X": 1.0}, "B.density": 0.1, "B.species": {"ATP": 3.0}},
        {"time": 5.0, "medium": {"N": 1.5, "X": 0.5}, "B.density": 0.2, "B.species": {"ATP": 2.0}},
    ]
    times, columns = columns_of(rows)
    assert list(times) == [0.0, 5.0]
    assert list(columns) == ["medium.N", "medium.X", "B.density", "B.species.ATP"]
    assert list(columns["B.species.ATP"]) == [3.0, 2.0]


# ---- a whole world -------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plan():
    from xeno_biomodels.world import load_plan

    short = copy.deepcopy(load_plan())
    short["time"].update(horizon=120.0, store_every=8)
    return short


@pytest.fixture(scope="module")
def world_core():
    from xeno_biomodels.world import make_core

    return make_core()


@needs_world
def test_the_steady_state_of_the_sbml_model_is_the_rest_state_of_the_world(plan, world_core):
    from xeno_biomodels.world import World

    world = World(plan, YEAST, "xeno", core=world_core)
    k, rho = world.dilution, world.biovolume
    assert world.sbml_medium == {"GLCo": 2.0}  # the level that the SBML file gives external glucose
    assert world.feed["GLCo"] == pytest.approx(2.0 + rho * world.uptake_at_rest / k)  # what holds it there
    state, moved, _ = world.rest()
    assert moved < 1e-10
    assert state["medium"]["GLCo"] == pytest.approx(2.0, rel=1e-12)
    assert state["B"]["growth_rate"] == pytest.approx(k, rel=1e-10)  # it grows at the dilution rate
    for name, value in world.sbml_species.items():
        assert state["B"]["species"][name] == pytest.approx(value, rel=1e-10)
    # the sender on its own substrate: growth is dilution, and what flows in is what it takes
    assert state["A"]["growth_rate"] == pytest.approx(k, rel=1e-10)
    supplied = k * (world.feed["X"] - state["medium"]["X"])
    assert state["A"]["density"] * state["A"]["uptake"] == pytest.approx(supplied, rel=1e-9)


@needs_world
def test_a_sender_that_shares_no_species_leaves_every_number_of_the_receiver_as_it_was(plan, world_core):
    from xeno_biomodels.world import World

    world = World(plan, YEAST, "xeno", core=world_core)
    state, _, _ = world.rest()
    _, base, problem = world.follow(world.document(state))
    assert problem is None
    for document in (world.document(state, activity=4.0), world.document(state, present=False)):
        _, run, _ = world.follow(document)
        assert not np.array_equal(run["medium.X"], base["medium.X"])  # the sender did something
        for name in base:
            if name.startswith("B.") or name == "medium.GLCo":
                assert np.array_equal(run[name], base[name]), name  # and the receiver has the same bits
    _, alone, _ = world.follow(world.document(state, receiver_present=False))
    for name in base:
        if name.startswith("A.") or name in ("medium.X", "medium.W"):
            assert np.array_equal(alone[name], base[name]), (
                name
            )  # the sender does not notice the receiver either


@needs_world
def test_a_sender_that_takes_the_nutrient_of_the_receiver_moves_it(plan, world_core):
    from xeno_biomodels.world import World

    world = World(plan, YEAST, "leak-1e-1", core=world_core)
    state, moved, _ = world.rest()
    assert moved < 1e-10
    assert state["medium"]["GLCo"] == pytest.approx(
        2.0, rel=1e-10
    )  # the receiver still sets its nutrient at rest
    assert state["B"]["density"] < world.biovolume  # and pays for the sharing with its density
    _, base, _ = world.follow(world.document(state))
    _, run, _ = world.follow(world.document(state, activity=4.0))
    assert np.min(run["medium.GLCo"]) < 0.95 * base["medium.GLCo"][0]
    assert not np.array_equal(run["B.species.ATP"], base["B.species.ATP"])


@needs_world
def test_a_leak_written_as_a_share_takes_that_share_at_rest(plan, world_core):
    from xeno_biomodels.world import World

    shared = copy.deepcopy(plan)
    shared["designs"] = {"xeno": 0.0, "share": {"share": 0.1}}
    alone = World(shared, YEAST, "xeno", core=world_core)
    world = World(shared, YEAST, "share", core=world_core)
    state, moved, _ = world.rest()
    assert moved < 1e-10
    assert state["medium"]["GLCo"] == pytest.approx(
        2.0, rel=1e-10
    )  # the nutrient stays where the SBML file has it
    assert state["B"]["density"] == pytest.approx(
        0.9 * world.biovolume, rel=1e-9
    )  # the receiver gives up the share
    xeno, _, _ = alone.rest()
    assert state["A"]["density"] == pytest.approx(
        xeno["A"]["density"], rel=1e-9
    )  # the sender gains nothing by it
    config = world._sender_config()
    assert config["also_km"] == {"GLCo": 2.0} and config["also_worth"] == {"GLCo": 0.0}
    taken = (
        config["also_takes"]["GLCo"] * config["vmax"] * 2.0 / (2.0 + 2.0)
    )  # per cell volume, at activity 1
    assert state["A"]["density"] * taken == pytest.approx(
        0.1 * world.biovolume * world.uptake_at_rest, rel=1e-9
    )


@needs_world
def test_a_schedule_moves_the_activity_of_the_sender_in_time(plan, world_core):
    from xeno_biomodels.world import World

    world = World(plan, YEAST, "leak-1e-1", core=world_core)
    state, _, _ = world.rest()
    times, run, problem = world.follow(world.document(state, schedule=([0.0, 30.0, 60.0], [2.0, 0.5, 4.0])))
    assert problem is None
    stored = dict(zip(times.tolist(), run["A.activity"].tolist(), strict=True))
    assert (stored[0.0], stored[28.0], stored[30.0], stored[58.0], stored[60.0], stored[120.0]) == (
        2.0,
        2.0,
        0.5,
        0.5,
        4.0,
        4.0,
    )
    _, step, _ = world.follow(world.document(state, activity=2.0))
    assert np.array_equal(
        run["medium.GLCo"][times <= 30.0], step["medium.GLCo"][times <= 30.0]
    )  # a step until it changes
    assert not np.array_equal(run["medium.GLCo"][times > 34.0], step["medium.GLCo"][times > 34.0])
