"""Several pairs in one vessel: the pouring, and who can reach whom.

The tests of a whole community need the checkout of viva-biomodels and the SBML files of the three receivers
under ``models/``; they are skipped where either is missing.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from xeno_biomodels import paths
from xeno_biomodels.community_report import chain

RECEIVERS = ("BIOMD0000000172", "BIOMD0000000217", "BIOMD0000000017")
HAVE = (paths.VIVA_BIOMODELS / "viva_biomodels" / "__init__.py").exists() and all(
    any((paths.MODELS / name).glob("*.xml")) for name in RECEIVERS
)
needs_models = pytest.mark.skipif(
    not HAVE, reason="needs third_party/viva-biomodels and the three SBML files"
)


# ---- who can reach whom, from what each organism reads and writes --------------------------------------------------


def member(reads, writes=None, **more):
    writes = reads if writes is None else writes
    return {"reads": sorted(reads), "writes": sorted(writes), "touches": sorted({*reads, *writes}), **more}


def test_a_chain_goes_through_what_one_writes_and_the_next_reads():
    members = {
        "Y": member({"glucose"}),
        "E": member({"NH4"}),
        "xY": member({"X", "NH4"}, {"X", "NH4", "W"}),
        "xE": member({"X", "glucose"}, {"X", "glucose", "W"}),
    }
    assert chain(members, "xE", "Y") == ["xE", "glucose", "Y"]
    assert chain(members, "xE", "E") == ["xE", "X", "xY", "NH4", "E"]  # through the other xeno, not through W
    assert chain(members, "Y", "E") == ["Y", "glucose", "xE", "X", "xY", "NH4", "E"]


def test_a_product_that_nothing_reads_carries_nothing_further():
    members = {
        "a": member({"X"}, {"X", "W"}),
        "b": member({"Z"}, {"Z", "W"}),  # excretes the same product, takes something else
        "c": member({"Z"}),
    }
    assert chain(members, "a", "b") == ["a", "W", "b"]  # b protects its product: the measure reads it
    assert chain(members, "a", "c") is None  # but b's state did not move, so nothing goes on to c
    assert chain(members, "c", "a") == [
        "c",
        "Z",
        "b",
        "W",
        "a",
    ]  # c does move b, and b's product is a's product


# ---- the organism of an SBML model under another name and unit -----------------------------------------------------


def test_sbml_organism_reads_a_chemical_under_its_own_name_and_unit(tmp_path):
    import tellurium
    from process_bigraph import allocate_core

    from xeno_biomodels.organisms import SBMLOrganism, register

    path = tmp_path / "tiny.xml"
    path.write_text(
        tellurium.antimonyToSBML(
            "model tiny\n compartment cell = 1; species $half in cell, S in cell;\n"
            " J1: $half -> S; cell * k1 * half;\n half = 3; S = 1; k1 = 0.5;\nend\n"
        ),
        encoding="utf-8",
    )
    config = {
        "model_file": str(path),
        "bulk_species": False,
        "medium_species": ["half"],
        "medium_keys": {"half": "whole"},
        "medium_per_unit": {"half": 2.0},
    }
    organism = SBMLOrganism(config=config, core=register(allocate_core()))
    change = organism.update({"medium": {"whole": 4.0}, "density": 0.01, "dilution": 0.002}, 0.5)
    assert change["uptake"]["half"] == pytest.approx(0.5 * 8.0)  # the model sees 2 * 4 of its halves
    assert change["medium"] == pytest.approx(
        {"whole": -0.01 * 4.0 / 2.0 * 0.5}
    )  # and takes half as many wholes


# ---- a whole community ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plan():
    from xeno_biomodels.world import load_plan

    short = copy.deepcopy(load_plan())
    short["time"].update(horizon=40.0, store_every=8)
    short["community"].update(assembly=40.0)
    return short


@pytest.fixture(scope="module")
def core():
    from xeno_biomodels.world import make_core

    return make_core()


def follow(community, state, **what):
    from xeno_biomodels.world import follow_document

    _, columns, problem = follow_document(community.core, community.document(state, **what), 40.0)
    assert problem is None
    return columns


def same(run, base, names) -> bool:
    return all(np.array_equal(run[key], base[key]) for key in base if key.split(".")[0] in names)


@needs_models
def test_pouring_takes_the_mean_of_the_vessels(plan, core):
    from xeno_biomodels.community import Community

    community = Community(plan, "shared", core=core)
    poured = community.pour()
    yeast, lactis = community.pairs[0].world, community.pairs[2].world
    assert poured["medium"]["glucose"] == pytest.approx(
        (2.0 + 0.0 + 30.0 / 2.0) / 3.0
    )  # halves counted as wholes
    assert poured["medium"]["NH4"] == pytest.approx(0.05 / 3.0)
    assert community.feed["glucose"] == pytest.approx(
        (yeast.feed["GLCo"] + lactis.feed["halfglucose"] / 2.0) / 3.0
    )
    assert poured["organisms"]["Y"]["density"] == pytest.approx(yeast.biovolume / 3.0)
    # three vessels with the same X: the mean is that X, and the three xenos together are one population at rest
    assert community.feed["X"] == pytest.approx(20.0)
    state, _, _ = yeast.rest()
    assert poured["medium"]["X"] == pytest.approx(state["medium"]["X"])
    assert sum(poured["organisms"][name]["density"] for name in ("xY", "xE", "xL")) == pytest.approx(
        state["A"]["density"]
    )


@needs_models
def test_with_substrates_of_their_own_a_xeno_moves_nobody(plan, core):
    from xeno_biomodels.community import Community

    community = Community(plan, "private", core=core)
    state = community.pour()
    base = follow(community, state)
    run = follow(community, state, activities={"xY": 4.0})
    assert not np.array_equal(run["medium.X_xY"], base["medium.X_xY"])  # it did something
    assert same(run, base, {"Y", "E", "L", "xE", "xL"})  # and no organism has another number for it
    for key in ("medium.glucose", "medium.NH4", "medium.X_xE", "medium.X_xL"):
        assert np.array_equal(run[key], base[key]), key
    gone = follow(community, state, absent=("Y",))  # the two that live on glucose are another matter
    assert not np.array_equal(gone["L.density"], base["L.density"])
    assert same(gone, base, {"E", "xY", "xE", "xL"})


@needs_models
def test_on_one_substrate_the_xenos_move_each_other_and_no_sbml_organism(plan, core):
    from xeno_biomodels.community import Community

    community = Community(plan, "shared", core=core)
    state = community.pour()
    base = follow(community, state)
    run = follow(community, state, activities={"xY": 4.0})
    assert run["xE.density"][-1] < base["xE.density"][-1] and run["xL.density"][-1] < base["xL.density"][-1]
    assert same(run, base, {"Y", "E", "L"})
    members = community.members()
    assert chain(members, "xY", "xE") == ["xY", "X", "xE"]
    assert chain(members, "xY", "Y") is None and chain(members, "Y", "xY") is None
    assert chain(members, "Y", "L") == ["Y", "glucose", "L"]


@needs_models
def test_a_xeno_that_is_orthogonal_to_its_partner_only_reaches_it_through_the_others(plan, core):
    from xeno_biomodels.community import Community

    community = Community(plan, "partner-only", core=core)
    members = community.members()
    assert not set(members["xE"]["touches"]) & set(members["E"]["touches"])  # the pair shares no species
    assert chain(members, "xE", "E") == ["xE", "X", "xY", "NH4", "E"]
    state = community.pour()
    base = follow(community, state)
    run = follow(community, state, activities={"xE": 8.0})
    assert not np.array_equal(run["medium.NH4"], base["medium.NH4"])
    assert not same(run, base, {"E"})  # its own partner has other numbers now
