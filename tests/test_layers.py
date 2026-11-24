"""What an organism protects: the two kinds of quantity, their layers, and the quantities that are derived."""

from __future__ import annotations

import copy
import csv

import numpy as np
import pytest

from xeno_biomodels import layers
from xeno_biomodels.orthogonality import measure, read_series
from xeno_biomodels.world import WorldError, write_series

PLAN = {
    "receivers": {"M1": {"nutrient": "N", "atp": "ATP", "carriers": ["ATP", "ADP"], "proteins": ["E"]}},
    "protect": {
        "common": {
            "medium": "medium.{food}",
            "particle": "{name}.atp_made",
            "molecule": "{name}.pool",
            "complex": "{uptake}",
            "individual": "{name}.growth_rate",
            "population": "{name}.density",
        },
        "specific": {
            "M1": {"molecule": "{name}.species.S", "complex": ["{name}.rates.J2", "{name}.species.E"]},
            "xeno": {"medium": "medium.{product}"},
        },
        "xeno_layers": ["medium", "complex", "individual", "population"],
        "tolerance": "rel:0.1",
        "from_function": ["{name}.growth_rate"],
    },
}

TINY = """
model tiny
  compartment cell = 1;
  species $N in cell, S in cell, T in cell, ATP in cell, ADP in cell, E in cell, $P in cell;
  J1: $N + ATP -> S + ADP; k * N * ATP;
  J2: S + 2 ADP -> T + 2 ATP; k * S * ADP;
  J3: ATP -> ADP; k * ATP;
  J4: T + ATP -> $P + ATP; k * T;
  N = 1; S = 1; T = 1; ATP = 1; ADP = 1; E = 1; P = 0; k = 1;
end
"""


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    import tellurium

    path = tmp_path_factory.mktemp("sbml") / "tiny.xml"
    path.write_text(tellurium.antimonyToSBML(TINY), encoding="utf-8")
    return path


def changed(**protect) -> dict:
    plan = copy.deepcopy(PLAN)
    plan["protect"].update(protect)
    return plan


# ---- what is protected ---------------------------------------------------------------------------------------------


def test_an_sbml_organism_protects_the_common_quantity_of_every_layer_and_its_own():
    entries = layers.protected(PLAN, layers.sbml_organism(PLAN, "M1", "B"))
    assert [(entry.column, entry.layer, entry.kind) for entry in entries] == [
        ("medium.N", "medium", "common"),
        ("B.atp_made", "particle", "common"),
        ("B.pool", "molecule", "common"),
        ("B.uptake.N", "complex", "common"),
        ("B.growth_rate", "individual", "common"),
        ("B.density", "population", "common"),
        ("B.species.S", "molecule", "specific"),
        ("B.rates.J2", "complex", "specific"),
        ("B.species.E", "complex", "specific"),
    ]
    specs = {entry.column: entry.spec for entry in entries}
    assert specs["B.growth_rate"] == "B.growth_rate=fn"  # its tolerance follows from the function
    assert specs.pop("B.growth_rate") and set(specs.values()) == {f"{name}=rel:0.1" for name in specs}


def test_in_a_common_medium_the_food_has_the_name_of_the_medium_and_the_uptake_that_of_the_model():
    entries = layers.protected(PLAN, layers.sbml_organism(PLAN, "M1", "L", food="glucose"))
    assert [entry.column for entry in entries[:4]] == ["medium.glucose", "L.atp_made", "L.pool", "L.uptake.N"]


def test_an_organism_of_laws_protects_the_layers_that_it_has():
    entries = layers.protected(PLAN, layers.xeno_organism("A", "X", "W"))
    assert [(entry.column, entry.layer, entry.kind) for entry in entries] == [
        ("medium.X", "medium", "common"),
        ("A.uptake", "complex", "common"),
        ("A.growth_rate", "individual", "common"),
        ("A.density", "population", "common"),
        ("medium.W", "medium", "specific"),
    ]
    assert entries[2].tolerance == "fn"


def test_what_the_plan_cannot_mean_is_refused():
    sbml, xeno = layers.sbml_organism(PLAN, "M1", "B"), layers.xeno_organism("A", "X", "W")
    with pytest.raises(WorldError, match="no layer organelle"):
        layers.protected(changed(common={"organelle": "{name}.density"}), sbml)
    with pytest.raises(WorldError, match="has no layer molecule"):
        layers.protected(changed(specific={"xeno": {"molecule": "{name}.density"}}), xeno)
    with pytest.raises(WorldError, match="product"):  # an SBML organism has no product to name
        layers.protected(changed(specific={"M1": {"medium": "medium.{product}"}}), sbml)
    with pytest.raises(WorldError, match="twice"):
        layers.protected(changed(specific={"M1": {"molecule": "{name}.pool"}}), sbml)
    assert (
        layers.protected(changed(specific={}), sbml)[-1].column == "B.density"
    )  # nothing of its own is fine


def test_a_model_in_which_the_plan_names_no_atp_has_no_particle_layer(tiny, monkeypatch):
    monkeypatch.setattr(layers, "sbml_file", lambda biomodel: tiny)
    plain = copy.deepcopy(PLAN)
    plain["receivers"]["M1"].update(atp=None, carriers=[], proteins=[])
    plain["protect"]["specific"] = {}
    organism = layers.sbml_organism(plain, "M1", "B")
    assert layers.layers_of(plain, organism) == ("medium", "molecule", "complex", "individual", "population")
    assert layers.layers_of(PLAN, layers.sbml_organism(PLAN, "M1", "B")) == layers.LAYERS
    assert layers.layers_of(PLAN, layers.xeno_organism("A", "X", "W")) == (
        "medium",
        "complex",
        "individual",
        "population",
    )
    assert "B.atp_made" not in [entry.column for entry in layers.protected(plain, organism)]
    derived = layers.derive(plain, organism, run_columns())
    assert list(derived) == ["B.pool"]
    assert derived["B.pool"] == pytest.approx(
        [1.0 + 0.5 + 7.0 * 3, 2.0 + 0.25 + 7.0 * 3]
    )  # every species: none set apart


def test_a_file_is_read_by_rule_where_the_plan_says_nothing(tiny):
    found = layers.annotate(tiny)
    assert found == {"atp": "ATP", "carriers": ["ADP", "ATP"], "proteins": []}


# ---- the groups in which the measure is read -----------------------------------------------------------------------


def test_groups_are_every_layer_each_kind_and_each_kind_at_each_layer():
    entries = layers.protected(PLAN, layers.sbml_organism(PLAN, "M1", "B"))
    groups = layers.groups(entries)
    assert groups["layer:molecule"] == ["B.pool", "B.species.S"]
    assert groups["molecule/common"] == ["B.pool"] and groups["molecule/specific"] == ["B.species.S"]
    assert groups["complex/specific"] == ["B.rates.J2", "B.species.E"]
    assert "medium/specific" not in groups and "particle/specific" not in groups
    assert groups["kind:specific"] == ["B.species.S", "B.rates.J2", "B.species.E"]
    assert len(groups["kind:common"]) == len(layers.LAYERS)
    everything = sorted(entry.column for entry in entries)
    by_layer = [name for label, names in groups.items() if label.startswith("layer:") for name in names]
    by_kind = [name for label, names in groups.items() if label.startswith("kind:") for name in names]
    by_both = [name for label, names in groups.items() if "/" in label for name in names]
    assert sorted(by_layer) == sorted(by_kind) == sorted(by_both) == everything  # each covers the whole once
    assert layers.split("layer:molecule") == ("molecule", "all")
    assert layers.split("kind:common") == ("all", "common")
    assert layers.split("complex/specific") == ("complex", "specific")


def test_an_organism_of_laws_has_no_group_at_the_layers_that_it_lacks():
    groups = layers.groups(layers.protected(PLAN, layers.xeno_organism("A", "X", "W")))
    assert not any("particle" in label or "molecule" in label for label in groups)
    assert groups["layer:medium"] == ["medium.X", "medium.W"]


# ---- the derived quantities ----------------------------------------------------------------------------------------


def test_atp_coefficients_are_what_a_reaction_forms_less_what_it_uses(tiny):
    assert layers.atp_coefficients(tiny, "ATP") == {
        "J1": -1.0,
        "J2": 2.0,
        "J3": -1.0,
    }  # J4 forms what it uses
    with pytest.raises(WorldError, match="no species GTP"):
        layers.atp_coefficients(tiny, "GTP")
    with pytest.raises(WorldError, match="no reaction forms N"):
        layers.atp_coefficients(tiny, "N")


def run_columns(name: str = "B") -> dict[str, np.ndarray]:
    return {
        "medium.N": np.array([1.0, 0.9]),
        f"{name}.rates.J1": np.array([1.0, -0.5]),  # at the second time it runs backwards, and forms ATP
        f"{name}.rates.J2": np.array([2.0, 3.0]),
        f"{name}.rates.J3": np.array([1.0, 1.0]),
        f"{name}.rates.J4": np.array([9.0, 9.0]),
        f"{name}.species.S": np.array([1.0, 2.0]),
        f"{name}.species.T": np.array([0.5, 0.25]),
        f"{name}.species.ATP": np.array([7.0, 7.0]),
        f"{name}.species.ADP": np.array([7.0, 7.0]),
        f"{name}.species.E": np.array([7.0, 7.0]),
        f"{name}.uptake.N": np.array([1.0, 0.5]),
        f"{name}.growth_rate": np.array([0.1, 0.05]),
        f"{name}.density": np.array([0.01, 0.01]),
    }


def test_atp_made_counts_what_forms_atp_and_the_pool_the_metabolites(tiny, monkeypatch):
    monkeypatch.setattr(layers, "sbml_file", lambda biomodel: tiny)
    derived = layers.derive(PLAN, layers.sbml_organism(PLAN, "M1", "B"), run_columns())
    assert list(derived) == ["B.atp_made", "B.pool"]
    assert derived["B.atp_made"] == pytest.approx(
        [2 * 2.0, 0.5 + 2 * 3.0]
    )  # only what forms it, at each time
    assert derived["B.pool"] == pytest.approx([1.5, 2.25])  # S + T: not the carriers, not the protein
    assert layers.derive(PLAN, layers.xeno_organism("A", "X", "W"), run_columns()) == {}
    everything_apart = copy.deepcopy(PLAN)
    everything_apart["receivers"]["M1"]["proteins"] = ["E", "S", "T"]
    with pytest.raises(WorldError, match="no metabolite is left"):
        layers.derive(everything_apart, layers.sbml_organism(everything_apart, "M1", "B"), run_columns())


def test_write_protected_keeps_what_the_measure_reads_and_nothing_else(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(layers, "sbml_file", lambda biomodel: tiny)
    organism = layers.sbml_organism(PLAN, "M1", "B")
    series = write_series(tmp_path / "run" / "series.csv", np.array([0.0, 10.0]), run_columns())
    target = layers.write_protected(
        series, tmp_path / "run" / "protected.csv", PLAN, [organism], ["B.rates.J4"]
    )
    with target.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    wanted = [entry.column for entry in layers.protected(PLAN, organism)]
    assert header == ["time", *wanted, "B.rates.J4"]
    times, columns = read_series(target)
    assert list(times) == [0.0, 10.0]
    assert columns["B.pool"] == pytest.approx([1.5, 2.25]) and columns["B.atp_made"] == pytest.approx(
        [4.0, 6.5]
    )
    assert np.array_equal(
        columns["B.species.S"], run_columns()["B.species.S"]
    )  # a stored column is not touched
    with pytest.raises(WorldError, match="no column"):
        layers.write_protected(series, tmp_path / "run" / "other.csv", PLAN, [organism], ["B.nothing"])


# ---- from the measure to the rows of a table -----------------------------------------------------------------------


def test_group_rows_name_the_layer_and_the_kind_of_every_group(tiny, tmp_path, monkeypatch):
    monkeypatch.setattr(layers, "sbml_file", lambda biomodel: tiny)
    organism = layers.sbml_organism(PLAN, "M1", "B")
    entries = layers.protected(PLAN, organism)
    times = np.array([0.0, 10.0])
    base = {name: np.full(2, values[0]) for name, values in run_columns().items()}
    moved = {**base, "B.species.S": np.array([1.0, 1.05]), "medium.N": np.array([1.0, 0.98])}
    files = {}
    for name, columns in (("baseline", base), ("u=2", moved)):
        series = write_series(tmp_path / name / "series.csv", times, columns)
        files[name] = layers.write_protected(series, tmp_path / name / "protected.csv", PLAN, [organism], [])
    result, _ = measure(
        files["baseline"],
        {"u=2": files["u=2"]},
        [entry.spec for entry in entries],
        function="B.growth_rate",
        qmin="rel:0.5",
        allocation="dependent",
        groups=layers.groups(entries),
    )
    rows = {(row["layer"], row["kind"]): row for row in layers.group_rows(result)}
    assert rows[("medium", "common")]["D"] == pytest.approx(0.2)  # 2 % of a quantity that tolerates 10 %
    assert rows[("molecule", "specific")]["D"] == pytest.approx(0.5)  # S: 5 % of 10 %
    assert rows[("molecule", "common")]["D"] == pytest.approx(0.05 / 0.15)  # the pool S + T moves by as much
    assert (
        rows[("molecule", "all")]["D"] == pytest.approx(0.5) and rows[("molecule", "all")]["quantities"] == 2
    )
    assert rows[("molecule", "all")]["D_quantity"] == "B.species.S"
    assert rows[("all", "common")]["D"] == pytest.approx(0.05 / 0.15)
    assert rows[("all", "specific")]["D"] == pytest.approx(0.5) == result["D_peak"]
    assert rows[("complex", "specific")]["D"] == 0.0 and rows[("complex", "specific")]["stage1_trajectory"]
    assert layers.group_rows({"status": "ok"}) == []
