"""The parts of the studies that are arithmetic: plans that differ in one thing, indices, schedules."""

from __future__ import annotations

import numpy as np
import pytest

from xeno_biomodels.studies import comparison, schedules
from xeno_biomodels.studies.common import changed, number, slug, truth


def test_a_changed_plan_differs_where_it_was_told_to_and_leaves_the_plan_alone():
    plan = {"supply": {"dilution": 0.5}, "receivers": {"a": {"biovolume": 1.0}, "b": {"biovolume": 2.0}}}
    other = changed(
        plan,
        {
            "supply.dilution": lambda value: 2.0 * value,
            "receivers.*.biovolume": 3.0,
            "receivers.a.growth": False,
        },
    )
    assert other["supply"]["dilution"] == 1.0
    assert other["receivers"] == {"a": {"biovolume": 3.0, "growth": False}, "b": {"biovolume": 3.0}}
    assert plan == {
        "supply": {"dilution": 0.5},
        "receivers": {"a": {"biovolume": 1.0}, "b": {"biovolume": 2.0}},
    }
    assert slug("sender: capacity V x0.25") == "sender_capacity_V_x0.25"
    assert number("") is None and number("None") is None and number("2.5") == 2.5
    assert truth("True") and not truth("False") and not truth("")


def test_niche_overlap_is_the_cosine_of_what_two_organisms_take():
    assert comparison.pianka({"X": 3.0}, {"N": 2.0}) == 0.0  # nothing in common
    assert comparison.pianka({"N": 3.0}, {"N": 0.2}) == pytest.approx(1.0)  # the same, whatever the amounts
    shares = comparison.pianka({"X": 9.0, "N": 1.0}, {"N": 5.0})
    assert shares == pytest.approx(0.1 / np.sqrt(0.9**2 + 0.1**2))
    assert (
        comparison.pianka({"X": 1.0}, {"X": 0.0}) == 0.0
    )  # an organism that takes nothing overlaps with nobody


def test_what_an_organism_of_laws_takes_follows_its_law():
    config = {"substrate": "X", "vmax": 20.0, "km": 1.0, "also_takes": {"N": 0.1}, "also_km": {"N": 2.0}}
    taken = comparison.taken_by_law(config, {"X": 3.0, "N": 2.0, "W": 9.0}, 2.0)
    assert taken == pytest.approx({"X": 2.0 * 20.0 * 3.0 / 4.0, "N": 0.1 * 2.0 * 20.0 * 2.0 / 4.0})


def test_rank_correlation_counts_ties_by_their_mean_rank():
    assert comparison.rank_correlation([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 25.0, 900.0]) == pytest.approx(1.0)
    assert comparison.rank_correlation([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    tied = comparison.rank_correlation([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0])
    assert tied == pytest.approx(np.corrcoef([0.5, 0.5, 2.5, 2.5], [0.0, 1.0, 2.0, 3.0])[0, 1])
    assert comparison.rank_correlation([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) is None
    assert comparison.rank_correlation([1.0, 2.0], [1.0, 2.0]) is None


def test_the_worst_switching_follows_the_sign_of_the_response_to_a_pulse():
    times = np.linspace(0.0, 100.0, 101)
    overshoot = np.where(times < 30.0, 1.0, -0.5)  # up for 30 minutes, then back
    starts, values = schedules.switching(overshoot, times, 0.5, 1.5)
    # the activity takes the sign of h(T - t): low while h(T - t) is negative, high for the last 30 minutes
    assert (starts, values) == ([0.0, 71.0], [0.5, 1.5])
    assert (
        schedules.switching(np.ones(101), times, 0.5, 1.5) is None
    )  # a response of one sign: the step itself
    nearly = np.ones(101)
    nearly[50:] = -1e-6  # as good as zero: not a change of sign
    assert schedules.switching(nearly, times, 0.5, 1.5) is None


def test_the_waves_of_the_plan_start_high_and_start_low():
    plan = {"time": {"horizon": 40.0}, "studies": {"schedules": {"low": 0.5, "high": 2.0, "periods": [20]}}}
    found = schedules.waves(plan)
    assert list(found) == ["step to high", "step to low", "wave 20 min, high first", "wave 20 min, low first"]
    tag, (times, values) = found["wave 20 min, high first"]
    assert (times, values) == ([0.0, 10.0, 20.0, 30.0], [2.0, 0.5, 2.0, 0.5])
    assert tag == {"schedule_kind": "wave", "period": 20.0, "starts": "high"}
    assert found["wave 20 min, low first"][1][1] == [0.5, 2.0, 0.5, 2.0]
    assert found["step to low"][1] == ([0.0], [0.5])


# ---- the worlds that the studies ask for ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def plan():
    from xeno_biomodels.world import load_plan

    return load_plan()


def test_sensitivity_changes_one_assumption_at_a_time(plan):
    from xeno_biomodels.studies import sensitivity

    found = sensitivity.variants(plan)
    assert found["reference"][1] is plan
    assert len(found) == 1 + len(sensitivity.PARAMETERS) * len(sensitivity.FACTORS) + 2 + len(
        sensitivity.HORIZONS
    )
    tag, variant = found["sender: capacity V x4"]
    assert tag == {"parameter": "sender: capacity V", "factor": 4.0}
    assert (
        variant["sender"]["vmax"] == 4.0 * plan["sender"]["vmax"]
        and variant["sender"]["km"] == plan["sender"]["km"]
    )
    thin = found["density of the receiver x0.25"][1]["receivers"]
    assert [body["biovolume"] for body in thin.values()] == [
        0.25 * body["biovolume"] for body in plan["receivers"].values()
    ]
    assert all(body["growth"] is False for body in found["density held"][1]["receivers"].values())
    assert found["horizon x2"][1]["time"]["horizon"] == 2.0 * plan["time"]["horizon"]


def test_numerics_stores_every_interval_on_the_times_of_the_plan(plan):
    from xeno_biomodels.studies import numerics

    for _, variant in numerics.variants(plan).values():
        time = variant["time"]
        assert (
            time["interval"] * time["store_every"] == plan["time"]["interval"] * plan["time"]["store_every"]
        )
    assert numerics.variants(plan)["tolerances looser (1e-08)"][1]["time"]["relative_tolerance"] == 1e-8


def test_channels_give_each_receiver_a_second_way_in(plan):
    from xeno_biomodels.studies import channels

    found = {(biomodel, design): (tag, variant) for tag, variant, biomodel, design in channels.variants(plan)}
    tag, variant = found[("BIOMD0000000017", "other-0.1")]
    assert tag == {"channel": "other", "species": "O2", "share": 0.1}
    assert variant["receivers"]["BIOMD0000000017"]["medium"] == ["halfglucose", "O2"]
    assert variant["designs"]["other-0.1"] == {"share": 0.1, "of": "O2"}
    assert variant["designs"]["nutrient-0.1"] == {"share": 0.1} and variant["designs"]["xeno"] == 0.0
    assert variant["protect"]["specific"]["BIOMD0000000017"]["medium"] == "medium.O2"
    assert variant["receivers"]["BIOMD0000000172"]["medium"] == [
        "GLCo"
    ]  # the others are as the plan has them
    assert (
        "medium" not in plan["protect"]["specific"]["BIOMD0000000017"]
    )  # and the plan itself is not touched


def test_receivers_are_taken_by_the_rule_of_the_plan(plan, monkeypatch):
    import json

    from xeno_biomodels.studies import receivers

    def sensed(rises: float, moves: float, uptake: float = 2.0) -> dict:
        return {"level": 1.0, "uptake": uptake, "uptake_change": rises, "largest_species_change": moves}

    screen = [
        {
            "biomodel_id": "A",
            "name": "reads it",
            "steady": "True",
            "largest_real_eigenvalue": "-0.5",
            "sensed": json.dumps({"N": sensed(0.02, 0.1), "M": sensed(0.0005, 0.1)}),
        },
        {
            "biomodel_id": "B",
            "name": "is not moved by it",
            "steady": "True",
            "largest_real_eigenvalue": "-0.5",
            "sensed": json.dumps({"N": sensed(0.02, 0.001)}),
        },
        {
            "biomodel_id": "C",
            "name": "gives it off",
            "steady": "True",
            "largest_real_eigenvalue": "-0.5",
            "sensed": json.dumps({"N": sensed(0.02, 0.1, uptake=-2.0)}),
        },
        {
            "biomodel_id": "D",
            "name": "does not settle",
            "steady": "False",
            "largest_real_eigenvalue": "",
            "sensed": "",
        },
        {
            "biomodel_id": "E",
            "name": "is stable by rounding alone",
            "steady": "True",
            "largest_real_eigenvalue": "-1e-17",
            "sensed": json.dumps({"N": sensed(0.02, 0.1)}),
        },
    ]
    monkeypatch.setattr(receivers, "read_rows", lambda path: screen)
    found = receivers.candidates(plan)
    assert [item["key"] for item in found] == ["A@N"]
    assert found[0]["uptake_rises_by"] == 0.02 and found[0]["is_a_receiver_of_the_plan"] is False


def test_the_time_of_half_the_peak_is_the_first_stored_time_at_or_above_it():
    from xeno_biomodels.studies import cascade

    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    assert cascade._half(times, np.array([0.0, 0.2, 0.5, 1.0, 0.3])) == 20.0
    assert cascade._half(times, np.zeros(5)) is None


def test_pruning_takes_away_the_directories_that_hold_nothing(tmp_path):
    from xeno_biomodels.experiment import prune

    (tmp_path / "world" / "u=2").mkdir(parents=True)
    (tmp_path / "world" / "baseline").mkdir()
    (tmp_path / "world" / "measure").mkdir()
    (tmp_path / "world" / "measure" / "result.json").write_text("{}", encoding="utf-8")
    prune(tmp_path / "world")
    left = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert left == ["world", "world/measure", "world/measure/result.json"]
