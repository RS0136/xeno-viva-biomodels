"""The information that a noisy record carries, the evaluations with tolerances from the function (adverse
changes, conserved pairs, the margin used), and a leak that is written as a share."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from xeno_biomodels import paths
from xeno_biomodels.paper import figures
from xeno_biomodels.studies import functional, information
from xeno_biomodels.studies.common import leak_of

ECOLI = "BIOMD0000000217"
HAVE_ECOLI = (paths.PROJECT / "third_party" / "viva-biomodels").exists() and any(
    (paths.MODELS / ECOLI).glob("*.xml")
)
needs_ecoli = pytest.mark.skipif(
    not HAVE_ECOLI, reason="needs third_party/viva-biomodels and models/BIOMD0000000217"
)


# ---- the information that a noisy record carries --------------------------------------------------------------------


def test_runs_that_coincide_say_nothing_and_runs_that_are_far_apart_say_which_one_it_was():
    same = np.zeros((5, 40))
    assert information.information(same, 1.0, samples=2000) == pytest.approx(0.0, abs=1e-12)
    apart = 1000.0 * np.eye(5)  # five runs, each far from every other against a noise of one
    assert information.information(apart, 1.0, samples=2000) == pytest.approx(np.log2(5.0), abs=1e-6)
    # two of the five coincide: an observer can tell four things apart, and the pair only as a pair
    apart[1] = apart[0]
    expected = np.log2(5.0) - (2.0 / 5.0) * 1.0  # two of five runs leave one bit open
    assert information.information(apart, 1.0, samples=2000) == pytest.approx(expected, abs=1e-6)


def test_the_information_of_two_runs_is_that_of_a_binary_input_with_gaussian_noise():
    distance, sigma = 2.0, 1.0
    points = np.array([[0.0], [distance]])
    # by quadrature: I = 1 - E log2(1 + exp(-(d^2 + 2 d sigma z) / (2 sigma^2))) for z standard normal
    z = np.linspace(-12.0, 12.0, 240001)
    weight = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    inside = np.log2(1.0 + np.exp(-(distance**2 + 2.0 * distance * sigma * z) / (2.0 * sigma**2)))
    exact = 1.0 - float(np.sum(weight * inside) * (z[1] - z[0]))
    assert information.information(points, sigma, samples=400000) == pytest.approx(exact, abs=3e-3)
    assert 0.0 < exact < 1.0


def test_the_information_depends_on_the_runs_only_through_their_distances():
    random = np.random.default_rng(7)
    points = random.normal(size=(4, 3))
    turn, _ = np.linalg.qr(random.normal(size=(50, 50)))  # an orthogonal map of a larger space
    longer = np.hstack([points, np.zeros((4, 47))]) @ turn.T + 3.0  # embedded, turned and moved
    # the same up to the draws of the noise (the two take their draws along different axes)
    assert information.information(longer, 0.8, samples=200000) == pytest.approx(
        information.information(points, 0.8, samples=200000), abs=0.01
    )
    # more noise, less said
    assert information.information(points, 4.0, samples=20000) < information.information(
        points, 0.8, samples=20000
    )


def test_the_noise_at_which_half_is_still_said_scales_with_the_runs():
    points = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]])
    at = information.half(points)
    assert at is not None
    assert information.half(10.0 * points) == pytest.approx(10.0 * at, rel=0.1)
    assert information.half(np.zeros((3, 2))) is None  # nothing is said at any noise


# ---- the evaluations with tolerances from the function --------------------------------------------------------------


def _courses():
    """A baseline and a run of two quantities and the growth rate, at three stored times."""
    base = {"a": np.full(3, 1.0), "b": np.full(3, 2.0), "B.growth_rate": np.full(3, 1.0)}
    run = {
        "a": np.array([1.0, 0.9, 1.0]),  # falls: against a function that rises with it
        "b": np.array([2.0, 1.6, 2.0]),  # falls: for a function that falls with it
        "B.growth_rate": np.array([1.0, 0.98, 1.05]),
    }
    return run, base, np.array([True, True, True])


def test_only_the_changes_that_lower_the_function_are_adverse():
    run, base, stored = _courses()
    slope, margin = {"a": 0.5, "b": -0.25}, 0.5
    delta = {key: margin / (2 * abs(value)) for key, value in slope.items()}  # 0.5 and 1.0
    found = functional.readings(run, base, stored, delta, slope, [], margin)
    assert found["two_sided"][:, 1] == pytest.approx([0.1 / 0.5, 0.4 / 1.0])
    assert found["adverse"][:, 1] == pytest.approx([0.1 / 0.5, 0.0])  # b moved in the direction that helps
    assert np.all(found["adverse"] <= found["two_sided"] + 1e-15)
    # the margin used is what the growth rate itself lost, and a growth rate that rose used none of it
    assert found["used"][0] == pytest.approx([0.0, 0.02 / 0.5, 0.0])


def test_a_conserved_pair_counts_once_and_is_read_along_the_direction_in_which_it_can_move():
    base = {"a": np.full(2, 1.0), "b": np.full(2, 3.0), "c": np.full(2, 5.0), "B.growth_rate": np.ones(2)}
    run = {**base, "a": np.array([1.0, 1.2]), "b": np.array([3.0, 2.8]), "c": np.array([5.0, 5.5])}
    slope, margin = {"a": 0.4, "b": -0.1, "c": 0.05}, 0.6
    delta = {key: margin / (3 * abs(value)) for key, value in slope.items()}
    stored = np.array([True, True])
    alone = functional.readings(run, base, stored, delta, slope, [], margin)["independent"]
    assert alone[:, 1] == pytest.approx([0.2 / delta["a"], 0.2 / delta["b"], 0.5 / delta["c"]])
    paired = functional.readings(run, base, stored, delta, slope, [("a", "b")], margin)["independent"]
    assert paired.shape == (2, 2)  # the single quantity, and the pair
    # two quantities share the margin, and the pair moves the function by (0.4 + 0.1) per unit of a
    assert paired[0, 1] == pytest.approx(0.5 / (margin / (2 * 0.05)))
    assert paired[1, 1] == pytest.approx(0.2 / (margin / (2 * 0.5)))


def test_a_function_that_turns_inside_the_box_is_not_monotone_there():
    rest = {"x": 1.0, "y": 1.0}

    def rising(state):
        return state["x"] * state["y"] / (1.0 + state["x"])

    def turning(state):  # largest at x = 1
        return state["y"] * state["x"] * (2.0 - state["x"])

    delta = {"x": 0.5, "y": 0.5}
    assert functional.monotone(rising, rest, delta)[0]
    assert not functional.monotone(turning, rest, delta)[0]
    # where it is monotone the least value is at a corner: the one where both are lowest
    assert functional.least_in_box(rising, rest, delta) == pytest.approx(0.5 * 0.5 / 1.5)


@needs_ecoli
def test_the_two_forms_of_a_carrier_are_found_as_a_pair_from_two_steady_states_of_the_model_alone():
    from xeno_biomodels.world import World, load_plan

    world = World(load_plan(), ECOLI, "xeno")
    names = [
        "medium.NH4",
        "B.species.GLN",
        "B.species.AMP",
        "B.species.GLU",
        "B.species.ATP",
        "B.species.ADP",
    ]
    assert functional.conserved_pairs(world, names) == [("B.species.ATP", "B.species.ADP")]
    assert functional.conserved_pairs(world, ["medium.NH4", "B.species.GLN", "B.species.GLU"]) == []


# ---- a leak that is written as a share ------------------------------------------------------------------------------


def test_a_leak_is_one_number_whether_it_is_a_specificity_or_a_share():
    assert leak_of(0.01) == 0.01
    assert leak_of("0.3") == 0.3
    assert leak_of({"share": 0.1}) == 0.1


def test_a_step_whose_added_capacity_has_no_bound_is_drawn_apart_by_its_activity_and_its_leak():
    places = sorted(
        figures.BEYOND[activity] * figures.APART[leak]
        for activity in figures.BEYOND
        for leak in figures.APART
    )
    assert places[0] > 2.8  # inside the strip, to the right of every step that has a capacity
    assert all(b / a > 1.15 for a, b in pairwise(places))  # and no two in one place
