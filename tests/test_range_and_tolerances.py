"""A run that stops before the horizon, the range of the activity and its closed forms, the SBML model alone,
tolerances that follow from the function, and a community that coexists stably."""

from __future__ import annotations

import numpy as np
import pytest

from xeno_biomodels import layers, paths
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.paper import figures, style, tables
from xeno_biomodels.studies import activity, coexistence, functional, quasistatic
from xeno_biomodels.world import WorldError, write_series

YEAST = "BIOMD0000000172"
HAVE_WORLD = (paths.PROJECT / "third_party" / "viva-biomodels").exists() and any(
    (paths.MODELS / YEAST).glob("*.xml")
)
needs_world = pytest.mark.skipif(
    not HAVE_WORLD, reason="needs third_party/viva-biomodels and models/BIOMD0000000172"
)


@pytest.fixture(scope="module")
def plan():
    from xeno_biomodels.world import load_plan

    return load_plan()


# ---- a run that stops before the horizon ---------------------------------------------------------------------------


def test_a_run_that_stopped_keeps_its_last_values_and_loses_the_receiver_at_the_next_stored_time(tmp_path):
    times = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    baseline = write_series(
        tmp_path / "baseline.csv", times, {"B.density": np.ones(5), "medium.S": np.full(5, 2.0)}
    )
    partial = tmp_path / "partial.csv"
    partial.write_text("time,B.density,medium.S\n0.0,1.0,2.0\n10.0,0.9,1.5\n20.0,0.8,1.1\n", encoding="utf-8")
    written = layers.write_collapsed(
        partial, baseline, tmp_path / "protected.csv", {}, [], ["B.density", "medium.S"], gone="B.density"
    )
    stored, columns = read_series(written)
    assert list(stored) == list(times)  # on the times of the baseline, as the measure asks
    assert list(columns["B.density"]) == [1.0, 0.9, 0.8, 0.0, 0.0]  # gone from the next stored time on
    assert list(columns["medium.S"]) == [
        2.0,
        1.5,
        1.1,
        1.1,
        1.1,
    ]  # nothing is read there, and nothing is missing


def test_a_run_that_was_stored_to_the_horizon_is_not_a_run_that_stopped(tmp_path):
    times = np.array([0.0, 10.0])
    baseline = write_series(tmp_path / "baseline.csv", times, {"B.density": np.ones(2)})
    partial = tmp_path / "partial.csv"
    partial.write_text("time,B.density\n0.0,1.0\n10.0,0.9\n", encoding="utf-8")
    with pytest.raises(WorldError):
        layers.write_collapsed(
            partial, baseline, tmp_path / "out.csv", {}, [], ["B.density"], gone="B.density"
        )


# ---- the range of the activity -------------------------------------------------------------------------------------


class _Vessel:
    """A world as far as ``activity`` asks it: a receiver that takes 0.05 of its nutrient per volume of medium
    when it is alone, beside a sender of three laws with a leak for that nutrient."""

    baseline = 1.0
    dilution = 1.0 / 600.0
    nutrient = "S"
    sbml_medium = {"S": 2.0}  # noqa: RUF012
    feed = {"X": 20.0}  # noqa: RUF012
    biovolume = 1e-3
    uptake_at_rest = 50.0

    def __init__(self, leak: float = 0.01, worth: float | None = None):
        self.leak, self.worth = leak, worth

    def _sender_config(self) -> dict:
        config = {
            "substrate": "X",
            "vmax": 20.0,
            "km": 1.0,
            "growth_yield": 2.5e-4,
            "maintenance": 2e-4,
            "also_takes": {"S": self.leak} if self.leak else {},
        }
        if self.worth is not None:
            config["also_worth"] = {"S": self.worth}
        return config


def _by_hand(vessel: _Vessel, activity: float) -> tuple[float, float]:
    """(density of the sender, what its population takes of the nutrient) at rest, written out."""
    k, level = vessel.dilution, vessel.sbml_medium["S"]
    needed = (k + 2e-4) / 2.5e-4
    taken = vessel.leak * activity * 20.0 * level / (1.0 + level)
    own = needed - (1.0 if vessel.worth is None else vessel.worth) * taken
    substrate = 1.0 * own / (activity * 20.0 - own)
    density = k * (20.0 - substrate) / own
    return density, density * taken


def test_what_the_sender_takes_at_rest_is_the_closed_form_of_its_three_laws():
    vessel = _Vessel()
    for step in (0.5, 1.0, 8.0):
        assert activity.sender_flux(vessel, step) == pytest.approx(_by_hand(vessel, step), rel=1e-12)
    # a sender that cannot use what it takes needs all of its own substrate, whatever it takes of the nutrient
    useless = _Vessel(worth=0.0)
    density, flux = activity.sender_flux(useless, 8.0)
    assert (density, flux) == pytest.approx(_by_hand(useless, 8.0), rel=1e-12)
    assert density < activity.sender_flux(vessel, 8.0)[0]  # and is the less dense for it


def test_a_sender_that_could_live_on_the_nutrient_alone_has_no_rest_state_at_the_level_of_the_file():
    greedy = _Vessel(leak=0.3)
    assert activity.sender_flux(greedy, 1.0) is not None
    assert activity.sender_flux(greedy, 4.0) is None  # 0.3 * 4 * 20 * 2/3 = 16 is more than the 7.47 it needs
    assert activity.rest_density(greedy, 4.0) is None
    assert activity.sender_flux(_Vessel(), 0.25) is None  # and at a quarter of its capacity it cannot grow


def test_the_receiver_pays_with_its_density_what_the_sender_takes():
    vessel = _Vessel()
    _, flux = _by_hand(vessel, 2.0)
    assert activity.rest_density(vessel, 2.0) == pytest.approx(1e-3 - flux / 50.0, rel=1e-12)
    assert vessel.baseline == 1.0


def test_the_activity_of_washout_is_where_the_sender_takes_all_that_the_receiver_took():
    vessel = _Vessel()
    critical = activity.washout(vessel)
    assert activity.rest_density(vessel, 0.999 * critical) is not None
    assert activity.rest_density(vessel, 1.001 * critical) is None
    _, flux = _by_hand(vessel, critical)
    assert flux == pytest.approx(vessel.biovolume * vessel.uptake_at_rest, rel=1e-6)
    assert activity.washout(_Vessel(leak=0.0)) is None  # a sender that takes nothing of the receiver


def test_the_peak_of_the_nutrient_is_that_of_the_two_equations_it_is_derived_from():
    k, beta = 1.0 / 600.0, 1.0 / 86.0
    times = np.linspace(0.0, 6000.0, 600001)
    course = (np.exp(-k * times) - np.exp(-beta * times)) / (beta - k)  # the nutrient after a unit step
    assert course.max() == pytest.approx(activity.peak_factor(k, beta) / beta, rel=1e-9)
    assert times[int(np.argmax(course))] == pytest.approx(np.log(beta / k) / (beta - k), abs=0.01)
    assert activity.peak_factor(k, k) == pytest.approx(np.exp(-1.0))  # the limit of equal rates
    assert activity.peak_factor(k, k * (1.0 + 1e-6)) == pytest.approx(np.exp(-1.0), rel=1e-5)


def test_the_linear_response_gives_each_quantity_its_own_share_of_the_swing_of_the_nutrient():
    vessel = _Vessel()
    coefficients = {"B.uptake.S": 0.4, "B.pool": 0.7, "B.rates.r": -0.2}
    tolerance = {"medium.S": 0.1, "B.pool": 0.1, "B.rates.r": 0.05, "B.density": 0.1, "B.unknown": 0.1}
    found = activity.linear(vessel, coefficients, tolerance)
    density = activity.rest_density(vessel, 1.0)
    beta = density * 0.4 * 50.0 / 2.0
    assert found["beta"] == pytest.approx(beta, rel=1e-12)
    assert found["t_peak"] == pytest.approx(np.log(beta / vessel.dilution) / (beta - vessel.dilution))
    swing = activity.peak_factor(vessel.dilution, beta) / 0.4  # of the nutrient, per unit of what B takes
    assert found["per_flux"]["medium.S"] == pytest.approx(swing / 0.1, rel=1e-12)
    assert found["per_flux"]["B.pool"] == pytest.approx(0.7 * swing / 0.1, rel=1e-12)
    assert found["per_flux"]["B.rates.r"] == pytest.approx(0.2 * swing / 0.05, rel=1e-12)  # without its sign
    assert found["per_flux"]["B.density"] == pytest.approx(10.0)  # in the end the density pays all of it
    assert "B.unknown" not in found["per_flux"]  # no response coefficient, no statement


# ---- tolerances that follow from the function ----------------------------------------------------------------------


def test_first_order_tolerances_keep_a_linear_function_over_the_whole_box():
    rest = {"x": 1.0, "y": 1.0}

    def q(state):  # Q = 1 at rest, slopes 0.5 and 0.25
        return 1.0 + 0.5 * (state["x"] - 1.0) + 0.25 * (state["y"] - 1.0)

    margin, q_min = 0.5, 0.5
    delta = {"x": margin / (2 * 0.5), "y": margin / (2 * 0.25)}  # margin / (m s_k)
    assert functional.least_in_box(q, rest, delta) == pytest.approx(q_min, rel=1e-5)  # the bound is exact
    assert functional.guaranteed_share(q, rest, delta, q_min * (1.0 - 1e-5)) == 1.0


def test_a_function_that_curves_downward_is_kept_over_a_share_of_its_first_order_tolerance_only():
    rest = {"x": 0.5}

    def q(state):  # concave: its tangent at rest lies above it on both sides
        return 2.0 * state["x"] - state["x"] ** 2

    q0, slope = 0.75, 1.0
    q_min = 0.5 * q0
    delta = {"x": (q0 - q_min) / slope}
    assert functional.least_in_box(q, rest, delta) == pytest.approx(2.0 * 0.125 - 0.125**2)  # below Q_min
    assert functional.least_in_box(q, rest, delta) < q_min
    # Q = Q_min at x = 1 - sqrt(1 - Q_min); the share is how far that is from rest, over the tolerance
    reached = 1.0 - np.sqrt(1.0 - q_min)
    assert functional.guaranteed_share(q, rest, delta, q_min) == pytest.approx(
        (0.5 - reached) / delta["x"], rel=1e-9
    )


def test_a_concentration_in_the_box_is_never_taken_to_zero_or_below():
    seen = []

    def q(state):
        seen.append(state["x"])
        return state["x"]

    functional.least_in_box(q, {"x": 2.0}, {"x": 5.0})  # a tolerance of more than the value itself
    assert min(seen) == pytest.approx(functional.FLOOR * 2.0)


# ---- the SBML model alone ------------------------------------------------------------------------------------------


@needs_world
def test_inside_the_organism_are_the_layers_that_a_model_has_without_a_medium_and_a_density(plan):
    entries = quasistatic.inside(plan, YEAST)
    assert {entry.layer for entry in entries} == set(quasistatic.INSIDE)
    assert not any(
        entry.column.startswith("medium.") or entry.column.endswith(".density") for entry in entries
    )


@needs_world
def test_the_model_alone_at_the_levels_of_its_file_is_at_the_steady_state_of_its_file(plan):
    from xeno_biomodels.isolated import Isolated
    from xeno_biomodels.world import World

    world = World(plan, YEAST, "xeno")
    alone = Isolated(world)
    assert alone.uptake() == pytest.approx(world.uptake_at_rest, rel=1e-9)
    found = alone.steady(dict(world.sbml_medium))
    assert found[f"B.uptake.{world.nutrient}"] == pytest.approx(world.uptake_at_rest, rel=1e-8)
    assert found["B.growth_rate"] == pytest.approx(
        world.dilution, rel=1e-8
    )  # it would grow at the dilution rate
    # with a tenth more glucose it takes up more, and far below what it needs it has no steady state
    more = alone.steady({world.nutrient: 1.1 * world.sbml_medium[world.nutrient]})
    assert more[f"B.uptake.{world.nutrient}"] > world.uptake_at_rest
    assert alone.steady({world.nutrient: 0.3 * world.sbml_medium[world.nutrient]}) is None
    assert alone.uptake() == pytest.approx(world.uptake_at_rest, rel=1e-9)  # and it is put back to its file


@needs_world
def test_the_growth_function_is_the_rate_law_of_the_uptake_and_reads_what_that_law_reads(plan):
    from xeno_biomodels.world import World

    world = World(plan, YEAST, "xeno")
    rows, delta, bound = functional.tolerances(world, 0.5)
    assert bound["function_at_rest"] == pytest.approx(world.dilution, rel=1e-9)
    assert bound["beta"] == pytest.approx(1.0)  # the tolerances spend the whole margin, to first order
    assert set(delta) == {"medium.GLCo", "B.species.GLCi"}  # glucose outside and inside, and nothing else
    for row in rows:  # the tolerance over the value is 1 / (2 m |elasticity|) with Q_min at half
        assert row["delta_over_value"] == pytest.approx(
            1.0 / (2 * len(rows) * abs(row["elasticity_of_the_uptake"])), rel=1e-6
        )
    assert bound["gamma"] > 0.0  # a rate law is not linear


# ---- a community that is at rest -----------------------------------------------------------------------------------


def test_the_community_at_rest_has_two_pairs_on_different_nutrients_and_xenos_that_coexist_stably(plan):
    study = coexistence.variant(plan)
    assert list(study["community"]["names"]) == list(coexistence.PAIRS)
    nutrients = {plan["receivers"][biomodel]["nutrient"] for biomodel in coexistence.PAIRS}
    assert len(nutrients) == len(coexistence.PAIRS)  # no two SBML organisms live on the same species
    designs = study["community"]["designs"]
    # every xeno has a substrate of its own: two xenos with the same laws on one substrate would coexist only
    # neutrally, and their densities would drift
    assert all(design["substrate"] == "own" for design in designs.values())
    assert designs["private"]["cross"] == 0.0 and "overlap" not in designs["private"]
    assert designs["overlapping"]["cross"] == 0.0 and 0.0 < designs["overlapping"]["overlap"] < 1.0
    # the chain to the own partner: a little of the other pair's nutrient, which the xeno cannot use
    assert designs["indirect"]["cross"] > 0.0 and designs["indirect"]["worth"] == 0.0
    assert designs["indirect"]["overlap"] == designs["overlapping"]["overlap"]
    assert plan["community"]["designs"] != designs  # the plan itself is left as it was


def test_the_design_indirect_is_swept_in_one_of_its_two_numbers_at_a_time():
    swept = coexistence.swept()
    reference = coexistence.DESIGNS["indirect"]
    for design in swept.values():
        differs = [key for key in reference if design[key] != reference[key]]
        assert len(differs) == 1 and differs[0] in ("cross", "overlap")
    assert {design["cross"] for design in swept.values()} == {*coexistence.SWEEP, reference["cross"]}
    assert {design["overlap"] for design in swept.values()} == {*coexistence.OVERLAPS, reference["overlap"]}


# ---- the figures and the tables ------------------------------------------------------------------------------------


def test_labels_are_kept_apart_and_under_the_top():
    assert figures._apart([0.1, 0.5, 0.9], 0.1) == [0.1, 0.5, 0.9]  # nothing to do
    assert figures._apart([0.5, 0.52, 0.53], 0.1) == pytest.approx([0.5, 0.6, 0.7])  # pushed upward
    assert figures._apart([0.9, 0.95, 0.99], 0.1) == pytest.approx([0.8, 0.9, 1.0])  # and back under the top


def test_the_steps_with_a_loss_are_named_without_their_times_and_causes():
    assert tables._lost_steps("") == "--"
    assert tables._lost_steps("u=8@30(extinction)") == "$u{=}8$"
    assert tables._lost_steps("u=2@40(extinction) u=4@20(function)") == "$u{=}2$, $u{=}4$"


def test_a_share_is_written_as_a_percentage_without_a_power_of_ten():
    assert tables._per_cent(0.5982) == "59.8\\,\\%"
    assert tables._per_cent(10.814728) == "1080\\,\\%"
    assert tables._per_cent(0.0081699) == "0.817\\,\\%"


def test_a_column_is_named_as_the_manuscript_names_the_quantity():
    assert style.quantity_name("medium.GLCo") == "glucose (medium)"
    assert style.quantity_name("B.uptake.NH4") == "uptake"
    assert style.quantity_name("B.species.AMP") == "GS-AMP"
    assert style.quantity_name("B.atp_made") == "ATP formation"
