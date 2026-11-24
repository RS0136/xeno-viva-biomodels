"""What a reading owes to the range of the activity.

The function of the sender is the substrate that its population converts, and it does not fall when the activity
rises: every activity above about 0.4 keeps it. The set of admissible activities therefore has no upper end, while
what a control takes of the nutrient of the receiver grows in proportion to epsilon times the activity. A supremum
over all that the sender may do is then the loss of the receiver for every control, and a reading of D is a
reading for a range of activities. This study says what the range does, from the runs of the plan and from the
closed form of the rest state:

    activities   D of every step alone: its size, when it peaks, whether the receiver is lost, and D over the
                 added capacity epsilon * |u - 1|, which is the same number for every step and every epsilon as
                 long as the answer is linear
    quantities   the same for every protected quantity: the profile of a step
    thresholds   for every control: the gain (D per unit of added capacity, from the weakest steps), the activity
                 at which the linear reading reaches the tolerance, the steps between which the runs reach it,
                 and the activity above which the receiver has no rest state (closed form)
    gain         the gain in closed form, from the linear response of the vessel (see ``linear``), beside the
                 gain that the runs give

The closed form. While the inside of the receiver follows the nutrient (study ``quasistatic``), a world is two
equations, for the nutrient S and the density n of the receiver, with the uptake J(S) of the model alone. Around
the rest state, a step of the sender is a step of what its population takes of the nutrient, by
phi(u) - phi(1), where phi(u) = n_A(u) q_S(u) is known in closed form. The linear system has the rates k (the
dilution) and beta = n R J*/S* (how fast the uptake of the population answers the nutrient; R is the response
coefficient of the uptake of the model alone), and the nutrient leaves its level by at most

    |s|max = |phi(u) - phi(1)| / beta * (k / beta) ** (k / (beta - k)),     at  t = ln(beta / k) / (beta - k)

A quantity y inside the organism moves by R_y |s|max / S* of its value, the density by (phi(u) - phi(1)) / J* in
the end, and D is the largest of these over the tolerances. With phi(u) - phi(1) = phi(1) (a(u) - 1), where
a(u) = u n_A(u) / n_A(1) is the capacity of the population of the sender for the nutrient, D = g eps |a(u) - 1|.

Written to ``runs/studies/activity/``: ``activities.csv``, ``quantities.csv``, ``thresholds.csv`` and ``gain.csv``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from xeno_biomodels import experiment
from xeno_biomodels.studies import cascade, quasistatic
from xeno_biomodels.studies.common import STUDIES, leak_of, read_rows, write_rows
from xeno_biomodels.world import World

OUT = STUDIES / "activity"
WEAK = (0.5, 2.0)  # the steps from which the gain is read: the nearest to the baseline on either side


def rest_density(world: World, activity: float) -> float | None:
    """The density of the receiver at the rest state of a world with the sender at an activity, in closed form;
    None where the two have no rest state together (the receiver is washed out, or the sender cannot settle).
    The receiver pays with its density what the population of the sender takes of its nutrient."""
    found = sender_flux(world, activity)
    if found is None:
        return None
    density = world.biovolume - found[1] / world.uptake_at_rest
    return float(density) if density > 0.0 else None


def washout(world: World, above: float = 1e6) -> float | None:
    """The activity above which the receiver has no rest state: bisection on the closed form. None where it has
    one up to ``above`` (a sender that takes nothing of it)."""
    low, high = world.baseline, float(above)
    if rest_density(world, low) is None:
        return low
    if rest_density(world, high) is not None:
        return None
    for _ in range(200):
        middle = math.sqrt(low * high)
        if rest_density(world, middle) is None:
            high = middle
        else:
            low = middle
    return low


def sender_flux(world: World, activity: float) -> tuple[float, float] | None:
    """(density of the sender, what its population takes of the nutrient per volume of medium and time) at the
    rest state of a world with the sender at an activity, in closed form; None where the sender cannot settle.
    The nutrient is at the level of the SBML file there, whether the receiver is still in the vessel or not."""
    config, k = world._sender_config(), world.dilution
    needed = (k + config["maintenance"]) / config["growth_yield"]
    level = world.sbml_medium[world.nutrient]
    share = config["also_takes"].get(world.nutrient, 0.0)
    half = config.get("also_km", {}).get(world.nutrient, config["km"])
    taken = share * activity * config["vmax"] * level / (half + level)
    own = needed - config.get("also_worth", {}).get(world.nutrient, 1.0) * taken
    if not 0.0 < own < activity * config["vmax"]:
        return None
    substrate = config["km"] * own / (activity * config["vmax"] - own)
    density = k * (world.feed[config["substrate"]] - substrate) / own
    return (density, density * taken) if density > 0.0 else None


def peak_factor(k: float, beta: float) -> float:
    """(k / beta) ** (k / (beta - k)): what is left of a step of the flux, over beta, at the peak of the nutrient
    (its limit where the two rates are equal is 1 / e)."""
    if abs(beta - k) < 1e-9 * k:
        return math.exp(-1.0)
    return (k / beta) ** (k / (beta - k))


def linear(world: World, coefficients: dict[str, float], tolerance: dict[str, float]) -> dict:
    """The linear response of a world to a step of what the sender takes: the rate beta, the time of the peak,
    and for every protected quantity the D per unit of flux, relative to what the receiver itself takes at rest.

    ``coefficients`` are the response coefficients of the model alone (column -> d ln y / d ln S), with that of
    the uptake among them; ``tolerance`` the tolerance of each protected column over its value at rest."""
    k, level, uptake = world.dilution, world.sbml_medium[world.nutrient], world.uptake_at_rest
    density = rest_density(world, world.baseline)
    response = coefficients[f"B.uptake.{world.nutrient}"]
    beta = density * response * uptake / level
    # per unit of flux taken by the sender, over what the receiver takes (density * uptake)
    swing = density * uptake / (beta * level) * peak_factor(k, beta)  # of the nutrient, relative, at its peak
    per_flux = {}
    for column, allowed in tolerance.items():
        if column == f"medium.{world.nutrient}":
            per_flux[column] = swing / allowed
        elif column == "B.density":
            per_flux[column] = 1.0 / allowed  # in the end: the density pays what the sender takes
        elif column in coefficients:
            per_flux[column] = abs(coefficients[column]) * swing / allowed
    return {
        "beta": beta,
        "time_constant": 1.0 / beta,
        "t_peak": math.log(beta / k) / (beta - k) if abs(beta - k) > 1e-9 * k else 1.0 / k,
        "density": density,
        "per_flux": per_flux,
    }


def run(
    plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT, runs: Path = experiment.OUT
) -> list[Path]:
    """``runs`` is where the worlds of ``plan`` were followed and measured."""
    del jobs, analyze_only  # this study reads runs that are there and one closed form
    baseline = float(plan["activities"]["baseline"])
    steps = experiment.run_names(plan)
    activities, quantities, thresholds, gains = [], [], [], []
    for biomodel in plan["receivers"]:
        gain_of_receiver = None
        closed_of_receiver = None
        inside = quasistatic.inside(plan, biomodel)
        alone = World(plan, biomodel, "xeno")
        uptake = f"B.uptake.{alone.nutrient}"
        coefficients = quasistatic.coefficients(alone, [*[entry.column for entry in inside], uptake])
        for design, written in plan["designs"].items():
            leak = leak_of(written)
            directory = runs / biomodel / design
            found = cascade.changes(plan, biomodel, directory)
            if found is None:
                continue
            times, entries, signed = found
            result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
            lost = {item["activity"]: item for item in result["losses"]}
            left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
            head = {"biomodel": biomodel, "design": design, "leak": float(leak)}
            mine = []
            for name, activity in steps.items():
                row = {**head, "step": name, "activity": activity, "admissible": name not in left_out}
                added = float(leak) * abs(activity - baseline)
                row["added_capacity"] = added
                if name in signed:
                    read = np.where(
                        np.isnan(signed[name]), 0.0, np.abs(signed[name])
                    )  # nothing is read after a loss
                    course = np.max(read, axis=0)  # over the quantities, at every time
                    peak = int(np.argmax(course))
                    row.update(
                        D=float(course[peak]),
                        t_peak=float(times[peak]),
                        receiver_lost_at=lost[name]["t"] if name in lost else None,
                        cause=lost[name]["cause"] if name in lost else "",
                        D_over_added_capacity=float(course[peak]) / added if added > 0.0 else None,
                    )
                    for n, entry in enumerate(entries):
                        one = read[n]
                        at = int(np.argmax(one))
                        quantities.append(
                            {
                                **head,
                                "step": name,
                                "activity": activity,
                                "quantity": entry.column,
                                "layer": entry.layer,
                                "kind": entry.kind,
                                "D": float(one[at]),
                                "t_peak": float(times[at]),
                                "up": bool(signed[name][n][at] > 0.0),
                                "D_over_added_capacity": float(one[at]) / added if added > 0.0 else None,
                            }
                        )
                activities.append(row)
                mine.append(row)
            if not float(leak) > 0.0:
                continue
            world = World(plan, biomodel, design)
            weak = [
                row["D_over_added_capacity"]
                for row in mine
                if row["activity"] in WEAK and row.get("D_over_added_capacity") is not None
            ]
            gain = float(np.mean(weak)) if weak else None
            gain_of_receiver = (
                gain if gain_of_receiver is None else gain_of_receiver
            )  # that of the weakest control
            # the closed form: the linear response of the vessel to what the population of the sender takes
            measured = read_rows(directory / "measure" / "deltas.csv")
            tolerance = {
                row["column"]: float(row["delta"]) / abs(float(row["y0"]))
                for row in measured
                if float(row["y0"])
            }
            found = linear(world, coefficients, tolerance)
            at_baseline = sender_flux(world, baseline)
            own = found["density"] * world.uptake_at_rest  # what the receiver takes, per volume of medium
            peaks = {column: value for column, value in found["per_flux"].items() if column != "B.density"}
            carried = max(peaks, key=peaks.get)
            per_capacity = at_baseline[1] / own / float(leak)  # flux per unit of capacity, relative
            closed = peaks[carried] * per_capacity
            closed_of_receiver = closed if closed_of_receiver is None else closed_of_receiver
            # of all that the receiver protects: the peak inside or in the medium, or what the density pays
            whole = max(found["per_flux"].values()) * per_capacity

            def reaches(activity: float, world=world, found=found, own=own, at_baseline=at_baseline) -> bool:
                """Whether the linear reading is at the tolerance or beyond it at an activity."""
                stepped = sender_flux(world, activity)
                if stepped is None:
                    return True
                settles = rest_density(world, activity) is not None
                forcing = abs(stepped[1] - at_baseline[1]) / own
                return (
                    max(v * forcing for c, v in found["per_flux"].items() if settles or c != "B.density")
                    >= 1.0
                )

            low, high = baseline, 1e6
            for _ in range(200):
                middle = math.sqrt(low * high)
                low, high = (low, middle) if reaches(middle) else (middle, high)
            at_tolerance = math.sqrt(low * high)
            from_runs = []
            for row in mine:
                stepped = sender_flux(world, row["activity"])
                if stepped is None:
                    continue
                capacity = row["activity"] * stepped[0] / (baseline * at_baseline[0])
                forcing = abs(stepped[1] - at_baseline[1]) / own
                settles = rest_density(world, row["activity"]) is not None
                row["capacity_of_the_population"] = capacity
                row["D_linear"] = max(
                    value * forcing
                    for column, value in found["per_flux"].items()
                    if settles or column != "B.density"
                )
                if row.get("D") is not None and row["admissible"] and row["receiver_lost_at"] is None:
                    row["D_over_capacity_of_the_population"] = row["D"] / (float(leak) * abs(capacity - 1.0))
                    from_runs.append(row["D_over_capacity_of_the_population"])
            gains.append(
                {
                    **head,
                    "response_coefficient_of_the_uptake": coefficients[uptake],
                    "beta": found["beta"],
                    "time_constant_minutes": found["time_constant"] / world.minute,
                    "t_peak_minutes": found["t_peak"] / world.minute,
                    "carried_by": carried,
                    "response_coefficient_of_that_quantity": 1.0
                    if carried.startswith("medium.")
                    else coefficients[carried],
                    "gain_closed_form": closed,
                    "gain_of_the_density": found["per_flux"]["B.density"] * per_capacity,
                    "gain_closed_form_of_the_whole": whole,
                    "activity_at_tolerance_closed_form": at_tolerance,
                    "least_gain_in_the_runs": min(from_runs) if from_runs else None,
                    "largest_gain_in_the_runs": max(from_runs) if from_runs else None,
                    "gain_closed_form_of_the_weakest_control": closed_of_receiver,
                }
            )
            up = sorted(
                (row["activity"], row["D"], row["receiver_lost_at"] is not None)
                for row in mine
                if row.get("D") is not None and row["activity"] > baseline
            )
            within = [u for u, d, gone in up if d <= 1.0 and not gone]
            beyond = [u for u, d, gone in up if d > 1.0 or gone]
            gone = washout(world)
            thresholds.append(
                {
                    **head,
                    "gain": gain,
                    "gain_of_the_weakest_control": gain_of_receiver,
                    "activity_at_tolerance_linear": baseline + 1.0 / (gain_of_receiver * float(leak)),
                    "largest_step_within_tolerance": max(within) if within else None,
                    "smallest_step_beyond_tolerance": min(beyond) if beyond else None,
                    "activity_of_washout": gone,
                    "capacity_of_washout": None if gone is None else float(leak) * gone,
                    "density_at_rest_over_alone": rest_density(world, baseline) / world.biovolume,
                    "steps_with_a_loss": " ".join(sorted(lost)),
                }
            )
    written = [
        write_rows(out / "activities.csv", activities),
        write_rows(out / "quantities.csv", quantities),
        write_rows(out / "thresholds.csv", thresholds),
        write_rows(out / "gain.csv", gains),
    ]
    return written
