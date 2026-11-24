"""Tolerances that follow from the function.

In the plan the function of a receiver is its growth rate, and the growth rate is itself protected, so its
tolerance of the kind ``fn`` is the whole margin and the lemma has nothing to do; every other tolerance is a tenth
of a value at rest, by convention. Here the function is written out as what it is in the world: the growth rate is
the uptake of the nutrient over its value at rest, and the uptake is a rate law of the SBML model, a function of
the level of the nutrient and of the species of the model that the law reads. Those are then the protected
quantities, each with a tolerance of the kind ``fn``, and nothing is left to convention:

    Q_B(S, x)   = k J(S, x) / J*,  J the rate law of the uptake, evaluated by the model at any state
    s_k         = |dQ_B/dy_k| at the rest state, by central differences, for S and for every species x_k
    delta_k     = margin / (m s_k),  m the number of quantities on which Q_B depends, margin = Q_B - Q_min

so that the tolerance of a quantity, relative to its value, is 1 / (2 m |elasticity of the uptake to it|) when
Q_min is half the function at rest. These tolerances spend the whole margin to first order (beta = 1), and a rate
law is not linear, so they are no guarantee. Whether they keep the function is asked of the rate law itself: Q_B
is evaluated at every corner of the box of tolerances and at points inside it, and where its least value there is
below Q_min, the tolerances are scaled by the largest share of the margin for which it is not. The study gives
the tolerances, D with them for every control of the plan, and the lemma on the runs:

    tolerances  every quantity that the uptake reads: its elasticity, its tolerance, and the same in per cent
    worlds      D with the tolerances of first order and with those that are guaranteed, what carries it, and
                the verdicts, beside D and the verdict of the plan
    lemma       the second-order term of the bound (the curvature of Q_B at rest over the tolerances), the least
                value of Q_B over the box, the share that is guaranteed; and over every stored time of every
                admissible run, the times at which every quantity is within its tolerance, how many of those
                have the function below Q_min, and the least margin kept at them

Written to ``runs/studies/functional/``: ``tolerances.csv``, ``worlds.csv``, ``activities.csv``, ``lemma.csv``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

from xeno_biomodels import experiment
from xeno_biomodels.isolated import Isolated
from xeno_biomodels.orthogonality import RELEVANT, STEP, read_series
from xeno_biomodels.studies.common import STUDIES, leak_of, read_rows, write_rows
from xeno_biomodels.world import BASELINE, REMOVAL, World

OUT = STUDIES / "functional"
INSIDE = 400  # points inside the box of tolerances at which the function is evaluated, beside its corners
VERIFY = 20000  # points inside the box that is kept in the end, to see that its least value is at a corner
MONOTONE = (
    2000  # points inside the box at which the direction of the function along each quantity is looked at
)
FLOOR = 1e-6  # a concentration is not taken below this share of its value at rest


def function_of_state(world: World) -> tuple[dict[str, float], callable]:
    """The rest state as one mapping (the nutrient under ``medium.``, the species under ``B.species.``), and Q_B
    as a function of such a mapping: the growth rate that the rate law of the uptake gives at that state."""
    alone = Isolated(world)
    rest = {f"medium.{world.nutrient}": world.sbml_medium[world.nutrient]}
    rest.update({f"B.species.{name}": value for name, value in world.sbml_species.items()})

    def q(state: dict[str, float]) -> float:
        species = {name[len("B.species.") :]: value for name, value in state.items() if name.startswith("B.")}
        levels = {world.nutrient: state[f"medium.{world.nutrient}"]}
        return alone.growth(species, levels)

    return rest, q


def tolerances(world: World, q_min_share: float) -> tuple[list[dict], dict[str, float], dict]:
    """The quantities on which Q_B depends with their functional tolerances: (rows, column -> delta, what the
    bound of the lemma comes to: beta and the second-order term gamma, both over the margin)."""
    rest, q = function_of_state(world)
    q0 = q(rest)
    margin = q0 - q_min_share * q0
    # a species that the model does not have (a level of zero, or of a denormal number) is stepped by an amount
    step = {name: STEP * (abs(value) if abs(value) > 1e-100 else 1.0) for name, value in rest.items()}

    def moved(changes: dict[str, float]) -> float:
        return q({**rest, **{name: rest[name] + change for name, change in changes.items()}})

    slope = {
        name: (moved({name: step[name]}) - moved({name: -step[name]})) / (2.0 * step[name]) for name in rest
    }
    depends = [
        name
        for name in rest
        if abs(rest[name]) > 1e-100 and abs(slope[name]) * abs(rest[name]) / abs(q0) > RELEVANT
    ]
    delta = {name: margin / (len(depends) * abs(slope[name])) for name in depends}
    # the second-order term: half the sum of |d2Q/dy_j dy_k| delta_j delta_k, at the rest state. The steps of the
    # second differences are a thousandth of the tolerances, large enough for the differences to be resolved.
    wide = {name: 1e-3 * delta[name] for name in depends}
    curvature = 0.0
    for i, a in enumerate(depends):
        for b in depends[i:]:
            if a == b:
                second = (moved({a: wide[a]}) - 2.0 * q0 + moved({a: -wide[a]})) / wide[a] ** 2
                curvature += 0.5 * abs(second) * delta[a] ** 2
            else:
                second = (
                    moved({a: wide[a], b: wide[b]})
                    - moved({a: wide[a], b: -wide[b]})
                    - moved({a: -wide[a], b: wide[b]})
                    + moved({a: -wide[a], b: -wide[b]})
                ) / (4.0 * wide[a] * wide[b])
                curvature += abs(second) * delta[a] * delta[b]
    rows = [
        {
            "biomodel": world.biomodel,
            "quantity": name,
            "at_rest": rest[name],
            "elasticity_of_the_uptake": slope[name] * rest[name] / q0,
            "delta": delta[name],
            "delta_over_value": delta[name] / abs(rest[name]) if rest[name] else None,
        }
        for name in depends
    ]
    bound = {
        "function_at_rest": q0,
        "margin": margin,
        "quantities": len(depends),
        "beta": sum(abs(slope[name]) * delta[name] for name in depends) / margin,
        "gamma": curvature / margin,
    }
    return rows, delta, bound


def least_in_box(
    q, rest: dict[str, float], delta: dict[str, float], share: float = 1.0, inside: int = INSIDE
) -> float:
    """The least value of Q_B over the states in which every quantity is within ``share`` times its tolerance: at
    every corner of that box and at points inside it (the same relative points for every share)."""
    names = list(delta)
    low = {name: max(rest[name] - share * delta[name], FLOOR * abs(rest[name])) for name in names}
    high = {name: rest[name] + share * delta[name] for name in names}
    least = np.inf
    for corner in itertools.product((False, True), repeat=len(names)):
        state = {name: high[name] if up else low[name] for name, up in zip(names, corner, strict=True)}
        least = min(least, q({**rest, **state}))
    random = np.random.default_rng(0)
    for point in random.random((inside, len(names))):
        state = {name: low[name] + x * (high[name] - low[name]) for name, x in zip(names, point, strict=True)}
        least = min(least, q({**rest, **state}))
    return float(least)


def guaranteed_share(q, rest: dict[str, float], delta: dict[str, float], q_min: float) -> float:
    """The largest share of the tolerances, up to 1, with which the function stays at Q_min or above everywhere
    in the box (bisection on ``least_in_box``)."""
    if least_in_box(q, rest, delta, 1.0) >= q_min:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(50):
        middle = 0.5 * (low + high)
        if least_in_box(q, rest, delta, middle) >= q_min:
            low = middle
        else:
            high = middle
    return low


def monotone(q, rest: dict[str, float], delta: dict[str, float], share: float = 1.0) -> tuple[bool, int]:
    """Whether Q_B rises or falls with each quantity in the same direction everywhere in the box of tolerances:
    the sign of its difference along every quantity, at every corner and at ``MONOTONE`` points inside. Where it
    does, the least value of Q_B over the box is at a corner. Returns (monotone, points looked at)."""
    names = list(delta)
    low = {name: max(rest[name] - share * delta[name], FLOOR * abs(rest[name])) for name in names}
    high = {name: rest[name] + share * delta[name] for name in names}
    random = np.random.default_rng(1)
    points = [np.array(corner, dtype=float) for corner in itertools.product((0.0, 1.0), repeat=len(names))]
    points += list(random.random((MONOTONE, len(names))))
    first = None
    for point in points:
        state = {name: low[name] + x * (high[name] - low[name]) for name, x in zip(names, point, strict=True)}
        here = q({**rest, **state})
        signs = []
        for name, x in zip(names, point, strict=True):
            width = high[name] - low[name]
            toward = -1.0 if x > 0.5 else 1.0  # a small step that stays inside the box
            moved = q({**rest, **state, name: state[name] + toward * 1e-4 * width})
            signs.append(np.sign(toward * (moved - here)))
        if first is None:
            first = signs
        elif any(a * b < 0.0 for a, b in zip(first, signs, strict=True)):
            return False, len(points)
    return True, len(points)


def conserved_pairs(world: World, names: list[str]) -> list[tuple[str, str]]:
    """Pairs of species whose sum is the same at two steady states of the model alone at which each of them
    differs: the two forms of one carrier (ATP and ADP, NAD and NADH), which are one quantity that can vary and
    not two."""
    alone, level = Isolated(world), world.sbml_medium[world.nutrient]
    low, high = alone.steady({world.nutrient: 0.8 * level}), alone.steady({world.nutrient: 1.25 * level})
    if low is None or high is None:
        return []
    pairs, taken = [], set()
    species = [name for name in names if name.startswith("B.species.")]
    for i, a in enumerate(species):
        for b in species[i + 1 :]:
            if a in taken or b in taken:
                continue
            moved = max(abs(high[a] - low[a]), abs(high[b] - low[b]))
            if moved > 0.0 and abs((high[a] + high[b]) - (low[a] + low[b])) < 1e-6 * moved:
                pairs.append((a, b))
                taken.update((a, b))
    return pairs


def readings(
    columns: dict[str, np.ndarray],
    base: dict[str, np.ndarray],
    stored: np.ndarray,
    delta: dict[str, float],
    slope: dict[str, float],
    pairs: list[tuple[str, str]],
    margin: float,
) -> dict[str, np.ndarray]:
    """Courses of a run in the tolerances that follow from the function, read in four ways (quantity x time):

    two_sided     |change| / tolerance, as the definition has it
    adverse       only the changes that lower the function count (it has to stay above a least value, not below
                  a largest one)
    independent   a conserved pair counts once: the margin is divided among the quantities that can vary
                  independently, and a pair is read along the direction in which it can move
    used          what the function itself lost, over the margin: one course, read from the growth rate
    """
    change = {key: columns[key][stored] - base[key][stored] for key in delta}
    two_sided = np.vstack([np.abs(change[key]) / delta[key] for key in delta])
    adverse = np.vstack([np.maximum(-np.sign(slope[key]) * change[key], 0.0) / delta[key] for key in delta])
    paired = {name for pair in pairs for name in pair}
    singles = [key for key in delta if key not in paired]
    m = len(singles) + len(pairs)
    rows = [np.abs(change[key]) / (margin / (m * abs(slope[key]))) for key in singles]
    for a, b in pairs:  # a rises by what b falls: the slope along that direction
        rows.append(np.abs(change[a]) / (margin / (m * abs(slope[a] - slope[b]))))
    used = np.maximum(base["B.growth_rate"][stored] - columns["B.growth_rate"][stored], 0.0) / margin
    return {
        "two_sided": two_sided,
        "adverse": adverse,
        "independent": np.vstack(rows),
        "used": used[np.newaxis, :],
    }


def run(
    plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT, runs: Path = experiment.OUT
) -> list[Path]:
    """``runs`` is where the worlds of ``plan`` were followed and measured (those of the plan itself, or those
    of a study that has receivers of its own)."""
    del jobs, analyze_only  # the runs are there; the rate law is asked without any integration
    share = float(str(plan["measure"]["qmin"]).partition(":")[2])  # Q_min as a share of the function at rest
    table, worlds, activities, lemma = [], [], [], []
    for biomodel in plan["receivers"]:
        world = World(plan, biomodel, "xeno")
        rows, delta, bound = tolerances(world, share)
        q_min, margin = share * bound["function_at_rest"], bound["margin"]
        slope = {
            row["quantity"]: row["elasticity_of_the_uptake"] * bound["function_at_rest"] / row["at_rest"]
            for row in rows
        }
        rest, q = function_of_state(world)
        bound["least_function_in_the_box_over_q_min"] = least_in_box(q, rest, delta) / q_min
        kept = guaranteed_share(q, rest, delta, q_min)
        bound["share_guaranteed"] = kept
        bound["monotone_in_the_box"], bound["points_for_monotone"] = monotone(q, rest, delta)
        bound["least_function_in_the_guaranteed_box_over_q_min"] = (
            least_in_box(q, rest, delta, kept, inside=VERIFY) / q_min
        )
        bound["points_in_the_guaranteed_box"] = VERIFY + 2 ** len(delta)
        curved = bound["gamma"]  # the curvature at rest: what the lemma would allow if it held over the box
        bound["share_from_the_curvature_at_rest"] = (
            1.0 if curved == 0.0 else min((-1.0 + np.sqrt(1.0 + 4.0 * curved)) / (2.0 * curved), 1.0)
        )
        for row in rows:
            row["delta_guaranteed_over_value"] = (
                None if row["delta_over_value"] is None else kept * row["delta_over_value"]
            )
        counted = {"times": 0, "within": 0, "below": 0, "least": np.inf, "beyond_and_kept": 0, "beyond": 0}
        sure = {"within": 0, "below": 0, "least": np.inf}
        pairs = conserved_pairs(world, list(delta))
        for design, written in plan["designs"].items():
            leak = leak_of(written)
            directory = runs / biomodel / design
            if not (directory / "measure" / "result.json").exists():
                continue
            result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
            lost = {item["activity"]: item["t"] for item in result["losses"]}
            left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
            times, base = read_series(directory / BASELINE / "series.csv")
            best: dict[str, tuple[float, str, str, float]] = {}
            removed: dict[str, float] = {}
            for name, activity in [*experiment.run_names(plan).items(), (REMOVAL, None)]:
                path = directory / name / "series.csv"
                if name in left_out:
                    continue
                if not path.exists():  # a run that stopped: what was stored until then
                    path = directory / name / "partial.csv"
                    if not path.exists():
                        continue
                    rows_ = read_rows(path)
                    columns = {
                        key: np.array([float(row[key]) for row in rows_]) for key in rows_[0] if key != "time"
                    }
                    stored = np.arange(len(rows_))
                else:
                    _, columns = read_series(path)
                    stored = np.arange(times.size)
                read = stored[times[stored] < lost[name]] if name in lost else stored
                found = readings(columns, base, stored, delta, slope, pairs, margin)
                every = found["two_sided"]
                if activity is None:  # the sender taken away: read apart, as D_removal is
                    removed = {key: float(np.max(course)) for key, course in found.items()}
                    continue
                if float(leak) > 0.0:
                    # The lemma is tried on every stored time, also after the receiver has lost its function:
                    # read only until the loss, the function could not be below Q_min at any time.
                    growth, inside = columns["B.growth_rate"][stored], np.max(every, axis=0) <= 1.0
                    counted["times"] += int(stored.size)
                    counted["within"] += int(np.sum(inside))
                    counted["below"] += int(np.sum(growth[inside] < q_min))
                    counted["beyond"] += int(np.sum(~inside))
                    counted["beyond_and_kept"] += int(np.sum(growth[~inside] >= q_min))
                    if np.any(inside):
                        counted["least"] = min(counted["least"], float(np.min(growth[inside] - q_min)))
                    tight = np.max(every, axis=0) <= kept  # within the tolerances that are guaranteed
                    sure["within"] += int(np.sum(tight))
                    sure["below"] += int(np.sum(growth[tight] < q_min))
                    if np.any(tight):
                        sure["least"] = min(sure["least"], float(np.min(growth[tight] - q_min)))
                if not read.size:
                    continue
                step = {
                    "biomodel": biomodel,
                    "design": design,
                    "leak": float(leak),
                    "step": name,
                    "activity": activity,
                    "receiver_lost_at": lost.get(name),
                }
                labels = {"two_sided": list(delta), "adverse": list(delta)}
                for key, whole in found.items():
                    course = whole[:, read]
                    largest = np.max(course, axis=0)
                    at = int(np.argmax(largest))
                    carried = labels[key][int(np.argmax(course[:, at]))] if key in labels else ""
                    step[key] = float(largest[at])
                    if key == "two_sided":
                        step.update(carried_by=carried, t_peak=float(times[read][at]))
                    if largest[at] > best.get(key, (0.0,))[0]:
                        best[key] = (float(largest[at]), carried, name, float(times[read][at]))
                step["margin_used_in_the_whole_run"] = float(np.max(found["used"]))
                activities.append(step)
            plan_row = next(
                row
                for row in read_rows(runs / "summary.csv")
                if (row["biomodel"], row["design"]) == (biomodel, design)
            )
            nothing = (0.0, "", "", 0.0)
            two, adverse = best.get("two_sided", nothing), best.get("adverse", nothing)
            worlds.append(
                {
                    "biomodel": biomodel,
                    "design": design,
                    "leak": float(leak),
                    "D_functional": two[0],
                    "carried_by": two[1],
                    "step": two[2],
                    "t_peak": two[3],
                    "tolerable_functional": bool(two[0] <= 1.0 and not lost),
                    "D_guaranteed": two[0] / kept,
                    "tolerable_guaranteed": bool(two[0] / kept <= 1.0 and not lost),
                    "D_adverse": adverse[0],
                    "adverse_carried_by": adverse[1],
                    "adverse_step": adverse[2],
                    "D_adverse_guaranteed": adverse[0] / kept,
                    "tolerable_adverse_guaranteed": bool(adverse[0] / kept <= 1.0 and not lost),
                    "D_independent": best.get("independent", nothing)[0],
                    "margin_used": best.get("used", nothing)[0],
                    "margin_used_step": best.get("used", nothing)[2],
                    "function_kept_in_every_run": not lost,
                    "D_removal_functional": removed.get("two_sided"),
                    "D_removal_guaranteed": None if not removed else removed["two_sided"] / kept,
                    "D_removal_adverse_guaranteed": None if not removed else removed["adverse"] / kept,
                    "margin_used_by_the_removal": removed.get("used"),
                    "D_of_the_plan": float(plan_row["D"]),
                    "tolerable_in_the_plan": plan_row["stage3_tolerable"] == "True",
                    "D_removal_of_the_plan": float(plan_row["D_removal"]),
                    "losses": plan_row["losses"],
                }
            )
        table += rows
        lemma.append(
            {
                "biomodel": biomodel,
                **bound,
                "conserved_pairs": " ".join(
                    f"{a.rpartition('.')[2]}+{b.rpartition('.')[2]}" for a, b in pairs or []
                ),
                "independent_quantities": len(delta) - len(pairs),
                "stored_times": counted["times"],
                "every_quantity_within_tolerance": counted["within"],
                "of_these_function_below_q_min": counted["below"],
                "least_margin_kept_over_margin": counted["least"] / bound["margin"]
                if counted["within"]
                else None,
                "a_quantity_beyond_tolerance": counted["beyond"],
                "of_these_function_kept": counted["beyond_and_kept"],
                "every_quantity_within_guaranteed_tolerance": sure["within"],
                "of_these_function_below_q_min_guaranteed": sure["below"],
                "least_margin_kept_over_margin_guaranteed": sure["least"] / bound["margin"]
                if sure["within"]
                else None,
            }
        )
    return [
        write_rows(out / "tolerances.csv", table),
        write_rows(out / "worlds.csv", worlds),
        write_rows(out / "activities.csv", activities),
        write_rows(out / "lemma.csv", lemma),
    ]
