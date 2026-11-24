"""An activity that changes in time, instead of a step.

D is a supremum over what the sender may do, and the worlds of the plan try five steps of one knob. Here the knob
moves in time between two activities, ``low`` and ``high`` of ``studies.schedules`` in the plan:

    step           to the high and to the low activity, held: what the plan does, for comparison
    wave           a square wave between the two with a period from minutes to days, starting high or starting low
    worst case     for each layer of the receiver, the switching between the two that the step responses say
                   moves its quantity most at the end of the run

The two activities lie as far below the baseline as above it, so a wave leaves the mean of the activity at the
baseline, and what it does is done by its moving alone.

The worst case comes from the linear picture. With g(t) the response of a quantity to a step of one unit of the
activity (the mean of the two step responses, each over its distance from the baseline), h = dg/dt is its
response to a pulse, and what an activity v(t) = u(t) - u_0 leaves at the end T is the integral of
h(T - t) v(t). Within the two bounds it is largest when v sits at the bound that has the sign of h(T - t) at
every time: it then leaves the
integral of |h|, where a step leaves at most the peak of |g| (``predicted_gain_over_the_step`` is the one over
the other). If h never changes sign that is the step itself and nothing is run; if it does (the quantity
overshoots or comes back), the switching is followed and measured like any other run.

Every run is measured alone against the baseline, layer by layer. The sender keeps its function if it keeps it on
average over the run (a wave that ends low has not stopped). Runs are stored once a minute.

Written to ``runs/studies/schedules/``: ``worlds.csv`` (one row per world and schedule) and ``layers.csv``.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from xeno_biomodels import layers
from xeno_biomodels.orthogonality import MeasureError, measure, read_series
from xeno_biomodels.studies.common import STUDIES, changed, write_rows
from xeno_biomodels.world import World, WorldError, write_series

OUT = STUDIES / "schedules"
BASELINE = "baseline"


def waves(plan: dict) -> dict[str, tuple[dict, tuple[list[float], list[float]]]]:
    """Name -> (what it is, (times, values)) for the steps and the square waves."""
    settings = plan["studies"]["schedules"]
    low, high, horizon = float(settings["low"]), float(settings["high"]), float(plan["time"]["horizon"])
    out = {
        "step to high": ({"schedule_kind": "step", "period": "", "starts": "high"}, ([0.0], [high])),
        "step to low": ({"schedule_kind": "step", "period": "", "starts": "low"}, ([0.0], [low])),
    }
    for period in settings["periods"]:
        times = [float(at) for at in np.arange(0.0, horizon, float(period) / 2.0)]
        for first, second, starts in ((high, low, "high"), (low, high, "low")):
            values = [first if n % 2 == 0 else second for n in range(len(times))]
            tag = {"schedule_kind": "wave", "period": float(period), "starts": starts}
            out[f"wave {float(period):g} min, {starts} first"] = (tag, (times, values))
    return out


def switching(
    pulse: np.ndarray, times: np.ndarray, low: float, high: float
) -> tuple[list[float], list[float]] | None:
    """The schedule between ``low`` and ``high`` that the response to a pulse says leaves most at the last time.

    ``pulse`` is h on ``times``, already of the sign that is to be made large. None when h keeps its sign: the
    schedule is then a step.
    """
    floor = 1e-3 * float(np.max(np.abs(pulse)))
    sign = np.where(pulse > floor, 1, np.where(pulse < -floor, -1, 0))
    for n in range(1, sign.size):  # where h is as good as zero, stay where it was
        if sign[n] == 0:
            sign[n] = sign[n - 1]
    for n in range(sign.size - 2, -1, -1):
        if sign[n] == 0:
            sign[n] = sign[n + 1]
    if np.all(sign >= 0) or np.all(sign <= 0):
        return None
    mirrored = sign[::-1]  # v(t) takes the sign of h(T - t)
    starts, values = [0.0], [high if mirrored[0] > 0 else low]
    for n in range(1, mirrored.size):
        if mirrored[n] != mirrored[n - 1]:
            starts.append(float(times[n]))
            values.append(high if mirrored[n] > 0 else low)
    return starts, values


def _world(item) -> tuple[list[dict], list[dict]]:
    """Every schedule of one world: followed, measured alone against the baseline, and let go."""
    plan, biomodel, design, out = item
    settings, read = plan["studies"]["schedules"], plan["measure"]
    low, high = float(settings["low"]), float(settings["high"])
    study = changed(plan, {"time.store_every": int(settings["store_every"])})
    tag = {"biomodel": biomodel, "design": design}
    directory = out / "worlds" / biomodel / design
    try:
        world = World(study, biomodel, design)
        state, moved, _ = world.rest()
    except WorldError as exc:
        return [{**tag, "status": f"WorldError: {str(exc)[:200]}"}], []
    receiver = layers.sbml_organism(plan, biomodel, "B")
    sender = layers.xeno_organism("A", plan["sender"]["substrate"], plan["sender"]["product"])
    entries = layers.protected(plan, receiver)
    groups = layers.groups(entries)

    def of(name: str, key: str) -> str:
        return read[key].replace("{name}", name)

    more = [of("B", "function"), of("B", "density"), of("A", "sender_function")]

    def follow(name: str, schedule) -> tuple[Path | None, str | None]:
        """A run: what the measure reads of it, and whether every number of the receiver is that of the baseline."""
        document = world.document(state) if schedule is None else world.document(state, schedule=schedule)
        times, columns, problem = world.follow(document)
        if problem is not None:
            return None, problem
        series = write_series(directory / name / "series.csv", times, columns)
        return layers.write_protected(
            series, directory / name / "protected.csv", plan, [receiver, sender], more
        ), None

    def read_run(name: str, path: Path) -> tuple[dict, list[dict], dict]:
        result, _ = measure(
            directory / BASELINE / "protected.csv",
            {name: path},
            [entry.spec for entry in entries],
            receiver_density=of("B", "density"),
            viable=float(plan["viable"]),
            function=of("B", "function"),
            qmin=read["qmin"],
            mode=read["mode"],
            allocation=read["allocation"],
            sender_function=of("A", "sender_function"),
            sender_qmin=read["sender_qmin"],
            sender_mode="time_average",
            groups=groups,
            sigma=float(read["sigma"]),
            stage_tol=float(read["stage_tol"]),
            bits_tol=float(read["bits_tol"]),
        )
        own = [f"B.{key}" for key in ("density", "growth_rate")] + [entry.column for entry in entries]
        _, base = read_series(directory / BASELINE / "protected.csv")
        _, mine = read_series(path)
        row = {
            "status": "ok",
            "D": result["D_peak"],
            "D_end": result["D_end"],
            "t_peak": result["t_peak"],
            "D_quantity": result["D_quantity"],
            "admissible": not result["excluded_by_sender_function"],
            "losses": " ".join(f"{loss['t']:g}({loss['cause']})" for loss in result["losses"]),
            "stage1_trajectory": result["1_trajectory"],
            "stage3_tolerable": result["3_tolerable"],
            "receiver_bit_identical": all(np.array_equal(base[key], mine[key]) for key in own if key in base),
        }
        return row, layers.group_rows(result), result

    rows, by_group = [], []
    _, problem = follow(BASELINE, None)
    if problem is not None:
        return [{**tag, "status": f"the baseline could not be followed: {problem}"}], []
    responses = {}
    for name, (what, schedule) in waves(plan).items():
        path, problem = follow(name, schedule)
        head = {**tag, "schedule": name, **what, "rest_moved": moved}
        if path is None:
            rows.append({**head, "status": f"not followed: {problem}"})
            continue
        try:
            row, found, result = read_run(name, path)
        except MeasureError as exc:
            rows.append({**head, "status": str(exc).splitlines()[0]})
            continue
        rows.append({**head, **row})
        by_group += [{**head, **line} for line in found]
        if what["schedule_kind"] == "step":
            responses[what["starts"]] = (path, result)

    # The worst case of each layer, from the two step responses.
    if len(responses) == 2 and any(row.get("D", 0.0) for row in rows if row.get("status") == "ok"):
        times, base = read_series(directory / BASELINE / "protected.csv")
        _, up = read_series(responses["high"][0])
        _, down = read_series(responses["low"][0])
        tried = {}
        for layer in layers.LAYERS:
            label = f"layer:{layer}"
            if label not in groups:
                continue
            stronger = max(responses.values(), key=lambda found: found[1]["groups"][label]["D_peak"])[1]
            quantity = stronger["groups"][label]["D_quantity"]
            if quantity is None:
                continue
            # the response to a step of one unit of the activity, and to a pulse
            step = 0.5 * (
                (up[quantity] - base[quantity]) / (high - world.baseline)
                + (down[quantity] - base[quantity]) / (low - world.baseline)
            )
            pulse = np.gradient(step, times)
            width = max(round(5.0 / float(times[1] - times[0])), 1)  # a mean over five minutes
            pulse = np.convolve(pulse, np.ones(width) / width, mode="same")
            if not float(np.max(np.abs(pulse))) > 0.0:
                continue
            direction = 1.0 if abs(float(np.max(step))) >= abs(float(np.min(step))) else -1.0
            schedule = switching(direction * pulse, times, low, high)
            spacing = float(times[1] - times[0])
            # what the linear picture promises: the switching leaves the integral of |h|, a step at most the peak of g
            at_peak, in_all = float(np.max(np.abs(step))), float(np.sum(np.abs(pulse))) * spacing
            what = {
                "schedule_kind": "worst case",
                "period": "",
                "starts": "",
                "for_layer": layer,
                "for_quantity": quantity,
                "predicted_gain_over_the_step": in_all / at_peak if at_peak > 0.0 else float("inf"),
            }
            head = {**tag, "schedule": f"worst case for {layer}", **what, "rest_moved": moved}
            if schedule is None:
                rows.append(
                    {
                        **head,
                        "status": "the step itself: the response to a pulse keeps its sign",
                        "switches": 0,
                    }
                )
                continue
            key = (tuple(schedule[0]), tuple(schedule[1]))
            if key not in tried:
                path, problem = follow(f"worst_{layer}", schedule)
                tried[key] = (path, problem)
            path, problem = tried[key]
            head["switches"] = len(schedule[0]) - 1
            if path is None:
                rows.append({**head, "status": f"not followed: {problem}"})
                continue
            try:
                row, found, _ = read_run(f"worst_{layer}", path)
            except MeasureError as exc:
                rows.append({**head, "status": str(exc).splitlines()[0]})
                continue
            rows.append({**head, **row})
            by_group += [{**head, **line} for line in found]
    shutil.rmtree(
        directory, ignore_errors=True
    )  # stored once a minute: the tables are kept, the series are not
    return rows, by_group


def run(plan: dict, jobs: int = 1, analyze_only: bool = False, out: Path = OUT) -> list[Path]:
    written = [out / "worlds.csv", out / "layers.csv"]
    if analyze_only:
        return written
    work = [
        (plan, biomodel, design, out)
        for biomodel in plan["receivers"]
        for design in plan["studies"]["schedules"]["designs"]
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(work)), mp_context=get_context("spawn")) as pool:
            found = list(pool.map(_world, work))
    else:
        found = [_world(item) for item in work]
    write_rows(out / "worlds.csv", [row for rows, _ in found for row in rows])
    write_rows(out / "layers.csv", [row for _, by_group in found for row in by_group])
    return written
