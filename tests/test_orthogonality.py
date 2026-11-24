"""The measure against cases worked by hand: tolerances, the lemma, losses, admissible activities, stages, bits."""

from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from xeno_biomodels.orthogonality import (
    Function,
    MeasureError,
    information_bounds,
    measure,
    parse_protected,
    read_series,
)

TIMES = np.linspace(0.0, 10.0, 11)


def write(path, columns: dict, times=TIMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", *columns])
        for k, t in enumerate(times):
            writer.writerow(
                [repr(float(t)), *(repr(float(np.broadcast_to(v, times.shape)[k])) for v in columns.values())]
            )
    return path


def flat(**values):
    return {name: np.full(TIMES.shape, float(value)) for name, value in values.items()}


# ---- how a tolerance is written ------------------------------------------------------------------------------------


def test_parse_protected_reads_the_three_forms():
    assert parse_protected("B.x=abs:0.5").spec == "abs:0.5"
    item = parse_protected("B.x=rel:0.1,floor:1e-3")
    assert (item.kind, item.value, item.floor) == ("rel", 0.1, 1e-3)
    assert parse_protected(" B.x = fn ").kind == "fn"


@pytest.mark.parametrize(
    "text", ["B.x", "B.x=", "B.x=abs:-1", "B.x=abs:0", "B.x=rel:0.1,ceil:2", "B.x=rel:x", "=abs:1", "B.x=fnn"]
)
def test_parse_protected_refuses_what_it_cannot_read(text):
    with pytest.raises(MeasureError):
        parse_protected(text)


def test_tolerances_abs_rel_and_floor(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": -2.0, "B.c": 0.0}))
    _, written = measure(
        base, [], ["B.a=abs:0.25", "B.b=rel:0.1", "B.c=rel:0.1,floor:0.003"], out=tmp_path / "m"
    )
    rows = {row["column"]: row for row in csv.DictReader((tmp_path / "m" / "deltas.csv").open())}
    assert float(rows["B.a"]["delta"]) == 0.25
    assert float(rows["B.b"]["delta"]) == pytest.approx(0.2)  # R |y(0)|
    assert float(rows["B.c"]["delta"]) == 0.003  # the floor, since R |y(0)| = 0
    assert {path.name for path in written} == {"orthogonality.csv", "deltas.csv", "result.json"}


def test_relative_tolerance_of_a_zero_needs_a_floor(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.c": 0.0}))
    with pytest.raises(MeasureError, match="floor"):
        measure(base, [], ["B.c=rel:0.1"])


def test_relative_tolerance_reads_time_zero_of_the_baseline(tmp_path):
    base = write(tmp_path / "base.csv", {"B.a": np.linspace(2.0, 8.0, TIMES.size)})
    run = write(tmp_path / "run.csv", {"B.a": np.linspace(2.0, 8.0, TIMES.size) + 0.1})
    result, _ = measure(base, [run], ["B.a=rel:0.1"])
    assert result["D_peak"] == pytest.approx(0.1 / (0.1 * 2.0))


# ---- tolerances from the function, and the lemma -------------------------------------------------------------------


def test_fn_tolerance_is_the_margin_shared_among_all(tmp_path):
    # Q = 3 a + 2 b = 16 on the baseline, Q_min = 10: margin 6. N = m = 2.
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0}))
    result, _ = measure(
        base, [], ["B.a=fn", "B.b=fn"], function="3 * B.a + 2 * B.b", qmin="abs:10", out=tmp_path / "m"
    )
    rows = {row["column"]: row for row in csv.DictReader((tmp_path / "m" / "deltas.csv").open())}
    assert float(rows["B.a"]["delta"]) == pytest.approx(6.0 / (2 * 3.0), rel=1e-8)
    assert float(rows["B.b"]["delta"]) == pytest.approx(6.0 / (2 * 2.0), rel=1e-8)
    assert result["margin"] == pytest.approx(6.0)
    assert result["bound_over_margin"] == pytest.approx(1.0, rel=1e-8)  # N / m = 1


def test_allocation_all_counts_every_protected_quantity_and_dependent_only_those_q_reads(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0, "B.c": 7.0}))
    protect = ["B.a=fn", "B.b=fn", "B.c=abs:1"]
    kwargs = {"function": "3 * B.a + 2 * B.b", "qmin": "abs:10"}
    every, _ = measure(base, [], protect, allocation="all", out=tmp_path / "all", **kwargs)
    some, _ = measure(base, [], protect, allocation="dependent", out=tmp_path / "dep", **kwargs)
    d_all = {r["column"]: float(r["delta"]) for r in csv.DictReader((tmp_path / "all" / "deltas.csv").open())}
    d_dep = {r["column"]: float(r["delta"]) for r in csv.DictReader((tmp_path / "dep" / "deltas.csv").open())}
    assert d_all["B.a"] == pytest.approx(6.0 / (3 * 3.0), rel=1e-8)  # m = N = 3
    assert d_dep["B.a"] == pytest.approx(6.0 / (2 * 3.0), rel=1e-8)  # m = 2: Q reads a and b
    assert every["bound_over_margin"] == pytest.approx(2.0 / 3.0, rel=1e-8)
    assert some["bound_over_margin"] == pytest.approx(1.0, rel=1e-8)


def test_fn_needs_a_function_that_depends_on_the_quantity(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.c": 7.0}))
    with pytest.raises(MeasureError, match="depends"):
        measure(base, [], ["B.a=fn", "B.c=fn"], function="3 * B.a", qmin="abs:10")
    with pytest.raises(MeasureError, match="needs the function"):
        measure(base, [], ["B.a=fn"])


def test_lemma_every_quantity_at_its_tolerance_leaves_the_function_at_its_least(tmp_path):
    # Linear Q: the first-order bound is exact. Each quantity moved by its whole tolerance, against Q.
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0}))
    kwargs = {"function": "3 * B.a + 2 * B.b", "qmin": "abs:10"}
    protect = ["B.a=fn", "B.b=fn"]
    at = write(tmp_path / "at.csv", flat(**{"B.a": 4.0 - 1.0, "B.b": 2.0 - 1.5}))
    result, _ = measure(base, [at], protect, **kwargs)
    assert result["D_peak"] == pytest.approx(1.0, rel=1e-7)
    assert result["receiver_margin_least"] == pytest.approx(
        0.0, abs=1e-6
    )  # Q = Q_min: kept, with nothing to spare
    inside = write(tmp_path / "inside.csv", flat(**{"B.a": 4.0 - 0.9, "B.b": 2.0 - 1.35}))
    result, _ = measure(base, [inside], protect, **kwargs)
    assert result["D_peak"] == pytest.approx(0.9, rel=1e-7)
    assert result["3_tolerable"] and not result["losses"]
    assert result["receiver_margin_least"] == pytest.approx(0.6, rel=1e-6)


def test_lemma_holds_to_first_order_for_a_curved_function(tmp_path):
    # Q = a b / (a + b). Changes of 1 % of each tolerance: the function may lose 1 % of the margin, and no more
    # than that plus a second-order term.
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0}))
    kwargs = {"function": "B.a * B.b / (B.a + B.b)", "qmin": "rel:0.5"}
    protect = ["B.a=fn", "B.b=fn"]
    first, _ = measure(base, [], protect, out=tmp_path / "m", **kwargs)
    deltas = {r["column"]: float(r["delta"]) for r in csv.DictReader((tmp_path / "m" / "deltas.csv").open())}
    share = 0.01
    run = write(
        tmp_path / "run.csv", flat(**{"B.a": 4.0 - share * deltas["B.a"], "B.b": 2.0 - share * deltas["B.b"]})
    )
    result, _ = measure(base, [run], protect, **kwargs)
    assert result["D_peak"] == pytest.approx(share, rel=1e-6)
    lost = first["margin"] - result["receiver_margin_least"]
    assert lost == pytest.approx(share * first["margin"], rel=0.02)
    assert lost <= share * first["margin"] * (1.0 + 0.02)


def test_margin_follows_the_mode(tmp_path):
    q = np.array([5.0, 4.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    base = write(tmp_path / "base.csv", {"B.q": q})
    margins = {
        mode: measure(base, [], ["B.q=abs:1"], function="B.q", qmin="abs:2", mode=mode)[0]["margin"]
        for mode in ("pointwise", "terminal", "time_average")
    }
    assert margins["pointwise"] == pytest.approx(3.0 - 2.0)
    assert margins["terminal"] == pytest.approx(11.0 - 2.0)
    assert margins["time_average"] == pytest.approx(np.trapezoid(q, TIMES) / 10.0 - 2.0)


# ---- the interference ----------------------------------------------------------------------------------------------


def test_baseline_alone_gives_zero_and_every_stage(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0}))
    result, _ = measure(base, [], ["B.a=abs:1"])
    assert result["D_peak"] == 0.0 and result["O"] == 1.0
    assert result["1_trajectory"] and result["3_tolerable"] and result["4_distribution"]
    assert result["bits_upper"] == 0.0 and result["bits_lower"] == 0.0


def test_d_is_the_largest_change_over_its_tolerance_over_runs_quantities_and_times(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0}))
    bump = np.zeros(TIMES.size)
    bump[3] = 0.3
    one = write(tmp_path / "one" / "series.csv", {"B.a": 4.0 + bump, "B.b": np.full(TIMES.size, 2.0)})
    two = write(
        tmp_path / "two" / "series.csv", {"B.a": np.full(TIMES.size, 4.0), "B.b": 2.0 - 0.1 * TIMES / 10.0}
    )
    result, written = measure(base, [one, two], ["B.a=abs:0.5", "B.b=abs:0.05"], out=tmp_path / "m")
    assert result["D_peak"] == pytest.approx(0.1 / 0.05)  # B.b of "two" at the end
    assert result["t_peak"] == 10.0 and result["activities"] == 2
    _, series = read_series(written[0])
    assert series["D[one]"][3] == pytest.approx(0.6) and series["D[two]"][-1] == pytest.approx(2.0)
    assert np.all(series["D"] >= 0.0) and series["D"][3] == pytest.approx(0.6)
    assert series["O"][-1] == pytest.approx(1.0 / 3.0)
    assert not result["3_tolerable"] and not result["1_trajectory"]


def test_stage_thresholds(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0}))
    tiny = write(tmp_path / "tiny.csv", flat(**{"B.a": 1.0 + 5e-7}))
    small = write(tmp_path / "small.csv", flat(**{"B.a": 1.0 + 0.5}))
    large = write(tmp_path / "large.csv", flat(**{"B.a": 1.0 + 1.5}))
    protect = ["B.a=abs:1"]
    assert measure(base, [tiny], protect)[0]["1_trajectory"]
    assert not measure(base, [tiny], protect, stage_tol=1e-7)[0]["1_trajectory"]
    middle = measure(base, [small], protect)[0]
    assert middle["3_tolerable"] and not middle["1_trajectory"] and not middle["4_distribution"]
    assert not measure(base, [large], protect)[0]["3_tolerable"]
    assert measure(base, [small], protect, bits_tol=100.0)[0]["4_distribution"]


def test_a_missing_number_is_not_read_as_no_change(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0, "B.n": 10.0}))
    holed = np.full(TIMES.size, 1.0)
    holed[4] = np.nan
    run = write(tmp_path / "run.csv", {"B.a": holed, "B.n": np.full(TIMES.size, 10.0)})
    with pytest.raises(MeasureError, match="no value at t = 4"):
        measure(base, [run], ["B.a=abs:1"])
    gone = np.full(TIMES.size, 10.0)
    gone[3:] = 0.0  # the receiver has died out before the hole: nothing is read there
    run = write(tmp_path / "run2.csv", {"B.a": holed, "B.n": gone})
    result, _ = measure(base, [run], ["B.a=abs:1"], receiver_density="B.n", viable=1.0)
    assert (
        result["losses"] == [{"activity": "run2", "t": 3.0, "cause": "extinction"}]
        and result["D_peak"] == 0.0
    )


def test_a_stride_reads_every_nth_stored_time(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0}))
    bump = np.full(TIMES.size, 1.0)
    bump[3], bump[4] = 1.8, 1.2  # the peak is at a time that a stride of two passes over
    run = write(tmp_path / "run.csv", {"B.a": bump})
    assert measure(base, [run], ["B.a=abs:1"])[0]["D_peak"] == pytest.approx(0.8)
    coarse, _ = measure(base, [run], ["B.a=abs:1"], stride=2)
    assert coarse["D_peak"] == pytest.approx(0.2) and coarse["t_peak"] == 4.0
    assert coarse["stored_times"] == 6
    times, _ = read_series(run, stride=5)
    assert list(times) == [0.0, 5.0, 10.0]


def test_runs_must_be_stored_on_the_times_of_the_baseline(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0}))
    other = write(tmp_path / "other.csv", {"B.a": np.ones(6)}, times=np.linspace(0.0, 10.0, 6))
    with pytest.raises(MeasureError, match="times of the baseline"):
        measure(base, [other], ["B.a=abs:1"])
    with pytest.raises(MeasureError, match="no column"):
        measure(base, [], ["B.zz=abs:1"])


# ---- losses of the receiver ----------------------------------------------------------------------------------------


def test_extinction_ends_the_reading_and_fails_the_stages(tmp_path):
    n = np.full(TIMES.size, 10.0)
    base = write(tmp_path / "base.csv", {"B.a": np.full(TIMES.size, 1.0), "B.n": n})
    dying = n.copy()
    dying[6:] = 0.5  # below viable from t = 6
    a = np.full(TIMES.size, 1.0)
    a[2] = 1.2  # 0.2 while B is there
    a[8] = 50.0  # read by nobody: B has died out
    run = write(tmp_path / "run.csv", {"B.a": a, "B.n": dying})
    result, written = measure(
        base, [run], ["B.a=abs:1"], receiver_density="B.n", viable=1.0, out=tmp_path / "m"
    )
    assert result["D_peak"] == pytest.approx(0.2)
    assert result["losses"] == [{"activity": "run", "t": 6.0, "cause": "extinction"}]
    assert not result["1_trajectory"] and not result["3_tolerable"] and not result["4_distribution"]
    assert result["bits_upper"] is None
    _, series = read_series(written[0])
    assert np.all(np.isnan(series["D[run]"][6:])) and np.all(series["D"][6:] == 0.0)  # u_0 is still a row


def test_function_read_pointwise_is_lost_at_the_first_time_below(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.q": 10.0}))
    q = np.full(TIMES.size, 10.0)
    q[1] = 9.0
    q[4:] = 3.0
    run = write(tmp_path / "run.csv", {"B.q": q})
    result, _ = measure(base, [run], ["B.q=abs:10"], function="B.q", qmin="rel:0.5")
    assert result["losses"] == [{"activity": "run", "t": 4.0, "cause": "function"}]
    assert result["D_peak"] == pytest.approx(0.1)  # only what was read before the loss
    assert not result["3_tolerable"]


def test_function_read_at_the_end_is_judged_on_the_run_and_every_time_is_read(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.q": 10.0}))
    dip = np.full(TIMES.size, 10.0)
    dip[4] = 3.0  # below Q_min for a moment, back at the end
    low = np.full(TIMES.size, 10.0)
    low[-1] = 3.0  # below Q_min at the end
    recovered = write(tmp_path / "recovered.csv", {"B.q": dip})
    ended = write(tmp_path / "ended.csv", {"B.q": low})
    kwargs = {"function": "B.q", "qmin": "rel:0.5", "mode": "terminal"}
    fine, _ = measure(base, [recovered], ["B.q=abs:10"], **kwargs)
    assert not fine["losses"] and fine["D_peak"] == pytest.approx(0.7) and fine["3_tolerable"]
    lost, _ = measure(base, [ended], ["B.q=abs:10"], **kwargs)
    assert lost["losses"] == [{"activity": "ended", "t": 10.0, "cause": "function"}]  # put at the horizon
    assert lost["D_peak"] == pytest.approx(0.7)  # the last time is read too
    assert not lost["3_tolerable"]


def test_a_receiver_lost_on_the_baseline_is_not_measured(tmp_path):
    base = write(tmp_path / "base.csv", {"B.q": np.linspace(10.0, 1.0, TIMES.size)})
    with pytest.raises(MeasureError, match="not_functional"):
        measure(base, [], ["B.q=abs:1"], function="B.q", qmin="rel:0.5")
    thin = write(tmp_path / "thin.csv", flat(**{"B.q": 10.0, "B.n": 0.1}))
    with pytest.raises(MeasureError, match="not_functional"):
        measure(thin, [], ["B.q=abs:1"], receiver_density="B.n", viable=1.0)


# ---- the activities of the sender ----------------------------------------------------------------------------------


def test_runs_in_which_the_sender_loses_its_function_are_left_out(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0, "A.q": 8.0}))
    working = write(tmp_path / "working.csv", flat(**{"B.a": 1.3, "A.q": 6.0}))
    stopped = write(tmp_path / "stopped.csv", flat(**{"B.a": 9.0, "A.q": 1.0}))
    result, written = measure(
        base,
        [working, stopped],
        ["B.a=abs:1"],
        sender_function="A.q",
        sender_qmin="rel:0.5",
        out=tmp_path / "m",
    )
    assert [row["activity"] for row in result["excluded_by_sender_function"]] == ["stopped"]
    assert result["activities"] == 1 and result["D_peak"] == pytest.approx(0.3)
    _, series = read_series(written[0])
    assert "D[working]" in series and "D[stopped]" not in series


def test_removal_is_read_apart_from_the_activities(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0, "A.q": 8.0}))
    working = write(tmp_path / "working.csv", flat(**{"B.a": 1.1, "A.q": 8.0}))
    removed = write(tmp_path / "removed.csv", flat(**{"B.a": 3.0, "A.q": 0.0}))
    result, written = measure(
        base,
        [working],
        ["B.a=abs:1"],
        sender_function="A.q",
        sender_qmin="rel:0.5",
        removal=removed,
        out=tmp_path / "m",
    )
    assert result["D_peak"] == pytest.approx(0.1) and result["D_removal"] == pytest.approx(2.0)
    assert result["3_tolerable"] and result["removal_lost_at"] is None
    _, series = read_series(written[0])
    assert series["D_removal"][0] == pytest.approx(2.0)


# ---- groups of protected quantities --------------------------------------------------------------------------------


def test_a_group_is_read_as_the_whole_is_on_its_own_quantities(tmp_path):
    still = np.full(TIMES.size, 1.0)
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0, "B.c": 1.0}))
    bump = np.zeros(TIMES.size)
    bump[3] = 0.3
    one = write(tmp_path / "one" / "series.csv", {"B.a": 4.0 + bump, "B.b": 2.0 * still, "B.c": still})
    two = write(
        tmp_path / "two" / "series.csv", {"B.a": 4.0 * still, "B.b": 2.0 - 0.1 * TIMES / 10.0, "B.c": still}
    )
    groups = {"inner": ["B.a"], "outer": ["B.b", "B.c"], "unmoved": ["B.c"]}
    protect = ["B.a=abs:0.5", "B.b=abs:0.05", "B.c=abs:1"]
    result, _ = measure(base, [one, two], protect, groups=groups)
    found = result["groups"]
    assert found["inner"]["D_peak"] == pytest.approx(0.6) and found["inner"]["t_peak"] == 3.0
    assert (found["inner"]["D_activity"], found["inner"]["D_quantity"]) == ("one", "B.a")
    assert found["outer"]["D_peak"] == pytest.approx(2.0)
    assert (found["outer"]["D_activity"], found["outer"]["D_quantity"]) == ("two", "B.b")
    assert found["inner"]["3_tolerable"] and not found["inner"]["1_trajectory"]
    assert not found["outer"]["3_tolerable"]
    unmoved = found["unmoved"]
    assert unmoved["D_peak"] == 0.0 and unmoved["O"] == 1.0 and unmoved["D_quantity"] is None
    assert unmoved["1_trajectory"] and unmoved["3_tolerable"] and unmoved["4_distribution"]
    assert [found[label]["protected"] for label in groups] == [1, 2, 1]
    # groups that cover the whole: its D is the largest of theirs
    assert result["D_peak"] == max(found["inner"]["D_peak"], found["outer"]["D_peak"])
    assert "groups" not in measure(base, [one, two], protect)[0]


def test_a_group_keeps_the_tolerances_that_its_quantities_have_in_the_whole(tmp_path):
    # Q = 3 a + 2 b = 16, Q_min = 10: the margin 6 is shared among the N = 3 protected quantities of the whole.
    base = write(tmp_path / "base.csv", flat(**{"B.a": 4.0, "B.b": 2.0, "B.c": 7.0}))
    run = write(tmp_path / "run.csv", flat(**{"B.a": 4.0 - 0.1, "B.b": 2.0, "B.c": 7.0}))
    result, _ = measure(
        base,
        [run],
        ["B.a=fn", "B.b=fn", "B.c=abs:1"],
        function="3 * B.a + 2 * B.b",
        qmin="abs:10",
        allocation="all",
        groups={"a": ["B.a"]},
    )
    alone = result["groups"]["a"]
    assert alone["D_peak"] == pytest.approx(0.1 / (6.0 / (3 * 3.0)), rel=1e-7)  # not 6 / (1 * 3)
    assert alone["D_peak"] == pytest.approx(result["D_peak"])


def test_the_losses_of_the_receiver_are_those_of_the_whole_in_every_group(tmp_path):
    n, still = np.full(TIMES.size, 10.0), np.full(TIMES.size, 1.0)
    base = write(tmp_path / "base.csv", {"B.a": still, "B.b": still, "B.n": n})
    dying = n.copy()
    dying[6:] = 0.5  # below viable from t = 6
    a = still.copy()
    a[2] = 1.2
    a[8] = 50.0  # read by nobody: B has died out
    run = write(tmp_path / "run.csv", {"B.a": a, "B.b": still, "B.n": dying})
    result, _ = measure(
        base,
        [run],
        ["B.a=abs:1", "B.b=abs:1"],
        receiver_density="B.n",
        viable=1.0,
        groups={"moved": ["B.a"], "unmoved": ["B.b"]},
    )
    assert result["groups"]["moved"]["D_peak"] == pytest.approx(0.2)  # read until the loss, as the whole is
    unmoved = result["groups"]["unmoved"]
    assert unmoved["D_peak"] == 0.0  # nothing of this group moved, and still no stage holds: B was lost
    assert not unmoved["1_trajectory"] and not unmoved["3_tolerable"] and not unmoved["4_distribution"]
    assert unmoved["bits_upper"] is None


def test_a_group_names_protected_quantities_and_at_least_one(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0, "B.b": 1.0}))
    with pytest.raises(MeasureError, match="not protected"):
        measure(base, [], ["B.a=abs:1"], groups={"g": ["B.b"]})
    with pytest.raises(MeasureError, match="no quantity"):
        measure(base, [], ["B.a=abs:1"], groups={"g": []})


def test_groups_are_read_for_the_removal_too_and_leave_their_files(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0, "B.b": 1.0}))
    run = write(tmp_path / "run.csv", flat(**{"B.a": 1.1, "B.b": 1.0}))
    removed = write(tmp_path / "removed.csv", flat(**{"B.a": 1.0, "B.b": 3.0}))
    result, written = measure(
        base,
        [run],
        ["B.a=abs:1", "B.b=abs:1"],
        removal=removed,
        groups={"a": ["B.a"], "b": ["B.b"]},
        out=tmp_path / "m",
    )
    assert result["groups"]["a"]["D_removal"] == 0.0
    assert result["groups"]["b"]["D_removal"] == pytest.approx(2.0) == result["D_removal"]
    names = {"orthogonality.csv", "deltas.csv", "result.json", "groups.csv", "groups_series.csv"}
    assert {path.name for path in written} == names
    rows = {row["group"]: row for row in csv.DictReader((tmp_path / "m" / "groups.csv").open())}
    assert float(rows["a"]["D_peak"]) == pytest.approx(0.1) and float(rows["b"]["D_peak"]) == 0.0
    _, series = read_series(tmp_path / "m" / "groups_series.csv")
    assert series["D[a]"] == pytest.approx(np.full(TIMES.size, 0.1)) and np.all(series["D[b]"] == 0.0)
    # and every quantity on its own, beside its tolerance
    deltas = {row["column"]: row for row in csv.DictReader((tmp_path / "m" / "deltas.csv").open())}
    assert float(deltas["B.a"]["D"]) == pytest.approx(0.1) and deltas["B.a"]["D_activity"] == "run"
    assert float(deltas["B.b"]["D"]) == 0.0 and float(deltas["B.b"]["D_removal"]) == pytest.approx(2.0)


# ---- information ---------------------------------------------------------------------------------------------------


def test_bits_upper_is_the_gaussian_channel_of_the_largest_response():
    deviation = np.full((2, 5), 0.3)  # N = 2 quantities, T = 5 times: M = 10, P = 10 * 0.09
    lower, upper = information_bounds([deviation], sigma=1.0)
    assert upper == pytest.approx(0.5 * 10 * math.log2(1.0 + 0.9 / 10))
    _, quieter = information_bounds([deviation], sigma=3.0)
    assert quieter == pytest.approx(0.5 * 10 * math.log2(1.0 + 0.9 / (10 * 9.0)))
    assert 0.0 <= lower <= upper


def test_bits_lower_goes_from_nothing_to_the_whole_choice():
    same = information_bounds([np.zeros((1, 4))])
    assert same == (0.0, 0.0)  # a response that is the baseline says nothing
    far = np.full((1, 4), 1e3)
    lower, upper = information_bounds([far])
    assert lower == pytest.approx(1.0, abs=1e-9)  # two responses told apart for certain: one bit
    assert lower <= upper
    three, _ = information_bounds([far, -far])
    assert three == pytest.approx(math.log2(3.0), abs=1e-9)


def test_bits_of_the_measure_use_the_changes_over_the_tolerances(tmp_path):
    base = write(tmp_path / "base.csv", flat(**{"B.a": 1.0, "B.b": 1.0}))
    run = write(tmp_path / "run.csv", flat(**{"B.a": 1.2, "B.b": 1.0}))
    result, _ = measure(base, [run], ["B.a=abs:0.4", "B.b=abs:1"])
    power = TIMES.size * 0.5**2  # B.a off by half a tolerance at each of the 11 times
    size = 2 * TIMES.size
    assert result["bits_upper"] == pytest.approx(0.5 * size * math.log2(1.0 + power / size))


# ---- expressions ---------------------------------------------------------------------------------------------------


def test_function_reads_dotted_names_and_quoted_names():
    f = Function('B.species.ATP / (B.species.ATP + col("B/ADP")) + exp(0) - 1')
    assert f.names() == ["B.species.ATP", "B/ADP"]
    values = f.values({"B.species.ATP": np.array([3.0, 1.0]), "B/ADP": np.array([1.0, 1.0])})
    assert values == pytest.approx([0.75, 0.5])


@pytest.mark.parametrize(
    "text", ["__import__('os')", "B.a if B.b else 1", "open(B.a)", "B.a[0]", "lambda: 1", "3", "B.a +"]
)
def test_function_refuses_anything_that_is_not_arithmetic(text):
    with pytest.raises(MeasureError):
        Function(text)


def test_function_needs_its_columns_and_a_positive_least_value():
    with pytest.raises(MeasureError, match="does not have"):
        Function("B.a + B.b").values({"B.a": np.ones(3)})
    with pytest.raises(MeasureError):
        Function("B.a", "abs:-1")
    with pytest.raises(MeasureError):
        Function("B.a", "half")
    with pytest.raises(MeasureError):
        Function("B.a", mode="sometimes")
