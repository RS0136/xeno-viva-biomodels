"""The measure of orthogonality of a sender A to a receiver B, from the stored series of runs.

A set of runs is compared: a baseline, and runs in which one activity of A was stepped at time zero, all from
the same initial conditions and under the same supply. Every run is a CSV file with a column ``time`` and one
column per stored quantity; all runs are stored on the same times.

Protected quantities and tolerances
    B protects y_k (k = 1..N), columns of the files. The tolerance delta_k of each is written in one of three ways:

        abs:X                delta_k = X
        rel:R[,floor:F]      delta_k = max(R |y_k(0)|, F), with y_k(0) of the baseline at time zero
        fn                   delta_k = margin / (m max_t |dQ_B/dy_k|)

    Q_B is the function that B must keep, an expression of its protected quantities, with Q_B >= Q_min. The margin
    is what the baseline has to spare: min_t Q_B - Q_min (mode ``pointwise``), Q_B(T) - Q_min (``terminal``) or
    mean_t Q_B - Q_min (``time_average``). m is N (allocation ``all``) or the number of protected quantities that
    Q_B depends on (``dependent``). The slopes are central differences along the baseline, and their largest
    absolute value over the stored times is taken.

Lemma (why a tolerance ``fn`` is not arbitrary)
    To first order in the changes of the protected quantities,

        |dQ_B| <= sum_k |dQ_B/dy_k| |dy_k| <= sum_k max_t |dQ_B/dy_k| delta_k

    whenever |dy_k| <= delta_k for every k. With every delta_k from ``fn`` the right side is (N / m) margin with
    allocation ``all`` (m = N, so it is the margin) and the margin with ``dependent`` (the terms of the quantities
    that Q_B does not depend on vanish). So D <= 1 implies Q_B >= Q_min, to first order, in the sense of the mode.
    For any mixture of tolerances the right side over the margin is reported as ``bound_over_margin``: the
    implication holds, to first order, when it is at most 1.

When the receiver is lost
    tau_B(u) is the first stored time at which the density of B is below ``viable``, or an earlier one at which
    Q_B < Q_min when the function is read pointwise. Read at the end or on average, the function is judged on the
    whole run: its loss is put at the horizon, and every stored time is read.

Admissible activities
    U_A is the baseline u_0 and the runs in which A keeps its own function, Q_A >= Q_A,min. A run that does not is
    left out (``excluded_by_sender_function``). Taking A away is outside U_A and is read apart (``D_removal``).

Interference
    D(t) = sup over u in U_A with t < tau_B(u) of max_k |y_k^u(t) - y_k^u0(t)| / delta_k
    D    = sup_t D(t),      O = 1 / (1 + D)

    u_0 is always among the rows, as a row of zeros, so D(t) >= 0 at every time. O is for display only.

Stages
    1 trajectory     no loss, and D <= stage_tol (1e-6 unless given)
    3 tolerable      no loss, and D <= 1
    4 distribution   no loss, and the upper bound on the bits <= bits_tol

Groups
    The protected quantities may be put into groups: the layers of the receiver, or the quantities that every
    organism has against those that are characteristic of one. A group is read exactly as the whole is, on its own
    quantities alone and with the tolerances that they have in the whole, so that D of the whole is the largest D
    of the groups that cover it. The losses of the receiver are those of the whole: a receiver is not lost at one
    layer and working at another.

Bits
    The response of each activity is a point: its changes over the tolerances, (N, T) flattened; the baseline is the
    origin. With observation noise of standard deviation sigma on each of the M = N T numbers, in tolerances, the
    information that the response gives about which activity was chosen is at most that of a Gaussian channel of
    the same peak power, (M / 2) log2(1 + P / (M sigma^2)) with P the largest squared norm of a point, and at least
    that of a uniform choice, bounded from below through the pairwise Bhattacharyya distances
    (Kolchinsky and Tracey 2017): -mean_i log2 mean_j exp(-|x_i - x_j|^2 / (8 sigma^2)).
"""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODES = ("pointwise", "terminal", "time_average")
ALLOCATIONS = ("all", "dependent")
STAGE_TOL = 1e-6
BITS_TOL = 1e-3
#: Q depends on a quantity when a relative change of the quantity moves Q by at least this share of Q.
RELEVANT = 1e-6
#: The step of a central difference, as a share of the size of the quantity.
STEP = 1e-6


class MeasureError(ValueError):
    """The comparison cannot be made as it was asked."""


# ---- series --------------------------------------------------------------------------------------------------------


def read_series(path: str | Path, stride: int = 1) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """The stored times of a run and its columns; with ``stride`` n, every n-th stored time from the first."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [row for row in reader if row][:: max(int(stride), 1)]
    if "time" not in header:
        raise MeasureError(f"{path}: no column 'time'")
    data = np.array(rows, dtype=np.float64) if rows else np.zeros((0, len(header)))
    columns = {name: data[:, i].copy() for i, name in enumerate(header)}
    times = columns.pop("time")
    if times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise MeasureError(f"{path}: needs two or more stored times, increasing")
    return times, columns


# ---- functions -----------------------------------------------------------------------------------------------------

_CALLS = {
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "tanh": np.tanh,
    "abs": np.abs,
    "minimum": np.minimum,
    "maximum": np.maximum,
}
_BINARY = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
}


def _dotted(node: ast.AST) -> str | None:
    """``B.species.ATP`` as one name, or None when the node is not a chain of names."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


class Function:
    """Q of one system: an expression of columns, the least value that it may take, and how it is read over time.

    The expression may use + - * / **, numbers, the functions exp, log, sqrt, tanh, abs, minimum and maximum, and
    the names of columns. A dotted name (``B.species.ATP``) is one column; a column whose name is not made of
    identifiers and dots is written ``col("...")``.
    """

    def __init__(self, text: str, qmin: str | None = None, mode: str = "pointwise"):
        if mode not in MODES:
            raise MeasureError(f"mode must be one of {', '.join(MODES)}, not {mode!r}")
        self.text, self.mode = str(text), mode
        try:
            self._tree = ast.parse(self.text.strip(), mode="eval").body
        except SyntaxError as exc:
            raise MeasureError(f"cannot read the function {text!r}: {exc.msg}") from None
        self._names: list[str] = []
        self._check(self._tree)
        if not self._names:
            raise MeasureError(f"the function {text!r} reads no column")
        self._kind, self._limit = None, None
        if qmin is not None:
            kind, _, value = str(qmin).partition(":")
            if kind not in ("abs", "rel") or not value:
                raise MeasureError(f"Q_min is written abs:X or rel:R, not {qmin!r}")
            self._kind, self._limit = kind, float(value)
            if not self._limit > 0.0:
                raise MeasureError("Q_min must be positive: the function must be worth keeping")

    def _check(self, node: ast.AST) -> None:
        name = _dotted(node)
        if name is not None:
            if name not in self._names:
                self._names.append(name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, (int, float))
            and not isinstance(node.value, bool)
        ):
            pass
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            self._check(node.left)
            self._check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            self._check(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            if node.func.id == "col":
                if (
                    len(node.args) != 1
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)
                ):
                    raise MeasureError(f'{self.text!r}: col takes one name in quotes, col("...")')
                if node.args[0].value not in self._names:
                    self._names.append(node.args[0].value)
            elif node.func.id in _CALLS:
                for argument in node.args:
                    self._check(argument)
            else:
                raise MeasureError(
                    f"{self.text!r}: the function {node.func.id} is not available; use {', '.join(_CALLS)}"
                )
        else:
            raise MeasureError(
                f"{self.text!r}: only numbers, columns, + - * / ** and {', '.join(_CALLS)} may be used"
            )

    def names(self) -> list[str]:
        """The columns that the expression reads."""
        return list(self._names)

    def _value(self, node: ast.AST, columns: dict[str, np.ndarray]):
        name = _dotted(node)
        if name is not None:
            return columns[name]
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.BinOp):
            return _BINARY[type(node.op)](self._value(node.left, columns), self._value(node.right, columns))
        if isinstance(node, ast.UnaryOp):
            value = self._value(node.operand, columns)
            return -value if isinstance(node.op, ast.USub) else value
        if node.func.id == "col":
            return columns[node.args[0].value]
        return _CALLS[node.func.id](*(self._value(argument, columns) for argument in node.args))

    def values(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Q along a run."""
        missing = [name for name in self._names if name not in columns]
        if missing:
            raise MeasureError(f"the function {self.text!r} reads {missing}, which the run does not have")
        read = {name: np.asarray(columns[name], dtype=np.float64) for name in self._names}
        shape = next(iter(read.values())).shape
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(self._value(self._tree, read), dtype=np.float64), shape).copy()

    def q_min(self, q0: float) -> float:
        """The least admissible Q: a value, or a share of Q of the baseline at time zero."""
        if self._kind is None:
            raise MeasureError(f"the function {self.text!r} has no Q_min")
        return self._limit if self._kind == "abs" else self._limit * float(q0)

    def margin(self, q: np.ndarray, times: np.ndarray, q_min: float) -> float:
        """Q - Q_min as the mode reads it: the least over time, at the end, or on average over time."""
        if q.size == 0:
            return float("nan")
        if self.mode == "terminal":
            return float(q[-1] - q_min)
        if self.mode == "time_average":
            span = float(times[-1] - times[0]) if times.size > 1 else 0.0
            return float(np.trapezoid(q, times) / span - q_min) if span > 0.0 else float(q[-1] - q_min)
        return float(np.min(q) - q_min)

    def slopes(self, columns: dict[str, np.ndarray], names: list[str]) -> dict[str, np.ndarray]:
        """dQ/dy along a run for each of ``names``, by central differences (zero for a name that Q does not read)."""
        out = {}
        for name in names:
            y = np.asarray(columns[name], dtype=np.float64)
            if name not in self._names:
                out[name] = np.zeros_like(y)
                continue
            size = float(np.max(np.abs(y)))
            step = STEP * (size if size > 0.0 else 1.0)
            high = self.values({**columns, name: y + step})
            low = self.values({**columns, name: y - step})
            out[name] = (high - low) / (2.0 * step)
        return out


# ---- tolerances ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Protected:
    """One protected quantity and how its tolerance is written."""

    column: str
    kind: str  # abs, rel or fn
    value: float | None = None  # X of abs, R of rel
    floor: float | None = None  # F of rel

    @property
    def spec(self) -> str:
        if self.kind == "fn":
            return "fn"
        if self.kind == "abs":
            return f"abs:{self.value!r}"
        return f"rel:{self.value!r}" + (f",floor:{self.floor!r}" if self.floor is not None else "")


def parse_protected(text: str) -> Protected:
    """``COLUMN=abs:X``, ``COLUMN=rel:R[,floor:F]`` or ``COLUMN=fn``."""
    column, mark, spec = str(text).rpartition("=")
    column, spec = column.strip(), spec.strip()
    if not mark or not column:
        raise MeasureError(
            f"a protected quantity is written COLUMN=abs:X, COLUMN=rel:R[,floor:F] or COLUMN=fn, not {text!r}"
        )
    try:
        if spec == "fn":
            return Protected(column, "fn")
        if spec.startswith("abs:"):
            value = float(spec[4:])
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError
            return Protected(column, "abs", value)
        if spec.startswith("rel:"):
            share, _, rest = spec[4:].partition(",")
            floor = None
            if rest:
                if not rest.startswith("floor:"):
                    raise ValueError
                floor = float(rest[6:])
                if not (math.isfinite(floor) and floor > 0.0):
                    raise ValueError
            value = float(share)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError
            return Protected(column, "rel", value, floor)
    except ValueError:
        pass
    raise MeasureError(
        f"{column}: the tolerance {spec!r} is not abs:X, rel:R[,floor:F] or fn with positive numbers"
    )


def tolerances(
    protected: list[Protected],
    times: np.ndarray,
    baseline: dict[str, np.ndarray],
    function: Function | None,
    q: np.ndarray | None,
    margin: float | None,
    allocation: str,
) -> tuple[np.ndarray, list[dict]]:
    """delta_k of every protected quantity, and for the record where each came from."""
    names = [item.column for item in protected]
    peak = dict.fromkeys(names, 0.0)
    depends = dict.fromkeys(names, False)
    if function is not None:
        size_q = max(float(np.max(np.abs(q))), np.finfo(float).tiny)
        for name, slope in function.slopes(baseline, names).items():
            peak[name] = float(np.max(np.abs(slope)))
            # a quantity that is zero on the baseline counts by its unit
            size = float(np.max(np.abs(baseline[name])))
            depends[name] = bool(peak[name] * (size if size > 0.0 else 1.0) / size_q > RELEVANT)
    shared = len(protected) if allocation == "all" else max(sum(depends.values()), 1)
    deltas, record = [], []
    for item in protected:
        y0 = float(baseline[item.column][0])
        if item.kind == "abs":
            delta = float(item.value)
        elif item.kind == "rel":
            delta = item.value * abs(y0)
            if item.floor is not None:
                delta = max(delta, item.floor)
            elif not delta > 0.0:
                raise MeasureError(
                    f"{item.column}: a relative tolerance of a quantity that is zero at time zero; give floor:F or abs:X"
                )
        else:
            if function is None:
                raise MeasureError(f"{item.column}: the tolerance fn needs the function of the receiver")
            if not depends[item.column]:
                raise MeasureError(
                    f"{item.column}: the tolerance fn needs a function that depends on this quantity "
                    f"({function.text!r} does not); write abs:X or rel:R instead"
                )
            delta = margin / (shared * peak[item.column])
        if not (math.isfinite(delta) and delta > 0.0):
            raise MeasureError(
                f"{item.column}: the tolerance came to {delta!r}; it must be finite and positive"
            )
        deltas.append(delta)
        record.append(
            {
                "column": item.column,
                "tolerance": item.spec,
                "delta": delta,
                "source": item.kind,
                "y0": y0,
                "slope_peak": peak[item.column],
                "function_depends": depends[item.column],
            }
        )
    return np.array(deltas), record


# ---- losses --------------------------------------------------------------------------------------------------------


def receiver_loss(
    function: Function | None,
    q_min: float | None,
    q: np.ndarray | None,
    density: np.ndarray | None,
    times: np.ndarray,
    viable: float | None,
) -> dict | None:
    """When the receiver stops being a system that works along one run; None when it works throughout.

    Returns ``index``, the first stored time that is not read (None when every time is read), ``t``, the time at
    which the loss is put, and ``cause``: function or extinction. Q is read only while there are cells to have it.
    """
    extinct = None
    if density is not None:
        dead = np.flatnonzero(~(np.asarray(density, dtype=np.float64) >= viable))
        extinct = int(dead[0]) if dead.size else None
    judged_on_the_run = False
    if function is not None:
        alive = np.asarray(q, dtype=np.float64)[:extinct]
        if function.mode == "pointwise":
            below = np.flatnonzero(~(alive >= q_min))
            if below.size:
                return {"index": int(below[0]), "t": float(times[below[0]]), "cause": "function"}
        else:
            judged_on_the_run = not function.margin(alive, times[:extinct], q_min) >= 0.0
    if extinct is not None:
        return {"index": extinct, "t": float(times[extinct]), "cause": "extinction"}
    if judged_on_the_run:
        return {"index": None, "t": float(times[-1]), "cause": "function"}
    return None


# ---- information ---------------------------------------------------------------------------------------------------


def information_bounds(deviations: list[np.ndarray], sigma: float = 1.0) -> tuple[float, float]:
    """Bits that the response gives about the activity: (lower bound for a uniform choice, upper bound).

    ``deviations`` holds, for each activity, the signed change of every protected quantity at every stored time
    over its tolerance, as an array (quantity, time). The baseline is added as the origin.
    """
    if not sigma > 0.0:
        raise MeasureError("the observation noise sigma must be positive")
    points = [np.asarray(item, dtype=np.float64).ravel() for item in deviations]
    points = [item for item in points if np.all(np.isfinite(item))]
    if not points:
        return 0.0, 0.0
    cloud = np.vstack([np.zeros_like(points[0]), *points])
    size = cloud.shape[1]
    power = float(np.max(np.sum(cloud**2, axis=1)))
    upper = 0.5 * size * math.log2(1.0 + power / (size * sigma**2))
    norms = np.sum(cloud**2, axis=1)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2.0 * cloud @ cloud.T, 0.0) / sigma**2
    np.fill_diagonal(squared, 0.0)
    lower = -float(np.mean(np.log(np.mean(np.exp(-squared / 8.0), axis=1)))) / math.log(2.0)
    return max(0.0, lower), max(0.0, upper)  # in this order: no negative zero


# ---- the measure ---------------------------------------------------------------------------------------------------


def _label(path: Path, taken: set[str]) -> str:
    name = path.parent.name if path.stem in ("series", "protected") else path.stem
    label, n = name, 2
    while label in taken:
        label, n = f"{name}#{n}", n + 1
    taken.add(label)
    return label


def _same_times(times: np.ndarray, other: np.ndarray, path) -> None:
    if other.shape != times.shape or not np.allclose(other, times, rtol=1e-12, atol=1e-12):
        raise MeasureError(f"{path}: not stored on the times of the baseline")


def measure(
    baseline: str | Path,
    activities: list[str | Path] | dict[str, str | Path],
    protect: list[str],
    *,
    receiver_density: str | None = None,
    viable: float | None = None,
    function: str | None = None,
    qmin: str | None = None,
    mode: str = "pointwise",
    allocation: str = "all",
    sender_function: str | None = None,
    sender_qmin: str | None = None,
    sender_mode: str = "pointwise",
    removal: str | Path | None = None,
    groups: dict[str, list[str]] | None = None,
    sigma: float = 1.0,
    stage_tol: float = STAGE_TOL,
    bits_tol: float = BITS_TOL,
    stride: int = 1,
    out: str | Path | None = None,
) -> tuple[dict, list[Path]]:
    """Orthogonality of a sender to a receiver, from the baseline run and the runs with an activity stepped.

    ``protect`` lists the protected quantities of the receiver as ``COLUMN=abs:X``, ``COLUMN=rel:R[,floor:F]`` or
    ``COLUMN=fn``. ``function`` is Q_B and ``qmin`` its least value (``abs:X``, or ``rel:R`` of Q_B of the baseline
    at time zero); ``sender_function`` and ``sender_qmin`` are those of A. ``receiver_density`` names the column of
    the density of B, which is lost below ``viable``. ``removal`` is the run without A. ``groups`` names sets of
    protected quantities that are also read on their own (``result["groups"]``). With ``stride`` n only every
    n-th stored time of every run is read.

    Returns the numbers and, when ``out`` is given, the files written there: ``orthogonality.csv`` (time, D, O and
    D of every admissible activity), ``deltas.csv`` (every tolerance, where it came from, and D of the quantity
    alone), ``result.json`` and, with groups, ``groups.csv`` and ``groups_series.csv`` (D of every group over time).
    """
    if allocation not in ALLOCATIONS:
        raise MeasureError(f"allocation must be one of {', '.join(ALLOCATIONS)}, not {allocation!r}")
    protected = [parse_protected(text) for text in protect]
    if not protected:
        raise MeasureError("no protected quantity was given")
    names = [item.column for item in protected]
    if len(set(names)) != len(names):
        raise MeasureError("a protected quantity is listed twice")
    if receiver_density is not None and viable is None:
        raise MeasureError("the density of the receiver needs the viable density")

    times, base = read_series(baseline, stride)
    wanted = list(names) + ([receiver_density] if receiver_density else [])
    missing = [name for name in wanted if name not in base]
    if missing:
        raise MeasureError(f"{baseline}: no column {missing}")
    density0 = base[receiver_density] if receiver_density else None

    # The function of the receiver on the baseline: it must hold there, or there is nothing to protect.
    q_function = Function(function, qmin, mode) if function is not None else None
    q0 = q_min = margin = None
    if q_function is not None:
        if qmin is None:
            raise MeasureError("the function of the receiver needs its least value, Q_min")
        outside = [name for name in q_function.names() if name not in names]
        q0 = q_function.values(base)
        q_min = q_function.q_min(q0[0])
        margin = q_function.margin(q0, times, q_min)
    lost0 = receiver_loss(q_function, q_min, q0, density0, times, viable)
    if lost0 is not None or (margin is not None and not margin >= 0.0):
        cause = lost0["cause"] if lost0 else "function"
        at = lost0["t"] if lost0 else float(times[-1])
        raise MeasureError(
            f"not_functional: the receiver is lost on the baseline itself ({cause}, at t = {at:g})"
        )
    if q_function is not None and not margin > 0.0 and any(item.kind == "fn" for item in protected):
        raise MeasureError("the baseline has no margin (Q_B = Q_min somewhere): a tolerance fn would be zero")

    deltas, record = tolerances(protected, times, base, q_function, q0, margin, allocation)
    bound_over_margin = None
    if q_function is not None and margin > 0.0:
        bound_over_margin = float(sum(row["slope_peak"] * row["delta"] for row in record) / margin)

    # The function of the sender: which runs are in U_A.
    a_function = Function(sender_function, sender_qmin, sender_mode) if sender_function is not None else None
    a_min = None
    if a_function is not None:
        if sender_qmin is None:
            raise MeasureError("the function of the sender needs its least value, Q_A,min")
        a_min = a_function.q_min(a_function.values(base)[0])

    taken: set[str] = set()
    if isinstance(activities, dict):
        runs = [(str(label), Path(path)) for label, path in activities.items()]
    else:
        runs = [(_label(Path(path), taken), Path(path)) for path in activities]

    y0 = np.vstack([base[name] for name in names])  # (quantity, time)

    def read(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, dict | None]:
        """A run: its columns, its changes over the tolerances with NaN where it is not read, and its loss."""
        ts, columns = read_series(path, stride)
        _same_times(times, ts, path)
        absent = [name for name in wanted if name not in columns]
        if absent:
            raise MeasureError(f"{path}: no column {absent}")
        signed = (np.vstack([columns[name] for name in names]) - y0) / deltas[:, None]
        q = q_function.values(columns) if q_function is not None else None
        density = columns[receiver_density] if receiver_density else None
        loss = receiver_loss(q_function, q_min, q, density, times, viable)
        read_until = None if loss is None else loss["index"]
        gaps = np.argwhere(~np.isfinite(signed[:, :read_until]))
        if gaps.size:  # a number that is missing while the receiver works is not a change of zero
            k, n = gaps[0]
            raise MeasureError(f"{path}: {names[k]} has no value at t = {times[n]:g}")
        if read_until is not None:
            signed = signed.copy()
            signed[:, read_until:] = np.nan
        return columns, signed, loss

    groups = {str(label): list(members) for label, members in (groups or {}).items()}
    stray = sorted({name for members in groups.values() for name in members} - set(names))
    if stray:
        raise MeasureError(f"a group names {stray}, which are not protected")
    empty = [label for label, members in groups.items() if not members]
    if empty:
        raise MeasureError(f"the groups {empty} have no quantity")

    excluded, losses, kept, receiver_margins = [], [], {}, []
    for label, path in runs:
        columns, signed, loss = read(path)
        if a_function is not None:
            holds = a_function.margin(a_function.values(columns), times, a_min)
            if not holds >= 0.0:
                excluded.append({"activity": label, "sender_margin": holds})
                continue
        kept[label] = (signed, loss)  # the changes of an admissible run: NaN from its loss on
        if loss is not None:
            losses.append({"activity": label, "t": loss["t"], "cause": loss["cause"]})
        if q_function is not None:
            receiver_margins.append(q_function.margin(q_function.values(columns), times, q_min))
    lost = bool(losses)
    away = read(Path(removal))[1:] if removal is not None else None  # the run without the sender, read apart

    def reading(rows: list[int]) -> tuple[dict, dict[str, np.ndarray]]:
        """The interference on the quantities ``rows`` alone, with the tolerances that they have in the whole."""
        each = {label: np.max(np.abs(signed[rows]), axis=0) for label, (signed, _) in kept.items()}
        d_t = np.nanmax(np.vstack([np.zeros_like(times), *each.values()]), axis=0)  # u_0 is the row of zeros
        at = int(np.argmax(d_t))
        peak = float(d_t[at])
        largest, activity, quantity = 0.0, None, None  # what makes D: the run and the quantity at the peak
        for label, (signed, _) in kept.items():
            size = np.where(np.isfinite(signed[rows]), np.abs(signed[rows]), 0.0)
            if size.size and float(size.max()) > largest:
                largest, activity = float(size.max()), label
                quantity = names[rows[int(np.unravel_index(int(np.argmax(size)), size.shape)[0])]]
        lower, upper = information_bounds(
            [signed[rows] for signed, loss in kept.values() if loss is None], sigma
        )
        numbers = {
            "D_peak": peak,
            "D_end": float(d_t[-1]),
            "t_peak": float(times[at]),
            "D_activity": activity,
            "D_quantity": quantity,
            "O": 1.0 / (1.0 + peak),
            "1_trajectory": bool(not lost and peak <= stage_tol),
            "3_tolerable": bool(not lost and peak <= 1.0),
            "4_distribution": bool(not lost and upper <= bits_tol),
            "bits_lower": None if lost else lower,
            "bits_upper": None if lost else upper,
            "protected": len(rows),
        }
        series = {"D": d_t, **{f"D[{label}]": values for label, values in each.items()}}
        if away is not None:
            series["D_removal"] = np.max(np.abs(away[0][rows]), axis=0)
            finite = series["D_removal"][np.isfinite(series["D_removal"])]
            numbers["D_removal"] = float(np.max(finite)) if finite.size else None
        return numbers, series

    whole, series = reading(list(range(len(names))))
    result = {
        "status": "ok",
        **{key: value for key, value in whole.items() if key not in ("protected", "D_removal")},
        "activities": len(kept),
        "excluded_by_sender_function": excluded,
        "losses": losses,
        "protected": len(protected),
        "stored_times": int(times.size),
        "horizon": float(times[-1]),
        "sigma": float(sigma),
        "stage_tol": float(stage_tol),
        "bits_tol": float(bits_tol),
        "allocation": allocation,
    }
    if q_function is not None:
        result.update(
            {
                "function": q_function.text,
                "mode": mode,
                "Q_baseline": float(q0[0]),
                "Q_min": float(q_min),
                "margin": float(margin),
                "bound_over_margin": bound_over_margin,
                "function_reads_unprotected": outside,
                "receiver_margin_least": float(min(receiver_margins)) if receiver_margins else None,
            }
        )
    if a_function is not None:
        result.update(
            {"sender_function": a_function.text, "sender_mode": sender_mode, "Q_A_min": float(a_min)}
        )
    if away is not None:
        result["D_removal"] = whole["D_removal"]
        result["removal_lost_at"] = None if away[1] is None else away[1]["t"]
        result["removal_loss_cause"] = None if away[1] is None else away[1]["cause"]

    for k, row in enumerate(record):  # every quantity on its own
        alone, _ = reading([k])
        row.update(D=alone["D_peak"], D_activity=alone["D_activity"], t_peak=alone["t_peak"])
        if away is not None:
            row["D_removal"] = alone["D_removal"]
    read_groups = {
        label: reading([names.index(name) for name in members]) for label, members in groups.items()
    }
    if groups:
        result["groups"] = {label: numbers for label, (numbers, _) in read_groups.items()}

    def table(path: Path, columns: dict[str, np.ndarray]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["time", *columns])
            for k, t in enumerate(times):
                writer.writerow([repr(float(t)), *(repr(float(values[k])) for values in columns.values())])

    written: list[Path] = []
    if out is not None:
        target = Path(out)
        target.mkdir(parents=True, exist_ok=True)
        ordered = {"D": series["D"], "O": 1.0 / (1.0 + series["D"])}
        ordered.update({key: values for key, values in series.items() if key != "D"})
        table(target / "orthogonality.csv", ordered)
        with (target / "deltas.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record[0]))
            writer.writeheader()
            writer.writerows(record)
        (target / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        written = [target / "orthogonality.csv", target / "deltas.csv", target / "result.json"]
        if groups:
            with (target / "groups.csv").open("w", encoding="utf-8", newline="") as handle:
                rows = [{"group": label, **numbers} for label, (numbers, _) in read_groups.items()]
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            table(
                target / "groups_series.csv",
                {f"D[{label}]": found["D"] for label, (_, found) in read_groups.items()},
            )
            written += [target / "groups.csv", target / "groups_series.csv"]
    return result, written


__all__ = [
    "BITS_TOL",
    "STAGE_TOL",
    "Function",
    "MeasureError",
    "Protected",
    "information_bounds",
    "measure",
    "parse_protected",
    "read_series",
    "receiver_loss",
    "tolerances",
]
