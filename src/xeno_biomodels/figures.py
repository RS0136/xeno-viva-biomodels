"""The figures of the study, from what ``select``, ``verify`` and ``worlds`` left under ``runs/``.

    selection.png   the agreement of the five engines on every model of the corpus, and where the receivers sit
    verify.png      each receiver run here by viva-biomodels: its time course, and how far the engines are apart
    worlds.png      each receiver with the sender: what the sender does, and what reaches the receiver
    summary.png     the interference against the specificity of the sender's transporter, for every receiver
    layers.png      the same interference layer by layer, for the quantities common to all and those of each
    community.png          the pairs poured into one vessel: the density of every organism over a long time
    community_matrix.png   who moves whom there: D and D_removal for every ordered pair of organisms
    community_layers.png   where it lands: every xeno against every SBML organism, layer by layer

The figures of the studies (``study_*.png``) are made by ``xeno_biomodels.studies.figures``.

Colour does one job per figure. The three receivers are three categories and take the first three hues of the
palette in a fixed order; the activities of the sender are levels of one knob and take one hue from light to dark.
Both sets were checked for separation under colour-vision deficiency before use.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from xeno_biomodels import paths
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.world import BASELINE, load_plan

SURFACE, INK, SECOND, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CATEGORY = ["#2a78d6", "#eb6834", "#1baf7a"]  # the receivers, in the order of the plan
LEVELS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # the activities of the sender, low to high
OUT = paths.RUNS / "figures"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": SECOND,
            "axes.titlecolor": INK,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": SECOND,
            "ytick.labelcolor": SECOND,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "lines.linewidth": 2.0,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "figure.dpi": 110,
            "savefig.dpi": 170,
        }
    )


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(plan: dict, biomodel: str) -> str:
    receiver = plan["receivers"][biomodel]
    return f"{receiver['organism']}, {receiver['what']}"


def _short(biomodel: str) -> str:
    return f"BIOMD{int(biomodel.replace('BIOMD', ''))}"


def _dot(axis, x, y, color, size=7.0, ring=SURFACE, width=1.6):
    """A marker with a ring of the surface around it, so that it stays legible where it crosses a line."""
    axis.plot([x], [y], marker="o", ms=size, mfc=color, mec=ring, mew=width, ls="none", zorder=5)


def selection(plan: dict, out: Path) -> Path:
    scores = _rows(paths.RUNS / "selection" / "corpus_scores.csv")
    five = sorted(float(row["max_variable_nrmse"]) for row in scores if int(row["n_engines"]) == 5)
    limit = float(plan["selection"]["variable_nrmse"])
    plateau = json.loads((paths.RUNS / "selection" / "summary.json").read_text(encoding="utf-8"))
    best = {}
    for row in scores:
        if row["biomodel_id"] in plan["receivers"] and int(row["n_engines"]) == 5:
            value = float(row["max_variable_nrmse"])
            best[row["biomodel_id"]] = min(value, best.get(row["biomodel_id"], np.inf))

    figure, axis = plt.subplots(figsize=(7.6, 4.3))
    rank = np.arange(1, len(five) + 1)
    inside = int(np.searchsorted(five, limit, side="right"))
    axis.axvspan(0.5, inside + 0.5, color="#f0efec", lw=0, zorder=0)
    axis.plot(rank, np.maximum(five, 1e-8), color=MUTED, lw=1.6, zorder=2)
    axis.axhline(limit, color=AXIS, lw=0.8, zorder=1)
    axis.set_yscale("log")
    axis.set_ylim(5e-8, 3.0)
    axis.set_xlim(0.5, len(five) + 0.5)
    axis.set_xlabel(f"time courses that all five engines ran ({len(five)}), in order of agreement")
    axis.set_ylabel("largest nRMSE of any variable\nbetween any two engines")
    axis.text(
        inside * 0.5,
        0.35,
        "the plateau: every variable within 1e-5\non all five engines",
        ha="center",
        va="center",
        color=SECOND,
        fontsize=8.5,
    )
    axis.text(len(five), limit * 1.4, "1e-5", ha="right", va="bottom", color=MUTED, fontsize=8)
    axis.text(
        len(five) * 0.86,
        4e-7,
        "float32 and 100 stored\npoints resolve nothing\nbelow about 1e-7",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.5,
    )
    for biomodel in plan["receivers"]:
        color = CATEGORY[list(plan["receivers"]).index(biomodel)]
        at = int(np.searchsorted(five, best[biomodel])) + 1
        axis.plot(
            [at],
            [best[biomodel]],
            marker="o",
            ms=7.5,
            mfc=color,
            mec=SURFACE,
            mew=1.6,
            ls="none",
            zorder=5,
            label=f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}  ({best[biomodel]:.1e})",
        )
    axis.legend(
        loc="center left",
        bbox_to_anchor=(0.03, 0.62),
        title="the receivers of the plan",
        title_fontsize=8.5,
        alignment="left",
    )
    axis.set_title(
        f"Agreement of five ODE engines on the viva-biomodels corpus ({plateau['plateau_models']} models on the plateau)"
    )
    figure.tight_layout()
    path = out / "selection.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def verify(plan: dict, out: Path) -> Path:
    receivers = list(plan["receivers"])
    figure, axes = plt.subplots(
        2, len(receivers), figsize=(11.0, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0]}
    )
    pairs = [("copasi", "tellurium"), ("copasi", "simbio"), ("simbio", "tellurium")]
    limit = float(plan["selection"]["variable_nrmse"])
    for column, biomodel in enumerate(receivers):
        series = {}
        for engine in ("copasi", "simbio", "tellurium"):
            series[engine] = read_series(paths.RUNS / "verify" / biomodel / f"{engine}.csv")
        times, values = series["tellurium"]
        model = [name for name in series["simbio"][1] if name in values and name in series["copasi"][1]]
        top = axes[0, column]
        for name in model:
            largest = float(np.max(np.abs(values[name])))
            if largest > 0.0 and float(np.ptp(values[name])) > 1e-6 * largest:
                top.plot(times, values[name] / largest, color=CATEGORY[column], lw=1.1, alpha=0.75)
        top.set_ylim(-0.03, 1.05)
        top.set_title(f"{_short(biomodel)}\n{_label(plan, biomodel)}", fontsize=9)
        if column == 0:
            top.set_ylabel("each species over\nits largest value")
        low = axes[1, column]
        gap = np.zeros_like(
            times
        )  # over the three pairs of engines and the variables that both of a pair report
        for first, second in pairs:
            a, b = series[first][1], series[second][1]
            for name in (name for name in a if name in b):
                size = max(float(np.max(np.abs(a[name]))), float(np.max(np.abs(b[name]))))
                if size > 0.0:
                    gap = np.maximum(gap, np.abs(a[name] - b[name]) / size)
        low.plot(times, np.maximum(gap, 1e-12), color=CATEGORY[column], lw=1.6)
        low.axhline(limit, color=AXIS, lw=0.8)
        low.set_yscale("log")
        low.set_ylim(1e-9, 1e-3)
        low.set_xlabel("time (min)")
        low.text(times[-1], limit * 1.5, "1e-5", ha="right", va="bottom", color=MUTED, fontsize=8)
        if column == 0:
            low.set_ylabel(
                "largest gap of any variable\nbetween any two engines at\nthat time, over its size"
            )
    figure.suptitle(
        "The receivers run here through the compare-simulators composite of viva-biomodels (COPASI, Tellurium, simbio)",
        x=0.01,
        ha="left",
        fontsize=10,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path = out / "verify.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _activities(directory: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """D of every admissible activity of a world, by name."""
    times, columns = read_series(directory / "measure" / "orthogonality.csv")
    return times, {name[2:-1]: values for name, values in columns.items() if name.startswith("D[")}


def worlds(plan: dict, out: Path, designs: tuple[str, ...] = ("xeno", "leak-1e-2", "leak-1e-1")) -> Path:
    receivers = list(plan["receivers"])
    steps = [f"u={value:g}" for value in plan["activities"]["steps"]]
    figure, axes = plt.subplots(
        len(receivers), 1 + len(designs), figsize=(12.4, 2.55 * len(receivers) + 0.9), sharex=True
    )
    shown: dict[str, object] = {}
    for row, biomodel in enumerate(receivers):
        root = paths.RUNS / "worlds" / biomodel
        _, admissible = _activities(root / "xeno")
        ramp = {name: LEVELS[n] for n, name in enumerate(name for name in steps if name in admissible)}
        # what the sender does, in the design that shares nothing
        axis = axes[row, 0]
        times, base = read_series(root / "xeno" / BASELINE / "series.csv")
        hours = times / 60.0
        for name in steps:
            if not (root / "xeno" / name / "series.csv").exists():
                continue
            _, run = read_series(root / "xeno" / name / "series.csv")
            color = ramp.get(name, AXIS)
            (line,) = axis.plot(
                hours, run["A.density"] / base["A.density"][0], color=color, lw=2.0 if name in ramp else 1.4
            )
            shown.setdefault(
                name.replace("u=", "u = ")
                + ("" if name in ramp else "  (the sender washes out: not in U_A)"),
                line,
            )
        (line,) = axis.plot(hours, base["A.density"] / base["A.density"][0], color=MUTED, lw=1.4)
        shown.setdefault("baseline, u = 1", line)
        axis.set_ylim(0.0, 1.25)
        axis.set_ylabel(
            f"{_short(biomodel)}\n{plan['receivers'][biomodel]['organism']}", color=INK, fontsize=9
        )
        if row == 0:
            axis.set_title("Sender A\nits density over its density at rest", fontsize=8.8)
        # what reaches the receiver, design by design
        for column, design in enumerate(designs, start=1):
            axis = axes[row, column]
            times, each = _activities(root / design)
            for name, values in each.items():
                axis.plot(times / 60.0, values, color=ramp.get(name, AXIS))
            axis.axhline(1.0, color=AXIS, lw=0.8)
            axis.set_yscale("symlog", linthresh=1e-3, linscale=0.6)
            axis.set_ylim(-2e-4, 60.0)
            axis.set_yticks([0.0, 1e-2, 1e-1, 1.0, 10.0])
            axis.set_yticklabels(["0", "0.01", "0.1", "1", "10"] if column == 1 else [])
            leak = float(plan["designs"][design])
            if row == 0:
                what = "A shares no species with B" if leak == 0.0 else f"A takes B's nutrient at {leak:g}"
                axis.set_title(f"Receiver B, design {design}\n{what}", fontsize=8.8)
            if leak == 0.0:
                axis.text(
                    0.5,
                    0.2,
                    "D(t) = 0 at every stored time,\nfor every activity",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color=SECOND,
                    fontsize=8.5,
                )
                axis.text(
                    0.98,
                    1.25,
                    "D = 1: the tolerance",
                    transform=axis.get_yaxis_transform(),
                    ha="right",
                    va="bottom",
                    color=MUTED,
                    fontsize=8,
                )
            if column == 1:
                axis.set_ylabel("largest change of a protected\nquantity of B over its tolerance")
    for axis in axes[-1]:
        axis.set_xlabel("time after the step (h)")
    figure.legend(
        shown.values(),
        shown.keys(),
        loc="upper left",
        bbox_to_anchor=(0.005, 0.962),
        ncols=len(shown),
        handlelength=1.6,
        columnspacing=1.4,
    )
    figure.suptitle(
        "One step of the activity u of the sender at time zero, in the same vessel under the same supply",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.935))
    path = out / "worlds.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def summary(plan: dict, out: Path) -> Path:
    rows = [row for row in _rows(paths.RUNS / "worlds" / "summary.csv") if row.get("status") == "ok"]
    receivers, designs = list(plan["receivers"]), list(plan["designs"])
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)
    panels = (
        ("D", "D: the largest interference of any admissible step of the activity"),
        ("D_removal", "D_removal: the sender taken away at time zero"),
    )
    for axis, (key, title) in zip(axes, panels, strict=True):
        ends = []
        for n, biomodel in enumerate(receivers):
            mine = {row["design"]: row for row in rows if row["biomodel"] == biomodel}
            xs = [
                i
                for i, design in enumerate(designs)
                if design in mine and mine[design][key] not in ("", "None")
            ]
            ys = [float(mine[designs[i]][key]) for i in xs]
            axis.plot(
                xs,
                ys,
                color=CATEGORY[n],
                label=f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}",
            )
            for x, y in zip(xs, ys, strict=True):
                flagged = key == "D" and (mine[designs[x]]["losses"] or mine[designs[x]]["not_followed"])
                if flagged:
                    _dot(axis, x, y, SURFACE, size=9.5, ring=CATEGORY[n], width=2.0)
                else:
                    _dot(axis, x, y, CATEGORY[n])
            ends.append((ys[-1], n, biomodel))
        for limit, text in (
            (1.0, "D = 1: tolerable below"),
            (float(plan["measure"]["stage_tol"]), "D = 1e-6: unmoved below"),
        ):
            axis.axhline(limit, color=AXIS, lw=0.8)
            axis.text(len(designs) - 0.62, limit / 1.6, text, ha="right", va="top", color=MUTED, fontsize=8)
        axis.set_yscale("symlog", linthresh=1e-7, linscale=0.5)
        axis.set_ylim(-3e-8, 400.0)
        axis.set_yticks([0.0, 1e-6, 1e-4, 1e-2, 1.0, 100.0])
        axis.set_yticklabels(["0", "1e-6", "1e-4", "0.01", "1", "100"])
        axis.set_xticks(range(len(designs)))
        axis.set_xticklabels(
            [
                f"{float(plan['designs'][design]):g}"
                + ("\n(xeno)" if float(plan["designs"][design]) == 0.0 else "")
                for design in designs
            ]
        )
        axis.set_xlim(-0.35, len(designs) - 0.55)
        axis.set_title(title, fontsize=9)
    axes[0].set_ylabel("interference, in tolerances")
    axes[1].legend(loc="center right", bbox_to_anchor=(1.0, 0.42), handlelength=1.6)
    axes[0].text(
        0.33,
        0.42,
        "open marker: in one of the steps the receiver was lost,\nor its SBML model could not be integrated to the end",
        transform=axes[0].transAxes,
        ha="left",
        va="center",
        color=MUTED,
        fontsize=7.8,
    )
    figure.supxlabel(
        "specificity of the transporter of A for the nutrient of B, relative to its own substrate",
        fontsize=9,
        color=SECOND,
    )
    figure.suptitle(
        "An organism that shares no species with the SBML model does not move it; one that shares its nutrient does",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path = out / "summary.png"
    figure.savefig(path)
    plt.close(figure)
    return path


SHADES = {"within": LEVELS[0], "beyond": LEVELS[1], "lost": LEVELS[3]}


def _tile(axis, x: float, y: float, value: float, lost: bool = False, marked: bool = False) -> None:
    """A number of the measure on the shade of what it means: nothing for an exact zero, then within the
    tolerance, beyond it, and a receiver that was lost. ``marked`` rings it."""
    from matplotlib.patches import FancyBboxPatch

    kind = "lost" if lost else "beyond" if value > 1.0 else "within" if value > 0.0 else None
    if kind:
        box = "round,pad=0,rounding_size=0.08"
        axis.add_patch(
            FancyBboxPatch((x - 0.44, y - 0.4), 0.88, 0.8, boxstyle=box, fc=SHADES[kind], ec="none")
        )
    text = "0" if value == 0.0 else f"{value:.2g}" if value < 9.95 else f"{value:.0f}"
    ink = "#ffffff" if kind in ("beyond", "lost") else INK
    axis.text(x, y + (0.1 if lost else 0.0), text, ha="center", va="center", fontsize=8.5, color=ink)
    if lost:
        axis.text(x, y - 0.2, "lost", ha="center", va="center", fontsize=6.8, color="#ffffff")
    if marked:
        box = "round,pad=0,rounding_size=0.1"
        axis.add_patch(
            FancyBboxPatch((x - 0.47, y - 0.43), 0.94, 0.86, boxstyle=box, fc="none", ec=INK, lw=1.3)
        )


def _shade_legend(figure, pair: bool, anchor=(0.01, 0.005)) -> None:
    from matplotlib.patches import Patch

    legend = [
        Patch(fc=SURFACE, ec=AXIS, label="0: every number of the receiver unchanged"),
        Patch(fc=SHADES["within"], label="moved, within its tolerance (D ≤ 1)"),
        Patch(fc=SHADES["beyond"], label="moved beyond it (D > 1)"),
        Patch(fc=SHADES["lost"], label="receiver lost in one of the runs"),
    ]
    if pair:
        legend.append(Patch(fc="none", ec=INK, lw=1.3, label="the two are a pair"))
    figure.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=anchor,
        ncols=len(legend),
        handlelength=1.4,
        columnspacing=1.4,
    )


def _bare(axis, columns: list[str], lines: list[str]) -> None:
    """The frame of a matrix: names along the top and down the side, and nothing else."""
    axis.set_xlim(-0.5, len(columns) - 0.5)
    axis.set_ylim(len(lines) - 0.5, -0.5)
    axis.set_aspect(0.55)
    axis.set_anchor("N")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(lines)))
    axis.set_yticklabels(lines)
    axis.grid(False)
    for side in ("left", "bottom", "top", "right"):
        axis.spines[side].set_visible(False)


def _group_lines(rows: list[dict]) -> list[tuple[str, str]]:
    """(layer, kind) of the groups that are one kind at one layer, in the order of the layers; first the whole."""
    from xeno_biomodels.layers import KINDS, LAYERS

    have = {(row["layer"], row["kind"]) for row in rows}
    return [(layer, kind) for layer in LAYERS for kind in KINDS if (layer, kind) in have]


def layers(plan: dict, out: Path) -> Path:
    """Each receiver with its sender, layer by layer: D of the common and of the specific quantities."""
    rows = _rows(paths.RUNS / "worlds" / "layers.csv")
    whole = {
        (row["biomodel"], row["design"]): row
        for row in _rows(paths.RUNS / "worlds" / "summary.csv")
        if row.get("status") == "ok"
    }
    receivers, designs = list(plan["receivers"]), list(plan["designs"])
    columns = [
        f"{float(plan['designs'][design]):g}" + ("\n(xeno)" if float(plan["designs"][design]) == 0.0 else "")
        for design in designs
    ]
    lines = _group_lines(rows)
    figure, axes = plt.subplots(
        1, len(receivers), figsize=(4.4 * len(receivers), 0.42 * (len(lines) + 1) + 2.2)
    )
    for column, (axis, biomodel) in enumerate(zip(axes, receivers, strict=True)):
        mine = {
            (row["layer"], row["kind"], row["design"]): row for row in rows if row["biomodel"] == biomodel
        }
        for j, design in enumerate(designs):
            if (biomodel, design) not in whole:
                continue
            lost = bool(whole[(biomodel, design)]["losses"])
            _tile(axis, j, 0, float(whole[(biomodel, design)]["D"]), lost)
            for i, (layer, kind) in enumerate(lines, start=1):
                if (layer, kind, design) in mine:
                    _tile(axis, j, i, float(mine[(layer, kind, design)]["D"]), lost)
        _bare(
            axis,
            columns,
            ["all layers"] + [f"{layer}, {kind}" for layer, kind in lines]
            if column == 0
            else [""] * (len(lines) + 1),
        )
        axis.axhline(0.5, color=AXIS, lw=0.8)
        axis.set_title(f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}", fontsize=9, pad=30)
    axes[len(receivers) // 2].set_xlabel(
        "specificity of the transporter of A for the nutrient of B, relative to its own substrate"
    )
    _shade_legend(figure, pair=False)
    figure.suptitle(
        "The same interference, layer by layer: quantities common to every organism, and those of each",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = out / "layers.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def community_layers(plan: dict, out: Path) -> Path | None:
    """In the common vessel, every xeno against every SBML organism, layer by layer. A design in which no xeno
    moves any SBML organism has no panel; None when that is every design."""
    rows = [
        row
        for row in _rows(paths.RUNS / "community" / "layers.csv")
        if row["sender_kind"] == "xeno" and row["receiver_kind"] == "sbml"
    ]
    names = plan["community"]["names"]
    receivers = [pair["organism"] for pair in names.values()]
    xenos = [pair["xeno"] for pair in names.values()]
    lines = _group_lines(rows)
    whole = {
        (row["layer"], row["design"], row["sender"], row["receiver"]): row
        for row in rows
        if row["kind"] == "all"
    }
    moving = [
        design
        for design in plan["community"]["designs"]
        if any(float(row["D"]) > 0.0 for row in rows if row["design"] == design)
    ]
    still = [design for design in plan["community"]["designs"] if design not in moving]
    if not moving:
        return None
    pairs = [(sender, receiver) for receiver in receivers for sender in xenos]
    figure, axes = plt.subplots(
        len(moving),
        1,
        figsize=(0.95 * len(pairs) + 3.2, (0.5 * len(lines) + 1.9) * len(moving) + 1.3),
        squeeze=False,
    )
    for axis, design in zip(axes[:, 0], moving, strict=True):
        mine = {
            (row["layer"], row["kind"], row["sender"], row["receiver"]): row
            for row in rows
            if row["design"] == design
        }
        for j, (sender, receiver) in enumerate(pairs):
            for i, (layer, kind) in enumerate(lines):
                found = mine.get((layer, kind, sender, receiver))
                if found:
                    _tile(
                        axis,
                        j,
                        i,
                        float(found["D"]),
                        bool(found["losses"]),
                        marked=found["same_pair"] == "True" and i == 0,
                    )
        _bare(
            axis,
            [f"{sender} to {receiver}" for sender, receiver in pairs],
            [f"{layer}, {kind}" for layer, kind in lines],
        )
        for edge in range(len(xenos), len(pairs), len(xenos)):
            axis.axvline(edge - 0.5, color=AXIS, lw=0.8)
        axis.set_title(
            f"{design}: D of every xeno against every SBML organism, at each layer", fontsize=9, pad=24
        )
    note = f"In {' and '.join(still)} every one of these numbers is 0. " if still else ""
    figure.text(
        0.01,
        0.075,
        note + "A ring on the first line marks a xeno against its own partner.",
        fontsize=8.5,
        color=SECOND,
    )
    _shade_legend(figure, pair=False)
    figure.suptitle(
        "Where the interference lands in the common vessel",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.11, 1, 0.95))
    path = out / "community_layers.png"
    figure.savefig(path)
    plt.close(figure)
    del whole
    return path


def _pair_colors(plan: dict) -> dict[str, str]:
    """An SBML organism and its xeno share the hue of their pair, in the order of the plan."""
    colors = {}
    for n, names in enumerate(plan["community"]["names"].values()):
        colors[names["organism"]] = colors[names["xeno"]] = CATEGORY[n]
    return colors


def community(plan: dict, out: Path) -> Path:
    """The fate of the poured community, design by design: the density of every organism over time."""
    root, designs = paths.RUNS / "community", list(plan["community"]["designs"])
    names = plan["community"]["names"]
    colors = _pair_colors(plan)
    figure, axes = plt.subplots(1, len(designs), figsize=(12.4, 4.6), sharey=True)
    handles = {}
    for axis, design in zip(axes, designs, strict=True):
        times, columns = read_series(root / design / "fate" / "series.csv")
        hours = times / 60.0
        ends: dict[str, float] = {}
        for n, pair in enumerate(names.values()):
            sbml, xeno = pair["organism"], pair["xeno"]
            (line,) = axis.plot(
                hours, columns[f"{sbml}.density"] / columns[f"{sbml}.density"][0], color=colors[sbml]
            )
            handles.setdefault(f"{sbml}: {plan['receivers'][list(names)[n]]['organism']}", line)
            # the xenos often lie on one another: dashes that take turns keep all three in view
            axis.plot(
                hours,
                columns[f"{xeno}.density"] / columns[f"{xeno}.density"][0],
                color=colors[xeno],
                lw=2.6,
                ls=(6.0 * n, (6.0, 12.0)),
            )
            handles.setdefault(
                f"{xeno}: its xeno", Line2D([0], [0], color=colors[xeno], lw=2.6, ls=(0, (2.2, 1.6)))
            )
            for name in (sbml, xeno):
                ends[name] = float(columns[f"{name}.density"][-1] / columns[f"{name}.density"][0])
        groups: list[list[str]] = []  # organisms that end close together share one label
        for name in sorted(ends, key=ends.get):
            if groups and np.log(ends[name] / ends[groups[-1][0]]) < 0.16:
                groups[-1].append(name)
            else:
                groups.append([name])
        for group in groups:
            level = float(np.exp(np.mean([np.log(ends[name]) for name in group])))
            label = ", ".join(sorted(group, key=list(ends).index))
            axis.annotate(
                label,
                (hours[-1], level),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        settings = plan["community"]["designs"][design]
        what = (
            "each xeno on a substrate of its own"
            if settings["substrate"] == "own"
            else "the xenos on one common substrate"
        )
        if float(settings["cross"]) > 0.0:
            what += f",\neach taking the other pairs' nutrients at {float(settings['cross']):g}"
        axis.set_title(f"{design}\n{what}", fontsize=8.8)
        axis.set_yscale("log")
        axis.set_ylim(0.03, 8.0)
        axis.set_yticks([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
        axis.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "5"])
        axis.set_xlim(0.0, hours[-1] * 1.2)
        axis.set_xticks(
            [tick for tick in axis.get_xticks() if 0.0 <= tick <= hours[-1]]
        )  # the room on the right is for the labels
        axis.set_xlabel("time after the pouring (h)")
    axes[0].set_ylabel("density over the density at the pouring")
    figure.legend(
        handles.values(),
        handles.keys(),
        loc="upper left",
        bbox_to_anchor=(0.005, 0.94),
        ncols=len(handles),
        handlelength=2.6,
        columnspacing=1.3,
    )
    figure.suptitle(
        "Three pairs poured into one vessel, with nothing stepped: who stays",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    path = out / "community.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def community_matrix(plan: dict, out: Path) -> Path:
    """Who moves whom in the community: D from the steps of every xeno, and D_removal for every organism."""
    rows = _rows(paths.RUNS / "community" / "summary.csv")
    designs = list(plan["community"]["designs"])
    names = plan["community"]["names"]
    order = [pair["organism"] for pair in names.values()] + [pair["xeno"] for pair in names.values()]
    xenos = [pair["xeno"] for pair in names.values()]

    def cell(axis, x, y, row, key):
        """One sender and one receiver."""
        if row is None or row.get("status") != "ok" or row[key] in ("", "None"):
            axis.text(
                x, y, "n/a" if row is not None else "", ha="center", va="center", color=MUTED, fontsize=8
            )
            return
        lost = bool(row["losses"]) if key == "D" else row["removal_lost_at"] not in ("", "None")
        _tile(axis, x, y, float(row[key]), lost, marked=row["same_pair"] == "True")

    figure, axes = plt.subplots(
        len(designs),
        2,
        figsize=(11.6, 2.05 * len(designs) + 2.6),
        gridspec_kw={"height_ratios": [1.0] * len(designs), "width_ratios": [1.0, 1.0]},
    )
    for line, design in enumerate(designs):
        mine = {(row["sender"], row["receiver"]): row for row in rows if row["design"] == design}
        for column, (senders, key) in enumerate(((xenos, "D"), (order, "D_removal"))):
            axis = axes[line, column]
            for i, sender in enumerate(senders):
                for j, receiver in enumerate(order):
                    if sender == receiver:
                        axis.text(j, i, "·", ha="center", va="center", color=AXIS, fontsize=10)
                    else:
                        cell(axis, j, i, mine.get((sender, receiver)), key)
            _bare(axis, order, senders)
            if line == 0:
                title = (
                    "D: the activity of the sender stepped"
                    if key == "D"
                    else "D_removal: the sender taken away"
                )
                axis.set_title(f"{title}\n(rows: sender, columns: receiver)", fontsize=9, pad=22)
            if column == 0:
                axis.text(
                    -0.16,
                    0.5,
                    design,
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color=INK,
                )
    _shade_legend(figure, pair=True)
    figure.suptitle(
        "Who moves whom in the common vessel, in tolerances of the receiver",
        x=0.01,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0.07, 0.05, 1, 0.95))
    path = out / "community_matrix.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def write(plan_path: str | Path | None = None, out: Path = OUT) -> list[Path]:
    """Every figure, and the report that sets them beside the numbers."""
    from xeno_biomodels import report

    plan = load_plan(plan_path)
    out.mkdir(parents=True, exist_ok=True)
    style()
    made = [selection(plan, out), verify(plan, out), worlds(plan, out), summary(plan, out)]
    made += [layers(plan, out), report.write(plan)]
    if (paths.RUNS / "community" / "summary.csv").exists():
        from xeno_biomodels import community_report

        made += [community(plan, out), community_matrix(plan, out)]
        landed = community_layers(plan, out)
        made += ([landed] if landed else []) + [community_report.write(plan)]
    if (paths.RUNS / "studies").exists():
        from xeno_biomodels.studies import report as studies_report

        made += studies_report.write(plan)
    return made
