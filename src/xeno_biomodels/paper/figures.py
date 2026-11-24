"""The figures of the manuscript and of its supplementary material, one function each.

Every function takes the plan, the directory of the manuscript and the name of the file, reads tables under
``runs/`` and returns the PDF that it wrote (a PNG lies beside it)."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.ticker import NullLocator

from xeno_biomodels import paths
from xeno_biomodels.layers import LAYERS
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.paper.style import (
    AXIS,
    CATEGORY,
    CELL,
    CELL_NAME,
    CELL_RUNS,
    CHEMICAL_NAME,
    CONTROL,
    GRID,
    INK,
    LAYER_COLOR,
    LEVELS,
    MUTED,
    SECOND,
    SHADES,
    WIDTH,
    bare_axis,
    italic,
    line_labels,
    panel_mark,
    quantity_name,
    receiver_name,
    save_figure,
    shade_legend,
    show_number,
    species_name,
    summary_rows,
    tile,
)
from xeno_biomodels.studies.common import LINES, STUDIES, number, read_rows
from xeno_biomodels.world import BASELINE

# ---- the figures of the text and of the supplement ----------------------------------------------------------------


def fig_groups(plan: dict, out: Path, name: str = "fig3") -> Path:
    """Each receiver with each design of the sender: D of the whole, and of each group of protected quantities
    (layer and kind). The receivers of the plan by the leak of the sender; the growing cell by the share that the
    sender takes at rest."""
    blocks = []
    rows = summary_rows(paths.RUNS / "worlds" / "layers.csv")
    whole = {
        (row["biomodel"], row["design"]): row
        for row in summary_rows(paths.RUNS / "worlds" / "summary.csv")
        if row.get("status") == "ok"
    }
    designs = list(plan["designs"])
    columns = [
        "0\n(xeno)" if float(plan["designs"][design]) == 0.0 else f"{float(plan['designs'][design]):g}"
        for design in designs
    ]
    for biomodel in plan["receivers"]:
        blocks.append((biomodel, designs, columns, rows, whole))
    if (CELL_RUNS / "layers.csv").exists():
        cell_rows = read_rows(CELL_RUNS / "layers.csv")
        cell_whole = {
            (row["biomodel"], row["design"]): row
            for row in read_rows(CELL_RUNS / "worlds" / "summary.csv")
            if row.get("status") == "ok"
        }
        cell_designs = list(dict.fromkeys(row["design"] for row in cell_rows))
        shown = ["0\n(xeno)" if d == "xeno" else f"{100 * float(d.split('-')[1]):g} %" for d in cell_designs]
        blocks.append((CELL, cell_designs, shown, cell_rows, cell_whole))
    lines = [line for line in LINES if any((row["layer"], row["kind"]) == line for row in rows)]
    widths = [len(block[1]) for block in blocks]
    figure, axes = plt.subplots(1, len(blocks), figsize=(WIDTH, 3.3), gridspec_kw={"width_ratios": widths})
    for n, (axis, (biomodel, mine_designs, labels, table, summary)) in enumerate(
        zip(axes, blocks, strict=True)
    ):
        mine = {
            (row["layer"], row["kind"], row["design"]): row for row in table if row["biomodel"] == biomodel
        }
        for j, design in enumerate(mine_designs):
            if (biomodel, design) not in summary:
                continue
            lost = bool(summary[(biomodel, design)]["losses"])
            tile(axis, j, 0, float(summary[(biomodel, design)]["D"]), lost, size=5.8)
            for i, (layer, kind) in enumerate(lines, start=1):
                if (layer, kind, design) in mine and number(mine[(layer, kind, design)].get("D")) is not None:
                    tile(axis, j, i, float(mine[(layer, kind, design)]["D"]), lost, size=5.8)
        bare_axis(
            axis, labels, line_labels(lines, True) if n == 0 else [""] * (len(lines) + 1), aspect="auto"
        )
        axis.tick_params(axis="x", labelsize=6.2)
        axis.axhline(0.5, color=AXIS, lw=0.6)
        axis.set_title(f"({'abcd'[n]})  {receiver_name(plan, biomodel)}", fontsize=8.2, pad=24, loc="left")
    figure.text(
        0.46,
        0.085,
        "leak $\\varepsilon$ of the sender for the nutrient of the receiver",
        ha="center",
        fontsize=8,
    )
    figure.text(0.905, 0.085, "share at baseline", ha="center", fontsize=8)
    figure.subplots_adjust(left=0.15, right=0.998, top=0.83, bottom=0.14, wspace=0.05)
    shade_legend(figure)
    return save_figure(figure, name, out)


def fig_routes(plan: dict, out: Path, name: str = "figS4") -> Path:
    """The layers of a receiver with the same share taken of its nutrient, or of another species that it reads."""
    settings = plan["studies"]["channels"]
    rows = read_rows(STUDIES / "channels" / "layers.csv")
    names = list(settings["other"])
    lines = [line for line in LINES if any((row["layer"], row["kind"]) == line for row in rows)]
    shares = [float(share) for share in settings["shares"]]
    columns = ["xeno"] + [f"{channel}-{share:g}" for share in shares for channel in ("nutrient", "other")]
    known = {
        "GLCo": "glucose",
        "EtOH": "ethanol",
        "NH4": r"NH$_4^+$",
        "KG": "2-OG",
        "halfglucose": "glucose",
        "O2": r"O$_2$",
    }
    figure, axes = plt.subplots(1, len(names), figsize=(WIDTH, 3.45))
    for n, (axis, biomodel) in enumerate(zip(axes, names, strict=True)):
        other = settings["other"][biomodel]["species"]
        nutrient = plan["receivers"][biomodel]["nutrient"]
        mine = {
            (row["design"], row["layer"], row["kind"]): row for row in rows if row["biomodel"] == biomodel
        }
        labels = ["none"] + [
            f"{known.get(nutrient if channel == 'nutrient' else other, other)}\n{100 * share:g} %"
            for share in shares
            for channel in ("nutrient", "other")
        ]
        for j, column in enumerate(columns):
            for i, line in enumerate(lines):
                row = mine.get((column, *line))
                if row is not None and number(row.get("D")) is not None:
                    tile(axis, j, i, float(row["D"]), bool(row.get("losses")))
        bare_axis(axis, labels, line_labels(lines, False) if n == 0 else [""] * len(lines))
        for edge in (0.5, 2.5):
            axis.axvline(edge, color=AXIS, lw=0.6)
        axis.set_title(
            f"({'abc'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5, pad=24, loc="left"
        )
    axes[1].set_xlabel(
        "species of which the sender takes a share, and the share that it takes at steady state", labelpad=6
    )
    figure.subplots_adjust(left=0.15, right=0.998, top=0.84, bottom=0.11, wspace=0.04)
    shade_legend(figure, lost=any(row.get("losses") for row in rows))
    return save_figure(figure, name, out)


def _box(axis, x, y, w, h, fc="white", ec=INK, lw=0.8, radius=2.0, ls="-"):
    axis.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", fc=fc, ec=ec, lw=lw, ls=ls
        )
    )


def _arrow(axis, start, end, color=INK, lw=1.0, ls="-", bend=0.0, head=7.0):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=f"-|>,head_length={head / 10:.2f},head_width={head / 22:.2f}",
            mutation_scale=10,
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={bend}",
        )
    )


def _node(axis, x, y, label, radius=4.6, fc="white"):
    from matplotlib.patches import Ellipse

    # the canvas is not square: an ellipse in its units is a circle on the page
    axis.add_patch(Ellipse((x, y), 2 * radius, 2 * radius * axis._ratio, fc=fc, ec=INK, lw=0.8))
    axis.text(x, y, label, ha="center", va="center", fontsize=8)


def fig_scenario(
    plan: dict, out: Path, name: str = "fig1", biomodel: str | None = None, design: str = CONTROL
) -> Path:
    """The world of a run, the layers at which the receiver is read, the measure on one protected quantity of one
    world, and the same world read at every layer."""
    height = 4.7
    figure = plt.figure(figsize=(WIDTH, height))

    # (a) the world
    axis = figure.add_axes([0.005, 0.47, 0.475, 0.52])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")
    box = axis.get_position()
    axis._ratio = (box.width * WIDTH) / (box.height * height)  # units of y per unit of x, for circles
    _box(axis, 8, 6, 83, 80, fc="#f4f4f4", ec=AXIS, lw=1.0, radius=3)
    axis.text(11, 81.5, "chemostat vessel: the medium", fontsize=7, color=SECOND, va="top")
    _arrow(axis, (0.5, 62), (8, 62), lw=1.2)
    axis.text(0.3, 66, "feed,\nrate $k$", fontsize=6.5, va="bottom", ha="left", linespacing=1.0)
    _arrow(axis, (91, 20), (98.5, 20), lw=1.2)
    axis.text(99.7, 24, "outflow,\nrate $k$", fontsize=6.5, va="bottom", ha="right", linespacing=1.0)
    _node(axis, 26, 62, "$S$")
    _node(axis, 64, 62, "$X$")
    _node(axis, 83.5, 47, "$W$")
    axis.text(26, 71.5, "nutrient of B", fontsize=6.5, ha="center", va="bottom")
    axis.text(64, 71.5, "substrate of A", fontsize=6.5, ha="center", va="bottom")
    axis.text(82, 56.5, "product of A", fontsize=6.5, ha="center", va="bottom")
    _box(axis, 11, 12, 30, 28, fc="white")
    axis.text(26, 33, "receiver B", fontsize=7.5, fontweight="bold", ha="center", va="center")
    axis.text(26, 21.5, "SBML model,\nunchanged", fontsize=6.8, ha="center", va="center", linespacing=1.05)
    _box(axis, 50, 12, 27, 28, fc="white")
    axis.text(63.5, 33, "sender A", fontsize=7.5, fontweight="bold", ha="center", va="center")
    axis.text(
        63.5, 21.5, "three laws,\nactivity $u$", fontsize=6.8, ha="center", va="center", linespacing=1.05
    )
    _arrow(axis, (26, 53.4), (26, 40.8))
    _arrow(axis, (64, 53.4), (64, 40.8))
    _arrow(axis, (77.3, 31), (81.2, 39.5))
    _arrow(axis, (30.6, 57.6), (52, 40.8), color=CATEGORY[1], ls=(0, (3, 2)), lw=1.2)
    axis.text(
        47.5,
        52.5,
        "leak $\\varepsilon$\n(xeno: $\\varepsilon$ = 0)",
        fontsize=6.5,
        color=CATEGORY[1],
        ha="center",
        va="bottom",
        linespacing=1.0,
    )
    figure.text(0.005, 0.985, "(a)", fontsize=9, fontweight="bold", va="top")

    # (b) the layers of the receiver, and what it protects at each
    axis = figure.add_axes([0.5, 0.47, 0.495, 0.52])
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 100)
    axis.axis("off")
    common = {
        "medium": "level of the nutrient",
        "particle": "rate of ATP formation",
        "molecule": "sum of the metabolite pools",
        "complex": "uptake of the nutrient",
        "individual": "growth rate",
        "population": "density",
    }
    specific = {"molecule": "one metabolite pool", "complex": "one or two enzyme steps"}
    top, tall, gap = 82.0, 11.2, 2.3
    axis.text(26, 89, "layer", fontsize=7, color=SECOND, va="center", ha="center")
    axis.text(46, 89, "protected quantity", fontsize=7, color=SECOND, va="center", ha="left")
    for n, layer in enumerate(LAYERS):
        y = top - n * (tall + gap)
        outside = layer == "medium"
        _box(axis, 11, y - tall, 30, tall, fc="#f4f4f4" if outside else "white", ec=AXIS if outside else INK)
        axis.text(26, y - tall / 2, layer, fontsize=7.2, fontweight="bold", va="center", ha="center")
        if layer in specific:
            axis.text(46, y - tall / 2 + 2.6, f"common: {common[layer]}", fontsize=6.5, va="center")
            axis.text(
                46,
                y - tall / 2 - 2.9,
                f"specific: {specific[layer]}",
                fontsize=6.5,
                va="center",
                color=CATEGORY[1],
            )
        else:
            axis.text(46, y - tall / 2, f"common: {common[layer]}", fontsize=6.5, va="center")
    low = top - (len(LAYERS) - 1) * (tall + gap) - tall
    _arrow(axis, (5, top), (5, low), lw=1.0)
    figure.text(0.5, 0.985, "(b)", fontsize=9, fontweight="bold", va="top")

    # (c) the measure on one protected quantity of one world
    biomodel = biomodel or next(iter(plan["receivers"]))
    root = paths.RUNS / "worlds" / biomodel / design
    nutrient = plan["receivers"][biomodel]["nutrient"]
    column = f"medium.{nutrient}"
    times, base = read_series(root / BASELINE / "series.csv")
    delta = {row["column"]: row for row in summary_rows(root / "measure" / "deltas.csv")}[column]
    tolerance, rest = float(delta["delta"]), float(base[column][0])
    _, admissible = _activities(root)
    axis = figure.add_axes([0.085, 0.105, 0.36, 0.3])
    hours = times / 60.0
    axis.axhspan(rest - tolerance, rest + tolerance, color="#e9e9e9", lw=0)
    axis.plot(hours, base[column], color=INK, lw=0.9)
    worst = (0.0, 0.0, 0.0)
    ramp = dict(zip(sorted(admissible, key=lambda step: float(step[2:])), LEVELS, strict=False))
    for step, color in ramp.items():
        _, run = read_series(root / step / "series.csv")
        axis.plot(hours, run[column], color=color, lw=1.3)
        k = int(np.argmax(np.abs(run[column] - base[column])))
        if abs(run[column][k] - base[column][k]) > worst[0]:
            worst = (abs(run[column][k] - base[column][k]), hours[k], run[column][k])
    axis.annotate(
        "",
        (worst[1], worst[2]),
        (worst[1], rest),
        arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.7, "shrinkA": 0, "shrinkB": 0},
    )
    axis.text(
        worst[1] * 2.6,
        rest - 1.22 * tolerance,
        f"largest change: {worst[0] / tolerance:.2f} $\\delta$",
        fontsize=6.5,
        ha="left",
        va="center",
    )
    axis.text(
        0.185,
        rest + tolerance * 0.9,
        "tolerance $\\pm\\delta$",
        fontsize=6.5,
        ha="left",
        va="top",
        color=SECOND,
    )
    axis.set_xscale("log")
    axis.set_xlim(10.0 / 60.0, hours[-1])
    axis.set_xticks([1.0, 10.0, 100.0])
    axis.set_xticklabels(["1", "10", "100"])
    axis.set_ylim(rest - 1.42 * tolerance, rest + 1.2 * tolerance)
    axis.grid(False)
    axis.set_xlabel("time after the step (h)")
    axis.set_ylabel("nutrient in the medium (mM)")
    handles = [Line2D([0], [0], color=INK, lw=0.9, label="baseline, $u$ = 1")]
    handles += [
        Line2D([0], [0], color=color, lw=1.3, label=f"$u$ = {step[2:]}") for step, color in ramp.items()
    ]
    axis.legend(
        handles=handles,
        loc="upper right",
        fontsize=6.2,
        handlelength=1.3,
        labelspacing=0.15,
        borderaxespad=0.1,
    )
    figure.text(0.005, 0.445, "(c)", fontsize=9, fontweight="bold", va="top")

    # (d) the same world read at every layer and for each kind of quantity
    groups = {
        (row["layer"], row["kind"]): float(row["D"])
        for row in summary_rows(paths.RUNS / "worlds" / "layers.csv")
        if (row["biomodel"], row["design"]) == (biomodel, design)
    }
    lines = [line for line in LINES if line in groups]
    axis = figure.add_axes([0.66, 0.105, 0.325, 0.3])
    for i, line in enumerate(lines):
        value = groups[line]
        axis.barh([i], [value], height=0.68, color=SHADES["beyond" if value > 1.0 else "within"], lw=0)
        axis.text(value + 0.025, i, show_number(value), va="center", ha="left", fontsize=6.5)
    axis.axvline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
    axis.text(1.0, -0.72, "$D$ = 1", fontsize=6.5, ha="center", va="bottom", color=SECOND)
    axis.set_yticks(range(len(lines)))
    axis.set_yticklabels([f"{layer}, {kind}" for layer, kind in lines])
    axis.set_ylim(len(lines) - 0.45, -0.6)
    axis.set_xlim(0.0, 1.45)
    axis.set_xticks([0.0, 0.5, 1.0])
    axis.grid(False)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("$D$ of the layer (tolerances)")
    figure.text(0.5, 0.445, "(d)", fontsize=9, fontweight="bold", va="top")
    return save_figure(figure, name, out)


def _activities(directory: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """D of every admissible activity of a world along the run, by name."""
    times, columns = read_series(directory / "measure" / "orthogonality.csv")
    return times, {name[2:-1]: values for name, values in columns.items() if name.startswith("D[")}


def fig_candidates(plan: dict, out: Path, name: str = "figS1") -> Path:
    """Every species and every rate of a model read alone, and where the picked quantities stand among them."""
    directory = STUDIES / "settings"
    rows = [row for row in read_rows(directory / "candidates.csv") if row["design"] == CONTROL]
    summary = {
        (row["biomodel"], row["layer"]): row
        for row in read_rows(directory / "candidates_summary.csv")
        if row["design"] == CONTROL and row.get("status") == "ok"
    }
    names, levels, floor = list(plan["receivers"]), ("particle", "molecule", "complex"), 1e-5
    figure, axes = plt.subplots(1, len(names), figsize=(WIDTH, 2.85), sharey=True)
    rng = np.random.default_rng(0)
    for n, (axis, biomodel) in enumerate(zip(axes, names, strict=True)):
        for x, layer in enumerate(levels):
            values = np.array(
                [float(row["D"]) for row in rows if row["biomodel"] == biomodel and row["layer"] == layer]
            )
            axis.plot(
                x + rng.uniform(-0.17, 0.17, values.size),
                np.maximum(values, floor),
                ls="none",
                marker="o",
                ms=3.6,
                mfc=CATEGORY[n],
                mec="white",
                mew=0.5,
                alpha=0.85,
                clip_on=False,
            )
            found = summary.get((biomodel, layer))
            if found is None:
                continue
            for kind, marker, shift in (("common", "D", 0.3), ("specific", "s", 0.42)):
                d = number(found.get(f"D_picked_{kind}"))
                if d is not None:
                    axis.plot(
                        [x + shift],
                        [max(d, floor)],
                        ls="none",
                        marker=marker,
                        ms=5.5,
                        mfc="white" if kind == "common" else INK,
                        mec=INK,
                        mew=1.0,
                        zorder=6,
                    )
        axis.axhline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
        axis.set_yscale("log")
        axis.set_ylim(floor / 1.6, 20.0)
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels(levels)
        axis.set_xlim(-0.45, len(levels) - 0.3)
        axis.grid(axis="x", visible=False)
        axis.set_title(f"({'abc'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5)
    axes[0].set_ylabel("$D$ of one quantity read alone\n($\\varepsilon$ = 0.01; tolerances)")
    handles = [
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=4,
            mfc=MUTED,
            mec="white",
            label="every species and every reaction rate of the model",
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="D",
            ms=5.5,
            mfc="white",
            mec=INK,
            mew=1.0,
            label="common quantity of the layer",
        ),
        Line2D(
            [0], [0], ls="none", marker="s", ms=5.5, mfc=INK, mec=INK, label="specific quantity of the layer"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        handlelength=1.0,
        columnspacing=1.6,
    )
    figure.subplots_adjust(left=0.1, right=0.99, top=0.9, bottom=0.2, wspace=0.07)
    return save_figure(figure, name, out)


#: colour and marker of each assumed number that is varied (a palette that is safe under colour-vision deficiency)
VARIED = {
    "dilution rate": ("#0072B2", "o", "dilution rate"),
    "density of the receiver": ("#E69F00", "s", "density of the receiver"),
    "sender: substrate in the feed": ("#009E73", "^", "sender: substrate in the feed"),
    "sender: capacity V": ("#D55E00", "v", "sender: capacity $V$"),
    "sender: half-saturation K": ("#CC79A7", "D", "sender: half-saturation $K$"),
    "sender: yield": ("#56B4E9", "P", "sender: yield"),
    "sender: maintenance": ("#000000", "X", "sender: maintenance"),
}


def _largest_line(rows: list[dict]) -> dict[tuple[str, str], tuple[str, str]]:
    """(variant, biomodel) -> the layer and kind at which D of the world is largest."""
    best: dict[tuple[str, str], tuple[float, tuple[str, str]]] = {}
    for row in rows:
        if row["kind"] == "all" or row["layer"] == "all" or number(row.get("D")) is None:
            continue
        key = (row["variant"], row["biomodel"])
        value = float(row["D"])
        if key not in best or value > best[key][0]:
            best[key] = (value, (row["layer"], row["kind"]))
    return {key: line for key, (_, line) in best.items()}


def fig_sensitivity(plan: dict, out: Path, name: str = "figS5") -> Path:
    """D of a control with every assumed number of the world at a quarter, a half, twice and four times its value."""
    directory = STUDIES / "sensitivity"
    rows = [row for row in read_rows(directory / "worlds.csv") if row["design"] == CONTROL]
    lines = _largest_line(
        [row for row in read_rows(directory / "worlds_layers.csv") if row["design"] == CONTROL]
    )
    names = list(plan["receivers"])
    figure, axes = plt.subplots(1, len(names), figsize=(WIDTH, 3.35), sharey=True)
    for n, (axis, biomodel) in enumerate(zip(axes, names, strict=True)):
        mine = {
            row["variant"]: row for row in rows if row["biomodel"] == biomodel and row.get("status") == "ok"
        }
        reference = float(mine["reference"]["D"])
        home = lines[("reference", biomodel)]
        for parameter, (color, marker, _) in VARIED.items():
            points = [(1.0, reference, True)]
            for row in mine.values():
                if row["parameter"] == parameter:
                    points.append(
                        (float(row["factor"]), float(row["D"]), lines[(row["variant"], biomodel)] == home)
                    )
            points.sort()
            # a line joins neighbouring factors only: where a world does not exist, the line stops
            factors = [0.25, 0.5, 1.0, 2.0, 4.0]
            have = {x: y for x, y, _ in points}
            for a, b in pairwise(factors):
                if a in have and b in have:
                    axis.plot([a, b], [have[a], have[b]], color=color, lw=1.0, zorder=3)
            for x, y, same in points:
                if x == 1.0:
                    continue
                axis.plot(
                    [x],
                    [y],
                    ls="none",
                    marker=marker,
                    ms=5.0,
                    mfc=color if same else "white",
                    mec=color,
                    mew=1.0,
                    zorder=4,
                )
        axis.plot([1.0], [reference], ls="none", marker="o", ms=6.5, mfc="white", mec=INK, mew=1.2, zorder=5)
        axis.axhline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0])
        axis.set_xticklabels(["1/4", "1/2", "1", "2", "4"])
        axis.set_xlim(0.2, 5.0)
        axis.set_ylim(0.25, 13.0)
        axis.set_yticks([0.3, 1.0, 3.0, 10.0])
        axis.set_yticklabels(["0.3", "1", "3", "10"])
        axis.set_title(f"({'abc'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5)
    axes[1].set_xlabel("assumed number, over its reference value")
    axes[0].set_ylabel("$D$ of the sender with $\\varepsilon$ = 0.01 (tolerances)")
    handles = [
        Line2D([0], [0], color=color, lw=1.0, marker=marker, ms=5, mfc=color, mec=color, label=label)
        for color, marker, label in VARIED.values()
    ]
    handles += [
        Line2D(
            [0], [0], ls="none", marker="o", ms=6.5, mfc="white", mec=INK, mew=1.2, label="reference scenario"
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="s",
            ms=5,
            mfc="white",
            mec=MUTED,
            mew=1.0,
            label="open: $D$ is largest in another group than in the reference scenario",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        handlelength=1.8,
        columnspacing=1.4,
    )
    figure.subplots_adjust(left=0.08, right=0.99, top=0.92, bottom=0.315, wspace=0.07)
    return save_figure(figure, name, out)


def _largest(rows: list[dict], layer: str, kind: str) -> dict[float, float]:
    """Period -> the largest D of a layer under the schedules of one kind (0 for those that have no period)."""
    found: dict[float, float] = {}
    for row in rows:
        if row["layer"] == layer and row["schedule_kind"] == kind and number(row.get("D")) is not None:
            key = float(row["period"]) if number(row.get("period")) is not None else 0.0
            found[key] = max(found.get(key, 0.0), float(row["D"]))
    return found


# where a step is drawn whose added capacity has no bound: by its activity, and a little apart by its leak
BEYOND = {2.0: 5.0, 4.0: 10.0, 8.0: 20.0}
APART = {0.1: 0.84, 0.3: 1.19}


def fig_activity(plan: dict, out: Path, name: str = "fig2") -> Path:
    """D of every step against the capacity that the step adds to the population of the sender, beside the
    line that the closed form of the vessel gives."""
    panels = [(biomodel, STUDIES / "activity") for biomodel in plan["receivers"]]
    if (CELL_RUNS / "activity" / "activities.csv").exists():
        panels.append((CELL, CELL_RUNS / "activity"))
    figure, axes = plt.subplots(1, len(panels), figsize=(WIDTH, 2.75), sharey=True)
    for n, (axis, (biomodel, directory)) in enumerate(zip(axes, panels, strict=True)):
        rows = [
            row
            for row in read_rows(directory / "activities.csv")
            if row["biomodel"] == biomodel and float(row["leak"]) > 0.0 and number(row.get("D")) is not None
        ]
        gains = [row for row in read_rows(directory / "gain.csv") if row["biomodel"] == biomodel]
        gone = {
            row["design"]: number(row["activity_of_washout"])
            for row in read_rows(directory / "thresholds.csv")
            if row["biomodel"] == biomodel
        }
        leaks = sorted({float(row["leak"]) for row in rows})
        shade = dict(zip(leaks, LEVELS, strict=False))
        gain = float(gains[0]["gain_closed_form_of_the_whole"])  # of the weakest control
        x = np.array([1e-4, 2.8])
        axis.plot(x, gain * x, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
        axis.axhline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
        unbounded = False
        for row in rows:
            if row["admissible"] != "True" or not float(row["D"]) > 0.0:
                continue
            capacity = number(row.get("capacity_of_the_population"))
            leak, activity = float(row["leak"]), float(row["activity"])
            lost = number(row.get("receiver_lost_at")) is not None
            beyond = gone[row["design"]] is not None and activity > gone[row["design"]]
            if capacity is None:
                # the sender would grow on the nutrient alone: its population has no rest state at the level of
                # the file, and the capacity that the step adds has no bound. Drawn in a strip of its own.
                unbounded, beyond = True, True
                at = BEYOND[min(BEYOND, key=lambda u: abs(u - activity))] * APART.get(leak, 1.0)
            else:
                at = leak * abs(capacity - 1.0)
            axis.plot(
                [at],
                [float(row["D"])],
                ls="none",
                marker="v" if activity < 1.0 else "o",
                ms=4.6,
                mfc="white" if beyond else shade[leak],
                mec=shade[leak],
                mew=1.0,
                zorder=4,
            )
            if lost:
                axis.plot([at], [float(row["D"])], ls="none", marker="x", ms=5.6, mec=INK, mew=0.9, zorder=5)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(2e-4, 40.0)
        axis.set_ylim(3e-3, 40.0)
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_xticks([1e-3, 1e-2, 1e-1, 1e0])
        if unbounded:
            axis.axvspan(2.8, 40.0, color=GRID, lw=0, zorder=0)
            axis.set_xticks([1e-3, 1e-2, 1e-1, 1e0, 10.0])
            axis.set_xticklabels(["$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$10^{0}$", "$\\infty$"])
        axis.set_title(f"({'abcd'[n]})  {receiver_name(plan, biomodel)}", fontsize=8.2)
        axis.text(
            0.76,
            0.04,
            f"dashed: $D$ = {gain:.3g} $\\varepsilon\\,|a-1|$",
            fontsize=6.3,
            color=SECOND,
            ha="right",
            va="bottom",
            transform=axis.transAxes,
        )
    figure.text(
        0.535,
        0.215,
        "capacity added to the population of the sender, $\\varepsilon\\,|a(u)-1|$",
        ha="center",
        fontsize=8,
    )
    axes[0].set_ylabel("$D$ of the step (tolerances)")
    plan_leaks = sorted(float(leak) for leak in plan["designs"].values() if float(leak) > 0.0)
    handles = [
        Line2D([0], [0], ls="none", marker="o", ms=4.6, mfc=LEVELS[i], mec=LEVELS[i], label=f"{leak:g}")
        for i, leak in enumerate(plan_leaks)
    ]
    handles += [
        Line2D(
            [0], [0], ls="none", marker="o", ms=4.6, mfc=MUTED, mec=MUTED, label="step up ($u$ = 2, 4, 8)"
        ),
        Line2D([0], [0], ls="none", marker="v", ms=4.6, mfc=MUTED, mec=MUTED, label="step down ($u$ = 0.5)"),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=4.6,
            mfc="white",
            mec=MUTED,
            mew=1.0,
            label="no steady state with the receiver",
        ),
        Line2D(
            [0], [0], ls="none", marker="x", ms=6.5, mec=INK, mew=1.0, label="receiver lost within the run"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=4,
        handlelength=1.0,
        columnspacing=1.2,
        title="leak $\\varepsilon$ in (a)-(c); in (d) the three shades are shares of 0.1 %, 1 % and 10 %",
        title_fontsize=6.6,
    )
    figure.subplots_adjust(left=0.08, right=0.995, top=0.9, bottom=0.36, wspace=0.07)
    return save_figure(figure, name, out)


def _swing(plan: dict, biomodel: str) -> dict[float, tuple[float, float]]:
    """How far the nutrient moves in the admissible runs of each control, until a loss: (lowest, highest) over
    its level at the steady state, by the specificity of the control."""
    out = {}
    for design, leak in plan["designs"].items():
        directory = paths.RUNS / "worlds" / biomodel / design
        if not float(leak) > 0.0 or not (directory / "measure" / "result.json").exists():
            continue
        result = json.loads((directory / "measure" / "result.json").read_text(encoding="utf-8"))
        lost = {item["activity"]: item["t"] for item in result["losses"]}
        left_out = {item["activity"] for item in result["excluded_by_sender_function"]}
        times, base = read_series(directory / BASELINE / "protected.csv")
        column = next(key for key in base if key.startswith("medium."))
        low = high = 1.0
        for path in sorted(directory.glob("u=*/protected.csv")):
            step = path.parent.name
            if step in left_out:
                continue
            _, columns = read_series(path)
            read = times < lost[step] if step in lost else np.ones(times.size, dtype=bool)
            if not np.any(read):
                continue
            ratio = columns[column][read] / base[column][read]
            low, high = min(low, float(np.min(ratio))), max(high, float(np.max(ratio)))
        out[float(leak)] = (low, high)
    return out


def _apart(wanted: list[float], gap: float, top: float = 1.0) -> list[float]:
    """Heights for labels that are wanted at given heights (ascending), at least ``gap`` apart and not above
    ``top``."""
    out = list(wanted)
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1] + gap)
    if out and out[-1] > top:
        out[-1] = top
        for i in range(len(out) - 2, -1, -1):
            out[i] = min(out[i], out[i + 1] - gap)
    return out


def _dose_panel(axis, plan: dict, biomodel: str, dose: list[dict], limits: dict, swing: dict | None) -> None:
    """The dose-response of one model alone: every protected quantity inside it over its value at rest."""
    span = (0.2, 5.0)
    mine = [row for row in dose if row["biomodel"] == biomodel and span[0] <= float(row["ratio"]) <= span[1]]
    ends, lowest, highest = [], 1.0, 1.0
    for quantity in dict.fromkeys(row["quantity"] for row in mine):
        rows = sorted(
            (row for row in mine if row["quantity"] == quantity), key=lambda row: float(row["ratio"])
        )
        at_file = next(float(row["value"]) for row in rows if float(row["ratio"]) == 1.0)
        if rows[0]["layer"] == "individual" and biomodel != CELL:
            continue  # the growth rate is the uptake over a constant: the same curve
        values = [float(row["value"]) / at_file for row in rows]
        color = LAYER_COLOR[rows[0]["layer"]]
        axis.plot(
            [float(row["ratio"]) for row in rows],
            values,
            color=color,
            lw=1.3,
            ls="-" if rows[0]["kind"] == "common" else (0, (4, 2)),
        )
        ends.append((values[-1], quantity_name(quantity), color))
        lowest, highest = min(lowest, *values), max(highest, *values)
    wide = highest / lowest > 30.0
    axis.set_xscale("log")
    if wide:
        axis.set_yscale("log")
        axis.set_ylim(lowest / 4.0, highest * 2.0)
    else:
        room = 0.08 * (highest - lowest)
        axis.set_ylim(lowest - 2.6 * room, highest + room)
    bottom, upper = axis.get_ylim()
    ends.sort()
    wanted = [
        float(
            np.log(value / bottom) / np.log(upper / bottom) if wide else (value - bottom) / (upper - bottom)
        )
        for value, _, _ in ends
    ]
    for at, (_, label, color) in zip(_apart(wanted, 0.075), ends, strict=True):
        axis.text(
            1.03, at, label, transform=axis.transAxes, fontsize=5.6, va="center", ha="left", color=color
        )
    if limits.get("no_steady_state_at_ratio") is not None:
        edge = float(limits.get("steady_state_down_to_ratio") or limits["lowest_ratio_with_a_steady_state"])
        axis.axvspan(span[0], edge, color="#e9e9e9", lw=0)
        axis.text(
            0.03,
            0.55,
            "no steady state",
            fontsize=6.0,
            color=SECOND,
            rotation=90,
            va="center",
            ha="left",
            transform=axis.transAxes,
        )
    for i, (_, (down, up)) in enumerate(sorted((swing or {}).items())):
        height = 0.022 + 0.03 * i
        axis.plot(
            [max(down, span[0]), min(up, span[1])],
            [height, height],
            color=LEVELS[i],
            lw=1.8,
            solid_capstyle="butt",
            transform=axis.get_xaxis_transform(),
            zorder=5,
        )
    axis.axvline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
    axis.set_xlim(*span)
    axis.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0])
    axis.set_xticklabels(["0.2", "0.5", "1", "2", "5"])
    axis.xaxis.set_minor_locator(NullLocator())
    if wide:
        axis.yaxis.set_minor_locator(NullLocator())


def fig_alone(plan: dict, out: Path, name: str = "fig4") -> Path:
    """The SBML models alone: their dose-response, D of every run beside D that the model alone gives, and one
    run of the growing cell time by time beside its model alone."""
    directory = STUDIES / "quasistatic"
    dose = read_rows(directory / "dose.csv")
    found = read_rows(directory / "prediction.csv")
    limits = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    names = list(plan["receivers"])
    cell = (CELL_RUNS / "quasistatic" / "dose.csv").exists()
    if cell:
        dose += read_rows(CELL_RUNS / "quasistatic" / "dose.csv")
        found += read_rows(CELL_RUNS / "quasistatic" / "prediction.csv")
        limits.update(json.loads((CELL_RUNS / "quasistatic" / "summary.json").read_text(encoding="utf-8")))
        names.append(CELL)
    colors = dict(zip(names, [*CATEGORY, "#8250c4"], strict=False))
    figure = plt.figure(figsize=(WIDTH, 5.2))
    grid = figure.add_gridspec(2, 3, left=0.075, right=0.885, top=0.95, bottom=0.2, wspace=0.85, hspace=0.5)
    for n, biomodel in enumerate(names):
        axis = figure.add_subplot(grid[n // 3, n % 3])
        swing = _swing(plan, biomodel) if biomodel != CELL else None
        _dose_panel(axis, plan, biomodel, dose, limits[biomodel], swing)
        axis.set_xlabel("nutrient, over its level $S^{*}$", fontsize=7.4)
        if n % 3 == 0:
            axis.set_ylabel("steady state of the model\nalone, over its value at $S^{*}$", fontsize=7.4)
        axis.set_title(f"({'abcd'[n]})  {receiver_name(plan, biomodel)}", fontsize=8.2)
    # (e) every run against the model alone
    axis = figure.add_subplot(grid[1, 1])
    for row in found:
        a, b = number(row.get("D_of_the_model_alone")), number(row.get("D_of_the_run"))
        if a is None or not (a > 0.0 and b > 0.0):
            continue
        axis.plot(
            [a],
            [b],
            ls="none",
            marker="o" if row["world"].startswith("plan") else "s",
            ms=2.8,
            mfc=colors[row["biomodel"]],
            mec="white",
            mew=0.3,
            alpha=0.9,
            zorder=4 if row["biomodel"] == CELL else 3,
        )
    axis.plot([1e-7, 30], [1e-7, 30], color=AXIS, lw=0.6, zorder=1)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(3e-6, 30)
    axis.set_ylim(3e-6, 30)
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xticks([1e-5, 1e-3, 1e-1, 1e1])
    axis.set_yticks([1e-5, 1e-3, 1e-1, 1e1])
    axis.set_xlabel("$D$ of the model alone, along\nthe medium of the run", fontsize=7.4)
    axis.set_ylabel("$D$ of the run", fontsize=7.4)
    axis.set_title("(e)  every run and quantity", fontsize=8.2)
    handles = [
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=4.2,
            mfc=colors[one],
            mec="white",
            label=receiver_name(plan, one),
        )
        for one in names
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        fontsize=5.8,
        handlelength=0.8,
        borderaxespad=0.1,
        labelspacing=0.25,
    )
    # (f) one run of the growing cell, time by time
    axis = figure.add_subplot(grid[1, 2])
    if cell and (CELL_RUNS / "quasistatic" / f"course_{CELL}.csv").exists():
        course = read_rows(CELL_RUNS / "quasistatic" / f"course_{CELL}.csv")
        hours = np.array([float(row["time"]) for row in course]) / 60.0
        medium = next(key for key in course[0] if key.startswith("medium."))
        shown = "B.rates.r_14"
        axis.plot(
            hours,
            [100.0 * (float(row[medium]) - 1.0) for row in course],
            color=MUTED,
            lw=1.0,
            label="nutrient",
        )
        axis.plot(
            hours,
            [100.0 * (float(row[shown]) - 1.0) for row in course],
            color=LAYER_COLOR["complex"],
            lw=1.3,
            label=f"{quantity_name(shown)}, run",
        )
        axis.plot(
            hours,
            [100.0 * (float(row[f"alone:{shown}"]) - 1.0) for row in course],
            color=LAYER_COLOR["complex"],
            lw=1.3,
            ls=(0, (1, 1.5)),
            label="same, model alone",
        )
        axis.set_xlim(0.0, 20.0)
        axis.set_xlabel("time after the step (h)", fontsize=7.4)
        axis.set_ylabel("change (%)", fontsize=7.4)
        low = axis.get_ylim()[0]
        axis.set_ylim(bottom=1.55 * low)
        axis.legend(loc="lower right", fontsize=5.6, handlelength=1.5, borderaxespad=0.4)
    axis.set_title(f"(f)  {CELL_NAME}, one run", fontsize=8.2)
    # the legends of (a)-(d), under the panels
    leaks = sorted(float(leak) for leak in plan["designs"].values() if float(leak) > 0.0)
    layers_ = [
        Line2D([0], [0], color=color, lw=1.3, label=f"{layer} layer") for layer, color in LAYER_COLOR.items()
    ]
    layers_ += [
        Line2D([0], [0], color=MUTED, lw=1.3, label="common quantity"),
        Line2D([0], [0], color=MUTED, lw=1.3, ls=(0, (4, 2)), label="specific quantity"),
    ]
    ranges = [
        Line2D([0], [0], color=LEVELS[i], lw=1.8, solid_capstyle="butt", label=f"$\\varepsilon$ = {leak:g}")
        for i, leak in enumerate(leaks)
    ]
    first = figure.legend(
        handles=layers_,
        loc="lower left",
        bbox_to_anchor=(0.07, 0.0),
        ncols=3,
        handlelength=2.0,
        columnspacing=1.2,
        title="(a)-(d): curves",
        title_fontsize=7,
        alignment="left",
    )
    figure.add_artist(first)
    figure.legend(
        handles=ranges,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.0),
        ncols=2,
        handlelength=1.6,
        columnspacing=1.2,
        title="(a)-(c): range of the nutrient in the runs",
        title_fontsize=7,
        alignment="left",
    )
    return save_figure(figure, name, out)


def fig_functional(plan: dict, out: Path, name: str = "fig5") -> Path:
    """The tolerances that the growth function gives, and three readings of the same runs: the convention of a
    tenth, the box of tolerances that follows from the function, and what the function itself lost."""
    sources = [(STUDIES / "functional", list(plan["receivers"]))]
    if (CELL_RUNS / "functional" / "worlds.csv").exists():
        sources.append((CELL_RUNS / "functional", [CELL]))
    names = [name for _, listed in sources for name in listed]
    colors = dict(zip(names, [*CATEGORY, "#8250c4"], strict=False))
    table = [row for directory, _ in sources for row in read_rows(directory / "tolerances.csv")]
    worlds = [row for directory, _ in sources for row in read_rows(directory / "worlds.csv")]
    figure = plt.figure(figsize=(WIDTH, 4.3))
    grid = figure.add_gridspec(
        1, 2, left=0.2, right=0.98, top=0.93, bottom=0.26, wspace=0.26, width_ratios=[0.8, 1.25]
    )
    axis = figure.add_subplot(grid[0, 0])
    labels, y = [], 0
    for biomodel in names:
        mine = [row for row in table if row["biomodel"] == biomodel]
        axis.text(
            3.4e4,
            y + 0.5 * (len(mine) - 1),
            receiver_name(plan, biomodel),
            fontsize=6.2,
            color=colors[biomodel],
            ha="right",
            va="center",
        )
        for row in mine:
            value = 100.0 * float(row["delta_over_value"])
            sure = 100.0 * float(row["delta_guaranteed_over_value"])
            axis.barh([y], [value], height=0.7, color=colors[biomodel], lw=0, alpha=0.45)
            axis.barh([y], [sure], height=0.7, color=colors[biomodel], lw=0)
            shown = f"{sure:.3g}" if sure < 1000.0 else f"{float(f'{sure:.3g}'):.0f}"
            axis.text(max(value, 3.3) * 1.12, y, f"{shown} %", va="center", ha="left", fontsize=5.8)
            labels.append(quantity_name(row["quantity"]))
            y += 1
        y += 0.6
        labels.append(None)
    ticks = list(zip(_positions(labels), [label for label in labels if label is not None], strict=True))
    axis.set_yticks([pos for pos, _ in ticks])
    axis.set_yticklabels([label for _, label in ticks], fontsize=6.0)
    axis.axvline(10.0, color=INK, lw=0.6, ls=(0, (1, 2)))
    axis.text(10.0, -0.9, "10 %", fontsize=6.0, ha="center", va="bottom", color=SECOND)
    axis.set_xscale("log")
    axis.set_xlim(0.5, 4e4)
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_ylim(max(pos for pos, _ in ticks) + 0.6, -1.0)
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="y", length=0)
    axis.set_xlabel("tolerance from the function,\nin per cent of the steady-state value", fontsize=7.4)
    axis.legend(
        handles=[
            Patch(fc=MUTED, alpha=0.45, lw=0, label="first order (whole margin)"),
            Patch(fc=MUTED, lw=0, label="whole box (number)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.2, -0.2),
        ncols=1,
        fontsize=6.4,
        handlelength=1.3,
    )
    panel_mark(axis, "a", dx=-88, dy=2)
    # (b) three readings, one small panel per receiver
    inner = grid[0, 1].subgridspec(1, len(names), wspace=0.1)
    styles = {
        "D_of_the_plan": {"marker": "s", "ms": 3.8, "mfc": "white", "mew": 1.0, "lw": 0.9, "ls": (0, (4, 2))},
        "D_guaranteed": {"marker": "o", "ms": 4.2, "mec": "white", "mew": 0.6, "lw": 1.4},
        "margin_used": {"marker": "D", "ms": 3.4, "mec": "white", "mew": 0.5, "lw": 0.9, "ls": (0, (1, 1.5))},
    }
    first = None
    for n, biomodel in enumerate(names):
        axis = figure.add_subplot(inner[0, n], sharey=first)
        first = first or axis
        mine = sorted(
            (row for row in worlds if row["biomodel"] == biomodel and float(row["leak"]) > 0.0),
            key=lambda row: float(row["leak"]),
        )
        x = [float(row["leak"]) for row in mine]
        for key, look in styles.items():
            axis.plot(x, [float(row[key]) for row in mine], color=colors[biomodel], **look)
        axis.axhline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_xticks([1e-3, 1e-2, 1e-1])
        axis.tick_params(axis="x", labelsize=6.2)
        axis.set_title(receiver_name(plan, biomodel), fontsize=7.0, color=colors[biomodel])
        if n:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel("over the admissible steps", fontsize=7.4)
            panel_mark(axis, "b", dx=-34, dy=12)
    figure.text(
        0.71,
        0.185,
        "leak $\\varepsilon$ of the sender (growing cell: share taken at the baseline)",
        ha="center",
        fontsize=7.4,
    )
    handles = [
        Line2D([0], [0], color=MUTED, label="$D$, convention of 10 %", **styles["D_of_the_plan"]),
        Line2D(
            [0],
            [0],
            color=MUTED,
            label="$D$, tolerances from the function (whole box)",
            **styles["D_guaranteed"],
        ),
        Line2D(
            [0], [0], color=MUTED, label="share of the margin that the function lost", **styles["margin_used"]
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.71, 0.0),
        ncols=1,
        fontsize=6.6,
        handlelength=2.4,
    )
    return save_figure(figure, name, out)


def _positions(labels: list) -> list[float]:
    """Where the bars of ``fig_functional`` sit: one unit apart, and a little more between receivers."""
    out, y = [], 0.0
    for label in labels:
        if label is None:
            y += 0.6
        else:
            out.append(y)
            y += 1.0
    return out


def fig_time(plan: dict, out: Path, name: str = "figS3") -> Path:
    """What the shared environment adds: when the interference is at each layer, and an activity that moves."""
    names = list(plan["receivers"])
    figure = plt.figure(figsize=(WIDTH, 5.3))
    top = figure.add_gridspec(1, 3, left=0.085, right=0.975, top=0.945, bottom=0.6, wspace=0.09)
    low = figure.add_gridspec(1, 3, left=0.085, right=0.975, top=0.44, bottom=0.185, wspace=0.09)
    families, first = {"medium": CATEGORY[0], "population": CATEGORY[2]}, None
    for n, biomodel in enumerate(names):
        axis = figure.add_subplot(top[0, n], sharey=first)
        first = first or axis
        rows = read_rows(STUDIES / "cascade" / "courses" / f"plan_{biomodel}_{CONTROL}.csv")
        times = np.array([float(row["time"]) for row in rows])
        for column in rows[0]:
            if column == "time":
                continue
            course = np.array([float(row[column]) for row in rows])
            peak = float(np.max(course))
            if not peak > 0.0:
                continue
            layer = column.split(",")[0]
            inside = layer not in families
            axis.plot(
                np.maximum(times, 5.0) / 60.0,
                course / peak,
                color=families.get(layer, CATEGORY[1]),
                lw=0.8 if inside else 2.4,
                zorder=4 if inside else 3,
            )
        axis.set_xscale("log")
        axis.set_xlim(10.0 / 60.0, float(times[-1]) / 60.0)
        axis.set_xticks([1.0, 10.0, 100.0])
        axis.set_xticklabels(["1", "10", "100"])
        axis.set_xlabel("time after the step (h)")
        axis.set_title(f"({'abc'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5)
        if n:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel("$D(t)$ of a layer over its peak")
    rows = [
        row
        for row in read_rows(STUDIES / "schedules" / "layers.csv")
        if row["design"] == CONTROL and row["kind"] == "common"
    ]
    shown = {"medium": CATEGORY[0], "molecule": CATEGORY[1], "population": CATEGORY[2]}
    periods = [float(period) for period in plan["studies"]["schedules"]["periods"]]
    edge, first = periods[-1] * 4.5, None
    for n, biomodel in enumerate(names):
        axis = figure.add_subplot(low[0, n], sharey=first)
        first = first or axis
        mine = [row for row in rows if row["biomodel"] == biomodel]
        for layer, color in shown.items():
            wave = _largest(mine, layer, "wave")
            xs = sorted(wave)
            axis.plot(xs, [wave[x] for x in xs], color=color, marker="o", ms=4.0, mec="white", mew=0.6)
            step, worst = _largest(mine, layer, "step"), _largest(mine, layer, "worst case")
            if step:
                axis.plot(
                    [edge],
                    [max(step.values())],
                    ls="none",
                    marker="s",
                    ms=5.0,
                    mfc=color,
                    mec="white",
                    mew=0.6,
                )
            if worst:
                axis.plot(
                    [edge * 2.4],
                    [max(worst.values())],
                    ls="none",
                    marker="^",
                    ms=5.5,
                    mfc=color,
                    mec="white",
                    mew=0.6,
                )
        axis.axhline(1.0, color=INK, lw=0.6, ls=(0, (1, 2)))
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([*periods, edge, edge * 2.4])
        axis.set_xticklabels(
            [f"{period:g}" for period in periods] + ["step", "worst"],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        axis.set_xlim(periods[0] / 1.6, edge * 2.4 * 1.6)
        axis.set_xlabel("period of the wave (min)")
        axis.set_title(f"({'def'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5)
        if n:
            axis.tick_params(labelleft=False)
        else:
            axis.set_ylabel("$D$ of a layer\n($\\varepsilon$ = 0.01; tolerances)")
    handles = [
        Line2D([0], [0], color=CATEGORY[0], lw=2.0, marker="o", ms=4, mec="white", label="medium"),
        Line2D(
            [0],
            [0],
            color=CATEGORY[1],
            lw=1.2,
            marker="o",
            ms=4,
            mec="white",
            label="inside the organism ((a)-(c): every layer; (d)-(f): molecule)",
        ),
        Line2D([0], [0], color=CATEGORY[2], lw=2.0, marker="o", ms=4, mec="white", label="population"),
        Line2D(
            [0], [0], ls="none", marker="s", ms=5, mfc=MUTED, mec="white", label="the larger of the two steps"
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="^",
            ms=5.5,
            mfc=MUTED,
            mec="white",
            label="worst switching of the linear picture",
        ),
    ]
    order = [
        handles[0],
        handles[3],
        handles[1],
        handles[4],
        handles[2],
    ]  # the legend fills its columns downwards
    figure.legend(
        handles=order,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncols=3,
        handlelength=1.8,
        columnspacing=1.3,
    )
    return save_figure(figure, name, out)


def fig_company(plan: dict, out: Path, name: str = "fig6") -> Path:
    """Two pairs in one vessel, at rest before anything is stepped: who moves whom, and by which way."""
    directory = STUDIES / "coexistence"
    rows = read_rows(directory / "community.csv")
    designs = list(dict.fromkeys(row["design"] for row in rows))
    order = list(dict.fromkeys(row["receiver"] for row in rows if row["receiver_kind"] == "sbml"))
    xenos = [name_ for name_ in dict.fromkeys(row["sender"] for row in rows if row["sender_kind"] == "xeno")]
    order = sorted(order) + sorted(xenos)
    xenos = sorted(xenos)

    def cell(axis, x, y, row, key):
        if row is None or row.get("status") != "ok" or row[key] in ("", "None"):
            return
        lost = bool(row["losses"]) if key == "D" else row["removal_lost_at"] not in ("", "None")
        tile(axis, x, y, float(row[key]), lost, marked=row["same_pair"] == "True", size=6.4)

    figure = plt.figure(figsize=(WIDTH, 4.4))
    top = figure.add_gridspec(1, len(designs), left=0.05, right=0.995, top=0.91, bottom=0.74, wspace=0.16)
    low = figure.add_gridspec(
        1, 2, left=0.05, right=0.975, top=0.645, bottom=0.215, wspace=0.36, width_ratios=[0.8, 1.0]
    )
    for n, design in enumerate(designs):
        axis = figure.add_subplot(top[0, n])
        mine = {(row["sender"], row["receiver"]): row for row in rows if row["design"] == design}
        for i, sender in enumerate(xenos):
            for j, receiver in enumerate(order):
                if sender != receiver:
                    cell(axis, j, i, mine.get((sender, receiver)), "D")
        bare_axis(axis, order, xenos, aspect=0.55)
        axis.set_title(f"({'abc'[n]})  {design}: $D$", fontsize=8.5, pad=14, loc="left")
    design = designs[-1]
    axis = figure.add_subplot(low[0, 0])
    mine = {(row["sender"], row["receiver"]): row for row in rows if row["design"] == design}
    for i, sender in enumerate(order):
        for j, receiver in enumerate(order):
            if sender != receiver:
                cell(axis, j, i, mine.get((sender, receiver)), "D_removal")
    bare_axis(axis, order, order, aspect=0.62)
    axis.set_title(f"(d)  {design}: $D_{{\\mathrm{{removal}}}}$", fontsize=8.5, pad=14, loc="left")
    # (e), (f) the way from a xeno to its own partner, in the run that carries D: the two links outside the
    # receiver, which move by orders of magnitude, and the two at the receiver, which move by per cent
    carried = next(
        row
        for row in rows
        if row["design"] == design
        and row["same_pair"] == "True"
        and row["sender_kind"] == "xeno"
        and number(row["D"])
        and float(row["D"]) > 1.0
    )
    sender, receiver = carried["sender"], carried["receiver"]
    other = next(name_ for name_ in xenos if name_ != sender)
    run = (
        f"{sender}.{carried['D_activity'].split('.')[-1]}"
        if "." not in carried["D_activity"]
        else carried["D_activity"]
    )
    times, base = read_series(directory / design / "baseline" / "series.csv")
    _, moved = read_series(directory / design / run / "series.csv")
    info = json.loads((directory / design / "community.json").read_text(encoding="utf-8"))
    nutrient = info["members"][receiver]["nutrient"]
    quantity = carried["D_quantity"]
    hours = np.maximum(times, 5.0) / 60.0
    pair = low[0, 1].subgridspec(2, 1, hspace=0.16)
    upper, lower = figure.add_subplot(pair[0, 0]), figure.add_subplot(pair[1, 0])
    theirs = info["members"][other][
        "substrate"
    ]  # the substrate of the other xeno, which the sender also takes
    upper.plot(
        hours,
        moved[f"medium.{theirs}"] / base[f"medium.{theirs}"],
        color=CATEGORY[1],
        lw=1.3,
        label=f"substrate of {other}",
    )
    upper.plot(
        hours,
        moved[f"{other}.density"] / base[f"{other}.density"],
        color=CATEGORY[1],
        lw=1.3,
        ls=(0, (4, 2)),
        label=f"density of {other}",
    )
    upper.set_yscale("log")
    upper.set_ylim(3e-3, 3.0)
    upper.yaxis.set_minor_locator(NullLocator())
    upper.legend(loc="lower left", fontsize=6.3, handlelength=2.2)
    upper.tick_params(labelbottom=False)
    lower.plot(
        hours,
        100.0 * (moved[f"medium.{nutrient}"] / base[f"medium.{nutrient}"] - 1.0),
        color=CATEGORY[0],
        lw=1.3,
        label=f"{CHEMICAL_NAME.get(nutrient, nutrient)} in the medium",
    )
    lower.plot(
        hours,
        100.0 * (moved[quantity] / base[quantity] - 1.0),
        color=CATEGORY[0],
        lw=1.3,
        ls=(0, (4, 2)),
        label=f"{quantity_name(quantity.replace(receiver + '.', 'B.'))} of {receiver}",
    )
    lower.axhline(10.0, color=INK, lw=0.6, ls=(0, (1, 2)))
    lower.legend(loc="upper left", fontsize=6.3, handlelength=2.2)
    lower.set_xlabel(f"time after the step of {sender} (h)")
    lower.set_ylabel("change (%)")
    upper.set_ylabel("ratio to\nthe baseline")
    for axis in (upper, lower):
        axis.set_xscale("log")
        axis.set_xlim(10.0 / 60.0, hours[-1])
        axis.set_xticks([1.0, 10.0, 100.0])
        axis.set_xticklabels(["1", "10", "100"])
        axis.xaxis.set_minor_locator(NullLocator())
    panel_mark(upper, "e", dx=-46, dy=1)
    panel_mark(lower, "f", dx=-46, dy=-3)
    shade_legend(figure, pair=True, columns=3, zero="0: nothing that the receiver protects changed")
    return save_figure(figure, name, out)


def fig_information(plan: dict, out: Path, name: str = "figS2") -> Path:
    """What the protected quantities of a receiver say of the activity of the sender, against the noise with
    which they are read, and against how many of the stored times are read."""
    directory = STUDIES / "information"
    noise = read_rows(directory / "noise.csv")
    readings = read_rows(directory / "readings.csv")
    names = list(plan["receivers"])
    leaks = sorted({float(row["leak"]) for row in noise if float(row["leak"]) > 0.0})
    shade = dict(zip(leaks, LEVELS, strict=False))
    most = np.log2(5.0)
    figure, axes = plt.subplots(2, len(names), figsize=(WIDTH, 4.3), sharey=True)
    for n, biomodel in enumerate(names):
        for row_axis, table, key in ((axes[0, n], noise, "sigma"), (axes[1, n], readings, "values_read")):
            for leak in leaks:
                mine = sorted(
                    (row for row in table if row["biomodel"] == biomodel and float(row["leak"]) == leak),
                    key=lambda row: float(row[key]),
                )
                if mine:
                    row_axis.plot(
                        [float(row[key]) for row in mine],
                        [float(row["bits"]) for row in mine],
                        color=shade[leak],
                        lw=1.3,
                        marker="o" if key == "values_read" else None,
                        ms=3.2,
                        mec="white",
                        mew=0.4,
                    )
            row_axis.axhline(most, color=INK, lw=0.6, ls=(0, (1, 2)))
            row_axis.set_xscale("log")
            row_axis.xaxis.set_minor_locator(NullLocator())
        axes[0, n].axvline(1.0, color=AXIS, lw=0.6)
        axes[0, n].set_title(f"({'abc'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5)
        axes[1, n].set_title(f"({'def'[n]})  {italic(species_name(plan, biomodel))}", fontsize=8.5)
        axes[0, n].set_xlabel("noise on every value (tolerances)")
        axes[1, n].set_xlabel("values read (noise of one tolerance)")
    axes[0, 0].set_ylabel("bits about the activity")
    axes[1, 0].set_ylabel("bits about the activity")
    handles = [
        Line2D([0], [0], color=shade[leak], lw=1.3, label=f"$\\varepsilon$ = {leak:g}") for leak in leaks
    ]
    handles.append(Line2D([0], [0], color=INK, lw=0.6, ls=(0, (1, 2)), label="all that five runs can say"))
    figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncols=5, handlelength=1.8)
    figure.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.2, wspace=0.08, hspace=0.6)
    return save_figure(figure, name, out)


#: the figures by the name of their file: those of the text by their number, those of the supplement with an S
FIGURES = {
    "fig1": fig_scenario,
    "fig2": fig_activity,
    "fig3": fig_groups,
    "fig4": fig_alone,
    "fig5": fig_functional,
    "fig6": fig_company,
    "figS1": fig_candidates,
    "figS2": fig_information,
    "figS3": fig_time,
    "figS4": fig_routes,
    "figS5": fig_sensitivity,
}
