"""The figures of the studies, from what they left under ``runs/studies/``.

    study_sensitivity.png   D of a control under every assumption changed, and the community under the cross
                            specificity swept
    study_candidates.png    every species and every rate of a model read alone, and where the quantities that
                            the plan picks stand among them
    study_numerics.png      what the coupling interval leaves in D
    study_channels.png      the layers of a receiver, with the interference entering by its nutrient or by another
                            species that its model reads
    study_cascade.png       when the interference is at each layer, after one step
    study_receivers.png     the receivers taken by rule, layer by layer
    study_schedules.png     an activity that changes in time: D at three layers against the period of a wave
    study_comparison.png    D beside the steady states and beside the niche overlap

The colours are those of ``xeno_biomodels.figures``: the three receivers of the plan are three categories, and a
number of the measure sits on the shade of what it means.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from xeno_biomodels import paths
from xeno_biomodels.figures import (
    AXIS,
    CATEGORY,
    INK,
    MUTED,
    SECOND,
    SURFACE,
    _bare,
    _shade_legend,
    _short,
    _tile,
    style,
)
from xeno_biomodels.studies import (
    cascade,
    channels,
    comparison,
    numerics,
    receivers,
    schedules,
    sensitivity,
    settings,
)
from xeno_biomodels.studies.common import LINES, number, read_rows

OUT = paths.RUNS / "figures"
CONTROL = "leak-1e-2"


def _title(figure, text: str) -> None:
    figure.suptitle(text, x=0.01, ha="left", fontsize=10.5, fontweight="bold")


def _save(figure, name: str, out: Path) -> Path:
    path = out / name
    figure.savefig(path)
    plt.close(figure)
    return path


def _nothing(axis, x: float, y: float, text: str = "no world") -> None:
    axis.text(x, y, text, ha="center", va="center", color=MUTED, fontsize=7.5)


def figure_sensitivity(plan: dict, out: Path) -> Path | None:
    if not (sensitivity.OUT / "worlds.csv").exists():
        return None
    rows = [row for row in read_rows(sensitivity.OUT / "worlds.csv") if row["design"] == CONTROL]
    names = list(plan["receivers"])
    variants = list(dict.fromkeys(row["variant"] for row in rows))
    found = {(row["variant"], row["biomodel"]): row for row in rows}
    together = (
        read_rows(sensitivity.OUT / "community.csv") if (sensitivity.OUT / "community.csv").exists() else []
    )
    swept = [
        row for row in together if row.get("parameter") == "cross specificity" and row.get("status") == "ok"
    ]
    figure = plt.figure(figsize=(13.4, 0.3 * len(variants) + 2.0))
    grid = figure.add_gridspec(
        1, 2, width_ratios=[1.0, 1.25], wspace=0.32, left=0.2, right=0.98, top=0.9, bottom=0.065
    )
    axis = figure.add_subplot(grid[0, 0])
    for i, variant in enumerate(variants):
        for j, biomodel in enumerate(names):
            row = found.get((variant, biomodel))
            if row is None or row.get("status") != "ok":
                _nothing(axis, j, i)
            else:
                _tile(axis, j, i, float(row["D"]), bool(row["losses"]))
    _bare(
        axis,
        [
            f"{_short(name)}\n{plan['receivers'][name]['organism'].split()[0][0]}. {plan['receivers'][name]['organism'].split()[1]}"
            for name in names
        ],
        variants,
    )
    axis.set_aspect("auto")
    axis.axhline(0.5, color=AXIS, lw=0.8)
    axis.set_title(f"D of the control {CONTROL}, one assumption changed at a time", fontsize=9, pad=34)
    right = figure.add_subplot(grid[0, 1])
    if swept:
        pairs = list(
            dict.fromkeys(
                (row["sender"], row["receiver"])
                for row in swept
                if row["same_pair"] == "True" and row["sender_kind"] == "xeno"
            )
        )
        for n, (sender, receiver) in enumerate(pairs):
            mine = sorted(
                (float(row["value"]), float(row["D"]))
                for row in swept
                if (row["sender"], row["receiver"]) == (sender, receiver)
            )
            right.plot(
                [x for x, _ in mine],
                [y for _, y in mine],
                color=CATEGORY[n % 3],
                marker="o",
                ms=5.5,
                mec=SURFACE,
                mew=1.2,
                label=f"{sender} to its own partner {receiver}",
            )
        across = {}
        for row in swept:
            if row["sender_kind"] == "xeno" and row["receiver_kind"] == "sbml":
                across[float(row["value"])] = max(across.get(float(row["value"]), 0.0), float(row["D"]))
        right.plot(
            list(across),
            list(across.values()),
            color=MUTED,
            lw=1.4,
            ls=(0, (4, 3)),
            label="the largest of any xeno to any SBML organism",
        )
        right.axhline(1.0, color=AXIS, lw=0.8)
        right.text(min(across), 1.12, "D = 1: the tolerance", color=MUTED, fontsize=8, va="bottom")
        right.set_xscale("log")
        right.set_yscale("log")
        right.set_xlabel("specificity of a xeno for the nutrients of the other pairs (design partner-only)")
        right.set_ylabel("D, in tolerances of the receiver")
        right.legend(loc="lower right")
        right.set_title("In the common vessel: a pair that is orthogonal alone", fontsize=9)
    else:
        right.axis("off")
    box = right.get_position()
    right.set_position([box.x0 + 0.03, box.y0 + 0.52 * box.height, box.width - 0.03, 0.44 * box.height])
    _shade_legend(figure, pair=False)
    _title(figure, "What the readings owe to the assumptions of the world")
    return _save(figure, "study_sensitivity.png", out)


def figure_candidates(plan: dict, out: Path) -> Path | None:
    if not (settings.OUT / "candidates.csv").exists():
        return None
    rows = [row for row in read_rows(settings.OUT / "candidates.csv") if row["design"] == CONTROL]
    summary = {
        (row["biomodel"], row["layer"]): row
        for row in read_rows(settings.OUT / "candidates_summary.csv")
        if row["design"] == CONTROL and row.get("status") == "ok"
    }
    names, levels = list(plan["receivers"]), ("particle", "molecule", "complex")
    figure, axes = plt.subplots(1, len(names), figsize=(12.0, 4.4), sharey=True)
    rng = np.random.default_rng(0)
    for n, (axis, biomodel) in enumerate(zip(axes, names, strict=True)):
        for x, layer in enumerate(levels):
            values = np.array(
                [float(row["D"]) for row in rows if row["biomodel"] == biomodel and row["layer"] == layer]
            )
            values = np.maximum(values, 1e-5)
            axis.plot(
                x + rng.uniform(-0.16, 0.16, values.size),
                values,
                ls="none",
                marker="o",
                ms=4.6,
                mfc=CATEGORY[n],
                mec=SURFACE,
                mew=0.8,
                alpha=0.8,
            )
            found = summary.get((biomodel, layer))
            if found is None:
                continue
            for kind, marker, shift in (("common", "D", 0.29), ("specific", "s", 0.41)):
                d = number(found.get(f"D_picked_{kind}"))
                if d is not None:
                    axis.plot(
                        [x + shift],
                        [max(d, 1e-5)],
                        ls="none",
                        marker=marker,
                        ms=7.5,
                        mfc=SURFACE if kind == "common" else INK,
                        mec=INK,
                        mew=1.4,
                        zorder=6,
                    )
        axis.axhline(1.0, color=AXIS, lw=0.8)
        axis.set_yscale("log")
        axis.set_xticks(range(len(levels)))
        axis.set_xticklabels(levels)
        axis.set_xlim(-0.5, len(levels) - 0.25)
        axis.set_title(f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}", fontsize=9)
    axes[0].set_ylabel(
        f"D of one quantity alone, design {CONTROL}\n(a quantity that does not move is shown at 1e-5)"
    )
    handles = [
        Line2D(
            [0],
            [0],
            ls="none",
            marker="o",
            ms=5,
            mfc=MUTED,
            mec=SURFACE,
            label="every species (particle, molecule) and every rate or state of a protein (complex) of the model",
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="D",
            ms=7,
            mfc=SURFACE,
            mec=INK,
            mew=1.4,
            label="the quantity common to all organisms",
        ),
        Line2D(
            [0], [0], ls="none", marker="s", ms=7, mfc=INK, mec=INK, label="the quantity picked as specific"
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        ncols=3,
        handlelength=1.2,
        columnspacing=1.6,
    )
    _title(figure, "Which quantity stands for a layer: all that could have been picked, and what was")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _save(figure, "study_candidates.png", out)


def figure_numerics(plan: dict, out: Path) -> Path | None:
    if not (numerics.OUT / "numerics.csv").exists():
        return None
    rows = [
        row
        for row in read_rows(numerics.OUT / "numerics.csv")
        if row.get("status") == "ok" and row["what"] == "coupling interval"
    ]
    limit = {
        (row["biomodel"], row["design"]): number(row.get("D_extrapolated"))
        for row in rows
        if number(row.get("D_extrapolated"))
    }
    names = list(plan["receivers"])
    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    dashes = {"leak-1e-2": "-", "leak-1e-1": (0, (4, 3))}
    for n, biomodel in enumerate(names):
        for design, dash in dashes.items():
            d0 = limit.get((biomodel, design))
            if not d0:
                continue
            mine = {
                float(row["value"]): float(row["D"])
                for row in rows
                if (row["biomodel"], row["design"]) == (biomodel, design)
            }
            reference = next(
                (
                    float(row["D_reference"])
                    for row in rows
                    if (row["biomodel"], row["design"]) == (biomodel, design)
                ),
                None,
            )
            if reference is not None:
                mine[float(plan["time"]["interval"])] = reference
            xs = sorted(mine)
            axis.plot(
                xs,
                [max(abs(mine[x] / d0 - 1.0), 1e-9) for x in xs],
                color=CATEGORY[n],
                ls=dash,
                marker="o",
                ms=5.5,
                mec=SURFACE,
                mew=1.2,
            )
            for (
                row
            ) in rows:  # a world in which another set of steps is admissible is another reading, and says so
                if (row["biomodel"], row["design"]) == (biomodel, design) and row.get(
                    "same_admissible"
                ) == "False":
                    x = float(row["value"])
                    axis.annotate(
                        "here a step of the sender\nis no longer admissible",
                        (x, max(abs(mine[x] / d0 - 1.0), 1e-9)),
                        xytext=(-12, -6),
                        textcoords="offset points",
                        ha="right",
                        va="top",
                        fontsize=8,
                        color=SECOND,
                    )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xticks([0.0625, 0.125, 0.25, 0.5])
    axis.set_xticklabels(["0.0625", "0.125", "0.25\n(the plan)", "0.5"])
    axis.set_xlabel("coupling interval (min)")
    axis.set_ylabel("|D over D at an interval of zero - 1|")
    handles = [
        Line2D([0], [0], color=CATEGORY[n], label=f"{_short(name)}  {plan['receivers'][name]['organism']}")
        for n, name in enumerate(names)
    ]
    handles += [Line2D([0], [0], color=MUTED, ls=dash, label=design) for design, dash in dashes.items()]
    axis.legend(handles=handles, loc="upper left", handlelength=2.2)
    axis.set_title("What the coupling interval leaves in D of the controls")
    figure.tight_layout()
    return _save(figure, "study_numerics.png", out)


def _layer_tiles(
    figure, axis, rows: dict, columns: list[str], labels: list[str], lines: list[tuple[str, str]], first: bool
) -> None:
    for j, column in enumerate(columns):
        for i, line in enumerate(lines):
            row = rows.get((column, *line))
            if row is not None and number(row.get("D")) is not None:
                _tile(axis, j, i, float(row["D"]), bool(row.get("losses")))
    _bare(axis, labels, [f"{layer}, {kind}" for layer, kind in lines] if first else [""] * len(lines))
    del figure


def figure_channels(plan: dict, out: Path) -> Path | None:
    if not (channels.OUT / "layers.csv").exists():
        return None
    rows = read_rows(channels.OUT / "layers.csv")
    names = list(plan["studies"]["channels"]["other"])
    lines = [line for line in LINES if any((row["layer"], row["kind"]) == line for row in rows)]
    shares = [float(share) for share in plan["studies"]["channels"]["shares"]]
    columns = ["xeno"] + [f"{channel}-{share:g}" for share in shares for channel in ("nutrient", "other")]
    figure, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 0.42 * len(lines) + 2.6))
    for n, (axis, biomodel) in enumerate(zip(axes, names, strict=True)):
        other = plan["studies"]["channels"]["other"][biomodel]["species"]
        nutrient = plan["receivers"][biomodel]["nutrient"]
        mine = {
            (row["design"], row["layer"], row["kind"]): row for row in rows if row["biomodel"] == biomodel
        }
        labels = ["none"] + [
            f"{nutrient if channel == 'nutrient' else other}\n{share:g}"
            for share in shares
            for channel in ("nutrient", "other")
        ]
        _layer_tiles(figure, axis, mine, columns, labels, lines, n == 0)
        axis.set_title(f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}", fontsize=9, pad=30)
        for edge in (0.5, 2.5):
            axis.axvline(edge, color=AXIS, lw=0.8)
    axes[len(names) // 2].set_xlabel(
        "what the sender takes of the receiver's, and the share of it that it takes at rest"
    )
    _shade_legend(figure, pair=False)
    _title(figure, "The same share taken by another way in: the layers of the receiver answer differently")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _save(figure, "study_channels.png", out)


def figure_cascade(plan: dict, out: Path) -> Path | None:
    names = list(plan["receivers"])
    files = {name: cascade.OUT / "courses" / f"plan_{name}_{CONTROL}.csv" for name in names}
    if not all(path.exists() for path in files.values()):
        return None
    figure, axes = plt.subplots(1, len(names), figsize=(12.2, 4.1), sharey=True)
    families = {"medium": CATEGORY[0], "population": CATEGORY[2]}
    for axis, biomodel in zip(axes, names, strict=True):
        rows = read_rows(files[biomodel])
        times = np.array([float(row["time"]) for row in rows])
        for name in rows[0]:
            if name == "time":
                continue
            course = np.array([float(row[name]) for row in rows])
            peak = float(np.max(course))
            if not peak > 0.0:
                continue
            layer = name.split(",")[0]
            inside = layer not in families
            # the thin lines of the inside go over the thick ones: where they lie on the medium, both are seen
            axis.plot(
                np.maximum(times, 5.0) / 60.0,
                course / peak,
                color=families.get(layer, CATEGORY[1]),
                lw=1.0 if inside else 3.2,
                alpha=0.9 if inside else 1.0,
                zorder=4 if inside else 3,
            )
        axis.set_xscale("log")
        axis.set_xlim(10.0 / 60.0, float(times[-1]) / 60.0)
        axis.set_xlabel("time after the step (h)")
        axis.set_title(f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}", fontsize=9)
    axes[0].set_ylabel("D of a layer over its own peak")
    handles = [
        Line2D([0], [0], color=CATEGORY[0], lw=3.2, label="medium: the level of the nutrient"),
        Line2D(
            [0],
            [0],
            color=CATEGORY[1],
            lw=1.0,
            label="inside the organism: particle, molecule, complex, individual (one line each)",
        ),
        Line2D([0], [0], color=CATEGORY[2], lw=3.2, label="population: the density"),
    ]
    figure.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        ncols=3,
        handlelength=1.8,
        columnspacing=1.6,
    )
    _title(figure, f"When the interference is at each layer: the step of the activity to 2, design {CONTROL}")
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    return _save(figure, "study_cascade.png", out)


def figure_receivers(plan: dict, out: Path) -> Path | None:
    if not (receivers.OUT / "worlds.csv").exists():
        return None
    worlds = read_rows(receivers.OUT / "worlds.csv")
    by_layer = {
        (row["biomodel"], row["design"], row["layer"]): row
        for row in read_rows(receivers.OUT / "layers.csv")
        if row["kind"] == "common"
    }
    keys = list(dict.fromkeys(row["biomodel"] for row in worlds))
    designs = [design for design in dict.fromkeys(row["design"] for row in worlds) if design != "xeno"]
    status = {(row["biomodel"], row["design"]): row for row in worlds}
    from xeno_biomodels.layers import LAYERS

    figure, axes = plt.subplots(1, len(designs), figsize=(5.4 * len(designs) + 2.6, 0.36 * len(keys) + 2.4))
    for n, (axis, design) in enumerate(zip(np.atleast_1d(axes), designs, strict=True)):
        for i, key in enumerate(keys):
            row = status.get((key, design))
            if row is None or row.get("status") != "ok":
                axis.text(
                    3.0,
                    i,
                    "no world" if row is None or "Error" in row.get("status", "") else "not measured",
                    ha="center",
                    va="center",
                    color=MUTED,
                    fontsize=7.5,
                )
                continue
            _tile(axis, 0, i, float(row["D"]), bool(row["losses"]))
            for j, layer in enumerate(LAYERS, start=1):
                found = by_layer.get((key, design, layer))
                if found is not None and number(found.get("D")) is not None:
                    _tile(axis, j, i, float(found["D"]), bool(row["losses"]))
        labels = [f"{_short(key.split('@')[0])}  {key.split('@')[1]}" for key in keys]
        _bare(axis, ["all", *LAYERS], labels if n == 0 else [""] * len(keys))
        axis.set_aspect("auto")
        axis.axvline(0.5, color=AXIS, lw=0.8)
        share = float(status[(keys[0], design)]["share"]) if (keys[0], design) in status else float("nan")
        axis.set_title(f"the sender takes {share:g} of what the organism takes", fontsize=9, pad=26)
    _shade_legend(figure, pair=False)
    _title(
        figure, "Every model of the plateau that reads what it takes up, read on the quantities common to all"
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.95))
    return _save(figure, "study_receivers.png", out)


def figure_schedules(plan: dict, out: Path) -> Path | None:
    if not (schedules.OUT / "layers.csv").exists():
        return None
    rows = [
        row
        for row in read_rows(schedules.OUT / "layers.csv")
        if row["design"] == CONTROL and row["kind"] == "common"
    ]
    names = list(plan["receivers"])
    shown = {"medium": CATEGORY[0], "molecule": CATEGORY[1], "population": CATEGORY[2]}
    periods = [float(period) for period in plan["studies"]["schedules"]["periods"]]
    figure, axes = plt.subplots(1, len(names), figsize=(12.2, 4.2), sharey=True)
    edge = periods[-1] * 4.0
    for axis, biomodel in zip(axes, names, strict=True):
        mine = [row for row in rows if row["biomodel"] == biomodel]
        for layer, color in shown.items():
            wave = {}
            for row in mine:
                if (
                    row["layer"] == layer
                    and row["schedule_kind"] == "wave"
                    and number(row.get("D")) is not None
                ):
                    wave[float(row["period"])] = max(wave.get(float(row["period"]), 0.0), float(row["D"]))
            xs = sorted(wave)
            axis.plot(
                xs, [max(wave[x], 1e-6) for x in xs], color=color, marker="o", ms=5.5, mec=SURFACE, mew=1.2
            )
            step = [
                float(row["D"])
                for row in mine
                if row["layer"] == layer
                and row["schedule_kind"] == "step"
                and number(row.get("D")) is not None
            ]
            if step:
                axis.plot(
                    [edge],
                    [max(max(step), 1e-6)],
                    ls="none",
                    marker="s",
                    ms=7,
                    mfc=color,
                    mec=SURFACE,
                    mew=1.2,
                )
            worst = [
                float(row["D"])
                for row in mine
                if row["layer"] == layer
                and row["schedule_kind"] == "worst case"
                and number(row.get("D")) is not None
            ]
            if worst:
                axis.plot(
                    [edge * 2.2],
                    [max(max(worst), 1e-6)],
                    ls="none",
                    marker="^",
                    ms=7.5,
                    mfc=color,
                    mec=SURFACE,
                    mew=1.2,
                )
        axis.axhline(1.0, color=AXIS, lw=0.8)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks([*periods, edge, edge * 2.2])
        axis.set_xticklabels([f"{period:g}" for period in periods] + ["step", "worst"])
        axis.set_xlabel("period of the wave (min)")
        axis.set_title(f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}", fontsize=9)
    axes[0].set_ylabel(f"D of a layer, design {CONTROL}")
    handles = [
        Line2D(
            [0], [0], color=color, marker="o", ms=5, mec=SURFACE, label=f"{layer}: the quantity common to all"
        )
        for layer, color in shown.items()
    ]
    handles += [
        Line2D(
            [0], [0], ls="none", marker="s", ms=7, mfc=MUTED, mec=SURFACE, label="the larger of the two steps"
        ),
        Line2D(
            [0],
            [0],
            ls="none",
            marker="^",
            ms=7,
            mfc=MUTED,
            mec=SURFACE,
            label="the switching that the step responses say is worst",
        ),
    ]
    figure.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.01, 0.0),
        ncols=3,
        handlelength=1.6,
        columnspacing=1.6,
    )
    settings_ = plan["studies"]["schedules"]
    _title(
        figure,
        f"An activity that moves between {settings_['low']:g} and {settings_['high']:g}: which layer feels which time scale",
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    return _save(figure, "study_schedules.png", out)


def figure_comparison(plan: dict, out: Path) -> Path | None:
    if not (comparison.OUT / "worlds.csv").exists():
        return None
    rows = [
        row
        for row in read_rows(comparison.OUT / "worlds.csv")
        if row.get("status") == "ok" and float(row["leak"]) > 0.0
    ]
    together = (
        [row for row in read_rows(comparison.OUT / "community.csv")]
        if (comparison.OUT / "community.csv").exists()
        else []
    )
    names = list(plan["receivers"])
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))
    axis = axes[0]
    for n, biomodel in enumerate(names):
        mine = [row for row in rows if row["biomodel"] == biomodel]
        axis.plot(
            [float(row["D_steady"]) for row in mine],
            [float(row["D"]) for row in mine],
            ls="none",
            marker="o",
            ms=7,
            mfc=CATEGORY[n],
            mec=SURFACE,
            mew=1.4,
            label=f"{_short(biomodel)}  {plan['receivers'][biomodel]['organism']}",
        )
        for row in mine:
            if row["losses"] or row["receiver_gone_at_rest"]:
                axis.plot(
                    [float(row["D_steady"])],
                    [float(row["D"])],
                    ls="none",
                    marker="o",
                    ms=11,
                    mfc="none",
                    mec=CATEGORY[n],
                    mew=1.4,
                )
    low = min(min(float(row["D_steady"]) for row in rows), min(float(row["D"]) for row in rows)) / 2.0
    high = max(max(float(row["D_steady"]) for row in rows), max(float(row["D"]) for row in rows)) * 2.0
    axis.plot([low, high], [low, high], color=AXIS, lw=0.8)
    axis.text(low * 2.2, low * 1.5, "the same", color=MUTED, fontsize=8, ha="left", va="top")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("the same reading on steady states alone, in tolerances")
    axis.set_ylabel("D of the runs, in tolerances")
    axis.legend(loc="lower right")
    axis.set_title(
        "A control world: the runs against the rest states of its steps\n(ring: the receiver is lost in a run, or gone at rest)",
        fontsize=9,
    )
    axis = axes[1]
    design = next(
        (
            name
            for name in reversed(list(plan["community"]["designs"]))
            if any(row["design"] == name for row in together)
        ),
        None,
    )
    mine = [row for row in together if row["design"] == design and number(row.get("D_removal")) is not None]
    if mine:
        floor = 1e-4
        for row in mine:
            way = number(row.get("organisms_on_the_way"))
            direct = way == 0.0
            axis.plot(
                [float(row["niche_overlap"])],
                [max(float(row["D_removal"]), floor)],
                ls="none",
                marker="o",
                ms=6.5,
                mfc=CATEGORY[0] if direct else CATEGORY[1],
                mec=SURFACE,
                mew=1.2,
            )
        axis.set_yscale("log")
        axis.axhline(1.0, color=AXIS, lw=0.8)
        axis.set_xlabel("niche overlap of the two organisms (Pianka)")
        axis.set_ylabel("D_removal of the ordered pair (0 shown at 1e-4)")
        handles = [
            Line2D(
                [0],
                [0],
                ls="none",
                marker="o",
                ms=6.5,
                mfc=CATEGORY[0],
                mec=SURFACE,
                label="the two take a species in common",
            ),
            Line2D(
                [0],
                [0],
                ls="none",
                marker="o",
                ms=6.5,
                mfc=CATEGORY[1],
                mec=SURFACE,
                label="they do not: the way goes through other organisms",
            ),
        ]
        axis.legend(handles=handles, loc="lower right")
        axis.set_title(f"The common vessel, design {design}: every ordered pair of organisms", fontsize=9)
    else:
        axis.axis("off")
    _title(figure, "D beside what a comparison of steady states and a niche overlap would say")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(figure, "study_comparison.png", out)


def write(plan: dict, out: Path = OUT) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    style()
    made = []
    for draw in (
        figure_sensitivity,
        figure_candidates,
        figure_numerics,
        figure_channels,
        figure_cascade,
        figure_receivers,
        figure_schedules,
        figure_comparison,
    ):
        path = draw(plan, out)
        if path is not None:
            made.append(path)
    return made
