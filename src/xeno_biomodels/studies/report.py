"""``runs/studies.md``: what the studies found, written from the tables under ``runs/studies/``.

As in the two reports of the runs, every number is read from a file that a study wrote, and a sentence that
depends on a result is written from that result. A study whose tables are not there has no section.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from xeno_biomodels import paths
from xeno_biomodels.report import _number, _table
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
from xeno_biomodels.studies import figures as study_figures
from xeno_biomodels.studies.common import LINES, number, read_rows, truth

CONTROL = "leak-1e-2"
#: what each function of the study ``settings`` reads, in words
READS = {
    "growth rate": "the growth rate",
    "work": "the density and the uptake of the nutrient",
    "share of ATP": "the pools of ATP and of ADP",
}


def _short(biomodel: str) -> str:
    return f"BIOMD{int(biomodel.split('@')[0].replace('BIOMD', ''))}"


def _show(value, digits: int = 3) -> str:
    """A number at so many digits, without an exponent where it can be written out."""
    if value is None:
        return "–"
    text = _number(str(value), digits)
    if "e+" in text and abs(float(text)) < 1e7:
        return f"{float(text):.0f}"
    return text


def _rest(value, digits: int = 3) -> str:
    """A reading on rest states: what is below 1e-9 is the rounding of two searches, and shown as that."""
    return "<1e-9" if value is not None and 0.0 < abs(float(value)) < 1e-9 else _show(value, digits)


def _span(values: list[float], digits: int = 3) -> str:
    """The least and the largest of some numbers, or the one number that they all are."""
    if not values:
        return "–"
    low, high = _show(min(values), digits), _show(max(values), digits)
    return low if low == high else f"{low} to {high}"


def _why(status: str) -> str:
    """Why a world could not be built or measured, without the name of the error and of the file."""
    text = status.split(": ", 1)[1] if ": " in status else status
    text = re.sub(r"^\S+: ", "", text)  # the model or the file that the message begins with
    return text.split(", at ")[0].split(" at virtual")[0].strip()[:150]


def _figure(add, name: str) -> None:
    if (paths.RUNS / "figures" / name).exists():
        add(f"![{name[:-4]}](figures/{name})\n")


def _line_of(rows: list[dict]) -> dict:
    """(layer, kind) -> row, for the groups that are one kind at one layer."""
    return {(row["layer"], row["kind"]): row for row in rows if (row["layer"], row["kind"]) in LINES}


def _choices(rows: list[dict]) -> list[str]:
    """The choices of the measure that a table of the study ``settings`` holds."""
    return list(dict.fromkeys(row["choice"] for row in rows))


def _horizon(plan: dict, row: dict) -> float:
    """How long the runs of a world of the study ``sensitivity`` were followed, in minutes."""
    longer = float(row["factor"]) if row.get("parameter") == "horizon of a run" else 1.0
    return float(plan["time"]["horizon"]) * longer


def _carrier(rows: list[dict]) -> str:
    """The layer and the kind at which D of a world is largest."""
    lines = _line_of(rows)
    found = [(number(row["D"]) or 0.0, key) for key, row in lines.items()]
    if not found or max(found)[0] == 0.0:
        return "–"
    return ", ".join(max(found)[1])


# ---- 1 ----------------------------------------------------------------------------------------------------------------


def _sensitivity(plan: dict, add) -> None:
    if not (sensitivity.OUT / "worlds.csv").exists():
        return
    rows = read_rows(sensitivity.OUT / "worlds.csv")
    by_group = read_rows(sensitivity.OUT / "worlds_layers.csv")
    names = list(plan["receivers"])
    add("## 1. The assumptions of the world\n")
    add(
        "Every number that the world adds to an SBML model is changed, one at a time, to a quarter, a half, twice "
        "and four times its value; the receiver is given two other growth laws; and a run is followed two and four "
        f"times as long. Each time the worlds of every receiver with the design `xeno` and with the control `{CONTROL}` "
        "are brought to rest, run and measured again.\n"
    )
    _figure(add, "study_sensitivity.png")
    good = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    xeno = [row for row in good if row["design"] == "xeno"]
    exact = [
        row
        for row in xeno
        if float(row["D"]) == 0.0
        and truth(row.get("receiver_bit_identical"))
        and truth(row["stage1_trajectory"])
    ]
    add(
        f"* **The zero does not depend on any of it.** Of {len(rows)} worlds, {len(good)} could be built. In "
        f"{len(exact)} of the {len(xeno)} that have the design `xeno`, D = 0, stage 1 holds, and every stored number "
        "of the receiver is that of the baseline, bit for bit."
    )
    if failed:
        why = {}
        for row in failed:
            reason = (
                "the sender cannot live at its baseline activity"
                if "cannot settle" in row["status"]
                else "the organism would exchange too much of its nutrient in one coupling interval"
                if "per interval" in row["status"]
                else row["status"][:80]
            )
            why.setdefault(reason, set()).add(row["variant"])
        add(
            f"* {len(failed)} worlds do not exist: "
            + "; ".join(f"{reason} ({', '.join(sorted(found))})" for reason, found in why.items())
            + ". These are the ends of the range in which the world of the plan is a world at all."
        )
    control = [row for row in good if row["design"] == CONTROL]
    reference = {row["biomodel"]: row for row in control if row["variant"] == "reference"}
    lines, elsewhere, flips, worlds_read = [], [], 0, 0
    for biomodel in names:
        mine = [row for row in control if row["biomodel"] == biomodel and row["variant"] != "reference"]
        base = reference.get(biomodel)
        if not mine or base is None:
            continue
        values = [float(row["D"]) for row in mine]
        flipped = [row["variant"] for row in mine if row["stage3_tolerable"] != base["stage3_tolerable"]]
        groups = {}
        for row in by_group:
            if row["biomodel"] == biomodel and row["design"] == CONTROL:
                groups.setdefault(row["variant"], []).append(row)
        home = _carrier(groups.get("reference", []))
        moved = [row["variant"] for row in mine if _carrier(groups.get(row["variant"], [])) != home]
        elsewhere += [
            f"{_short(biomodel)} {row['variant']}: {_carrier(groups.get(row['variant'], []))}"
            + (", at its peak at the end of the run" if float(row["t_peak"]) >= _horizon(plan, row) else "")
            for row in mine
            if row["variant"] in moved
        ]
        flips += len(flipped)
        worlds_read += len(mine)
        lines.append(
            [
                f"{_short(biomodel)} *{plan['receivers'][biomodel]['organism']}*",
                _show(float(base["D"])),
                _span(values),
                f"{len(flipped)} of {len(mine)}"
                + (f" ({', '.join(flipped)})" if 0 < len(flipped) <= 6 else ""),
                home,
                f"{len(moved)} of {len(mine)}" + (f" ({', '.join(moved)})" if 0 < len(moved) <= 6 else ""),
            ]
        )
    add("")
    add(
        _table(
            [
                "receiver",
                f"D of `{CONTROL}` in the plan",
                "D over all the variants",
                "variants in which stage 3 says otherwise",
                "D is largest at",
                "variants in which it is largest elsewhere",
            ],
            lines,
        )
        + "\n"
    )
    add(
        "* **The size of D of a control is owed to the assumptions.** It moves by the factors in the table, and the "
        f"verdict of stage 3 moves with it in {flips} of {worlds_read} control worlds: those of the receivers whose D "
        "is near 1 in the plan."
    )
    add(
        f"* **Where D lands is owed to them far less.** The layer at which D is largest is that of the plan in "
        f"{worlds_read - len(elsewhere)} of {worlds_read} control worlds"
        + (". In the others it is: " + "; ".join(elsewhere) + "." if elsewhere else ".")
    )
    longer = [row for row in control if row["parameter"] == "horizon of a run"]
    if longer:
        same = all(
            abs(float(row["D"]) / float(reference[row["biomodel"]]["D"]) - 1.0) < 1e-9
            for row in longer
            if row["biomodel"] in reference
        )
        add(
            "* A run two and four times as long gives "
            + (
                "the same D in every control: the peak lies inside the horizon of the plan."
                if same
                else "another D in some controls: there the peak does not lie inside the horizon of the plan (table `worlds.csv`)."
            )
        )
    if (sensitivity.OUT / "community.csv").exists():
        together = [row for row in read_rows(sensitivity.OUT / "community.csv") if row.get("status") == "ok"]
        swept = [row for row in together if row["parameter"] == "cross specificity"]
        if swept:
            values = sorted({float(row["value"]) for row in swept})
            pairs = list(
                dict.fromkeys(
                    (row["sender"], row["receiver"])
                    for row in swept
                    if truth(row["same_pair"]) and row["sender_kind"] == "xeno"
                )
            )
            lines = []
            for sender, receiver in pairs:
                mine = {
                    float(row["value"]): row
                    for row in swept
                    if (row["sender"], row["receiver"]) == (sender, receiver)
                }
                lines.append(
                    [
                        f"{sender} to {receiver}",
                        *[
                            _show(float(mine[value]["D"])) + (" (lost)" if mine[value]["losses"] else "")
                            if value in mine
                            else "–"
                            for value in values
                        ],
                    ]
                )
            zeros = [
                str(
                    sum(truth(row["receiver_bit_identical"]) for row in swept if float(row["value"]) == value)
                )
                for value in values
            ]
            lines.append(["ordered pairs with an exact zero, of 30", *zeros])
            add("")
            add(
                "In the common vessel (design `partner-only`), D of each xeno against its own partner, with which it "
                "shares no species, against the specificity of the xenos for the nutrients of the other pairs:\n\n"
                + _table(["", *[f"{value:g}" for value in values]], lines)
                + "\n"
            )
            if all(zero == "0" for zero in zeros):
                add(
                    "* **That a pair which is orthogonal alone is not so in company does not hang on the 1 % of the "
                    "plan.** At every specificity above zero no ordered pair is left with an exact zero; what the "
                    "specificity sets is the size, and with it where D of a pair passes its tolerance."
                )
            else:
                add(
                    "* At some of these specificities ordered pairs with an exact zero are left: see the last line of the table."
                )
        diluted = [row for row in together if row["parameter"] == "dilution rate"]
        if diluted:
            found = []
            for value in sorted({float(row["value"]) for row in diluted}):
                counts = [
                    f"{sum(truth(row['receiver_bit_identical']) for row in diluted if float(row['value']) == value and row['design'] == design)} in `{design}`"
                    for design in plan["community"]["designs"]
                ]
                found.append(f"at {value:g} times the dilution rate {', '.join(counts)}")
            add(
                "* The exact zeros of the three community designs are a matter of who shares a species with whom, and stay: "
                + "; ".join(found)
                + " (of 30 ordered pairs each)."
            )
    add("")


# ---- 2 ----------------------------------------------------------------------------------------------------------------


def _settings(plan: dict, add) -> None:
    if not (settings.OUT / "worlds.csv").exists():
        return
    rows = read_rows(settings.OUT / "worlds.csv")
    together = read_rows(settings.OUT / "community.csv")
    add("## 2. The choices of the measure\n")
    add(
        "Nothing is run here. The runs of the 15 worlds and of the 90 ordered pairs of the community are read again "
        "with one choice of the measure changed at a time.\n"
    )
    lines = []
    for choice in dict.fromkeys(row["choice"] for row in rows):
        mine = [row for row in rows if row["choice"] == choice and row.get("status") == "ok"]
        pairs = [row for row in together if row["choice"] == choice and row.get("status") == "ok"]
        ratios = [float(row["D_over_the_plan"]) for row in mine if number(row.get("D_of_the_plan"))]
        zero = all(float(row["D"]) == 0.0 for row in mine if row["design"] == "xeno")
        lines.append(
            [
                choice,
                _span(ratios, 4),
                f"{sum(not truth(row['same_stages']) for row in mine)}",
                f"{sum(not truth(row['same_losses']) for row in mine)}",
                f"{sum(not truth(row['same_admissible']) for row in mine)}",
                f"{sum(not truth(row.get('same_stages')) for row in pairs)}",
                f"{sum(not truth(row.get('same_losses')) for row in pairs)}",
                "yes"
                if zero
                and all(
                    truth(row["receiver_bit_identical"]) == (number(row.get("D_removal")) == 0.0)
                    for row in pairs
                    if row["sender_kind"] == "xeno" and number(row.get("D")) == 0.0
                )
                else "no",
            ]
        )
    add(
        _table(
            [
                "choice",
                "D over D of the plan (worlds with D > 0)",
                "worlds: stages differ (of 15)",
                "worlds: runs with a loss differ",
                "worlds: admissible runs differ",
                "pairs: stages differ (of 90)",
                "pairs: runs with a loss differ",
                "zeros stay zeros",
            ],
            lines,
        )
        + "\n"
    )
    good = [row for row in rows if row.get("status") == "ok" and number(row.get("D_of_the_plan"))]
    scaled = {
        choice: sorted({round(float(row["D_over_the_plan"]), 9) for row in good if row["choice"] == choice})
        for choice in ("tolerance 5 %", "tolerance 20 %")
    }
    if scaled.get("tolerance 5 %") == [2.0] and scaled.get("tolerance 20 %") == [0.5]:
        add(
            "* **D is a number in tolerances and scales with them, exactly**: half the tolerance, twice D, in every "
            "world. A verdict of stage 3 can change only where D of the plan lies between the two."
        )
    else:
        add("* With another tolerance D does not scale by the same factor in every world: see the table.")
    unmoved = [
        choice
        for choice in dict.fromkeys(row["choice"] for row in good)
        if all(abs(float(row["D_over_the_plan"]) - 1.0) < 1e-12 for row in good if row["choice"] == choice)
    ]
    add(
        f"* {len(unmoved)} of the {len(_choices(rows))} choices leave D of every world where it is; they decide when a "
        "receiver counts as lost and which runs of the sender are admissible. Reading fewer stored times lowers D "
        "by what the peak loses between two of them."
    )
    with_bits = [
        row for row in rows if row.get("status") == "ok" and number(row.get("bits_upper")) is not None
    ]
    if with_bits and all(float(row["D"]) == 0.0 for row in with_bits if truth(row["stage4_distribution"])):
        add(
            "* **The bits are a count per stored number.** Their upper bound grows with the number of stored times "
            "and falls with the noise (table `worlds.csv`: `bits_upper`), so stage 4 is a statement about a run as "
            "it was stored and read. Under every choice here it holds only in the worlds in which nothing moves."
        )
    if (settings.OUT / "lemma_summary.csv").exists():
        add("\n### The lemma on the runs\n")
        add(
            "The lemma says that where every quantity is within a tolerance `fn`, the function is kept, to first "
            "order. It is tried on every stored time of every run of the controls, for three functions of the "
            "receiver, each with a tolerance `fn` on the quantities that it reads:\n"
        )
        lines = []
        for row in read_rows(settings.OUT / "lemma_summary.csv"):
            lines.append(
                [
                    row["function"],
                    READS.get(row["function"], f"`{row['reads']}`"),
                    row["times"],
                    row["times_within_tolerance"],
                    row["of_these_function_lost"],
                    _show(number(row["least_margin_kept_within_tolerance"]), 3),
                    row["times_fell_beyond_bound"],
                    _show(number(row["largest_excess_over_margin_within_tolerance"]), 2),
                ]
            )
        add(
            _table(
                [
                    "function",
                    "reads",
                    "stored times",
                    "of these, every quantity within tolerance",
                    "of these, function below Q_min",
                    "least share of the margin kept there",
                    "times the function fell by more than the first-order bound",
                    "largest change beyond the bound while within tolerance, in margins",
                ],
                lines,
            )
            + "\n"
        )
        found = read_rows(settings.OUT / "lemma_summary.csv")
        broken = sum(int(row["of_these_function_lost"]) for row in found)
        fell = sum(int(row["times_fell_beyond_bound"]) for row in found)
        over = [row["function"] for row in found if int(row["times_beyond_first_order_bound"]) > 0]
        add(
            (
                "* **On these runs the implication of the lemma has no exception**: at no stored time is every quantity within its tolerance and the function below Q_min."
                if broken == 0
                else f"* On these runs the implication of the lemma fails at {broken} stored times: every quantity is within its tolerance there and the function is below Q_min."
            )
            + (
                " And the function never falls by more than the first-order bound allows."
                if fell == 0
                else f" The function falls by more than the first-order bound at {fell} stored times."
            )
            + (
                f" Where a function changes by more than the bound ({', '.join(over)}), it does so upward: that is the second order of a product whose factors rise together."
                if over and fell == 0
                else ""
            )
            + " The growth rate is its own function, so for it the bound is exact and the test says little; the two other functions read two quantities each."
        )
    if (settings.OUT / "candidates_summary.csv").exists():
        add("\n### Which quantities are protected\n")
        add(
            "The quantities that stand for a layer were picked: one common to all organisms, and for three layers one "
            "that is characteristic of the organism. Here every species and every reaction rate of a model is read "
            f"alone with the tolerance of the plan (design `{CONTROL}`): carriers at `particle`, the other species at "
            "`molecule`, rates and states of proteins at `complex`.\n"
        )
        _figure(add, "study_candidates.png")
        lines = []
        for row in read_rows(settings.OUT / "candidates_summary.csv"):
            if row["design"] != CONTROL or row.get("status") != "ok":
                continue
            lines.append(
                [
                    _short(row["biomodel"]),
                    row["layer"],
                    row["candidates"],
                    _show(number(row["least"])),
                    _show(number(row["median"])),
                    _show(number(row["largest"])),
                    f"{100 * float(row['share_beyond_tolerance']):.0f} %",
                    _show(number(row.get("D_picked_common"))),
                    (
                        f"{_show(number(row.get('D_picked_specific')))} (`{row['picked_specific'].replace('B.', '')}`)"
                        if row.get("picked_specific")
                        else "–"
                    ),
                    (
                        f"{100 * float(row['candidates_below_picked_specific']):.0f} %"
                        if row.get("picked_specific")
                        else "–"
                    ),
                ]
            )
        add(
            _table(
                [
                    "receiver",
                    "layer",
                    "candidates",
                    "least D",
                    "median",
                    "largest",
                    "beyond tolerance",
                    "D of the common quantity",
                    "D of the specific quantity",
                    "candidates below it",
                ],
                lines,
            )
            + "\n"
        )
        found = [
            row
            for row in read_rows(settings.OUT / "candidates_summary.csv")
            if row["design"] == CONTROL and row.get("status") == "ok"
        ]
        spread = [float(row["largest"]) / float(row["median"]) for row in found if float(row["median"]) > 0.0]
        high = [
            f"`{row['picked_specific'].replace('B.', '')}` of {_short(row['biomodel'])} (above {100 * float(row['candidates_below_picked_specific']):.0f} % of the {row['layer']} candidates)"
            for row in found
            if row.get("picked_specific") and float(row["candidates_below_picked_specific"]) >= 0.75
        ]
        low = [
            f"`{row['picked_specific'].replace('B.', '')}` of {_short(row['biomodel'])} (above {100 * float(row['candidates_below_picked_specific']):.0f} %)"
            for row in found
            if row.get("picked_specific") and float(row["candidates_below_picked_specific"]) <= 0.25
        ]
        verdicts = [row for row in found if 0.0 < float(row["share_beyond_tolerance"]) < 1.0]
        add(
            f"* **Within a layer the quantities of a model are far apart**: the largest D of a layer is {_span(spread, 2)} times its "
            "median, so what a layer says depends on which quantity stands for it."
            + (
                f" In {len(verdicts)} of the {len(found)} layers some candidates are within their tolerance and others beyond it."
                if verdicts
                else ""
            )
            + (
                " Picks that are among the most sensitive of their layer: " + "; ".join(high) + "."
                if high
                else ""
            )
            + (" Picks that are among the least sensitive: " + "; ".join(low) + "." if low else "")
            + " The common quantities are there to set organisms side by side; a specific one says what happens to the thing by which the organism is known, and is no summary of its layer."
        )
    add("")


# ---- 3 ----------------------------------------------------------------------------------------------------------------


def _numerics(plan: dict, add) -> None:
    if not (numerics.OUT / "numerics.csv").exists():
        return
    rows = read_rows(numerics.OUT / "numerics.csv")
    good = [row for row in rows if row.get("status") == "ok"]
    add("## 3. The numerical method\n")
    add(
        f"The organisms are coupled once per interval ({float(plan['time']['interval']):g} min in the plan), and the SBML "
        "model is integrated by CVODE within it. Both are changed, and the worlds with the designs "
        f"{', '.join(f'`{name}`' for name in numerics.DESIGNS)} are run again.\n"
    )
    _figure(add, "study_numerics.png")
    lines = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        mine = [row for row in good if row["variant"] == variant]
        controls = [row for row in mine if row["design"] != "xeno"]
        zero = [row for row in mine if row["design"] == "xeno"]
        lines.append(
            [
                variant,
                f"{len(mine)} of {sum(row['variant'] == variant for row in rows)}",
                f"{sum(float(row['D']) == 0.0 and truth(row.get('receiver_bit_identical')) for row in zero)} of {len(zero)}",
                (f"{100 * max(abs(float(row['D_change'])) for row in controls):.2g} %" if controls else "–"),
                (f"{100 * max(float(row['D_course_gap']) for row in controls):.2g} %" if controls else "–"),
                f"{sum(not truth(row['same_verdict']) for row in controls)}",
                f"{sum(not truth(row.get('same_admissible')) for row in controls)}",
            ]
        )
    add(
        _table(
            [
                "variant",
                "worlds measured",
                "`xeno`: exact zeros",
                "controls: largest change of D",
                "largest gap along D(t), over the peak",
                "controls with other stages or losses",
                "controls with other admissible steps",
            ],
            lines,
        )
        + "\n"
    )
    other = [row for row in good if row["design"] != "xeno" and not truth(row.get("same_admissible"))]
    if other:
        add(
            "* **The interval of the plan is small enough, and twice it is not, for the sender.** In "
            + "; ".join(f"{_short(row['biomodel'])} `{row['design']}` at {row['variant']}" for row in other)
            + " the steps that are admissible are no longer those of the plan ("
            + "; ".join(
                f"left out of U_A: {row['excluded_by_sender_function'] or 'none'}, in the plan {row['admissible_in_the_plan'] or 'none'}"
                for row in other
            )
            + "): at its largest activity the sender empties its own substrate within a few intervals, and the "
            "explicit coupling no longer follows it. D of that world is then a supremum over other runs, and that is "
            "the change in the table. At the interval of the plan and below it the admissible steps are the same."
        )
    limits = [
        row
        for row in good
        if number(row.get("reference_over_extrapolated")) is not None and row["design"] != "xeno"
    ]
    if limits:
        ratio = [
            float(row["error_ratio_h_over_half"])
            for row in limits
            if number(row.get("error_ratio_h_over_half")) is not None
        ]
        add(
            "* **The coupling is of first order in the interval, and what it leaves in D of a control is "
            f"{100 * min(abs(float(row['reference_over_extrapolated'])) for row in limits):.2g} to "
            f"{100 * max(abs(float(row['reference_over_extrapolated'])) for row in limits):.2g} %.** That is D at the interval of "
            "the plan over D at an interval of zero, the latter from the readings at a half and at a quarter of the "
            "interval. The error at the interval of the plan over the error at half of it is "
            f"{_span(ratio, 3)} (2 for a method of first order)."
        )
    zero = [row for row in good if row["design"] == "xeno"]
    if zero and all(float(row["D"]) == 0.0 and truth(row.get("receiver_bit_identical")) for row in zero):
        add(
            "* **The zero of the design `xeno` is a zero at every interval and every tolerance**: no number of the "
            "receiver depends on how the sender was integrated."
        )
    else:
        add("* The design `xeno` is not at an exact zero in every variant: see the table.")
    if (numerics.OUT / "layers.csv").exists():
        by_group = [
            row
            for row in read_rows(numerics.OUT / "layers.csv")
            if number(row.get("D_change")) is not None and number(row.get("D_reference"))
        ]
        changed = {(row["variant"], row["biomodel"], row["design"]) for row in other}
        found = {}
        for row in by_group:
            if (row["layer"], row["kind"]) in LINES and (
                row["variant"],
                row["biomodel"],
                row["design"],
            ) not in changed:
                found[row["variant"]] = max(found.get(row["variant"], 0.0), abs(float(row["D_change"])))
        if found:
            add(
                "* Layer by layer, in the worlds whose admissible steps are those of the plan, the largest change of D of a layer is "
                + "; ".join(f"{100 * value:.2g} % ({variant})" for variant, value in found.items())
                + "."
            )
    add("")


# ---- 4 ----------------------------------------------------------------------------------------------------------------


def _channels(plan: dict, add) -> None:
    if not (channels.OUT / "layers.csv").exists():
        return
    worlds = {(row["biomodel"], row["design"]): row for row in read_rows(channels.OUT / "worlds.csv")}
    by_group = read_rows(channels.OUT / "layers.csv")
    study = plan["studies"]["channels"]
    add("## 4. Where the interference enters\n")
    add(
        "In the worlds of the plan the sender can reach the receiver through its nutrient only, so the interference "
        "always enters at the medium and goes inward by one way. Here each receiver has a second way in: another "
        "species of the medium that its model reads ("
        + "; ".join(
            f"{_short(name)}: `{item['species']}`, {item['what']}" for name, item in study["other"].items()
        )
        + "). The world lets that species vary too, the receiver protects its level (`medium, specific`), and the "
        "sender takes at rest the same share of the one or of the other. The leak is in the units of no model: the "
        "transporter is half-saturated at the level of the SBML file, and what it takes is worth nothing to the sender.\n"
    )
    _figure(add, "study_channels.png")
    good = [row for row in worlds.values() if row.get("status") == "ok"]
    xeno = [row for row in good if row["design"] == "xeno"]
    add(
        f"* With two species of the medium to share, the design `xeno` is still exact: D = 0 and every number of the "
        f"receiver the same in {sum(float(row['D']) == 0.0 and truth(row.get('receiver_bit_identical')) for row in xeno)} of {len(xeno)} worlds."
    )
    other_top, cases, supply_only, inside_only, by_kind = 0, 0, [], [], []
    for biomodel, item in study["other"].items():
        nutrient = plan["receivers"][biomodel]["nutrient"]
        for share in study["shares"]:
            lines = {}
            for channel in ("nutrient", "other"):
                design = f"{channel}-{float(share):g}"
                if worlds.get((biomodel, design), {}).get("status") != "ok":
                    continue
                lines[channel] = _line_of(
                    [row for row in by_group if (row["biomodel"], row["design"]) == (biomodel, design)]
                )
            if len(lines) < 2:
                continue
            both = [key for key in LINES if key in lines["nutrient"] and key in lines["other"]]
            d = {channel: {key: float(found[key]["D"]) for key in both} for channel, found in lines.items()}
            apart = [key for key in both if (d["nutrient"][key] <= 1.0) != (d["other"][key] <= 1.0)]
            top = {channel: max(both, key=lambda key, c=channel: d[c][key]) for channel in d}
            inner = [key for key in both if key[0] not in ("medium", "population")]
            deepest = {channel: max(inner, key=lambda key, c=channel: d[c][key]) for channel in d}
            cases += 1
            other_top += top["nutrient"] != top["other"]
            for channel, species in (("nutrient", nutrient), ("other", item["species"])):
                where = f"{_short(biomodel)} by `{species}` at {float(share):g}"
                supply = d[channel][("medium", "common")]
                if supply <= 1.0 < d[channel][deepest[channel]]:
                    inside_only.append(
                        f"{where} (the nutrient {_show(supply)}, {', '.join(deepest[channel])} {_show(d[channel][deepest[channel]])})"
                    )
                elif d[channel][deepest[channel]] <= 1.0 < supply:
                    supply_only.append(
                        f"{where} (the nutrient {_show(supply)}, inside at most {_show(d[channel][deepest[channel]])})"
                    )
                # of the organism itself: the levels in the medium apart, which the sender takes
                own = {
                    kind: [key for key in both if key[1] == kind and key[0] != "medium"]
                    for kind in ("common", "specific")
                }
                if own["specific"]:
                    common = max(d[channel][key] for key in own["common"])
                    specific = max(own["specific"], key=lambda key, c=channel: d[c][key])
                    if common <= 1.0 < d[channel][specific]:
                        by_kind.append(
                            f"{where} (common at most {_show(common)}; {', '.join(specific)} {_show(d[channel][specific])})"
                        )
            add(
                f"* {_short(biomodel)} at a share of {float(share):g}: by `{nutrient}` D is largest at {', '.join(top['nutrient'])} "
                f"({_show(d['nutrient'][top['nutrient']])}) and inside the organism at {', '.join(deepest['nutrient'])} ({_show(d['nutrient'][deepest['nutrient']])}); "
                f"by `{item['species']}` at {', '.join(top['other'])} ({_show(d['other'][top['other']])}) and inside at "
                f"{', '.join(deepest['other'])} ({_show(d['other'][deepest['other']])})."
                + (
                    " Within tolerance by one way and beyond it by the other: "
                    + "; ".join(
                        f"{', '.join(key)} ({_show(d['nutrient'][key])} against {_show(d['other'][key])})"
                        for key in apart
                    )
                    + "."
                    if apart
                    else ""
                )
            )
    moved = [row for row in good if row["design"] != "xeno" and number(row.get("rest_moved")) is not None]
    if moved:
        add(
            f"* The worlds of this study are at rest to {_show(max(float(row['rest_moved']) for row in moved), 2)} of the size of what "
            "still moves per residence time. With a second species to settle, the closed form is no longer the rest "
            "state, and the search goes ahead to the limit of its steps (`World._ahead`)."
        )
    if cases:
        add(
            f"* **Where the interference enters decides where it lands.** In {other_top} of the {cases} comparisons (a "
            "receiver, a share) the largest D is at another layer or kind when the same share is taken by the other way in."
            + (
                " The nutrient within its tolerance and a quantity inside the organism beyond its own: "
                + "; ".join(inside_only)
                + "."
                if inside_only
                else ""
            )
            + (
                " The other way round, the nutrient beyond its tolerance and everything inside within: "
                + "; ".join(supply_only)
                + "."
                if supply_only
                else ""
            )
        )
        if by_kind:
            add(
                "* **A sender can leave what every organism has within tolerance and move what the organism is known "
                "by.** Of the quantities of the organism itself (the levels in the medium apart), the common ones "
                "within tolerance and a specific one beyond it: " + "; ".join(by_kind) + "."
            )
    add("")


# ---- 5 ----------------------------------------------------------------------------------------------------------------


def _cascade(plan: dict, add) -> None:
    if not (cascade.OUT / "arrival.csv").exists():
        return
    arrival = read_rows(cascade.OUT / "arrival.csv")
    dimension = read_rows(cascade.OUT / "dimension.csv")
    similarity = read_rows(cascade.OUT / "similarity.csv")
    names = list(plan["receivers"])
    add("## 5. What the layers add to one another\n")
    add(
        "A world of the plan has one way in, and its layers are stations on it. Two things are read from the runs: "
        "when the interference is at each layer, and how many independent readings the protected quantities are.\n"
    )
    _figure(add, "study_cascade.png")
    lines = []
    for key in LINES:
        cells = []
        for biomodel in names:
            found = [
                row
                for row in arrival
                if row["study"] == "plan"
                and (row["biomodel"], row["design"], row["runs"], row["layer"], row["kind"])
                == (biomodel, CONTROL, cascade.STEP, *key)
            ]
            if not found or number(found[0]["t_half"]) is None:
                cells.append("–")
            else:
                row = found[0]
                cells.append(
                    f"{float(row['t_half']) / 60:.2g} / {float(row['t_peak']) / 60:.3g} / {100 * float(row['D_end_over_peak']):.0f} %"
                )
        if any(cell != "–" for cell in cells):
            lines.append([", ".join(key), *cells])
    add(
        f"After the step of the activity to 2 in the design `{CONTROL}`: hours until D of a layer is half its peak / "
        "hours until the peak / what is left of the peak at the end of the run:\n\n"
        + _table(
            ["layer", *[f"{_short(name)} *{plan['receivers'][name]['organism']}*" for name in names]], lines
        )
        + "\n"
    )
    scales = []
    for biomodel in names:
        mine = [
            row
            for row in arrival
            if row["study"] == "plan"
            and (row["biomodel"], row["design"], row["runs"]) == (biomodel, CONTROL, cascade.STEP)
            and number(row["t_half"]) is not None
        ]
        inside = [row for row in mine if row["layer"] not in ("medium", "population")]
        density = [row for row in mine if row["layer"] == "population"]
        if inside and density:
            halves = [float(row["t_half"]) / 60.0 for row in inside]
            scales.append(
                f"{_short(biomodel)}: inside the organism D is half its peak after {_span(halves, 2)} h and "
                f"{100 * max(float(row['D_end_over_peak']) for row in inside):.2g} % of the peak or less is left at the end; the density after "
                f"{float(density[0]['t_half']) / 60:.2g} h, with {100 * float(density[0]['D_end_over_peak']):.0f} % left"
            )
    if scales:
        add(
            "* **Two time scales.** "
            + ". ".join(scales)
            + ". What is inside the organism follows the medium and "
            "goes back with it as far as the medium does; the density comes later and stays. A reading at one "
            "time, early or late, sees one of the two."
        )
    lines = []
    for row in dimension:
        if row["study"] != "plan" or row["design"] not in (CONTROL, "leak-1e-1"):
            continue
        lines.append(
            [
                _short(row["biomodel"]),
                row["design"],
                row["quantities"],
                _show(number(row["participation_as_they_are"])),
                f"{100 * float(row['first_course_share_as_they_are']):.1f} %",
                _show(number(row["participation_unit_length"])),
                row["courses_for_99_percent_unit_length"],
            ]
        )
    extra = [row for row in dimension if row["study"] == "channels" and row["channel"] != "none"]
    for row in extra:
        lines.append(
            [
                _short(row["biomodel"]),
                f"{row['design']} ({row['species']})" if row["species"] else row["design"],
                row["quantities"],
                _show(number(row["participation_as_they_are"])),
                f"{100 * float(row['first_course_share_as_they_are']):.1f} %",
                _show(number(row["participation_unit_length"])),
                row["courses_for_99_percent_unit_length"],
            ]
        )
    add(
        "How many readings the protected quantities are: the participation ratio of their changes over all admissible "
        "runs and stored times (1: one course with a gain for each quantity), as they are and with every quantity "
        "brought to the same length:\n\n"
        + _table(
            [
                "receiver",
                "world",
                "protected quantities that move",
                "participation",
                "share of the first course",
                "participation, equal lengths",
                "courses for 99 %",
            ],
            lines,
        )
        + "\n"
    )
    ones = {}
    for row in similarity:
        if row["study"] == "plan" and row["design"] == CONTROL and abs(float(row["cosine"])) > 0.9999:
            ones.setdefault(row["biomodel"], []).append(
                f"{row['quantity_a'].replace('B.', '')} and {row['quantity_b'].replace('B.', '')}"
            )
    one_way = [row for row in dimension if row["study"] == "plan"]
    two_ways = [row for row in dimension if row["study"] == "channels" and row["channel"] == "both"]
    alone = [
        row for row in dimension if row["study"] == "channels" and row["channel"] in ("nutrient", "other")
    ]
    if one_way:
        add(
            f"* **In a world with one way in, the {_span([float(row['quantities']) for row in one_way], 2)} protected quantities "
            f"are {_span([float(row['participation_unit_length']) for row in one_way], 3)} readings**, by the participation "
            "ratio with every quantity at the same length ("
            f"{_span([float(row['courses_for_99_percent_unit_length']) for row in one_way], 1)} courses hold 99 % of what they do). "
            "D of a layer is therefore not an independent test of the organism: most layers say what another has said."
        )
    add(
        f"* Quantities whose courses are the same but for a factor (cosine above 0.9999, design `{CONTROL}`): "
        + "; ".join(
            f"{_short(name)}: {len(found)} pairs"
            + (f" ({', '.join(found[:4])}{', ...' if len(found) > 4 else ''})" if found else "")
            for name, found in ones.items()
        )
        + ". The growth rate is the uptake by the growth law of the world, and in a pathway at quasi-steady state "
        "one flux follows another."
    )
    if two_ways and alone:
        text = (
            f"* **A second way in adds a reading.** With the runs of both ways in of a receiver together, the participation is "
            f"{_span([float(row['participation_unit_length']) for row in two_ways], 3)} (each way alone: "
            f"{_span([float(row['participation_unit_length']) for row in alone], 3)})."
        )
        if (cascade.OUT / "profiles.csv").exists():
            found = read_rows(cascade.OUT / "profiles.csv")
            text += (
                " The cosine between the two profiles of a receiver, D of every layer and kind by the one way and by the other, is "
                + "; ".join(
                    f"{_show(float(row['cosine_of_the_two_profiles']), 2)} ({_short(row['biomodel'])} at {float(row['share']):g})"
                    for row in found
                )
                + ": 1 would be the same profile."
            )
        add(text)
    add("")


# ---- 6 ----------------------------------------------------------------------------------------------------------------


def _receivers(plan: dict, add) -> None:
    if not (receivers.OUT / "worlds.csv").exists():
        return
    worlds = read_rows(receivers.OUT / "worlds.csv")
    found = read_rows(receivers.OUT / "candidates.csv")
    study = plan["studies"]["receivers"]
    add("## 6. Other SBML models as receivers\n")
    add(
        "The three receivers of the plan were taken by hand. Here a rule takes them: every model of the plateau that "
        "settles to a stable state, holds a species constant that it takes up, and with 10 % more of that species "
        f"takes up {100 * float(study['uptake_rises_by']):g} % more or more of it while a species of the model moves by "
        f"{100 * float(study['state_moves_by']):g} % or more. {len(found)} (model, species) pass, "
        f"{len({row['biomodel_id'] for row in found})} models. Each gets a world in which the species is the nutrient, in "
        "the units of no model: the organism takes "
        f"{100 * float(study['consumes']):g} % of what is fed, the times of the plan are turned into the unit of time of "
        "the model, and the leak of a control is a share of what the organism takes. What it protects is the "
        "quantities common to all organisms, at the layers that it has (`particle` where the file names an ATP that "
        "a reaction forms). The rule does not ask what a model is a model of: several are an enzyme or a signalling "
        "pathway, and an organism here is a model with an environment that it consumes.\n"
    )
    _figure(add, "study_receivers.png")
    good = [row for row in worlds if row.get("status") == "ok"]
    failed = [row for row in worlds if row.get("status") != "ok"]
    xeno = [row for row in good if row["design"] == "xeno"]
    exact = [row for row in xeno if float(row["D"]) == 0.0 and truth(row.get("receiver_bit_identical"))]
    add(
        f"* **The zero holds for every model that could be given a world**: {len(exact)} of {len(xeno)} with the design "
        "`xeno` have D = 0 and every number of the receiver unchanged, bit for bit."
    )
    if failed:
        why = {}
        for row in failed:
            why.setdefault(_why(row["status"]), set()).add(f"{_short(row['biomodel'])} `{row['species']}`")
        add(
            "* No world: "
            + "; ".join(f"{', '.join(sorted(names))} ({reason})" for reason, names in why.items())
            + "."
        )
    by_layer = {}
    for row in read_rows(receivers.OUT / "layers.csv"):
        if row["kind"] == "common" and row["layer"] != "all":
            by_layer.setdefault((row["biomodel"], row["design"]), {})[row["layer"]] = number(row["D"])
    for design in dict.fromkeys(row["design"] for row in good if row["design"] != "xeno"):
        mine = [row for row in good if row["design"] == design]
        tops = {}
        for row in mine:
            layers_ = by_layer.get((row["biomodel"], design), {})
            if layers_ and max(value or 0.0 for value in layers_.values()) > 0.0:
                top = max(layers_, key=lambda layer, found=layers_: found[layer] or 0.0)
                tops[top] = tops.get(top, 0) + 1
        add(
            f"* `{design}` (the sender takes {float(mine[0]['share']):g} of what the organism takes): D from "
            f"{_span([float(row['D']) for row in mine])} over {len(mine)} worlds, beyond the tolerance in "
            f"{sum(float(row['D']) > 1.0 for row in mine)}; the receiver is lost in a run in {sum(bool(row['losses']) for row in mine)}, and "
            f"a run could not be followed to the end in {sum(bool(row['not_followed']) for row in mine)}. The layer with the largest D: "
            + ", ".join(
                f"{layer} in {count}" for layer, count in sorted(tops.items(), key=lambda item: -item[1])
            )
            + "."
        )
    designs = list(dict.fromkeys(row["design"] for row in good if row["design"] != "xeno"))
    if designs:
        from xeno_biomodels.layers import LAYERS

        floor = float(plan["measure"]["stage_tol"])  # below it a quantity counts as one that does not move
        lines, still = [], {}
        for layer in LAYERS:
            cells = []
            for design in designs:
                values = [
                    found[layer]
                    for (key, name), found in by_layer.items()
                    if name == design
                    and found.get(layer) is not None
                    and any(row["biomodel"] == key and row["design"] == design for row in good)
                ]
                moving = [value for value in values if value > floor]
                still[(layer, design)] = len(values) - len(moving)
                cells.append(
                    (
                        f"{_span(moving, 2)} ({len(values)}"
                        + (f", {len(values) - len(moving)} unmoved" if len(moving) < len(values) else "")
                        + ")"
                    )
                    if values
                    else "–"
                )
            lines.append([layer, *cells])
        add(
            "\nD of the quantity common to all, over the worlds (in brackets: how many worlds have the layer, and in how "
            f"many of them the quantity does not move, D below {floor:g}):\n\n"
            + _table(["layer", *designs], lines)
            + "\n"
        )
        spans = {}
        for layer in LAYERS:
            values = [
                found[layer]
                for (key, name), found in by_layer.items()
                if name == designs[0] and (found.get(layer) or 0.0) > floor
            ]
            if len(values) > 1:
                spans[layer] = max(values) / min(values)
        if spans:
            tight = min(spans, key=spans.get)
            wide = max(spans, key=spans.get)
            add(
                f"* **The models differ least at `{tight}` and most at `{wide}`.** Over the worlds of `{designs[0]}` the largest D "
                "of a layer over the smallest (among the worlds in which the quantity moves) is "
                + ", ".join(f"{_show(value, 2)} at `{layer}`" for layer, value in spans.items())
                + "."
                + (
                    " The density is what the balance of a chemostat makes of it: what the sender takes from the "
                    "supply, the receiver loses in density, whatever the model. How far the nutrient falls and what "
                    "moves inside depends on how the model reads what it takes up."
                    if tight == "population"
                    else ""
                )
            )
        unmoved = still.get(("molecule", designs[0]), 0)
        if unmoved:
            add(
                f"* In {unmoved} of the worlds the sum of the pools does not move (below {floor:g}): a model whose species are "
                "the states of a fixed total has a pool that is constant by construction, and the common quantity "
                "of that layer says nothing of it."
            )
    add("")


# ---- 7 ----------------------------------------------------------------------------------------------------------------


def _schedules(plan: dict, add) -> None:
    if not (schedules.OUT / "worlds.csv").exists():
        return
    rows = read_rows(schedules.OUT / "worlds.csv")
    by_group = read_rows(schedules.OUT / "layers.csv")
    study = plan["studies"]["schedules"]
    names = list(plan["receivers"])
    add("## 7. An activity that changes in time\n")
    add(
        "D is a supremum over what the sender may do, and the worlds of the plan try steps. Here the activity moves "
        f"between {float(study['low']):g} and {float(study['high']):g}: as a step to either, as a square wave with a period of "
        f"{', '.join(f'{float(period):g}' for period in study['periods'])} min starting high or starting low, and as the switching "
        "between the two that the step responses say leaves most at the end of the run, for each layer (the linear "
        "picture; it is then run in the world as it is). Every run is measured alone against the baseline.\n"
    )
    _figure(add, "study_schedules.png")
    good = [row for row in rows if row.get("status") == "ok"]
    xeno = [row for row in good if row["design"] == "xeno"]
    add(
        f"* **No time course reaches a receiver that shares no species with the sender**: under every one of the "
        f"{len(xeno)} schedules of the design `xeno`, D = 0 and every number of the receiver is that of the baseline in "
        f"{sum(float(row['D']) == 0.0 and truth(row['receiver_bit_identical']) for row in xeno)}."
    )
    over_wave, over_step, slow_over_fast = [], [], {}
    for biomodel in names:
        mine = [row for row in good if row["biomodel"] == biomodel and row["design"] == CONTROL]
        steps = [row for row in mine if row["schedule_kind"] == "step"]
        wave = [row for row in mine if row["schedule_kind"] == "wave"]
        worst = [
            row
            for row in rows
            if row["biomodel"] == biomodel
            and row["design"] == CONTROL
            and row["schedule_kind"] == "worst case"
        ]
        if not steps or not wave:
            continue
        step = max(float(row["D"]) for row in steps)
        top = max(wave, key=lambda row: float(row["D"]))
        text = (
            f"* {_short(biomodel)}, `{CONTROL}`: the larger of the two steps gives D = {_show(step)}; the largest of the waves "
            f"{_show(float(top['D']))} (period {float(top['period']):g} min, starting {top['starts']}, at {top['D_quantity'].replace('B.', '')})."
        )
        layers_ = {}
        for row in by_group:
            if (
                (row["biomodel"], row["design"]) == (biomodel, CONTROL)
                and row["kind"] == "common"
                and row["schedule_kind"] == "wave"
                and number(row.get("D")) is not None
            ):
                found = layers_.setdefault(row["layer"], {})
                found[float(row["period"])] = max(found.get(float(row["period"]), 0.0), float(row["D"]))
        if layers_:
            text += (
                " Fastest against slowest wave, D of the layer: "
                + ", ".join(
                    f"{layer} {_show(found[min(found)])} against {_show(found[max(found)])}"
                    for layer, found in layers_.items()
                    if layer in ("medium", "molecule", "population")
                )
                + "."
            )
        ran = [row for row in worst if row.get("status") == "ok"]
        same = [row for row in worst if row.get("status", "").startswith("the step itself")]
        if ran:
            best = max(ran, key=lambda row: float(row["D"]))
            over_step.append(float(best["D"]) / step)
            switches = _span([float(row["switches"]) for row in ran], 1)
            text += (
                f" The worst case: for {len(same)} of {len(worst)} layers the response to a pulse keeps its sign and the step is "
                f"the worst; for the others a switching was run ({switches} {'switch' if switches == '1' else 'switches'}), and the largest D "
                f"of those runs is {_show(float(best['D']))}, {_show(float(best['D']) / step, 3)} times the larger step (the linear picture "
                f"promised {_span([float(row['predicted_gain_over_the_step']) for row in ran], 3)} times)."
            )
        elif same:
            text += f" The worst case: for all {len(same)} layers the response to a pulse keeps its sign, so within these bounds the step is the worst that the linear picture knows."
        over_wave.append(float(top["D"]) / step)
        for layer, found in layers_.items():
            if min(found) != max(found) and found[max(found)] > 0.0:
                slow_over_fast.setdefault(layer, []).append(
                    found[max(found)] / max(found[min(found)], 1e-300)
                )
        add(text)
    if over_wave:
        add(
            f"* **The steps of the plan are not the supremum.** Within the same two bounds the largest wave gives "
            f"{_span(over_wave, 3)} times D of the larger step"
            + (
                f", and the switching that the step responses point to {_span(over_step, 3)} times"
                if over_step
                else ""
            )
            + ". Where the answer to a step comes back within the run, what moves a quantity most is not a step "
            "from the baseline but the swing from one bound to the other, after the receiver has settled at the "
            "first; where it does not come back, the step is close to the worst. D of the plan, a supremum over "
            "steps, is a lower bound of the D of the definition, by a factor of this size for activities within "
            "these bounds."
        )
    if slow_over_fast:
        others = [
            value for layer, values in slow_over_fast.items() if layer != "population" for value in values
        ]
        density = slow_over_fast.get("population", [])
        add(
            "* **Which layer feels an activity depends on how fast the activity moves.** D of a layer under the "
            "slowest wave over D under the fastest: "
            + ", ".join(
                f"{layer} {_span(values, 2)}"
                for layer, values in slow_over_fast.items()
                if layer in ("medium", "molecule", "complex", "population")
            )
            + ". The mean of a wave is the baseline, so what is left of a fast wave is its ripple."
            + (
                " The density, which adds up what the growth rate loses, keeps the least of it: in every receiver a "
                "fast wave is further below a slow one there than at any other layer."
                if density and others and min(density) > max(others)
                else ""
            )
            + " Orthogonality at a layer is a statement about a class of activities, and the steps of the plan are "
            "the slow end of it."
        )
    add("")


# ---- 8 ----------------------------------------------------------------------------------------------------------------


def _comparison(plan: dict, add) -> None:
    if not (comparison.OUT / "worlds.csv").exists():
        return
    rows = [row for row in read_rows(comparison.OUT / "worlds.csv") if row.get("status") == "ok"]
    by_line = read_rows(comparison.OUT / "layers.csv")
    summary = (
        json.loads((comparison.OUT / "summary.json").read_text(encoding="utf-8"))
        if (comparison.OUT / "summary.json").exists()
        else {}
    )
    add("## 8. D beside other ways to say how much two organisms have to do with each other\n")
    add(
        "Three numbers that are in use are computed for the same worlds: the niche overlap of the two organisms "
        "(Pianka's index on the species of the medium, at rest); the linear response of the steady state to the "
        "activity (what C F⁻¹ G gives, from the rest states at activities 1 % above and below the baseline), times "
        "the size of each admissible step; and the steady state of each admissible step itself. The last two are "
        "read with the tolerances of the measure, so they are numbers of the kind of D.\n"
    )
    _figure(add, "study_comparison.png")
    lines = []
    for row in rows:
        lines.append(
            [
                _short(row["biomodel"]),
                row["design"],
                _show(float(row["niche_overlap"]), 2),
                _rest(float(row["D_linear"])),
                _rest(float(row["D_steady"])),
                f"**{_show(float(row['D']))}**",
                _show(float(row["D_end"])),
                "; ".join(
                    part
                    for part in (
                        f"in a run: {row['losses']}" if row["losses"] else "",
                        f"gone at rest: {row['receiver_gone_at_rest']}"
                        if row["receiver_gone_at_rest"]
                        else "",
                        f"no rest state found: {row['no_rest_state']}" if row.get("no_rest_state") else "",
                    )
                    if part
                )
                or "–",
            ]
        )
    add(
        _table(
            [
                "receiver",
                "design",
                "niche overlap",
                "linear response of the steady state",
                "steady states of the steps",
                "D of the runs",
                "D at the end of the runs",
                "receiver lost",
            ],
            lines,
        )
        + "\n"
    )
    control = [row for row in by_line if row["design"] == CONTROL]
    if control:
        names = list(plan["receivers"])
        lines = []
        for key in LINES:
            cells = []
            for biomodel in names:
                found = [
                    row for row in control if (row["biomodel"], row["layer"], row["kind"]) == (biomodel, *key)
                ]
                cells.append(
                    f"{_rest(number(found[0]['D_steady']), 2)} / **{_show(number(found[0]['D']), 3)}**"
                    if found
                    else "–"
                )
            if any(cell != "–" for cell in cells):
                lines.append([", ".join(key), *cells])
        add(
            f"Layer by layer in the design `{CONTROL}`: the steady states of the steps / **D of the runs**:\n\n"
            + _table(
                ["layer", *[f"{_short(name)} *{plan['receivers'][name]['organism']}*" for name in names]],
                lines,
            )
            + "\n"
        )
    steps = read_rows(comparison.OUT / "steps.csv") if (comparison.OUT / "steps.csv").exists() else []
    controls = [row for row in rows if float(row["leak"]) > 0.0]
    if controls:
        floor = float(plan["measure"]["stage_tol"])
        # the controls in which the receiver is there at the rest state of every admissible step, and the others
        whole = [row for row in controls if not row["receiver_gone_at_rest"] and not row.get("no_rest_state")]
        gone = [row for row in controls if row not in whole]
        inside = {}
        for row in controls:
            mine = [
                line
                for line in by_line
                if (line["biomodel"], line["design"]) == (row["biomodel"], row["design"])
                and line["layer"] != "population"
            ]
            inside[(row["biomodel"], row["design"])] = max(
                (number(line["D_steady"]) or 0.0 for line in mine), default=0.0
            )
        still = [row for row in whole if inside[(row["biomodel"], row["design"])] <= floor]
        if whole:
            above = [
                float(row["D"]) / float(row["D_steady"]) for row in whole if float(row["D_steady"]) > 0.0
            ]
            add(
                f"* **A comparison of steady states sees the density and {'nothing' if len(still) == len(whole) else 'little'} "
                f"else.** In {len(whole)} of the {len(controls)} controls the receiver is there at the rest state of every "
                f"admissible step, and in {len(still)} of these every protected quantity but its density is then where it "
                f"was (within {floor:g} of its tolerance). In a chemostat the nutrient at rest is where the growth law puts "
                "it, whatever else takes it; what is inside the organism goes back with it, and only the density pays. "
                "Read on steady states, those controls are orthogonal at every layer but the population. The runs say "
                f"what the receiver went through on the way: D of the runs is {_span(above, 3)} times the reading on "
                "steady states."
            )
            agree = [
                float(row["D_linear"]) / float(row["D_steady"])
                for row in whole
                if float(row["D_steady"]) > 0.0
            ]
            add(
                f"* In the same {len(whole)} controls the linear response is {_span(agree, 3)} times the reading on the "
                "steady states themselves: the difference is what a step of this size does beyond the first order, "
                "at rest."
            )
        if gone:
            kept = [row for row in gone if not row["losses"]]
            add(
                f"* **Where a step ends in the loss of the receiver, the steady states say so before the runs do.** In "
                f"{len(gone)} of the {len(controls)} controls the rest state of an admissible step has no receiver in it, or "
                "the SBML model cannot be followed to it: "
                + "; ".join(
                    f"{_short(row['biomodel'])} `{row['design']}` ("
                    + ", ".join(
                        part
                        for part in (
                            f"gone: {row['receiver_gone_at_rest']}" if row["receiver_gone_at_rest"] else "",
                            f"not followed: {row['no_rest_state']}" if row.get("no_rest_state") else "",
                        )
                        if part
                    )
                    + ")"
                    for row in gone
                )
                + f". In {len(kept)} of them no run of the plan, ten residence times long, has lost the receiver yet: "
                "what a run reports depends on how long it is followed, and D of these controls is a reading on the "
                "way to a loss."
            )
        slow = [
            f"{_short(row['biomodel'])} `{row['design']}` {row['step']}"
            for row in steps
            if row.get("status") == "ok"
            and truth(row["admissible"])
            and truth(row["receiver_there_at_rest"])
            and float(row["rest_moved"]) > 1e3 * float(plan["time"]["rest_tolerance"])
        ]
        if slow:
            add(
                "* The search for a rest state had not settled when it ended for these admissible steps, and their "
                "steady states are where the search stopped: " + "; ".join(slow) + "."
            )
        lost = [row for row in controls if row["losses"] and not row["receiver_gone_at_rest"]]
        if lost:
            add(
                "* A receiver that loses its function for a while and has it again at rest is not lost for a steady state: "
                + "; ".join(f"{_short(row['biomodel'])} `{row['design']}` ({row['losses']})" for row in lost)
                + "."
            )
        add(
            f"* The niche overlap of the controls is {_span([float(row['niche_overlap']) for row in controls], 2)}: it counts what share "
            "of its uptake the sender has in common with the receiver, not what that does to the receiver, and has "
            "no direction and no layers."
        )
    for design, found in (summary.get("community") or {}).items():
        add(
            f"* Common vessel, `{design}`: of {found['ordered_pairs']} ordered pairs, {found['overlap_zero_and_moved_by_removal']} "
            f"have a niche overlap of zero and are moved when the sender is taken away ({found['overlap_zero_and_moved_by_a_step']} "
            f"by a step of a xeno), and {found['overlap_above_zero_and_not_moved']} overlap and are not moved; the rank "
            f"correlation of the overlap with D_removal is {_show(found['rank_correlation_overlap_D_removal'], 2)}."
        )
    apart = sum(
        found["overlap_zero_and_moved_by_removal"] for found in (summary.get("community") or {}).values()
    )
    if apart:
        add(
            "* **An overlap of zero is not orthogonality.** Two organisms that take no species in common are moved "
            "by each other when a third takes something of each: the overlap is a property of two organisms, D of "
            "two organisms in a vessel. Where no organism stands between two others (the designs in which every "
            "pair with an overlap of zero is unmoved), the overlap and D_removal rank the pairs alike."
        )
    for design, found in (summary.get("receivers") or {}).items():
        rises = found["rank_correlation_D_uptake_rises_by"]
        add(
            f"* Receivers taken by rule, `{design}` ({found['worlds']} worlds): the rank correlation of D with how much a species "
            f"of the model moves at its steady state for 10 % more nutrient is {_show(found['rank_correlation_D_state_moves_by'], 2)}, and with "
            f"how much its uptake rises {_show(rises, 2)}."
            + (
                " The less the uptake of a model answers to its nutrient, the further the nutrient has to fall "
                "before the model gives way to the sender: what the screen measured at the steady state says "
                "something of D, and with the other sign than one might expect."
                if rises is not None and rises < -0.5
                else ""
            )
        )
    add("")


def write(plan: dict, out: Path | None = None) -> list[Path]:
    made = study_figures.write(plan)
    text: list[str] = []
    add = text.append
    add("# What stands behind the readings\n")
    add(
        "Written by `xvb figures` from the tables under `runs/studies/`, which `xvb studies` writes. The readings "
        "themselves are in [`report.md`](report.md) (each SBML organism with its xeno) and in "
        "[`community.md`](community.md) (the pairs in one vessel). Here each of them is put against what it could "
        "depend on: the assumptions of the world, the choices of the measure, the numerical method, the way by which "
        "the interference enters, the layers themselves, the SBML models, the kind of activity, and other indices.\n"
    )
    for section in (
        _sensitivity,
        _settings,
        _numerics,
        _channels,
        _cascade,
        _receivers,
        _schedules,
        _comparison,
    ):
        section(plan, add)
    add("## What the studies do not cover\n")
    add(
        "* **One thing at a time.** Each assumption and each choice is changed alone, from the plan. What two of "
        "them do together is not run."
    )
    add(
        "* **The laws of the world are varied, not replaced.** The receiver grows by its uptake with or without a "
        "maintenance, or does not grow; a world with another kind of supply (a batch, a biofilm) is another study."
    )
    add(
        "* **Activities within two bounds.** The waves and the switchings move between two activities; the worst "
        "case is the worst of the linear picture, run in the world as it is. A supremum over all that a sender "
        "could do in time is not computed."
    )
    add(
        "* **The receivers taken by rule are read on the common quantities only**, and the rule reads names: which "
        "species of a model are carriers is taken from their identifiers, and no species counts as a state of a "
        "protein."
    )
    add(
        "* **No measurement of an organism enters anywhere.** Every number here is a property of SBML models in "
        "assumed worlds."
    )
    add("")
    target = Path(out or paths.RUNS / "studies.md")
    target.write_text("\n".join(text) + "\n", encoding="utf-8")
    return [*made, target]
