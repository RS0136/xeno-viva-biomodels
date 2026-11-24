"""``runs/community.md``: what happens when the pairs share one vessel, written from the files under ``runs/community``.

As in ``report``, every number is read from a file that a command wrote, and a sentence that depends on a
result is written from that result.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from xeno_biomodels import layers, paths
from xeno_biomodels.orthogonality import read_series
from xeno_biomodels.report import _number, _table


def chain(members: dict, sender: str, receiver: str) -> list[str] | None:
    """The shortest way by which a sender can reach a receiver through the medium, as organism, species,
    organism, ...; None when there is none.

    A step goes from an organism to another through a species that the first writes and the second reads: only
    then does the state of the second move, and with it what the second writes. The last step may also end on a
    species that the receiver merely protects (its own product), since the measure reads that too.
    """
    ways, front = {sender: [sender]}, [sender]
    while front:
        following = []
        for name in front:
            last = sorted(set(members[name]["writes"]) & set(members[receiver]["touches"]))
            if name != receiver and last:
                # prefer a species that the receiver reads over one that it only protects
                read = [key for key in last if key in members[receiver]["reads"]]
                return [*ways[name], (read or last)[0], receiver]
            for other, body in members.items():
                common = sorted(set(members[name]["writes"]) & set(body["reads"]))
                if other not in ways and common:
                    ways[other] = [*ways[name], common[0], other]
                    following.append(other)
        front = following
    return None


def _matrix(rows: dict, senders: list[str], receivers: list[str], key: str) -> str:
    lines = []
    for sender in senders:
        cells = []
        for receiver in receivers:
            row = rows.get((sender, receiver))
            if sender == receiver:
                cells.append("·")
            elif row is None or row.get("status") != "ok" or row[key] in ("", "None"):
                cells.append("n/a")
            else:
                lost = bool(row["losses"]) if key == "D" else row["removal_lost_at"] not in ("", "None")
                text = _number(row[key]) + (" (lost)" if lost else "")
                cells.append(f"**{text}**" if row["same_pair"] == "True" else text)
        lines.append([f"**{sender}**", *cells])
    return _table(["sender \\ receiver", *receivers], lines)


def _layers(plan: dict, add, designs: list[str], order: list[str], xenos: list[str]) -> None:
    """Where the interference of a xeno lands in an SBML organism, layer by layer and by kind of quantity."""
    path = paths.RUNS / "community" / "layers.csv"
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["sender_kind"] == "xeno"
            and row["receiver_kind"] == "sbml"
            and row["D"] not in ("", "None")
        ]
    lines = [(layer, kind) for layer in layers.LAYERS for kind in layers.KINDS]
    for design in designs:
        mine = {
            (row["sender"], row["receiver"], row["layer"], row["kind"]): row
            for row in rows
            if row["design"] == design
        }
        if not any(float(row["D"]) > 0.0 for row in mine.values()):
            continue
        table, tops, apart = [], {}, []
        for receiver in order:
            for sender in xenos:
                found = {
                    key: float(mine[(sender, receiver, *key)]["D"])
                    for key in lines
                    if (sender, receiver, *key) in mine
                }
                if not found or max(found.values()) == 0.0:
                    continue
                top = max(found, key=found.get)
                tops[top] = tops.get(top, 0) + 1
                beyond = [", ".join(key) for key, value in found.items() if value > 1.0]
                common, specific = (
                    mine.get((sender, receiver, "all", "common")),
                    mine.get((sender, receiver, "all", "specific")),
                )
                if common and specific and float(common["D"]) <= 1.0 < float(specific["D"]):
                    apart.append(
                        f"{sender} to {receiver} (common {_number(common['D'])}, specific {_number(specific['D'])})"
                    )
                table.append(
                    [
                        f"{sender} to {receiver}"
                        + (" (its partner)" if mine[(sender, receiver, *top)]["same_pair"] == "True" else ""),
                        _number(str(found[top])),
                        ", ".join(top),
                        "; ".join(beyond) or "none",
                        _number(common["D"]) if common else "–",
                        _number(specific["D"]) if specific else "–",
                    ]
                )
        add(
            f"  * `{design}`, layer by layer (a receiver protects what it protects alone, `report.md` section 4):\n\n"
            + _table(
                [
                    "xeno to SBML organism",
                    "D",
                    "largest at",
                    "layers beyond their tolerance",
                    "D of the common quantities",
                    "D of the specific ones",
                ],
                table,
            )
            + "\n"
        )
        add(
            "    The layer at which D is largest: "
            + ", ".join(
                f"{', '.join(key)} in {count} of {len(table)}"
                for key, count in sorted(tops.items(), key=lambda item: -item[1])
            )
            + "."
            + (
                " Within tolerance on the quantities that every organism has, and beyond it on those of its own: "
                + "; ".join(apart)
                + "."
                if apart
                else ""
            )
        )


def write(plan: dict, out: Path | None = None) -> Path:
    root = paths.RUNS / "community"
    settings = plan["community"]
    designs = list(settings["designs"])
    with (root / "summary.csv").open(encoding="utf-8", newline="") as handle:
        every = list(csv.DictReader(handle))
    infos = {
        design: json.loads((root / design / "community.json").read_text(encoding="utf-8"))
        for design in designs
    }
    first = infos[designs[0]]
    order = [name for name, body in first["members"].items() if body["kind"] == "sbml"]
    xenos = [name for name, body in first["members"].items() if body["kind"] == "xeno"]
    names = [*order, *xenos]
    k = float(plan["supply"]["dilution"])
    text: list[str] = []
    add = text.append

    add("# Three pairs in one vessel\n")
    add(
        "Written by `xvb figures` from the files under `runs/community/`. The plan is `configs/experiment.yaml` "
        "(section `community`); time is in minutes, concentrations in mM, a density is a volume of cells per volume "
        "of medium. The single pairs are in [`report.md`](report.md).\n"
    )

    # ---- 1. how it is made ------------------------------------------------------------------------------------------
    add("## 1. How the community is made\n")
    add(
        f"A pair is an SBML organism and its xeno. Each of the {len(order)} pairs is first brought to rest in a vessel "
        "of its own, as in the report of the single pairs (design `xeno`: D = 0 within every pair). The vessels are "
        "then poured together in equal volumes. That fixes the community without a new number: every density and "
        "every concentration becomes the mean over the vessels, the feed the mean of the feeds, and the dilution "
        "rate stays. A species that two SBML models hold is one species of the common medium "
        "(`same_chemical` in the plan).\n"
    )
    lines = []
    for name in order:
        body = first["members"][name]
        partner = first["members"][body["partner"]]
        receiver = plan["receivers"][body["biomodel"]]
        lines.append(
            [
                f"**{name}**",
                f"{body['biomodel']}: *{receiver['organism']}*, {receiver['what']}",
                f"`{body['nutrient']}`"
                + (
                    f" (`{receiver['nutrient']}` in the model)"
                    if receiver["nutrient"] != body["nutrient"]
                    else ""
                ),
                f"**{body['partner']}**",
                _number(str(first["poured"]["organisms"][name]["density"])),
                _number(str(first["poured"]["organisms"][body["partner"]]["density"])),
            ]
        )
        del partner
    add(
        _table(
            [
                "organism",
                "SBML model",
                "lives on",
                "its xeno",
                "density at the pouring",
                "density of its xeno",
            ],
            lines,
        )
        + "\n"
    )
    shared = {}
    for key, members in (settings.get("same_chemical") or {}).items():
        shared[key] = ", ".join(
            f"`{item['species']}` of {biomodel} ({item['per_mM']:g} per mM)"
            for biomodel, item in members.items()
        )
    if shared:
        add(
            "One chemical under two names: "
            + "; ".join(f"`{key}` is {value}" for key, value in shared.items())
            + ".\n"
        )
    lines = []
    for key in first["poured"]["medium"]:
        vessels = [_number(str(first["vessels"][name]["medium"].get(key, 0.0)), 4) for name in order]
        feeds = [_number(str(first["vessels"][name]["feed"].get(key, 0.0)), 4) for name in order]
        lines.append(
            [
                f"`{key}`",
                " / ".join(vessels),
                _number(str(first["poured"]["medium"][key]), 4),
                " / ".join(feeds),
                _number(str(first["feed"][key]), 4),
            ]
        )
    add(
        f"The medium and the feed of the design `{designs[0]}`, vessel by vessel ({' / '.join(order)}) and after the pouring:\n\n"
        + _table(
            [
                "species",
                "level in the vessels",
                "level after the pouring",
                "feed of the vessels",
                "merged feed",
            ],
            lines,
        )
        + "\n"
    )

    # ---- 2. the designs ---------------------------------------------------------------------------------------------
    add("## 2. Three ways for the xenos to stand to the other pairs\n")
    add(
        "In every design each xeno takes nothing of its own partner, so every pair alone is exactly orthogonal. The "
        "designs differ in what a xeno has in common with the organisms of the other pairs.\n"
    )
    lines = []
    for design in designs:
        members = infos[design]["members"]
        touched = "; ".join(
            f"{name}: `{members[name]['substrate']}`"
            + (
                " and "
                + ", ".join(f"`{key}` at {value:g}" for key, value in members[name]["also_takes"].items())
                if members[name]["also_takes"]
                else ""
            )
            for name in xenos
        )
        lines.append(
            [f"`{design}`", infos[design]["xenos_on"], _number(str(infos[design]["cross"])), touched]
        )
    add(
        _table(
            [
                "design",
                "the xenos live on",
                "specificity for the nutrients of the other pairs' SBML organisms",
                "what each xeno takes",
            ],
            lines,
        )
        + "\n"
    )

    # ---- 3. the fate ------------------------------------------------------------------------------------------------
    add("## 3. With nothing stepped: who stays\n")
    add("![community](figures/community.png)\n")
    hours = float(settings["fate"]) / 60.0
    lines, stays, falls = [], {}, {}
    for design in designs:
        times, columns = read_series(root / design / "fate" / "series.csv")
        row = [f"`{design}`"]
        for name in names:
            density = columns[f"{name}.density"]
            ratio = float(density[-1] / density[0])
            # the rate at which the density still changes at the end, per residence time
            slope = float(np.log(density[-1] / density[-11]) / (times[-1] - times[-11]) / k)
            row.append(f"{_number(str(ratio))} ({slope:+.3f})")
            (falls if slope < -0.005 else stays).setdefault(design, []).append(name)
        row.append(
            " ".join(
                f"`{key}` {_number(str(columns['medium.' + key][-1]), 4)}"
                for key in sorted({infos[design]["members"][name]["nutrient"] for name in order})
            )
        )
        lines.append(row)
    add(
        f"Density after {hours:g} h over the density at the pouring, and in brackets how fast it still changes then "
        "(per residence time; 0 is rest, -1 is an organism that does not grow at all):\n\n"
        + _table(["design", *names, "nutrients at the end (mM)"], lines)
        + "\n"
    )
    add("What the runs show:\n")
    for design in designs:
        add(
            f"* `{design}`: still falling at the end: {', '.join(falls.get(design, [])) or 'no organism'}; "
            f"the others are at rest or growing."
        )
    takers: dict[str, list[str]] = {}
    for name in order:
        takers.setdefault(first["members"][name]["nutrient"], []).append(name)
    for key, group in takers.items():
        if len(group) > 1:
            levels = ", ".join(
                f"{name} at {_number(str(first['vessels'][name]['medium'][key]), 4)} mM" for name in group
            )
            add(
                f"* {' and '.join(group)} live on the same `{key}`, and in one vessel only one level of it can hold. "
                f"Each grows at the dilution rate at the level of its own vessel ({levels}); the one that does so at "
                "the lower level draws the nutrient down to it and the other is washed out. **Which of the two that "
                "is was decided by the levels at which the two SBML files hold the nutrient**, through the yield "
                "that each organism was given (growth at the dilution rate at the steady state of its file). It is "
                "not a statement about the two species."
            )
    add("")

    # ---- 4. who moves whom ------------------------------------------------------------------------------------------
    add("## 4. Who moves whom\n")
    steps = ", ".join(_number(str(value)) for value in plan["activities"]["steps"])
    add(
        f"The reading starts {float(settings['assembly']):g} min after the pouring, when the fast changes are over "
        f"(the slow ones of section 3 go on, so the baseline is not at rest). From that state: the baseline; each "
        f"xeno with its activity stepped to {steps}; each of the {len(names)} organisms taken away. Every run is "
        f"followed for {float(plan['time']['horizon']):g} min. The measure is the one of the single pairs, applied to "
        "every ordered pair of organisms: D from the steps of a xeno (an SBML organism has no knob), D_removal from "
        "taking the sender away. A receiver protects what it protects alone (`protect` in the plan): the quantities "
        "common to every organism, layer by layer, and those characteristic of it; its function is its growth "
        "rate, at half its value at the start or more. D of every layer and of each kind is in "
        "`runs/community/layers.csv`. Bold: the two organisms are a pair.\n"
    )
    add("![matrix](figures/community_matrix.png)\n")
    if (paths.RUNS / "figures" / "community_layers.png").exists():
        add("![layers](figures/community_layers.png)\n")
    by_design = {
        design: {(row["sender"], row["receiver"]): row for row in every if row["design"] == design}
        for design in designs
    }
    for design in designs:
        rows = by_design[design]
        add(f"### `{design}`\n")
        add("D, the activity of the sender stepped:\n\n" + _matrix(rows, xenos, names, "D") + "\n")
        add("D_removal, the sender taken away:\n\n" + _matrix(rows, names, names, "D_removal") + "\n")

    add("### What the runs show\n")
    for design in designs:
        rows = by_design[design]
        good = [row for row in rows.values() if row.get("status") == "ok"]
        exact = [row for row in good if row["receiver_bit_identical"] == "True"]
        members = infos[design]["members"]
        without = [row for row in good if chain(members, row["sender"], row["receiver"]) is None]
        agree = {(row["sender"], row["receiver"]) for row in exact} == {
            (row["sender"], row["receiver"]) for row in without
        }
        across = [row for row in good if row["sender_kind"] == "xeno" and row["receiver_kind"] == "sbml"]
        within = [row for row in across if row["same_pair"] == "True"]
        largest = max(within, key=lambda row: float(row["D"]))
        if not agree:
            structure = (
                "These are not exactly the ordered pairs without a way through the medium: see `matrix.csv`. "
            )
        elif exact:
            structure = (
                "These are exactly the ordered pairs in which the sender has no way to the receiver through the "
                "medium (no chain of species that one organism writes and the next reads). "
            )
        else:
            structure = "And there is no ordered pair in which the sender has no way to the receiver through the medium. "
        add(
            f"* **`{design}`**: in {len(exact)} of the {len(good)} ordered pairs every stored number of the receiver "
            "is the same in every run of the sender, bit for bit. "
            + structure
            + f"A xeno against an SBML organism: D from {_number(min(across, key=lambda row: float(row['D']))['D'])} to "
            f"{_number(max(across, key=lambda row: float(row['D']))['D'])}; against its own partner: "
            + ", ".join(f"{row['sender']} to {row['receiver']} {_number(row['D'])}" for row in within)
            + "."
        )
        if float(largest["D"]) > 0.0:
            path = chain(members, largest["sender"], largest["receiver"])
            run = f"{largest['sender']}.{largest['D_activity']}"
            _, base = read_series(root / design / "baseline" / "series.csv")
            times, moved = read_series(root / design / run / "series.csv")
            trail = []
            for item in path[1:-1]:
                column = f"medium.{item}" if f"medium.{item}" in base else f"{item}.density"
                change = moved[column] / base[column] - 1.0
                at = int(np.nanargmax(np.abs(change)))
                trail.append(
                    f"{'`' + item + '`' if column.startswith('medium.') else 'the density of ' + item} by {100 * float(change[at]):+.0f} % (largest after {times[at] / 60:.0f} h)"
                )
            add(
                f"  * The pair that is orthogonal alone is not so here. {largest['sender']} takes "
                + ", ".join(f"`{key}`" for key in members[largest["sender"]]["reads"])
                + f" and {largest['receiver']} reads `{members[largest['receiver']]['nutrient']}`: they share no species. "
                f"The shortest way from one to the other is {' → '.join(path)}. In the run that carries D ({run}, "
                f"by {largest['D_quantity']}), what lies on that way moves against the baseline: "
                + "; ".join(trail)
                + "."
            )
    _layers(plan, add, designs, order, xenos)
    sbml_pairs = [(a, b) for a in order for b in order if a != b]
    moved = {
        design: [
            f"{a} to {b} {_number(by_design[design][(a, b)]['D_removal'])}"
            for a, b in sbml_pairs
            if by_design[design][(a, b)].get("status") == "ok"
            and float(by_design[design][(a, b)]["D_removal"]) > 1.0
        ]
        for design in designs
    }
    add(
        "* Between the SBML organisms themselves, taking one away moves another beyond its tolerance in: "
        + "; ".join(f"`{design}`: {', '.join(found) or 'no ordered pair'}" for design, found in moved.items())
        + "."
    )
    add("")

    # ---- 5. limits --------------------------------------------------------------------------------------------------
    add("## 5. What this does not show\n")
    add(
        "* **Everything of the single pairs holds here too**: the worlds are assumed, an SBML organism can be "
        "reached only through its one nutrient, and the numbers of the designs that are not exact zeros depend on "
        "every assumed value (section 5 of `report.md`)."
    )
    add(
        "* **Who wins among the SBML organisms is an outcome of the calibration**, not of biology (section 3). The "
        "yields were fixed so that each organism is at rest in its own vessel, at the nutrient level of its own SBML file."
    )
    add(
        "* **Pouring in equal volumes is one way to put pairs together.** Other proportions, another feed or "
        "another dilution rate give another community. The exact zeros do not depend on it; everything else does."
    )
    add(
        "* **The baseline of the reading is not at rest.** The slow changes of section 3 go on under every run, and "
        "D compares each run with a baseline that moves. A zero is exact all the same; a number that is not zero "
        "is a change relative to that moving baseline, over the horizon and no longer."
    )
    cross = [design for design in designs if infos[design]["cross"] > 0.0]
    if cross:
        add(
            f"* **`{'`, `'.join(cross)}` is an example, not a finding about xenos.** That a xeno which is orthogonal to its "
            "partner takes a hundredth of what its transporter can carry of the nutrients of other organisms is assumed, "
            "to show what follows if it does."
        )
    add(
        "* **The SBML organisms have no knob**, so their effect on others is read by removal only, and D is given "
        "for the xenos alone."
    )
    lost = {design: infos[design]["not_followed"] for design in designs if infos[design]["not_followed"]}
    if lost:
        add(
            "* **Runs that could not be followed** are left out of the reading: "
            + "; ".join(
                f"`{design}`: "
                + ", ".join(f"{name} ({item['problem'].split(':')[0]})" for name, item in runs.items())
                for design, runs in lost.items()
            )
            + "."
        )
    add("")
    target = Path(out or paths.RUNS / "community.md")
    target.write_text("\n".join(text) + "\n", encoding="utf-8")
    return target
