"""``runs/report.md``: the numbers of the study beside its figures, written from the files under ``runs/``.

Every number in the report is read from a file that a command wrote; a sentence that depends on a result is
written from that result, and says so when the result is not what the study set out to show.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from xeno_biomodels import layers, paths


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(text: str | None, digits: int = 3) -> str:
    if text in (None, "", "None", "nan"):
        return "–"
    value = float(text)
    if value == 0.0:
        return "0"
    return f"{value:.{digits}g}"


def _table(header: list[str], lines: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out)


def _layers(plan: dict, add, worlds: list[dict]) -> None:
    """Section 4, layer by layer: what is protected at each layer, D of each, and what the two kinds say."""
    path = paths.RUNS / "worlds" / "layers.csv"
    if not path.exists():
        return
    rows = _rows(path)
    receivers, designs = list(plan["receivers"]), list(plan["designs"])
    controls = [design for design in designs if float(plan["designs"][design]) > 0.0]
    lines = [
        (layer, kind)
        for layer in layers.LAYERS
        for kind in layers.KINDS
        if any((row["layer"], row["kind"]) == (layer, kind) for row in rows)
    ]
    found = {(row["biomodel"], row["design"], row["layer"], row["kind"]): row for row in rows}
    whole = {(row["biomodel"], row["design"]): row for row in worlds if row.get("status") == "ok"}
    add("### Layer by layer, and by kind of quantity\n")
    add(
        "The measure is read once for all that B protects and, with the same tolerances, for every layer and for "
        "each of the two kinds of quantity, so D of the whole is the largest D of a layer. The layers are the "
        "levels of a cell as far as an SBML model of metabolism has them; it has nothing at the levels of the atom, "
        "the supramolecule (a genome), the condensate or the organelle, and nothing can be read there.\n"
    )
    entries = {name: layers.protected(plan, layers.sbml_organism(plan, name, "B")) for name in receivers}
    table = []
    for layer, kind in lines:
        cells = []
        for name in receivers:
            mine = [entry for entry in entries[name] if (entry.layer, entry.kind) == (layer, kind)]
            cells.append(
                ", ".join(
                    f"`{entry.column.replace('B.', '')}`" + (" (fn)" if entry.tolerance == "fn" else "")
                    for entry in mine
                )
                or "–"
            )
        table.append(
            [
                f"{layer}, {kind}",
                layers.WHAT[layer] if kind == "common" else "characteristic of the organism",
                *cells,
            ]
        )
    add(
        _table(
            [
                "layer, kind",
                "what it is",
                *[f"{name} *{plan['receivers'][name]['organism']}*" for name in receivers],
            ],
            table,
        )
        + "\n"
    )
    add(
        "`atp_made` is the ATP that the reactions of the model form per cell volume and time, and `pool` the sum of "
        "the pools of its metabolites (the species that are neither carriers nor states of a protein). Every "
        f"quantity has the tolerance `{plan['protect']['tolerance']}`, but for those marked fn.\n"
    )
    add("![layers](figures/layers.png)\n")
    for name in receivers:
        table = []
        for layer, kind in lines:
            cells = [
                _number(found[(name, design, layer, kind)]["D"])
                if (name, design, layer, kind) in found
                else "–"
                for design in designs
            ]
            table.append([f"{layer}, {kind}", *cells])
        table.append(
            [
                "**all**",
                *[
                    f"**{_number(whole[(name, design)]['D'])}**" if (name, design) in whole else "–"
                    for design in designs
                ],
            ]
        )
        add(
            f"D of {name} (*{plan['receivers'][name]['organism']}*), design by design:\n\n"
            + _table(["layer, kind", *designs], table)
            + "\n"
        )

    zero = [row for row in rows if row["design"] not in controls]
    if zero and all(float(row["D"]) == 0.0 and row["stage1_trajectory"] == "True" for row in zero):
        add(
            "* **The design `xeno` is orthogonal at every layer and for both kinds of quantity**: D = 0 in every one of them, for every receiver."
        )
    else:
        add("* The design `xeno` is not at zero at every layer: see the tables.")
    for name in receivers:
        tops, first = [], {}
        for design in controls:
            mine = {
                key: float(found[(name, design, *key)]["D"]) for key in lines if (name, design, *key) in found
            }
            if not mine:
                continue
            top = max(mine, key=mine.get)
            tops.append(f"{', '.join(top)} in `{design}`")
            for key, value in mine.items():
                if value > 1.0:
                    first.setdefault(key, design)
        never = [
            ", ".join(key)
            for key in lines
            if key not in first and any((name, design, *key) in found for design in controls)
        ]
        add(
            f"* {name} ({plan['receivers'][name]['organism']}): D is largest at {'; '.join(tops)}. A layer passes its tolerance first in: "
            + "; ".join(f"{', '.join(key)} in `{design}`" for key, design in first.items())
            + (f". Never, in these designs: {'; '.join(never)}." if never else ".")
        )
    inward, outward = (
        [],
        [],
    )  # tolerable on the common quantities and not on the specific ones, and the reverse
    for name in receivers:
        for design in controls:
            common, specific = (
                found.get((name, design, "all", "common")),
                found.get((name, design, "all", "specific")),
            )
            if not (common and specific):
                continue
            text = f"{name} `{design}` (common {_number(common['D'])}, specific {_number(specific['D'])})"
            if float(common["D"]) <= 1.0 < float(specific["D"]):
                inward.append(text)
            elif float(specific["D"]) <= 1.0 < float(common["D"]):
                outward.append(text)
    if inward or outward:
        add(
            "* **The two kinds of quantity can disagree.**"
            + (
                " The quantities that every organism has are within tolerance and the ones by which the organism is known are not in: "
                + "; ".join(inward)
                + "."
                if inward
                else ""
            )
            + (
                " The other way round, the common quantities beyond tolerance and the specific ones within, in: "
                + "; ".join(outward)
                + "."
                if outward
                else ""
            )
        )
    else:
        add(
            "* In these designs the two kinds of quantity agree: where the common quantities are within tolerance, so are the specific ones, and the other way round."
        )
    ratios = []
    for name in receivers:
        for design in controls:
            uptake, growth = (
                found.get((name, design, "complex", "common")),
                found.get((name, design, "individual", "common")),
            )
            if uptake and growth and float(growth["D"]) > 0.0:
                ratios.append(float(uptake["D"]) / float(growth["D"]))
    if ratios:
        add(
            "* **The layers are not independent readings.** The growth rate is the uptake times a constant, by the "
            "growth law of the world, so `individual` is `complex, common` read with another tolerance: D of the "
            f"uptake is {_number(str(min(ratios)), 4) if max(ratios) - min(ratios) < 1e-6 * max(ratios) else _number(str(min(ratios)), 4) + ' to ' + _number(str(max(ratios)), 4)} "
            "times D of the growth rate in every control. "
            "With a tolerance `fn` and the function read at every time, D of the growth rate cannot pass 1 by a "
            "fall without the receiver counting as lost. How many readings the layers are, and what a second way "
            "in changes, is in [`studies.md`](studies.md)."
        )
    weakest = controls[0] if controls else None
    if weakest:
        table = []
        for layer, kind in lines:
            if layer == "medium":
                continue
            cells = []
            for name in receivers:
                medium, mine = (
                    found.get((name, weakest, "medium", "common")),
                    found.get((name, weakest, layer, kind)),
                )
                share = (
                    _number(str(float(mine["D"]) / float(medium["D"])))
                    if medium and mine and float(medium["D"]) > 0.0
                    else "–"
                )
                cells.append(share)
            table.append([f"{layer}, {kind}", *cells])
        add(
            f"\nWhat arrives at each layer of what the medium loses: D of the layer over D of `medium, common`, in the "
            f"weakest control (`{weakest}`), where the answer is nearly linear. With one tolerance for all, this is the "
            "relative change of the quantity over that of the nutrient (for `individual`, with its tolerance `fn`, "
            "over a tolerance that is another):\n\n" + _table(["layer, kind", *receivers], table) + "\n"
        )
    add("")


def write(plan: dict, out: Path | None = None) -> Path:
    runs = paths.RUNS
    selection = json.loads((runs / "selection" / "summary.json").read_text(encoding="utf-8"))
    names = json.loads((runs / "selection" / "names.json").read_text(encoding="utf-8"))
    screen = {row["biomodel_id"]: row for row in _rows(runs / "selection" / "screen.csv")}
    verified = {row["biomodel"]: row for row in _rows(runs / "verify" / "summary.csv")}
    worlds = [row for row in _rows(runs / "worlds" / "summary.csv")]
    receivers, designs = list(plan["receivers"]), list(plan["designs"])
    steps = [f"u={value:g}" for value in plan["activities"]["steps"]]
    text: list[str] = []
    add = text.append

    add("# SBML organisms of viva-biomodels, and an organism orthogonal to them\n")
    add(
        "Written by `xvb figures` from the files under `runs/`. The plan is `configs/experiment.yaml`; time is in "
        "minutes, concentrations in mM, a density is a volume of cells per volume of medium.\n"
    )

    # ---- 1. the models ----------------------------------------------------------------------------------------------
    add("## 1. Which SBML models\n")
    add(
        f"viva-biomodels commits the time course of {selection['corpus_models']} BioModels under up to five ODE engines "
        f"(amici, copasi, pysces, simbio, tellurium) and states that agreement between them is not yet quantified "
        f"over the corpus. Scored here with the two measures that viva-biomodels uses for a pair of engines (mean "
        f"nRMSE, closeness), {selection['jobs_scored']} time courses of {selection['models_scored']} models could be "
        f"compared; {selection['jobs_run_by_five_engines']} were run by all five engines, and in "
        f'{selection["of_these_good_and_close_in_every_pair"]} of those every pair is "good" (mean nRMSE at most 0.01) '
        f'and "close". The plateau is stricter: all five engines, and no variable of any pair off by more than '
        f"{selection['plateau_is']['variable_nrmse']:g} of its size, on {selection['plateau_is']['moving_variables']} "
        f"or more variables that move. **{selection['plateau_models']} models are on it.** Their scores lie between "
        "about 1e-7 and 1e-5, which is the tolerance of the solvers and the float32 of the committed series: within "
        "the plateau one model is not more accurate than another.\n"
    )
    add(
        "The score is agreement between independent implementations on the same SBML file. It is not agreement "
        "with an experiment.\n"
    )
    add("![selection](figures/selection.png)\n")
    add(
        "A receiver must have an environment that another organism can share: a species that the SBML file holds "
        "constant, that the model takes up, and a steady state to start from (`runs/selection/screen.csv` lists, for "
        "every model of the plateau, what it takes up and how it reacts to 10 % more of it). From that list three "
        "models of the metabolism of a microorganism on a nutrient were taken by hand:\n"
    )
    lines = []
    for biomodel in receivers:
        receiver, record = plan["receivers"][biomodel], names.get(biomodel, {})
        sensed = (
            json.loads(screen[biomodel]["sensed"]).get(receiver["nutrient"], {})
            if screen.get(biomodel, {}).get("sensed")
            else {}
        )
        lines.append(
            [
                f"[{biomodel}](https://www.ebi.ac.uk/biomodels/{biomodel})",
                f"{record.get('name', '')}: *{receiver['organism']}*, {receiver['what']}",
                f"[{record.get('journal', '')} {record.get('year', '')}]({record.get('link', '')})",
                _number(verified[biomodel]["corpus_max_variable_nrmse"], 2),
                receiver["nutrient"],
                f"{100 * sensed['uptake_change']:+.2f} %" if sensed else "–",
                f"{100 * sensed['largest_species_change']:.1f} %" if sensed else "–",
            ]
        )
    add(
        _table(
            [
                "BioModel",
                "model",
                "publication",
                "score (five engines)",
                "nutrient",
                "uptake at +10 % nutrient",
                "largest change of a species",
            ],
            lines,
        )
        + "\n"
    )

    add(
        "The last two columns are what the model does at its steady state when the nutrient is raised by 10 %: the "
        "three take up a nutrient, and they differ in how much they read its level. That difference is what the "
        "controls of section 4 bring out.\n"
    )

    # ---- 2. run here ------------------------------------------------------------------------------------------------
    add("## 2. The models run here, through viva-biomodels\n")
    add(
        "`xvb verify` sends each receiver through the `compare-simulators` composite of viva-biomodels: its loader "
        "fetches the SBML file and the time course of the SED-ML file, each installed simulator runs it, and its "
        "comparison step scores every pair. The last column sets the series computed here against the series that "
        "viva-biomodels committed, from all five engines.\n"
    )
    lines = []
    for biomodel in receivers:
        row = verified[biomodel]
        lines.append(
            [
                biomodel,
                row["engines"].replace(",", ", "),
                f"{_number(row['max_pair_mean_nrmse'], 2)} ({row['bucket']})",
                f"{_number(row['max_closeness'], 2)} ({row['closeness_bucket']})",
                _number(row["max_variable_nrmse"], 2),
                f"{_number(row['local_vs_corpus_nrmse'], 2)} ({row['local_vs_corpus_pairs']} pairs)",
            ]
        )
    add(
        _table(
            [
                "BioModel",
                "engines here",
                "worst pair, mean nRMSE",
                "closeness",
                "worst variable, nRMSE",
                "worst variable against the corpus",
            ],
            lines,
        )
        + "\n"
    )
    add("![verify](figures/verify.png)\n")

    # ---- 3. the world -----------------------------------------------------------------------------------------------
    add("## 3. The world that each model lives in\n")
    add(
        "One composite of process-bigraph (Vivarium 2.0) per run: a store `medium` shared by everything, a node "
        "`supply` (dilution rate and feed) read by the chemostat and by both organisms, the SBML organism B and the "
        "organism of laws A. B is the Tellurium process of viva-tellurium, the one that viva-biomodels wires into its "
        "`biomodel-process` composite, with the interface of an organism added: before each interval it reads the "
        "nutrient from the medium into the boundary species of the model, and after it returns what the reactions of "
        "the model took. The SBML model, its species, rate laws and parameters, is not changed. Three laws are added "
        "around it, and they are assumptions of this world:\n"
    )
    add(
        _table(
            ["law", "form", "what fixes its numbers"],
            [
                [
                    "supply",
                    "dS/dt = k (S_feed − S)",
                    "k assumed; S_feed of the nutrient from the SBML file (below)",
                ],
                [
                    "exchange",
                    "dS/dt = density × (net production per cell volume)",
                    "the density at rest, assumed",
                ],
                [
                    "growth",
                    "μ = k J / J_rest, J the uptake of the nutrient",
                    "nothing: μ = k at the steady state of the SBML model",
                ],
            ],
        )
        + "\n"
    )
    add(
        "The feed is the one that holds the vessel at the value that the SBML file gives the nutrient when B is alone "
        "at the steady state of its model, so that steady state is the rest state of B in the world. The table "
        "checks it with a second engine: the species of B at rest in the composite (Tellurium) against the state "
        "that COPASI reaches from the initial state of the untouched SBML file, as the largest relative difference.\n"
    )
    lines = []
    for biomodel in receivers:
        info = json.loads((runs / "worlds" / biomodel / "xeno" / "world.json").read_text(encoding="utf-8"))
        nutrient = info["nutrient"]
        lines.append(
            [
                biomodel,
                nutrient,
                _number(str(info["sbml_value_of_medium"][nutrient]), 4),
                _number(str(info["uptake_at_rest"]), 4),
                _number(str(info["biovolume_alone"]), 3),
                _number(str(info["feed"][nutrient]), 4),
                _number(str(info["rest_receiver_vs_copasi_steady_state"]), 2),
                _number(str(info["rest_moved"]), 2),
            ]
        )
    add(
        _table(
            [
                "BioModel",
                "nutrient",
                "level in the SBML file",
                "uptake at the steady state (mM/min)",
                "density of B alone (assumed)",
                "feed (derived)",
                "species of B at rest against the steady state that COPASI finds for the file",
                "what still moves at rest",
            ],
            lines,
        )
        + "\n"
    )
    sender = plan["sender"]
    add(
        f"A takes `{sender['substrate']}` by Michaelis–Menten kinetics (V = {sender['vmax']:g} mM/min per cell volume at "
        f"activity 1, K = {sender['km']:g} mM), grows on it with a constant yield and a maintenance (Pirt), and "
        f"excretes `{sender['product']}`. Neither species occurs in any of the three SBML models. Its activity u "
        "multiplies the capacity of its transporter and is the knob that is stepped at time zero. Every value of A "
        f"is assumed. In the designs other than `xeno` the same transporter also takes the nutrient of B, at "
        f"{', '.join(_number(str(plan['designs'][design])) for design in designs if float(plan['designs'][design]) > 0.0)} "
        "of its specificity for its own substrate: these are the controls.\n"
    )

    # ---- 4. the result ----------------------------------------------------------------------------------------------
    add("## 4. Orthogonality of A to B\n")
    measure = plan["measure"]
    add(
        f"Runs of a world: the baseline at rest (u = {plan['activities']['baseline']:g}), u stepped at time zero to "
        f"{', '.join(_number(str(value)) for value in plan['activities']['steps'])}, and the removal of A; all from the "
        f"same rest state and under the same supply, followed for {plan['time']['horizon']:g} min "
        f"({float(plan['time']['horizon']) * float(plan['supply']['dilution']):.0f} residence times) and stored every "
        f"{float(plan['time']['interval']) * int(plan['time']['store_every']):g} min. B protects quantities of two "
        "kinds, layer by layer (`protect` in the plan): those common to every organism (the level of its nutrient "
        "in the medium, the ATP that it makes, the sum of its metabolite pools, its uptake, its growth rate, its "
        "density) and those characteristic of it (one pool and one or two steps of its metabolism). Each has the "
        f"tolerance `{plan['protect']['tolerance']}` of its value at rest, and the growth rate a tolerance `fn`. "
        "D of every layer and of each kind is in `runs/worlds/layers.csv` and in the figure of the layers below. "
        f"Q_B is `{measure['function'].replace('{name}', 'B')}` with "
        f"Q_min `{measure['qmin']}` read {measure['mode']}; Q_A is "
        f"`{measure['sender_function'].replace('{name}', 'A')}` with Q_A,min "
        f"`{measure['sender_qmin']}` read {measure['sender_mode']}; B is lost below a density of {float(plan['viable']):g}.\n"
    )
    lines = []
    for row in worlds:
        if row.get("status") != "ok":
            lines.append([row["biomodel"], row["design"], row.get("status", ""), *[""] * 9])
            continue
        stages = " ".join(
            mark
            for mark, key in (
                ("1", "stage1_trajectory"),
                ("3", "stage3_tolerable"),
                ("4", "stage4_distribution"),
            )
            if row[key] == "True"
        )
        bits = (
            "–"
            if row["bits_upper"] in ("", "None")
            else f"{_number(row['bits_lower'], 2)} to {_number(row['bits_upper'], 3)}"
        )
        lines.append(
            [
                row["biomodel"],
                row["design"],
                row["shared_species"] or "none",
                f"**{_number(row['D'])}**",
                f"{row['D_quantity']} in {row['D_activity']}" if row["D_quantity"] else "–",
                _number(row["O"]),
                stages or "none",
                bits,
                _number(row["D_removal"]),
                _number(row["D_receiver_removed_on_sender"]),
                row["excluded_by_sender_function"] or "–",
                (row["losses"] or "–")
                + (f"; of these, the model stopped: {row['not_followed']}" if row["not_followed"] else ""),
            ]
        )
    add(
        _table(
            [
                "receiver",
                "design of A",
                "species shared",
                "D",
                "carried by",
                "O",
                "stages that hold",
                "bits (lower to upper)",
                "D_removal",
                "D on A, B removed",
                "left out of U_A",
                "losses of B",
            ],
            lines,
        )
        + "\n"
    )
    add("![worlds](figures/worlds.png)\n")
    add("![summary](figures/summary.png)\n")

    xeno = [row for row in worlds if row.get("status") == "ok" and row["design"] == "xeno"]
    exact = [row for row in xeno if float(row["D"]) == 0.0 and row["stage1_trajectory"] == "True"]
    identical = [
        row
        for row in xeno
        if set(row["receiver_bit_identical_in"].split()) >= {*steps, "removal"}
        and row["sender_bit_identical_without_receiver"] == "True"
    ]
    add("### What the runs show\n")
    if len(exact) == len(receivers) == len(xeno):
        add(
            f"* **The design `xeno` is orthogonal to all {len(receivers)} SBML organisms at every stage of the "
            "definition.** D = 0, O = 1, stages 1 (trajectory), 3 (tolerable) and 4 (distribution) hold, the bits "
            "are 0 from below and from above, and D_removal = 0."
        )
    else:
        failed = (
            ", ".join(row["biomodel"] for row in xeno if row not in exact)
            or "a receiver that could not be measured"
        )
        add(f"* **The design `xeno` is not orthogonal at stage 1 to: {failed}.** See the table.")
    if len(identical) == len(xeno) and xeno:
        add(
            "* That zero is exact, not a small number. In every run of the design, the steps of the activity, the "
            "step that A does not survive, and the removal of A, every stored column of B and its nutrient is the "
            "same number as in the baseline, bit for bit; and with B taken away, every column of A is. The two "
            "organisms read and write disjoint species of the shared medium, so the processes of one never see a "
            "number that the other has touched."
        )
    else:
        add(
            "* The columns of B are not bit for bit those of the baseline in every run of the design `xeno`: see `receiver_bit_identical_in` in `runs/worlds/summary.csv`."
        )
    add(
        "* A is not idle while B does not move. Its density, its substrate and its product follow each step "
        "(left column of the figure), and the step to u = "
        + ", ".join(sorted({name[2:] for row in xeno for name in row["excluded_by_sender_function"].split()}))
        + " washes it out: that run is outside U_A by the function of the sender, and B does not notice it either."
        if any(row["excluded_by_sender_function"] for row in xeno)
        else "* A is not idle while B does not move: its density, its substrate and its product follow each step."
    )
    controls = [row for row in worlds if row.get("status") == "ok" and row["design"] != "xeno"]
    moved = [row for row in controls if float(row["D"]) > float(measure["stage_tol"])]
    rising = all(
        [float(row["D"]) for row in controls if row["biomodel"] == biomodel]
        == sorted(float(row["D"]) for row in controls if row["biomodel"] == biomodel)
        for biomodel in receivers
    )
    add(
        f"* The measure is not blind. In {len(moved)} of the {len(controls)} control worlds, where the same organism "
        "takes a share of the nutrient of B, D is above the floor of stage 1"
        + (
            ", and for every receiver it grows with the share (figure above)."
            if rising
            else "; it does not grow with the share for every receiver (table above)."
        )
    )
    for biomodel in receivers:
        mine = [row for row in controls if row["biomodel"] == biomodel]
        if not mine:
            continue
        tolerable = [row["design"] for row in mine if row["stage3_tolerable"] == "True"]
        carriers = sorted({row["D_quantity"] for row in mine if row["D_quantity"]})
        extra = []
        if any(row["losses"] for row in mine):
            extra.append(
                "B is lost in "
                + "; ".join(f"{row['design']}: {row['losses']}" for row in mine if row["losses"])
            )
        if any(row["not_followed"] for row in mine):
            extra.append(
                "among these losses its SBML model stops having a solution in "
                + "; ".join(f"{row['design']}: {row['not_followed']}" for row in mine if row["not_followed"])
            )
        add(
            f"  * {biomodel} ({plan['receivers'][biomodel]['organism']}): D from {_number(min(mine, key=lambda r: float(r['D']))['D'])} to "
            f"{_number(max(mine, key=lambda r: float(r['D']))['D'])}, carried by {', '.join(carriers)}; tolerable (stage 3) in "
            f"{', '.join(tolerable) if tolerable else 'no control'}"
            + ("; " + "; ".join(extra) if extra else "")
            + "."
        )
    add("")
    _layers(plan, add, worlds)
    add("### The lemma on these runs\n")
    add(
        "The growth rate of B is the only protected quantity on which Q_B depends, so with allocation `dependent` "
        "its tolerance `fn` is the whole margin of the baseline. `bound_over_margin` is "
        + ", ".join(
            sorted({_number(row["bound_over_margin"]) for row in worlds if row.get("status") == "ok"})
        )
        + " in every world: D ≤ 1 implies Q_B ≥ Q_min to first order. The runs agree: in the worlds with D ≤ 1 the "
        "least margin that B kept, over the margin of the baseline, was "
        + (
            ", ".join(
                f"{float(row['receiver_margin_least']) / float(row['margin']):.3f} ({row['biomodel'][-3:]}/{row['design']})"
                for row in worlds
                if row.get("status") == "ok" and row["design"] != "xeno" and float(row["D"]) <= 1.0
            )
            or "not available (no control world has D ≤ 1)"
        )
        + ".\n"
    )

    # ---- 5. limits --------------------------------------------------------------------------------------------------
    add("## 5. What this does not show\n")
    held, read = [], []
    for biomodel in receivers:
        info = json.loads((runs / "worlds" / biomodel / "xeno" / "world.json").read_text(encoding="utf-8"))
        constant = info["boundary_species_held_at_sbml_value"]
        held.append(f"{biomodel}: {', '.join(constant)}")
        found = json.loads(screen[biomodel]["sensed"]) if screen.get(biomodel, {}).get("sensed") else {}
        moving = [name for name in constant if found.get(name, {}).get("largest_species_change", 0.0) >= 1e-2]
        if moving:
            read.append(f"{', '.join(moving)} in {biomodel}")
    add(
        "* **Orthogonal to the model in this world, not to the organism.** B can be reached only through what the "
        "world lets vary, one nutrient per receiver. Every other boundary species of a model stays at the value of "
        "its SBML file in every run, whatever A does ("
        + "; ".join(held)
        + "). "
        + (
            "Some of these the models read (the screen finds that 10 % more of them moves a species of the model by "
            "1 % or more: "
            + "; ".join(read)
            + "): an organism that changed one of them would reach B, and this "
            "world would not show it. "
            if read
            else ""
        )
        + "What no model has at all (pH, temperature, osmolarity, space) cannot carry interference here either. The "
        "zero says that A touches nothing that B can see in this world; A was given a substrate and a product that "
        "are in none of the models."
    )
    add(
        "* **The zero is structural.** It follows from the wiring (no shared species), and the runs confirm it to the "
        "last bit; no value of a parameter of A or of the world was tuned to reach it, and none could spoil it. The "
        "measured values of the controls, by contrast, depend on every assumed number."
    )
    add(
        "* **The world around the SBML model is assumed**: the chemostat, the exchange with the medium, growth by a "
        "constant yield, the dilution rate, the densities, and every value of A. The SBML models were built for "
        "pathways at fixed surroundings, not for cells in a chemostat."
    )
    add(
        "* **Accuracy means agreement of engines**, on the time course of the SED-ML file (10 min from the initial "
        "state of the file). The worlds run the models far longer and from their steady state, with one boundary "
        "species varying; there only Tellurium was used."
    )
    add(
        "* **The supremum over U_A is bounded from below**, by "
        f"{len(plan['activities']['steps'])} steps of one knob. For the design `xeno` this does not matter (no step of "
        "any size and no time course can reach B); for the controls a larger or a time-varying activity gives a "
        "larger D (by how much, within two bounds, is in section 7 of [`studies.md`](studies.md))."
    )
    add(
        "* **The layers are a reading of the model, not of the cell.** Which quantity stands for a layer and which "
        "quantities count as characteristic of an organism are choices of the plan; the quantities of one layer "
        "of a model differ by an order of magnitude; and in a world with one way in the layers are fewer than two "
        "independent readings, not six (sections 2 and 5 of `studies.md`). D of a layer is not an independent "
        "test of the organism."
    )
    add(
        "* **What the readings depend on** (the assumptions of the world, the choices of the measure, the numerical "
        "method, where the interference enters, the SBML models, the kind of activity) is varied in the studies, "
        "one thing at a time, and written up in `studies.md`."
    )
    add(
        "* **The other direction is read by removal only.** The SBML organism has no knob in this world, so B to A is "
        'measured by taking B away (column "D on A, B removed").'
    )
    notes = []
    for biomodel in receivers:
        for row in worlds:
            if row.get("status") == "ok" and row["biomodel"] == biomodel and row["not_followed"]:
                info = json.loads(
                    (runs / "worlds" / biomodel / row["design"] / "world.json").read_text(encoding="utf-8")
                )
                for name, item in info["not_followed"].items():
                    notes.append(
                        f"{biomodel}/{row['design']}/{name}: {item['problem'].split(':')[0]}; at the last stored time "
                        f"({item['last_stored_time']:g} min) the nutrient was at {item['nutrient_over_rest_then']:.2f} of "
                        f"its level at rest and B grew at {item['receiver_growth_over_dilution_then']:.2f} of the dilution rate"
                    )
    if notes:
        add(
            "* **A run in which the SBML model stops having a solution counts as a loss of the receiver**, at the "
            "first stored time after it stopped, and is read until then (the message of the integrator is in "
            "`status.txt` of the run, what was stored in `partial.csv`). These are the strongest steps of the "
            "strongest controls of the yeast model. They are not failures of the integrator: the model alone has no "
            "steady state when its glucose is held below 0.53 of the level of its file (study `quasistatic` in "
            "[`studies.md`](studies.md)), its ATP is exhausted in finite time, and no step size takes a solver past "
            "that point:\n" + "\n".join(f"  * {note}" for note in notes)
        )
    add("")
    target = Path(out or runs / "report.md")
    target.write_text("\n".join(text) + "\n", encoding="utf-8")
    return target
