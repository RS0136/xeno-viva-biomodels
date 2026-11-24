"""The tables of the manuscript, written from the summary tables of the runs and from the tables of the three
reports, cell by cell, so that no number is typed. Each file under ``paper/tables/`` is one ``tabular``."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from xeno_biomodels import paths
from xeno_biomodels.paper.style import (
    CELL,
    CELL_NAME,
    CELL_RUNS,
    quantity_name,
    species_name,
    summary_rows,
)
from xeno_biomodels.studies.common import STUDIES, number, read_rows

ORGANISM = {"BIOMD172": "S. cerevisiae", "BIOMD217": "E. coli", "BIOMD17": "L. lactis"}
#: a quantity of a model as the manuscript names it
NAMED = {
    "B.density": "density",
    "B.species.AMP": "GS-AMP",
    "species.AcAld": "acetaldehyde",
    "rates.PDC": "PDC",
    "species.GLN": "glutamine",
    "rates.vgs species.AMP": "GS, GS-AMP",
    "species.pyruvate": "pyruvate",
    "rates.R2": "LDH",
}


def tex_number(value: float, digits: int = 3) -> str:
    """A number as the reports show it (``digits`` significant figures), with a power of ten where they use one."""
    if value == 0.0:
        return "0"
    text = f"{value:.{digits}g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def tex_cell(text: str) -> str:
    """A cell of a table of a report as LaTeX: its marks of emphasis, its signs and its powers of ten."""
    text = text.strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", lambda found: NAMED.get(found.group(1), found.group(1)), text)
    text = re.sub(
        r"BIOMD(\d+)\b(\s+\*[^*]+\*)?",
        lambda found: f"\\textit{{{ORGANISM.get('BIOMD' + found.group(1), 'BIOMD' + found.group(1))}}}",
        text,
    )
    text = re.sub(r"\*([^*]+)\*", r"\\textit{\1}", text)
    text = re.sub(r"leak-(\de-\d)", lambda found: f"{float(found.group(1)):g}", text)
    text = re.sub(
        r"(?<![\w.])(\d+(?:\.\d+)?)e([+-]?\d+)",
        lambda found: f"${found.group(1)}\\times10^{{{int(found.group(2))}}}$",
        text,
    )
    text = re.sub(r"\bx(\d)", r"$\\times$\1", text)
    text = text.replace("<$", "$<").replace("_", "\\_").replace("&", "\\&")
    text = re.sub(r"\s*%", r"\\,\\%", text)
    text = text.replace("–", "--").replace("≤", "$\\leq$")
    return text


def md_tables(path: Path) -> list[tuple[list[str], list[list[str]]]]:
    """Every table of a report: its head and its lines, cell by cell."""
    tables, block = [], []
    for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
        if line.strip().startswith("|"):
            block.append([cell.strip() for cell in line.strip().strip("|").split("|")])
            continue
        if len(block) > 2:
            tables.append((block[0], block[2:]))
        block = []
    return tables


def md_table(path: Path, *head: str, nth: int = 0) -> tuple[list[str], list[list[str]]]:
    """The table of a report whose first cells of the head begin with ``head``."""
    found = [
        table
        for table in md_tables(path)
        if all(cell.startswith(wanted) for cell, wanted in zip(table[0], head, strict=False))
    ]
    if len(found) <= nth:
        raise LookupError(f"{path.name}: no table {nth} whose head begins with {head}")
    return found[nth]


def tabular(columns: str, head: list[str], lines: list[list[str]], rules_before: tuple[int, ...] = ()) -> str:
    """A ``tabular`` with the rules of booktabs; ``rules_before`` are the lines before which a thin rule goes."""
    out = [f"\\begin{{tabular}}{{{columns}}}", "\\toprule", " & ".join(head) + " \\\\", "\\midrule"]
    for n, line in enumerate(lines):
        if n in rules_before:
            out.append("\\addlinespace")
        out.append(" & ".join(line) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(out)


def _step(text: str) -> str:
    """``u=8`` as mathematics."""
    return re.sub(r"u=([\d.]+)", r"$u{=}\1$", text)


def _carried(row: dict) -> str:
    """The quantity and the step at which D of a scenario is attained."""
    if not row["D_quantity"]:
        return "--"
    quantity = "nutrient" if row["D_quantity"].startswith("medium.") else quantity_name(row["D_quantity"])
    return f"{quantity}, {_step(row['D_activity'])}"


def _lost_steps(text: str) -> str:
    """``u=8@30(extinction) u=4@20(function)`` as the steps alone."""
    return ", ".join(_step(item.split("@")[0]) for item in text.split()) or "--"


def _per_cent(value: float) -> str:
    """A share as a percentage with three figures, and no power of ten."""
    shown = 100.0 * value
    return (f"{shown:.3g}" if shown < 1000.0 else f"{float(f'{shown:.3g}'):.0f}") + "\\,\\%"


def _sources(name: str) -> list[Path]:
    """Where a study wrote a table: for the receivers of the plan, and for the growing cell."""
    found = [STUDIES / name]
    if (CELL_RUNS / name).exists():
        found.append(CELL_RUNS / name)
    return found


def _summaries() -> list[dict]:
    """The scenarios of the plan and those of the growing cell, as their summaries have them."""
    rows = [row for row in summary_rows(paths.RUNS / "worlds" / "summary.csv") if row.get("status") == "ok"]
    path = CELL_RUNS / "worlds" / "summary.csv"
    if path.exists():
        rows += [row for row in read_rows(path) if row.get("status") == "ok"]
    return rows


def _name(plan: dict, biomodel: str) -> str:
    """A receiver as a table names it."""
    return CELL_NAME if biomodel == CELL else f"\\textit{{{species_name(plan, biomodel)}}}"


def _leak(row: dict) -> str:
    """The leak of a scenario as a table shows it: a specificity, or for the growing cell a share."""
    if row["biomodel"] == CELL:
        return "0" if not float(row["leak"]) > 0.0 and row["design"] == "xeno" else _per_cent(_share(row))
    return f"{float(row['leak']):g}"


def _share(row: dict) -> float:
    return float(row["design"].split("-")[1]) if row["design"].startswith("share-") else 0.0


def _blocks(plan: dict, rows: list[dict], cells) -> tuple[list[list[str]], tuple[int, ...]]:
    """Lines of a table by receiver: the name once, and a little room before the next receiver."""
    lines, breaks, last = [], [], None
    for row in rows:
        name = _name(plan, row["biomodel"])
        if name != last and lines:
            breaks.append(len(lines))
        lines.append([name if name != last else "", *cells(row)])
        last = name
    return lines, tuple(breaks)


def table_scenarios(plan: dict) -> str:
    """Every scenario: D over the admissible steps and where it is attained, the steps with a loss, the removal
    of the sender and what its presence costs, and the two activities that the closed form gives."""
    marks, gains = {}, {}
    for directory in _sources("activity"):
        marks.update({(r["biomodel"], r["design"]): r for r in read_rows(directory / "thresholds.csv")})
        gains.update({(r["biomodel"], r["design"]): r for r in read_rows(directory / "gain.csv")})

    def cells(row: dict) -> list[str]:
        key = (row["biomodel"], row["design"])
        mark, gain = marks.get(key), gains.get(key)
        lost = bool(row["losses"])
        return [
            _leak(row),
            tex_number(float(row["D"])) + ("$^{\\dagger}$" if lost else ""),
            _carried(row),
            _lost_steps(row["losses"]),
            tex_number(float(row["D_removal"])),
            "1" if mark is None else tex_number(float(mark["density_at_rest_over_alone"])),
            "--" if gain is None else tex_number(float(gain["activity_at_tolerance_closed_form"])),
            "--"
            if mark is None or number(mark["activity_of_washout"]) is None
            else tex_number(float(mark["activity_of_washout"])),
        ]

    lines, breaks = _blocks(plan, _summaries(), cells)
    head = [
        "Receiver",
        "leak",
        "$D$",
        "attained at",
        "receiver lost at",
        "$D_{\\mathrm{removal}}$",
        "density with the sender, over alone",
        "$u_1$",
        "$u_{\\mathrm{c}}$",
    ]
    return tabular("@{}lllL{2.9cm}L{2.1cm}lC{2.0cm}ll@{}", head, lines, breaks)


def table_criteria(plan: dict) -> str:
    """The scenarios of the plan under the trajectory and the tolerable criterion, and what their runs say of
    the activity in bits: computed, and between its two bounds."""
    bits = {(r["biomodel"], r["design"]): r for r in read_rows(STUDIES / "information" / "bits.csv")}
    mark = {"True": "$\\bullet$", "False": "$\\circ$"}

    def cells(row: dict) -> list[str]:
        found = bits.get((row["biomodel"], row["design"]), {})
        said = number(found.get("bits"))
        bounds = "--"
        if said is not None:
            bounds = f"{tex_number(float(found['bits_lower_bound']), 2)} to {tex_number(float(found['bits_upper_bound']), 2)}"
        return [
            _leak(row),
            tex_number(float(row["D"])) + ("$^{\\dagger}$" if row["losses"] else ""),
            mark[row["stage1_trajectory"]],
            mark[row["stage3_tolerable"]],
            # two decimals: the draws of the noise leave about 0.002 bits open
            "--" if said is None else ("0" if said == 0.0 else f"{said:.2f}"),
            bounds,
            "--"
            if number(found.get("noise_at_half")) is None
            else tex_number(float(found["noise_at_half"]), 2),
        ]

    rows = [row for row in summary_rows(paths.RUNS / "worlds" / "summary.csv") if row.get("status") == "ok"]
    lines, breaks = _blocks(plan, rows, cells)
    head = [
        "Receiver",
        "leak",
        "$D$",
        "trajectory",
        "tolerable",
        "bits",
        "bounds on the bits",
        "noise at half",
    ]
    return tabular("@{}lllccllC{1.6cm}@{}", head, lines, breaks)


def table_comparison(plan: dict) -> str:
    """D beside the niche overlap, the linear response of the steady state and the steady states of the steps."""
    rows = [row for row in read_rows(STUDIES / "comparison" / "worlds.csv") if row.get("status") == "ok"]

    def cells(row: dict) -> list[str]:
        steady = float(row["D_steady"])
        gone = []
        if row["receiver_gone_at_rest"]:
            gone.append("gone: " + _step(row["receiver_gone_at_rest"]).replace(" ", ", "))
        if row["no_rest_state"]:
            gone.append("not reached: " + _step(row["no_rest_state"]).replace(" ", ", "))
        return [
            f"{float(row['leak']):g}",
            tex_number(float(row["niche_overlap"]), 2),
            tex_number(float(row["D_linear"])),
            "$<10^{-9}$" if 0.0 < steady < 1e-9 else tex_number(steady),
            tex_number(float(row["D"])),
            tex_number(float(row["D_end"])),
            "; ".join(gone) or "--",
        ]

    lines, breaks = _blocks(plan, rows, cells)
    head = [
        "Receiver",
        "leak",
        "niche overlap",
        "linear response",
        "steady states",
        "$D$ (runs)",
        "$D$ at the end",
        "receiver at the steady state of a step",
    ]
    return tabular("@{}llC{1.25cm}C{1.5cm}C{1.3cm}C{1.1cm}C{1.3cm}L{4.3cm}@{}", head, lines, breaks)


def table_steps(plan: dict) -> str:
    """D of every admissible step alone, scenario by scenario, beside what the closed form gives for it; a
    dagger where the receiver is lost in the run."""
    rows = []
    for directory in _sources("activity"):
        rows += [row for row in read_rows(directory / "activities.csv") if float(row["leak"]) > 0.0]
    steps = sorted(
        {float(r["activity"]) for r in rows if r["admissible"] == "True" and number(r.get("D")) is not None}
    )
    keys = list(dict.fromkeys((row["biomodel"], row["design"]) for row in rows))

    def cells(key_row: dict) -> list[str]:
        out = [_leak(key_row)]
        for step in steps:
            found = next(
                (
                    row
                    for row in rows
                    if (row["biomodel"], row["design"]) == (key_row["biomodel"], key_row["design"])
                    and float(row["activity"]) == step
                ),
                None,
            )
            if found is None or number(found.get("D")) is None:
                out.append("--")
                continue
            lost = number(found.get("receiver_lost_at")) is not None
            shown = "--" if lost and float(found["D"]) == 0.0 else tex_number(float(found["D"]))
            closed = number(found.get("D_linear"))
            out.append(
                shown
                + ("$^{\\dagger}$" if lost else "")
                + ("" if closed is None else f" ({tex_number(closed)})")
            )
        return out

    listed = [
        {
            "biomodel": biomodel,
            "design": design,
            "leak": next(r["leak"] for r in rows if r["design"] == design),
        }
        for biomodel, design in keys
    ]
    lines, breaks = _blocks(plan, listed, cells)
    head = ["Receiver", "leak", *[f"$u = {step:g}$" for step in steps]]
    return tabular("@{}ll" + "l" * len(steps) + "@{}", head, lines, breaks)


def table_gain(plan: dict) -> str:
    """The linear response of the vessel in closed form: its two rates, the time of the peak, the gain, and the
    gain that the runs give."""
    rows = []
    for directory in _sources("activity"):
        rows += read_rows(directory / "gain.csv")

    def cells(row: dict) -> list[str]:
        low, high = number(row["least_gain_in_the_runs"]), number(row["largest_gain_in_the_runs"])
        whole, peak = float(row["gain_closed_form_of_the_whole"]), float(row["gain_closed_form"])
        return [
            _leak(row),
            tex_number(float(row["response_coefficient_of_the_uptake"])),
            tex_number(float(row["time_constant_minutes"])),
            tex_number(float(row["t_peak_minutes"])),
            "density" if whole > peak else quantity_name(row["carried_by"]),
            tex_number(whole),
            "--"
            if low is None
            else (
                tex_number(low)
                if tex_number(low) == tex_number(high)
                else f"{tex_number(low)} to {tex_number(high)}"
            ),
        ]

    lines, breaks = _blocks(plan, rows, cells)
    head = [
        "Receiver",
        "leak",
        "$R_J$",
        "$1/\\beta$ (min)",
        "$t^{*}$ (min)",
        "largest in",
        "$g$, closed form",
        "$g$, runs",
    ]
    return tabular("@{}llcccL{2.4cm}cl@{}", head, lines, breaks)


def table_response(plan: dict) -> str:
    """The response coefficients of the models alone beside the same ratios in the runs of the weakest control."""
    rows = []
    for directory in _sources("quasistatic"):
        rows += read_rows(directory / "response.csv")
    rows = [row for row in rows if row["layer"] != "individual" or row["biomodel"] == CELL]

    def cells(row: dict) -> list[str]:
        coefficient = number(row["response_coefficient"])
        return [
            quantity_name(row["quantity"]),
            f"{row['layer']}, {row['kind']}",
            "--" if coefficient is None else tex_number(coefficient),
            tex_number(float(row["in_the_runs_of_the_weakest_control"])),
        ]

    lines, breaks = _blocks(plan, rows, cells)
    head = [
        "Receiver",
        "quantity",
        "group",
        "response coefficient of the model alone",
        "ratio of the peaks in the runs of the weakest leaky sender",
    ]
    return tabular("@{}lllC{3.2cm}C{3.4cm}@{}", head, lines, breaks)


def _functional() -> list[dict]:
    rows = []
    for directory in _sources("functional"):
        rows += [row for row in read_rows(directory / "worlds.csv") if float(row["leak"]) > 0.0]
    return rows


def table_functional(plan: dict) -> str:
    """Three readings of the same runs: the convention of a tenth, the box of tolerances that follows from the
    function, and what the function itself lost."""
    mark = {"True": "$\\bullet$", "False": "$\\circ$"}

    def cells(row: dict) -> list[str]:
        return [
            _leak(row),
            tex_number(float(row["D_of_the_plan"])),
            mark[row["tolerable_in_the_plan"]],
            tex_number(float(row["D_guaranteed"])),
            mark[row["tolerable_guaranteed"]],
            tex_number(float(row["D_adverse_guaranteed"])),
            mark[row["tolerable_adverse_guaranteed"]],
            tex_number(float(row["margin_used"])),
            mark[row["function_kept_in_every_run"]],
            _lost_steps(row["losses"]),
        ]

    lines, breaks = _blocks(plan, _functional(), cells)
    head = [
        "Receiver",
        "leak",
        "\\multicolumn{2}{@{}l}{convention}",
        "\\multicolumn{2}{@{}l}{box}",
        "\\multicolumn{2}{@{}l}{box, adverse}",
        "\\multicolumn{2}{@{}l}{margin used}",
        "receiver lost at",
    ]
    return tabular("@{}llr@{\\ }lr@{\\ }lr@{\\ }lr@{\\ }lL{2.4cm}@{}", head, lines, breaks)


def table_readings(plan: dict) -> str:
    """The evaluations with tolerances from the function in full: of first order and guaranteed, with a
    conserved pair counted once, where they are attained, and for the removal of the sender. (What the removal
    is in the convention of a tenth is in the table of the scenarios, and it uses none of the margin of the
    function in any scenario, which the caption says.)"""

    def cells(row: dict) -> list[str]:
        removed = number(row.get("D_removal_guaranteed"))
        return [
            _leak(row),
            tex_number(float(row["D_functional"])),
            tex_number(float(row["D_guaranteed"])),
            f"{quantity_name(row['carried_by'])}, {_step(row['step'])}",
            tex_number(float(row["D_independent"])),
            tex_number(float(row["D_adverse"])),
            quantity_name(row["adverse_carried_by"]) if row["adverse_carried_by"] else "--",
            "--" if removed is None else tex_number(removed),
        ]

    lines, breaks = _blocks(plan, _functional(), cells)
    head = [
        "Receiver",
        "leak",
        "first order",
        "whole box",
        "attained at",
        "pairs once",
        "adverse",
        "attained in",
        "removal, box",
    ]
    return tabular("@{}llllL{2.4cm}llL{1.9cm}l@{}", head, lines, breaks)


def table_tolerances(plan: dict) -> str:
    """The quantities that the growth function of each receiver reads, with the tolerances that it gives them."""
    rows = []
    for directory in _sources("functional"):
        rows += read_rows(directory / "tolerances.csv")

    def cells(row: dict) -> list[str]:
        return [
            quantity_name(row["quantity"]),
            tex_number(float(row["at_rest"])),
            tex_number(float(row["elasticity_of_the_uptake"])),
            _per_cent(float(row["delta_over_value"])),
            _per_cent(float(row["delta_guaranteed_over_value"])),
        ]

    lines, breaks = _blocks(plan, rows, cells)
    head = [
        "Receiver",
        "quantity",
        "steady-state value",
        "elasticity of the function",
        "tolerance, first order",
        "tolerance, whole box",
    ]
    return tabular("@{}llC{2.3cm}C{2.2cm}C{2.2cm}C{2.2cm}@{}", head, lines, breaks)


def table_bound(plan: dict) -> str:
    """Lemma 1 with the tolerances that the function gives: the curvature at rest and what it would allow, the
    box, and what the runs did."""
    rows = []
    for directory in _sources("functional"):
        rows += read_rows(directory / "lemma.csv")

    def cells(row: dict) -> list[str]:
        least = float(row["least_function_in_the_box_over_q_min"])
        return [
            f"{row['quantities']} ({row['independent_quantities']})",
            tex_number(float(row["gamma"])),
            tex_number(float(row["share_from_the_curvature_at_rest"])),
            "$<10^{-6}$" if least < 1e-6 else tex_number(least),
            tex_number(float(row["share_guaranteed"])),
            "yes" if row["monotone_in_the_box"] == "True" else "no",
            row["stored_times"],
            row["every_quantity_within_tolerance"],
            row["of_these_function_below_q_min"],
            row["a_quantity_beyond_tolerance"],
            row["of_these_function_kept"],
        ]

    lines, breaks = _blocks(plan, rows, cells)
    lines = [line for line in lines]
    head = [
        "Receiver",
        "$m$",
        "$\\gamma_0$",
        "$\\theta(\\gamma_0)$",
        "least $Q_{\\mathrm{B}}/Q_{\\min}$ in the box",
        "$\\theta^{*}$",
        "mono\\-tone",
        "stored times",
        "all within tolerance",
        "of these, $Q_{\\mathrm{B}} < Q_{\\min}$",
        "one beyond tolerance",
        "of these, $Q_{\\mathrm{B}} \\geq Q_{\\min}$",
    ]
    del breaks
    return tabular("@{}lcccC{1.5cm}cC{0.9cm}C{0.9cm}C{1.2cm}C{1.3cm}C{1.2cm}C{1.3cm}@{}", head, lines)


def table_monolith(plan: dict) -> str:
    """Every control as one system of equations, under two integrators, beside the composite: scenario by
    scenario, the largest difference among its steps."""
    rows = read_rows(STUDIES / "monolith" / "monolith.csv")
    keys = list(dict.fromkeys((row["biomodel"], row["design"]) for row in rows))
    engines = list(dict.fromkeys(row["engine"] for row in rows))
    listed = []
    for biomodel, design in keys:
        mine = [row for row in rows if (row["biomodel"], row["design"]) == (biomodel, design)]
        listed.append({"biomodel": biomodel, "design": design, "leak": mine[0]["leak"], "rows": mine})

    def cells(item: dict) -> list[str]:
        out = [_leak(item)]
        compared = [row for row in item["rows"] if not row.get("note") and row["engine"] == engines[0]]
        out.append(f"{len(compared)}")
        for engine in engines:
            mine = [row for row in item["rows"] if row["engine"] == engine and not row.get("note")]
            whole = [
                abs(float(r["relative_difference_of_D"]))
                for r in mine
                if number(r.get("receiver_lost_at")) is None
            ]
            cut = [
                abs(float(r["relative_difference_of_D"]))
                for r in mine
                if number(r.get("receiver_lost_at")) is not None
            ]
            out.append("--" if not whole else f"{100.0 * max(whole):.3f}\\,\\%")
            out.append("--" if not cut else f"{100.0 * max(cut):.2f}\\,\\%")
        gaps = [
            float(r["largest_gap_of_a_protected_quantity_in_tolerances"])
            for r in item["rows"]
            if not r.get("note")
        ]
        out.append("--" if not gaps else tex_number(max(gaps), 2))
        return out

    lines, breaks = _blocks(plan, listed, cells)
    head = ["Receiver", "leak", "steps"]
    for engine in engines:
        short = "CVODE" if "CVODE" in engine else "LSODA"
        head += [f"{short}: runs without a loss", f"{short}: runs with a loss"]
    head.append("largest gap of a quantity (tolerances)")
    return tabular("@{}llc" + "C{1.9cm}C{1.7cm}" * len(engines) + "C{2.0cm}@{}", head, lines, breaks)


def table_numerics(plan: dict) -> str:
    """The numerical method: what the coupling interval and the tolerances of the integrator leave in D."""
    rows = [row for row in read_rows(STUDIES / "numerics" / "numerics.csv") if row.get("status") == "ok"]
    lines = []
    for variant in dict.fromkeys(row["variant"] for row in rows):
        mine = [row for row in rows if row["variant"] == variant]
        controls = [row for row in mine if row["design"] != "xeno"]
        same = [row for row in controls if row["same_admissible"] == "True"]
        zeros = [row for row in mine if row["design"] == "xeno"]
        label = re.sub(
            r"\((\d(?:\.\d+)?)e(-?\d+)\)", lambda found: f"($10^{{{int(found.group(2))}}}$)", variant
        )
        exact = sum(float(row["D"]) == 0.0 and row["receiver_bit_identical"] == "True" for row in zeros)
        lines.append(
            [
                label,
                f"{exact} of {len(zeros)}",
                f"{len(same)} of {len(controls)}",
                f"{100.0 * max(abs(float(row['D_change'])) for row in same):.2g}\\,\\%",
                f"{100.0 * max(abs(float(row['D_change'])) for row in controls):.2g}\\,\\%",
                f"{100.0 * max(abs(float(row['D_course_gap'])) for row in same):.2g}\\,\\%",
            ]
        )
    head = [
        "Variant",
        "xeno: exact zeros",
        "leaky senders with the same admissible steps",
        "largest change of $D$ among these",
        "largest change of $D$, all leaky senders",
        "largest gap along $D(t)$",
    ]
    return tabular("@{}L{3.6cm}C{1.5cm}C{2.5cm}C{2.2cm}C{2.2cm}C{2.0cm}@{}", head, lines)


def _chain(text: str) -> str:
    """A chain of dependence as a table shows it."""
    text = re.sub(r"X_x([A-Z])", r"X$_{\\mathrm{\1}}$", text)
    return text.replace(" > ", " $\\to$ ").replace("NH4", "NH$_4^+$")


def table_company(plan: dict) -> str:
    """The community of the design ``indirect``: every xeno on every SBML organism, step by step, and by which
    chain."""
    directory = STUDIES / "coexistence"
    rows = [row for row in read_rows(directory / "community.csv") if row["design"] == "indirect"]
    stepped = [row for row in read_rows(directory / "steps.csv") if row["design"] == "indirect"]
    steps = sorted({row["step"] for row in stepped}, key=lambda step: float(step.split("=")[1]))
    lines = []
    for row in rows:
        if row["sender_kind"] != "xeno" or row["receiver_kind"] != "sbml":
            continue
        mine = {
            item["step"]: item
            for item in stepped
            if (item["sender"], item["receiver"]) == (row["sender"], row["receiver"])
        }
        lines.append(
            [
                f"{row['sender']} to {row['receiver']}"
                + (" (partner)" if row["same_pair"] == "True" else ""),
                *[tex_number(float(mine[step]["D"])) if step in mine else "--" for step in steps],
                tex_number(float(row["D_removal"])),
                tex_number(float(row["niche_overlap"]), 2),
                _chain(row["way"]),
            ]
        )
    head = [
        "Sender to receiver",
        *[f"$D$, {_step(step)}" for step in steps],
        "$D_{\\mathrm{removal}}$",
        "niche overlap",
        "chain of dependence",
    ]
    return tabular("@{}L{2.3cm}" + "c" * len(steps) + "cC{1.1cm}L{3.5cm}@{}", head, lines)


def table_sweep(plan: dict) -> str:
    """The community of the design ``indirect`` with one of its two numbers changed: D of every xeno on every
    SBML organism, and where the density of the other xeno ends at the largest step."""
    rows = [row for row in read_rows(STUDIES / "coexistence" / "sweep.csv") if row["receiver_kind"] == "sbml"]
    designs = list(dict.fromkeys(row["design"] for row in rows))
    designs.sort(
        key=lambda d: (
            next(float(r["overlap"]) for r in rows if r["design"] == d) != 0.1,
            next(float(r["overlap"]) for r in rows if r["design"] == d),
            next(float(r["specificity"]) for r in rows if r["design"] == d),
        )
    )
    pairs = list(dict.fromkeys((row["sender"], row["receiver"]) for row in rows))
    lines = []
    for design in designs:
        mine = {(row["sender"], row["receiver"]): row for row in rows if row["design"] == design}
        one = next(iter(mine.values()))
        lines.append(
            [
                f"{float(one['specificity']):g}",
                f"{float(one['overlap']):g}",
                *[
                    tex_number(float(mine[pair]["D"])) + ("$^{\\dagger}$" if mine[pair].get("losses") else "")
                    for pair in pairs
                ],
            ]
        )
    head = [
        "for the nutrient of the other pair",
        "for the substrate of the other xeno",
        *[f"{sender} to {receiver}" for sender, receiver in pairs],
    ]
    return tabular("@{}C{3.0cm}C{3.0cm}" + "c" * len(pairs) + "@{}", head, lines)


def table_models(plan: dict) -> str:
    """The models that the rule selects: their BioModels entry, the species that the scenario lets vary, and D
    with the sender taking a hundredth and a tenth of what the model takes."""
    rows = [row for row in read_rows(STUDIES / "receivers" / "worlds.csv") if row.get("status") == "ok"]
    keys = list(dict.fromkeys(row["biomodel"] for row in rows))
    lines = []
    for key in keys:
        mine = {row["design"]: row for row in rows if row["biomodel"] == key}
        one = next(iter(mine.values()))
        name = one["name"].replace("_", "\\_").replace("&", "\\&")
        lines.append(
            [
                one["biomodel_id"],
                name,
                f"\\species{{{one['species'].replace('_', '\\_')}}}",
                *[
                    "--"
                    if design not in mine
                    else tex_number(float(mine[design]["D"]))
                    + ("$^{\\dagger}$" if mine[design]["losses"] else "")
                    for design in ("xeno", "share-0.01", "share-0.1")
                ],
            ]
        )
    head = ["BioModels entry", "name of the entry", "species", "xeno", "1\\,\\%", "10\\,\\%"]
    return tabular("@{}lL{7.4cm}lccc@{}", head, lines)


def _from_report(
    report: str,
    head: tuple[str, ...],
    columns: str,
    titles: list[str],
    keep: list[int],
    nth: int = 0,
    first: str | None = None,
    fix=None,
) -> str:
    """A table of a report, with the cells of the columns ``keep`` and other words at its head. ``first`` names a
    column of receivers: the name is written once for the lines that follow one another. ``fix`` has the last
    word on a cell: it is given the column and the cell as LaTeX."""
    _, lines = md_table(paths.RUNS / report, *head, nth=nth)
    out, breaks, last = [], [], None
    for line in lines:
        cells = [tex_cell(line[k]) for k in keep]
        if fix is not None:
            cells = [fix(k, cell) for k, cell in enumerate(cells)]
        if first is not None:
            if cells[0] != last and out:
                breaks.append(len(out))
            cells[0], last = ("" if cells[0] == last else cells[0]), cells[0]
        out.append(cells)
    return tabular(columns, titles, out, tuple(breaks))


CHEMICAL = {
    "GLCo": "glucose",
    "EtOH": "ethanol",
    "NH4": "ammonium",
    "KG": "2-oxoglutarate",
    "halfglucose": "glucose",
    "O2": "oxygen",
}


def _taken(cell: str) -> str:
    """What the sender takes in a world, as the manuscript says it: ``nutrient-0.01 (GLCo)`` is 1 % of the glucose."""
    found = re.fullmatch(r"(?:nutrient|other)-([\d.]+) \((\w+)\)", cell)
    if found:
        return f"{CHEMICAL[found.group(2)]}, {100 * float(found.group(1)):g}\\,\\%"
    found = re.fullmatch(r"nutrient and other, ([\d.]+)", cell)
    if found:
        return f"both species, {100 * float(found.group(1)):g}\\,\\%"
    return f"nutrient, $\\varepsilon = {cell}$"


def all_tables(plan: dict) -> dict[str, str]:
    """The name of every table of the manuscript and its ``tabular``."""
    made = {
        "scenarios": table_scenarios(plan),
        "criteria": table_criteria(plan),
        "comparison": table_comparison(plan),
        "steps": table_steps(plan),
        "gain": table_gain(plan),
        "response": table_response(plan),
        "functional": table_functional(plan),
        "readings": table_readings(plan),
        "tolerances": table_tolerances(plan),
        "bound": table_bound(plan),
        "monolith": table_monolith(plan),
        "company": table_company(plan),
        "sweep": table_sweep(plan),
        "models": table_models(plan),
    }
    made["choices"] = _from_report(
        "studies.md",
        ("choice",),
        "@{}L{5.6cm}C{2.4cm}C{2.2cm}C{2.2cm}C{1.6cm}@{}",
        [
            "Choice changed",
            "$D$ over the reference $D$",
            "scenarios whose criteria differ (of 15)",
            "scenarios whose runs with a loss differ",
            "zeros stay",
        ],
        [0, 1, 2, 3, 7],
        fix=lambda k, cell: cell.replace("Q\\_min", "$Q_{\\min}$"),
    )
    made["lemma"] = _from_report(
        "studies.md",
        ("function", "reads"),
        "@{}L{1.6cm}L{3.3cm}C{1.2cm}C{1.5cm}C{1.6cm}C{1.3cm}C{1.5cm}C{1.2cm}@{}",
        [
            "Function",
            "reads",
            "stored times",
            "all within tolerance",
            "of these, $Q<Q_{\\min}$",
            "least margin kept",
            "falls beyond the bound",
            "largest excess",
        ],
        [0, 1, 2, 3, 4, 5, 6, 7],
    )
    made["numerics"] = table_numerics(plan)
    made["sensitivity"] = _from_report(
        "studies.md",
        ("receiver", "D of `leak-1e-2` in the plan"),
        "@{}lC{1.5cm}C{2.5cm}C{2.5cm}C{3.2cm}C{2.2cm}@{}",
        [
            "Receiver",
            "reference $D$",
            "$D$ over the variants",
            "tolerable verdict differs",
            "$D$ is largest at",
            "largest elsewhere",
        ],
        [0, 1, 2, 3, 4, 5],
        fix=lambda k, cell: (
            re.sub(r"\s*\(.*\)$", "", cell) if k in (3, 5) else cell
        ),  # which worlds: in the text
    )
    made["candidates"] = _from_report(
        "studies.md",
        ("receiver", "layer", "candidates"),
        "@{}llccccccL{2.5cm}c@{}",
        [
            "Receiver",
            "layer",
            "$n$",
            "least",
            "median",
            "largest",
            "beyond",
            "common",
            "specific",
            "below it",
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        first="receiver",
    )
    made["arrival"] = _from_report(
        "studies.md",
        ("layer", "BIOMD172"),
        "@{}lccc@{}",
        ["Layer, kind", "\\textit{S. cerevisiae}", "\\textit{E. coli}", "\\textit{L. lactis}"],
        [0, 1, 2, 3],
    )
    made["dimension"] = _from_report(
        "studies.md",
        ("receiver", "world", "protected quantities that move"),
        "@{}llccccc@{}",
        [
            "Receiver",
            "what the sender takes",
            "$n$",
            "participation",
            "first course",
            "equal lengths",
            "courses",
        ],
        [0, 1, 2, 3, 4, 5, 6],
        first="receiver",
        fix=lambda k, cell: _taken(cell) if k == 1 else cell,
    )
    made["receivers"] = _from_report(
        "studies.md",
        ("layer", "share-0.01"),
        "@{}lcc@{}",
        ["Layer (common quantity)", "share 1\\,\\%", "share 10\\,\\%"],
        [0, 1, 2],
    )
    return made
