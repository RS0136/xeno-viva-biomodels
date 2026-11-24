"""The parts of the manuscript's figures and tables that are arithmetic: numbers and cells as LaTeX, and the tables
of a report read back."""

from __future__ import annotations

import pytest

from xeno_biomodels.paper import style, tables


def test_a_number_is_written_as_the_reports_write_it_with_a_power_of_ten_where_they_use_one():
    assert tables.tex_number(0.0) == "0"
    assert tables.tex_number(1.2389550965722318) == "1.24"
    assert tables.tex_number(17.074446278916827) == "17.1"
    assert tables.tex_number(0.018021837469681667) == "0.018"
    assert tables.tex_number(8924.869410605897, 2) == "$8.9\\times10^{3}$"
    assert tables.tex_number(2.52e-05) == "$2.52\\times10^{-5}$"


def test_a_number_in_a_tile_has_two_figures_and_fits_the_tile():
    assert style.show_number(0.0) == "0"
    assert style.show_number(0.1296) == "0.13"
    assert style.show_number(3.99) == "4"
    assert style.show_number(17.07) == "17"
    assert style.show_number(0.000536) == "5.4e-4"  # 0.00054 would not fit
    assert style.show_number(0.0075) == "0.0075"


def test_a_cell_of_a_report_becomes_latex():
    assert tables.tex_cell("**0.13**") == "0.13"
    assert tables.tex_cell("BIOMD172 *Saccharomyces cerevisiae*") == "\\textit{S. cerevisiae}"
    assert tables.tex_cell("BIOMD17") == "\\textit{L. lactis}"  # not the beginning of BIOMD172
    assert tables.tex_cell("leak-1e-2") == "0.01"
    assert tables.tex_cell("2.3e-06 %") == "$2.3\\times10^{-6}$\\,\\%"
    assert tables.tex_cell("0.464 (`species.AcAld`)") == "0.464 (acetaldehyde)"
    assert tables.tex_cell("7.88 (`rates.vgs species.AMP`)") == "7.88 (GS, GS-AMP)"
    assert tables.tex_cell("dilution rate x4") == "dilution rate $\\times$4"
    assert tables.tex_cell("–") == "--"
    assert tables.tex_cell("Q_min of the sender a quarter") == "Q\\_min of the sender a quarter"
    assert tables.tex_cell("40 %") == "40\\,\\%"


def test_what_the_sender_takes_is_named_by_its_chemical_and_its_share():
    assert tables._taken("nutrient-0.01 (GLCo)") == "glucose, 1\\,\\%"
    assert tables._taken("other-0.1 (KG)") == "2-oxoglutarate, 10\\,\\%"
    assert tables._taken("nutrient and other, 0.1") == "both species, 10\\,\\%"
    assert tables._taken("0.01") == "nutrient, $\\varepsilon = 0.01$"


def test_the_tables_of_a_report_are_read_back_cell_by_cell(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "# A report\n\n"
        "| receiver | D |\n| --- | --- |\n| BIOMD172 | **1.24** |\n| BIOMD217 | **7.88** |\n\n"
        "Some words.\n\n"
        "| layer | share-0.01 | share-0.1 |\n| --- | --- | --- |\n| medium | 0.55 to 3.2 (18) | 3.2 to 7.9 (18) |\n",
        encoding="utf-8",
    )
    found = tables.md_tables(report)
    assert [head for head, _ in found] == [["receiver", "D"], ["layer", "share-0.01", "share-0.1"]]
    head, lines = tables.md_table(report, "layer", "share-0.01")
    assert head == ["layer", "share-0.01", "share-0.1"]
    assert lines == [["medium", "0.55 to 3.2 (18)", "3.2 to 7.9 (18)"]]
    assert tables.md_table(report, "receiver")[1][1] == ["BIOMD217", "**7.88**"]
    with pytest.raises(LookupError):
        tables.md_table(report, "no such table")


def test_a_tabular_has_the_rules_of_booktabs_and_room_before_the_named_lines():
    text = tables.tabular("@{}ll@{}", ["a", "b"], [["1", "2"], ["3", "4"], ["5", "6"]], rules_before=(2,))
    assert text.splitlines() == [
        "\\begin{tabular}{@{}ll@{}}",
        "\\toprule",
        "a & b \\\\",
        "\\midrule",
        "1 & 2 \\\\",
        "3 & 4 \\\\",
        "\\addlinespace",
        "5 & 6 \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
