"""The figures and the tables of the manuscript, from what the other commands left under ``runs/``.

    xvb paper       writes paper/figures/fig1.pdf ... fig6.pdf, figS1.pdf ... figS5.pdf (with a PNG of each)
                    and paper/tables/*.tex

The manuscript (``paper/main.tex``) and its supplementary material (``paper/supplement.tex``) are written by hand
and take these files in. Nothing is run here: every number comes from a table under ``runs/`` or, for some of the
tables, from the reports that ``xvb figures`` writes, so that the manuscript and the reports cannot say different
things.

The figures follow the artwork rules of the journal: vector PDF with embedded TrueType fonts, a sans-serif face
close to Arial, no title inside the figure (the caption has it), panels marked (a), (b), ... Colour does one job
per figure, as in ``xeno_biomodels.figures``: the three receivers are three categories in a fixed order, and a
number of the measure sits on the shade of what it means.
"""

from __future__ import annotations

from pathlib import Path

from xeno_biomodels.paper.figures import FIGURES
from xeno_biomodels.paper.style import PAPER, apply_style
from xeno_biomodels.paper.tables import all_tables
from xeno_biomodels.world import load_plan


def write(
    plan_path: str | Path | None = None, out: Path = PAPER, only: list[str] | None = None
) -> list[Path]:
    """Every figure and every table of the manuscript (or the figures named in ``only``: fig2, figS1, ...)."""
    plan = load_plan(plan_path)
    apply_style()
    made = [draw(plan, out, name) for name, draw in FIGURES.items() if not only or name in only]
    if not only:
        (out / "tables").mkdir(parents=True, exist_ok=True)
        for name, text in all_tables(plan).items():
            path = out / "tables" / f"{name}.tex"
            path.write_text(text, encoding="utf-8")
            made.append(path)
    return made


__all__ = ["FIGURES", "PAPER", "all_tables", "write"]
