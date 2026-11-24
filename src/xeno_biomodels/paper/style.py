"""What the figures and the tables of the manuscript share: where they are written, colours and type, the
tiles that carry a number of the measure, and the names by which the manuscript calls a model or a quantity."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Patch

from xeno_biomodels import paths
from xeno_biomodels.studies.common import STUDIES

PAPER = paths.PROJECT / "paper"
WIDTH = 7.0  # inches: a figure over the full width of the page
INK, SECOND, MUTED, GRID, AXIS = "#000000", "#333333", "#6b6b6b", "#e3e3e3", "#9a9a9a"
CATEGORY = ["#2a78d6", "#eb6834", "#1baf7a"]  # the receivers, in the order of the plan
LEVELS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # the activities of the sender, low to high
SHADES = {"within": "#bcd6f5", "beyond": "#3987e5", "lost": "#0d366b"}
CONTROL = "leak-1e-2"
SHORT = {
    "Saccharomyces cerevisiae": "S. cerevisiae",
    "Escherichia coli": "E. coli",
    "Lactococcus lactis": "L. lactis",
}


def _family() -> str:
    """A sans-serif face close to Arial, as the journal asks of artwork: Liberation Sans where the system has it
    (its files are added by hand, since the font cache of matplotlib may be older than they are), or the face
    that matplotlib brings."""
    for directory in ("/usr/share/fonts/truetype/liberation", "/usr/share/fonts/liberation-sans"):
        for file in sorted(Path(directory).glob("LiberationSans-*.ttf")):
            font_manager.fontManager.addfont(str(file))
    for name in ("Liberation Sans", "Arial", "Helvetica"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
        except ValueError:
            continue
        return name
    return "DejaVu Sans"


def apply_style() -> None:
    family = _family()
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": family,
            "font.size": 8,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "text.color": INK,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK,
            "ytick.labelcolor": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 110,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure, name: str, out: Path) -> Path:
    """The figure as a PDF for the manuscript, and as a PNG beside it to look at."""
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pdf"
    figure.savefig(path, metadata={"CreationDate": None, "ModDate": None})  # the same file from the same runs
    figure.savefig(directory / f"{name}.png", dpi=200)
    plt.close(figure)
    return path


def panel_mark(axis, letter: str, x: float = 0.0, y: float = 1.0, dx: float = 0.0, dy: float = 6.0) -> None:
    """The mark of a panel, above its upper left corner."""
    axis.annotate(
        f"({letter})",
        (x, y),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )


def species_name(plan: dict, biomodel: str) -> str:
    """The organism of a receiver as it is written in a figure: the genus as a letter."""
    organism = plan["receivers"][biomodel]["organism"]
    return SHORT.get(organism, organism)


#: the receiver that grows at a rate of its own (study ``cell``), and where its runs are
CELL = "MODEL1502270000"
CELL_RUNS = STUDIES / "cell"
CELL_NAME = "growing cell"


def receiver_name(plan: dict, biomodel: str) -> str:
    """A receiver as a figure names it: its species in italics, or the growing cell."""
    return CELL_NAME if biomodel == CELL else italic(species_name(plan, biomodel))


def italic(text: str) -> str:
    """A name in italics, in the mathtext of matplotlib."""
    return r"$\mathit{" + text.replace(" ", r"\ ") + "}$"


def summary_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# ---- numbers on shades ---------------------------------------------------------------------------------------------


def show_number(value: float) -> str:
    """A number of the measure in a tile: two figures, and none after the point from ten on."""
    if value == 0.0:
        return "0"
    if value >= 9.95:
        return f"{value:.0f}"
    if value < 0.001:  # 0.00054 would not fit a tile: 5.4e-4
        mantissa, exponent = f"{value:.1e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{value:.2g}"


def tile(axis, x: float, y: float, value: float, lost: bool = False, marked: bool = False, size: float = 6.6):
    """A number of the measure on the shade of what it means: none for an exact zero, then within the tolerance,
    beyond it, and a receiver that was lost. ``marked`` rings it."""
    kind = "lost" if lost else "beyond" if value > 1.0 else "within" if value > 0.0 else None
    box = "round,pad=0,rounding_size=0.06"
    axis.add_patch(
        FancyBboxPatch(
            (x - 0.46, y - 0.42),
            0.92,
            0.84,
            boxstyle=box,
            fc=SHADES[kind] if kind else "white",
            ec="none" if kind else "#c9c9c9",
            lw=0.5,
        )
    )
    ink = "#ffffff" if kind in ("beyond", "lost") else INK
    axis.text(
        x, y + (0.13 if lost else 0.0), show_number(value), ha="center", va="center", fontsize=size, color=ink
    )
    if lost:
        axis.text(x, y - 0.21, "lost", ha="center", va="center", fontsize=size - 1.6, color="#ffffff")
    if marked:
        axis.add_patch(
            FancyBboxPatch(
                (x - 0.49, y - 0.45),
                0.98,
                0.9,
                boxstyle="round,pad=0,rounding_size=0.08",
                fc="none",
                ec=INK,
                lw=1.0,
            )
        )


def bare_axis(axis, columns: list[str], lines: list[str], aspect: float | str = 0.62) -> None:
    """The frame of a matrix: names along the top and down the side, and nothing else."""
    axis.set_xlim(-0.5, len(columns) - 0.5)
    axis.set_ylim(len(lines) - 0.5, -0.5)
    axis.set_aspect(aspect)
    axis.set_anchor("N")
    axis.set_xticks(range(len(columns)))
    axis.set_xticklabels(columns)
    axis.xaxis.tick_top()
    axis.set_yticks(range(len(lines)))
    axis.set_yticklabels(lines)
    axis.tick_params(length=0, pad=2)
    axis.grid(False)
    for side in ("left", "bottom", "top", "right"):
        axis.spines[side].set_visible(False)


def shade_legend(
    figure,
    pair: bool = False,
    anchor=(0.5, 0.0),
    lost: bool = True,
    columns: int = 0,
    zero: str = "0: the receiver unchanged, bit for bit",
) -> None:
    legend = [
        Patch(fc="white", ec="#c9c9c9", lw=0.5, label=zero),
        Patch(fc=SHADES["within"], label=r"moved, within tolerance ($D \leq 1$)"),
        Patch(fc=SHADES["beyond"], label=r"moved beyond tolerance ($D > 1$)"),
    ]
    if lost:
        legend.append(Patch(fc=SHADES["lost"], label="receiver lost in one of the runs"))
    if pair:
        legend.append(Patch(fc="none", ec=INK, lw=1.0, label="sender and receiver are a pair"))
    figure.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=anchor,
        ncols=columns or len(legend),
        handlelength=1.3,
        handleheight=0.9,
        handletextpad=0.5,
        columnspacing=1.1,
        fontsize=6.8,
    )


def line_labels(lines: list[tuple[str, str]], whole: bool) -> list[str]:
    return (["all layers"] if whole else []) + [f"{layer}, {kind}" for layer, kind in lines]


LAYER_COLOR = {"particle": "#E69F00", "molecule": "#009E73", "complex": "#0072B2", "individual": "#CC79A7"}
CHEMICAL_NAME = {"GLCo": "glucose", "NH4": "ammonium", "halfglucose": "glucose", "s0": "nutrient"}
QUANTITY_NAME = {
    "atp_made": "ATP formation",
    "pool": "metabolite pools",
    "growth_rate": "growth rate",
    "species.AcAld": "acetaldehyde",
    "rates.PDC": "PDC",
    "species.GLN": "glutamine",
    "rates.vgs": "GS",
    "species.AMP": "GS-AMP",
    "species.pyruvate": "pyruvate",
    "rates.R2": "LDH",
    "species.GLCi": "glucose (inside)",
    "species.GLU": "glutamate",
    "species.ATP": "ATP",
    "species.ADP": "ADP",
    "species.NAD": "NAD",
    "species.NADH": "NADH",
    "rates.r_14": "ribosome synthesis",
    "species.a": "energy",
    "species.rmr": "ribosomes on r-mRNA",
    "species.rmq": "ribosomes on q-mRNA",
    "species.rmt": "ribosomes on t-mRNA",
    "species.rmm": "ribosomes on m-mRNA",
    "species.si": "nutrient (inside)",
}


def quantity_name(column: str) -> str:
    """A column of a run as the manuscript names the quantity."""
    if column.startswith("medium."):
        return CHEMICAL_NAME.get(column[7:], column[7:]) + " (medium)"
    name = column.split(".", 1)[1]
    if name.startswith("uptake."):
        return "uptake"
    return QUANTITY_NAME.get(name, name)
