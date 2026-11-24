"""Commands: ``xvb select | verify | worlds | community | studies | figures | paper | all | orthogonality``.

xvb select                  score the viva-biomodels corpus, list the models on which all engines agree, and
                            screen them for what could be a receiver
xvb verify                  run the receivers of the plan through the comparison composite of viva-biomodels
xvb worlds [-j N]           every receiver with every design of the sender: the runs, then the measure
xvb worlds --analyze        measure again from the runs that are there
xvb community [-j N]        the pairs poured into one vessel: its fate, every step, every removal, then the
                            measure for every ordered pair of organisms (--analyze as above)
xvb studies [--only NAME ...] [-j N]
                            how far the readings depend on what was assumed and chosen: sensitivity, numerics,
                            channels, receivers, schedules, settings, cascade, comparison (all unless named)
xvb figures                 the figures and the reports from what the commands above left
xvb paper [--only FIG ...]  the figures and the tables of the manuscript (paper/), from the runs and the reports
xvb all [-j N]              the five in order
xvb orthogonality BASELINE --activity RUN ... --protect COLUMN=TOLERANCE ... [--group NAME=COLUMN,COLUMN ...]
                            the measure on any series files (see ``xvb orthogonality -h``)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path


def _select(args) -> None:
    from xeno_biomodels import corpus, screen

    out, _, _ = corpus.write(plan_path=args.plan)
    print(json.dumps(json.loads((out / "summary.json").read_text(encoding="utf-8")), indent=2))
    print(f"{out / 'corpus_scores.csv'}\n{out / 'candidates.csv'}")
    print(screen.write(out))


def _verify(args) -> None:
    from xeno_biomodels import verify

    for row in verify.run(plan_path=args.plan):
        print(
            f"{row['biomodel']}: {row['engines']}; worst pair {row['worst_pair']} mean nRMSE {row['max_pair_mean_nrmse']:.2e} "
            f"({row['bucket']}), closeness {row['max_closeness']:.2g} ({row['closeness_bucket']}); "
            f"against the corpus {row['local_vs_corpus_nrmse']:.2e}"
        )


def _worlds(args) -> None:
    from xeno_biomodels import experiment
    from xeno_biomodels.world import load_plan

    rows = experiment.run(load_plan(args.plan), jobs=args.jobs, analyze_only=args.analyze, only=args.only)
    for row in rows:
        if row.get("status") != "ok":
            print(f"{row['biomodel']}/{row['design']}: {row.get('status')}")
            continue
        stages = "".join(
            mark if row[key] else "-"
            for mark, key in (
                ("1", "stage1_trajectory"),
                ("3", "stage3_tolerable"),
                ("4", "stage4_distribution"),
            )
        )
        print(
            f"{row['biomodel']}/{row['design']}: D = {row['D']:.4g}, O = {row['O']:.4g}, stages {stages}, "
            f"D_removal = {row['D_removal']:.4g}" + (f", losses {row['losses']}" if row["losses"] else "")
        )
    print(experiment.OUT / "summary.csv")


def _community(args) -> None:
    from xeno_biomodels import community
    from xeno_biomodels.world import load_plan

    designs = args.only if args.command == "community" else None  # under ``all``, --only names receivers
    rows = community.run(load_plan(args.plan), jobs=args.jobs, analyze_only=args.analyze, only=designs)
    for row in rows:
        if row.get("status") != "ok":
            print(f"{row['design']}: {row['sender']} to {row['receiver']}: {row.get('status')}")
            continue
        steps = "" if row["D"] is None else f"D = {row['D']:.4g}, "
        removal = "–" if row["D_removal"] is None else f"{row['D_removal']:.4g}"
        print(
            f"{row['design']}: {row['sender']} to {row['receiver']}: {steps}D_removal = {removal}"
            + (f", losses {row['losses']}" if row["losses"] else "")
        )
    print(community.OUT / "summary.csv")


def _studies(args) -> None:
    from xeno_biomodels import studies
    from xeno_biomodels.world import load_plan

    for path in studies.run(load_plan(args.plan), args.only, jobs=args.jobs, analyze_only=args.analyze):
        print(path)


def _figures(args) -> None:
    from xeno_biomodels import figures

    for path in figures.write(plan_path=args.plan):
        print(path)


def _paper(args) -> None:
    from xeno_biomodels import paper

    for path in paper.write(plan_path=args.plan, only=args.only):
        print(path)


def _all(args) -> None:
    for step in (_select, _verify, _worlds, _community, _figures):
        step(args)


def _orthogonality(args) -> None:
    from xeno_biomodels.orthogonality import MeasureError, measure

    groups = {}
    for text in args.group:
        label, mark, columns = text.partition("=")
        if not mark or not label.strip():
            raise SystemExit(f"xvb orthogonality: a group is written NAME=COLUMN,COLUMN, not {text!r}")
        groups[label.strip()] = [column.strip() for column in columns.split(",") if column.strip()]
    try:
        result, written = measure(
            args.baseline,
            args.activity,
            args.protect,
            receiver_density=args.receiver_density,
            viable=args.viable,
            function=args.function,
            qmin=args.qmin,
            mode=args.mode,
            allocation=args.allocation,
            sender_function=args.sender_function,
            sender_qmin=args.sender_qmin,
            sender_mode=args.sender_mode,
            removal=args.removal,
            groups=groups,
            sigma=args.sigma,
            stage_tol=args.stage_tol,
            bits_tol=args.bits_tol,
            stride=args.stride,
            out=args.out,
        )
    except MeasureError as exc:
        raise SystemExit(f"xvb orthogonality: {exc}") from None
    print(json.dumps(result, indent=2))
    for path in written:
        print(path)


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(
        prog="xvb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def planned(name: str, function, text: str):
        command = commands.add_parser(name, help=text)
        command.add_argument("--plan", type=Path, default=None, help="The plan (configs/experiment.yaml).")
        command.set_defaults(handler=function)
        return command

    planned("select", _select, "Score the corpus and list the models on which all engines agree.")
    planned("verify", _verify, "Run the receivers through the comparison composite of viva-biomodels.")
    for name, function, text in (
        ("worlds", _worlds, "The runs of every world and their measure."),
        ("community", _community, "The pairs in one vessel: its runs and the measure of every ordered pair."),
        ("studies", _studies, "How far the readings depend on what was assumed and chosen."),
        ("all", _all, "select, verify, worlds, community, figures."),
    ):
        command = planned(name, function, text)
        command.add_argument(
            "-j", "--jobs", type=int, default=1, help="Worlds or runs followed at the same time."
        )
        command.add_argument(
            "--analyze", action="store_true", help="Only measure, from the runs that are there."
        )
        what = {"community": "Community designs", "studies": "Studies"}.get(name, "Receivers")
        command.add_argument("--only", nargs="*", help=f"{what} to restrict the runs to.")
    planned("figures", _figures, "The figures and the reports.")
    command = planned("paper", _paper, "The figures and the tables of the manuscript, under paper/.")
    command.add_argument(
        "--only", nargs="*", help="Figures to draw (fig1 ... fig8, figS1, figS2); the tables are then left."
    )

    command = commands.add_parser("orthogonality", help="The measure on any series files.")
    command.add_argument(
        "baseline", type=Path, help="CSV of the baseline run: time and one column per quantity."
    )
    command.add_argument(
        "--activity",
        type=Path,
        action="append",
        default=[],
        help="CSV of a run with an activity stepped (repeat).",
    )
    command.add_argument(
        "--protect",
        action="append",
        default=[],
        required=True,
        metavar="COLUMN=TOLERANCE",
        help="A protected quantity: abs:X, rel:R[,floor:F] or fn (repeat).",
    )
    command.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="NAME=COLUMN,COLUMN",
        help="Protected quantities that are also read on their own, with the tolerances of the whole (repeat).",
    )
    command.add_argument("--receiver-density", help="Column of the density of the receiver.")
    command.add_argument("--viable", type=float, help="Density below which the receiver is lost.")
    command.add_argument("--function", help="Q_B, an expression of columns.")
    command.add_argument(
        "--qmin", help="Least value of Q_B: abs:X, or rel:R of Q_B of the baseline at time zero."
    )
    command.add_argument("--mode", default="pointwise", choices=("pointwise", "terminal", "time_average"))
    command.add_argument("--allocation", default="all", choices=("all", "dependent"))
    command.add_argument("--sender-function", help="Q_A, an expression of columns.")
    command.add_argument("--sender-qmin", help="Least value of Q_A.")
    command.add_argument(
        "--sender-mode", default="pointwise", choices=("pointwise", "terminal", "time_average")
    )
    command.add_argument("--removal", type=Path, help="CSV of the run without the sender.")
    command.add_argument("--sigma", type=float, default=1.0, help="Observation noise, in tolerances.")
    command.add_argument("--stage-tol", type=float, default=1e-6)
    command.add_argument("--bits-tol", type=float, default=1e-3)
    command.add_argument("--stride", type=int, default=1, help="Read every n-th stored time only.")
    command.add_argument(
        "--out",
        type=Path,
        help="Directory for orthogonality.csv, deltas.csv, result.json and, with groups, groups.csv.",
    )
    command.set_defaults(handler=_orthogonality)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
