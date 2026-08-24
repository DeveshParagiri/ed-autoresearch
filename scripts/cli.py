"""Discoverable command surface for the ED-Fire autoresearch loop."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from scripts import ablate, evaluate, figures, optuna


Command = Callable[[argparse.Namespace], int]

TOOLS = (
    (
        "optuna",
        "Tune SEARCH_SPACE parameters against GFED5; print every trial and running best.",
    ),
    (
        "evaluate",
        "Run official ILAMB once; record global and regional metrics; return diagnostics.",
    ),
    (
        "ablate",
        "Attribute COMPONENTS across every ordering against GFED5; keep PARAMS fixed.",
    ),
    (
        "figures",
        "Recreate the GFED5 comparison map and observed/model seasonal cycle.",
    ),
)


def _list_tools(_: argparse.Namespace) -> int:
    width = max(len(name) for name, _ in TOOLS)
    for name, description in TOOLS:
        print(f"{name:<{width}}  {description}")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="ar",
        description="ED-Fire autoresearch tools. Run from autoresearch/.",
    )
    commands = root.add_subparsers(dest="command", metavar="COMMAND", required=True)

    listing = commands.add_parser("list", help="list the tools available to the model")
    listing.set_defaults(handler=_list_tools)

    optimize = commands.add_parser(
        "optuna",
        help="tune SEARCH_SPACE parameters against GFED5 and print every trial",
    )
    optimize.add_argument("--trials", type=int, default=500)
    optimize.add_argument(
        "--patience",
        type=int,
        default=50,
        help="stop after this many completed trials without a new three-decimal best; 0 disables",
    )
    optimize.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel trial threads; 1 is the memory-safe default",
    )
    optimize.add_argument("--seed", type=int, default=0)
    optimize.set_defaults(handler=optuna.run)

    official = commands.add_parser(
        "evaluate",
        help="run official ILAMB once, record metrics, and return diagnostics",
    )
    official.add_argument("--description", required=True)
    official.set_defaults(handler=evaluate.run)

    diagnostic = commands.add_parser(
        "ablate",
        help="run exact Shapley attribution of global and regional GFED5 scores",
        description=(
            "Evaluate every COMPONENTS subset against GFED5 with the current PARAMS, "
            "then report exact global and regional Shapley attribution plus leave-one-out."
        ),
    )
    diagnostic.set_defaults(handler=ablate.run)

    visual = commands.add_parser(
        "figures",
        help="recreate the GFED5 comparison map and seasonal cycle",
    )
    visual.set_defaults(handler=figures.run)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    handler: Command = args.handler
    try:
        return handler(args)
    except NotImplementedError as error:
        print(f"ar {args.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
