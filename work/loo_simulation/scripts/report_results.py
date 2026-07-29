"""Generate paper-ready tables and figures from a complete Monte Carlo run."""

from __future__ import annotations

import argparse
from pathlib import Path

from loo_sim import load_monte_carlo_results
from loo_sim.reporting import build_production_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/full_ladder_production/merged"),
        help="Complete merged Monte Carlo result directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/production"),
        help="Destination for generated tables and figures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = load_monte_carlo_results(args.input)
    files = build_production_report(result, args.output)
    print(
        f"Generated {len(files)} production-report files in "
        f"{args.output.resolve()}."
    )


if __name__ == "__main__":
    main()
