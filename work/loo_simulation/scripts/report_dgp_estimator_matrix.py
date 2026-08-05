"""Generate the five-DGP by four-estimator report bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from loo_sim import load_monte_carlo_results
from loo_sim.matrix_reporting import build_matrix_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/dgp_estimator_matrix/merged"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dgp_estimator_matrix"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("DGP_ESTIMATOR_RESULTS.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = load_monte_carlo_results(args.input)
    files = build_matrix_report(result, args.output, args.markdown)
    print(
        f"Generated {len(files)} report files and "
        f"{args.markdown.resolve()}."
    )


if __name__ == "__main__":
    main()
