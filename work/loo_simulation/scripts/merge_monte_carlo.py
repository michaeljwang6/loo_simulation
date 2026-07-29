"""Strictly validate and merge saved Monte Carlo result shards."""

from __future__ import annotations

import argparse
from pathlib import Path

from loo_sim import merge_saved_monte_carlo_results


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge disjoint Monte Carlo shards with configuration, "
            "coverage, and duplicate validation."
        )
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        required=True,
        help="Two or more saved shard directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for the merged result tables.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a validated merge that does not cover every replication.",
    )
    args = parser.parse_args()
    if len(args.inputs) < 2:
        parser.error("--inputs requires at least two shard directories.")
    return args


def main() -> None:
    args = _arguments()
    output = merge_saved_monte_carlo_results(
        args.inputs,
        args.output,
        require_complete=not args.allow_incomplete,
    )
    print(f"Saved validated merged results to {output.resolve()}.")


if __name__ == "__main__":
    main()
