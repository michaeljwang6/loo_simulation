"""Run and persist a JSON-configured Monte Carlo experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from loo_sim import (
    load_monte_carlo_config,
    run_monte_carlo,
    save_monte_carlo_results,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LOO-project, FE/KSS, BS20, and grouped-BLM numerical "
            "comparisons."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a Monte Carlo JSON configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output directory. Defaults to results/<configuration stem>."
        ),
    )
    parser.add_argument(
        "--replications",
        type=int,
        help="Optional override of the replication count in the JSON file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional override of the master seed in the JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress one-line replication progress updates.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config = load_monte_carlo_config(args.config)
    if args.replications is not None:
        config = replace(config, replications=args.replications)
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    output = (
        args.output
        if args.output is not None
        else Path("results") / args.config.stem
    )
    progress = None if args.quiet else print
    result = run_monte_carlo(config, progress=progress)
    saved = save_monte_carlo_results(result, output)
    failures = sum(
        attempt.status == "failure" for attempt in result.attempts
    )
    unstable = sum(
        attempt.status == "unstable" for attempt in result.attempts
    )
    print(
        f"Saved {len(result.records)} records and "
        f"{len(result.attempts)} attempts to {saved.resolve()}."
    )
    print(
        f"Diagnostics: {failures} failed and {unstable} unstable "
        "estimator attempts."
    )


if __name__ == "__main__":
    main()
