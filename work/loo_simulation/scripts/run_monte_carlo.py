"""Run and persist a JSON-configured Monte Carlo experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from loo_sim import (
    load_monte_carlo_config,
    run_monte_carlo,
    save_monte_carlo_results,
    shard_replication_indices,
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
        "--shard-index",
        type=int,
        help="Zero-based shard index; requires --shard-count.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        help="Total number of round-robin replication shards.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress one-line replication progress updates.",
    )
    args = parser.parse_args()
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error(
            "--shard-index and --shard-count must be supplied together."
        )
    return args


def main() -> None:
    args = _arguments()
    config = load_monte_carlo_config(args.config)
    if args.replications is not None:
        config = replace(config, replications=args.replications)
    if args.seed is not None:
        config = replace(config, seed=args.seed)

    replication_indices = None
    if args.shard_index is not None:
        replication_indices = shard_replication_indices(
            config.replications,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    if args.output is not None:
        output = args.output
    elif args.shard_index is None:
        output = Path("results") / args.config.stem
    else:
        shard_name = (
            f"shard_{args.shard_index:04d}_of_"
            f"{args.shard_count:04d}"
        )
        output = Path("results") / args.config.stem / shard_name
    progress = (
        None
        if args.quiet
        else lambda message: print(message, flush=True)
    )
    result = run_monte_carlo(
        config,
        replication_indices=replication_indices,
        progress=progress,
    )
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
