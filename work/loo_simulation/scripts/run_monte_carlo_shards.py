"""Run resumable Monte Carlo shards with limited process parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from loo_sim import (
    MonteCarloConfig,
    config_fingerprint,
    load_monte_carlo_config,
    load_monte_carlo_results,
    merge_saved_monte_carlo_results,
    run_monte_carlo,
    save_monte_carlo_results,
    shard_replication_indices,
)


def _shard_name(shard_index: int, shard_count: int) -> str:
    return f"shard_{shard_index:04d}_of_{shard_count:04d}"


def _run_one_shard(
    config: MonteCarloConfig,
    output_directory: str,
    shard_index: int,
    shard_count: int,
) -> tuple[int, int, int]:
    indices = shard_replication_indices(
        config.replications,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    result = run_monte_carlo(
        config,
        replication_indices=indices,
    )
    save_monte_carlo_results(result, output_directory)
    return shard_index, len(result.records), len(result.attempts)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run independently saved, resumable Monte Carlo shards and "
            "strictly merge them after completion."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Concurrent shard processes, between one and four.",
    )
    parser.add_argument(
        "--replications",
        type=int,
        help="Optional override of the configuration replication count.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional override of the configuration master seed.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and skip already completed shard directories.",
    )
    args = parser.parse_args()
    if args.shard_count < 1:
        parser.error("--shard-count must be positive.")
    if args.workers < 1 or args.workers > 4:
        parser.error("--workers must lie between one and four.")
    return args


def _validate_existing_shard(
    path: Path,
    *,
    config: MonteCarloConfig,
    expected_indices: tuple[int, ...],
) -> None:
    result = load_monte_carlo_results(path)
    if config_fingerprint(result.config) != config_fingerprint(config):
        raise ValueError(f"Existing shard has the wrong config: {path}.")
    if result.replication_indices != expected_indices:
        raise ValueError(
            f"Existing shard has the wrong replication indices: {path}."
        )


def main() -> None:
    args = _arguments()
    config = load_monte_carlo_config(args.config)
    if args.replications is not None:
        config = replace(config, replications=args.replications)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    if args.shard_count > config.replications:
        raise ValueError(
            "shard_count cannot exceed the number of replications."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[int, Path]] = []
    shard_directories: list[Path] = []
    for shard_index in range(args.shard_count):
        path = args.output_root / _shard_name(
            shard_index,
            args.shard_count,
        )
        shard_directories.append(path)
        indices = shard_replication_indices(
            config.replications,
            shard_index=shard_index,
            shard_count=args.shard_count,
        )
        if path.exists() and any(path.iterdir()):
            if not args.resume:
                raise FileExistsError(
                    f"Shard directory already contains files: {path}. "
                    "Use --resume to validate and reuse it."
                )
            _validate_existing_shard(
                path,
                config=config,
                expected_indices=indices,
            )
            print(
                f"Reusing {_shard_name(shard_index, args.shard_count)}.",
                flush=True,
            )
        else:
            pending.append((shard_index, path))

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _run_one_shard,
                    config,
                    str(path),
                    shard_index,
                    args.shard_count,
                ): shard_index
                for shard_index, path in pending
            }
            for future in as_completed(futures):
                shard_index, records, attempts = future.result()
                print(
                    f"Completed "
                    f"{_shard_name(shard_index, args.shard_count)}: "
                    f"{records} records, {attempts} attempts.",
                    flush=True,
                )

    merged_output = args.output_root / "merged"
    merge_saved_monte_carlo_results(
        shard_directories,
        merged_output,
        require_complete=True,
    )
    print(
        f"Saved complete validated merge to "
        f"{merged_output.resolve()}.",
        flush=True,
    )


if __name__ == "__main__":
    main()
