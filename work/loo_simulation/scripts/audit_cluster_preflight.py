#!/usr/bin/env python3
"""Apply the declared acceptance gate to one cluster preflight shard."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import re

from loo_sim import load_monte_carlo_results


FIXED_RANK_PATTERN = re.compile(r"^project_plugin_r(?P<rank>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a cluster preflight. Numerical warnings are permitted "
            "only for deliberately over-ranked fixed-rank project fits."
        )
    )
    parser.add_argument(
        "result_directory",
        type=Path,
        help="Saved Monte Carlo shard containing attempts.csv and records.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = load_monte_carlo_results(args.result_directory)
    scenario_config = {
        scenario.name: scenario for scenario in result.config.scenarios
    }
    problems: list[str] = []
    allowed_overrank_warnings: list[str] = []

    expected_attempts: set[tuple[str, int, str]] = set()
    for scenario in result.config.scenarios:
        estimator_names: list[str] = []
        if result.config.estimators.run_low_rank:
            estimator_names.extend(
                f"project_plugin_r{rank}"
                for rank in sorted(set(scenario.plugin_ranks))
            )
            if result.config.estimators.run_bic:
                estimator_names.append("project_plugin_bic")
        if result.config.estimators.run_fe_kss:
            estimator_names.extend(("akm_fe", "kss_ho", "kss_he"))
        if result.config.estimators.run_bs20:
            estimator_names.append("bs20")
        if result.config.estimators.run_blm:
            estimator_names.extend(
                f"blm_{variant}"
                for variant in result.config.estimators.blm_variants
            )
        for replication in result.replication_indices:
            expected_attempts.update(
                (scenario.name, replication, estimator)
                for estimator in estimator_names
            )

    attempt_keys = [
        (attempt.scenario, attempt.replication, attempt.estimator)
        for attempt in result.attempts
    ]
    actual_attempts = set(attempt_keys)
    for missing in sorted(expected_attempts - actual_attempts):
        problems.append(
            "Missing estimator attempt: "
            f"{missing[0]}, replication {missing[1]}, {missing[2]}."
        )
    for unexpected in sorted(actual_attempts - expected_attempts):
        problems.append(
            "Unexpected estimator attempt: "
            f"{unexpected[0]}, replication {unexpected[1]}, "
            f"{unexpected[2]}."
        )
    for key, count in Counter(attempt_keys).items():
        if count > 1:
            problems.append(
                "Duplicate estimator attempt: "
                f"{key[0]}, replication {key[1]}, {key[2]} "
                f"appears {count} times."
            )

    for attempt in result.attempts:
        scenario = scenario_config[attempt.scenario]
        label = (
            f"{attempt.scenario}, replication {attempt.replication}, "
            f"{attempt.estimator}"
        )
        if attempt.status in {"failure", "unsupported"}:
            problems.append(f"{label}: {attempt.status}: {attempt.message}")
            continue
        if attempt.status == "success":
            continue

        fixed_rank = FIXED_RANK_PATTERN.fullmatch(attempt.estimator)
        if (
            attempt.status == "unstable"
            and fixed_rank is not None
            and int(fixed_rank.group("rank")) > scenario.true_rank
        ):
            allowed_overrank_warnings.append(label)
        else:
            problems.append(
                f"{label}: unexpected instability: {attempt.message}"
            )

    selected_rank_records = [
        record
        for record in result.records
        if record.estimator == "project_plugin_bic"
        and record.metric == "selected_rank"
        and record.target_type == "rank_diagnostic"
    ]
    if result.config.estimators.run_bic:
        expected_rank_records = len(result.replication_indices) * len(
            result.config.scenarios
        )
        if len(selected_rank_records) != expected_rank_records:
            problems.append(
                "Expected "
                f"{expected_rank_records} BIC rank records but found "
                f"{len(selected_rank_records)}."
            )
        for record in selected_rank_records:
            if record.estimate != record.target:
                problems.append(
                    f"{record.scenario}, replication "
                    f"{record.replication}: BIC selected rank "
                    f"{record.estimate:g}, true rank {record.target:g}."
                )
    elif selected_rank_records:
        problems.append(
            "Found BIC rank records even though run_bic is false."
        )

    print(
        f"Audited {len(result.attempts)} attempts across "
        f"{len(result.replication_indices)} replication indices."
    )
    print(
        "Permitted deliberately over-ranked warnings: "
        f"{len(allowed_overrank_warnings)}."
    )
    for label in allowed_overrank_warnings:
        print(f"  ALLOWED: {label}")

    if problems:
        print(f"PREFLIGHT FAILED with {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1

    print("PREFLIGHT PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
