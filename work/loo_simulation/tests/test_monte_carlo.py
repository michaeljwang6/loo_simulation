import json
from dataclasses import replace

import pytest

from loo_sim import (
    EstimatorConfig,
    MonteCarloConfig,
    ScenarioConfig,
    config_fingerprint,
    config_from_dict,
    config_to_dict,
    load_monte_carlo_config,
    load_monte_carlo_results,
    merge_monte_carlo_results,
    merge_saved_monte_carlo_results,
    run_monte_carlo,
    save_monte_carlo_results,
    shard_replication_indices,
)


def _lightweight_config() -> MonteCarloConfig:
    return MonteCarloConfig(
        scenarios=(
            ScenarioConfig(
                name="test_additive",
                population_kind="free_factor",
                population_kwargs={
                    "n_workers": 24,
                    "n_firms": 6,
                    "rank": 0,
                    "singular_values": (),
                    "common_sorting": 0.4,
                },
                panel_kwargs={
                    "n_periods": 6,
                    "redraw_probability": 0.8,
                    "error_sd": 0.25,
                },
                true_rank=0,
                plugin_ranks=(0,),
            ),
        ),
        replications=2,
        seed=9917,
        estimators=EstimatorConfig(
            run_low_rank=True,
            run_bic=False,
            run_fe_kss=False,
            run_bs20=False,
            run_blm=False,
            low_rank_n_starts=1,
            low_rank_max_iterations=100,
        ),
    )


def test_runner_is_reproducible_and_keeps_target_types_separate() -> None:
    config = _lightweight_config()

    first = run_monte_carlo(config)
    second = run_monte_carlo(config)

    assert first.records == second.records
    assert first.attempts == second.attempts
    assert len(first.attempts) == 2
    assert all(
        attempt.estimator == "project_plugin_r0"
        for attempt in first.attempts
    )
    assert all(
        "functionally_stable=" in attempt.message
        and "near_optimal_starts=" in attempt.message
        and "edge_mean_rmse=" in attempt.message
        for attempt in first.attempts
    )
    assert {record.target_type for record in first.records} == {
        "analysis_sample_project",
        "population_project",
    }
    assert {record.metric for record in first.records} == {
        "q_f",
        "h_f",
        "rho_h",
        "c_assign",
    }

    summary = next(
        row
        for row in first.summaries()
        if row.metric == "q_f"
        and row.target_type == "population_project"
    )
    assert summary.n_attempts == 2
    assert summary.n_estimates == 2
    assert summary.n_failure == 0
    assert summary.bias_monte_carlo_se == pytest.approx(
        summary.error_standard_deviation
        / summary.n_estimates**0.5
    )


def test_conditional_summary_excludes_unstable_returned_values() -> None:
    result = run_monte_carlo(_lightweight_config())
    attempts = (
        result.attempts[0],
        replace(result.attempts[1], status="unstable"),
    )
    mixed = replace(result, attempts=attempts)

    unconditional = next(
        row
        for row in mixed.summaries()
        if row.metric == "q_f"
        and row.target_type == "population_project"
    )
    stable = next(
        row
        for row in mixed.summaries(
            included_statuses=("success",)
        )
        if row.metric == "q_f"
        and row.target_type == "population_project"
    )

    assert unconditional.n_attempts == 2
    assert unconditional.n_estimates == 2
    assert stable.n_attempts == 2
    assert stable.n_estimates == 1
    assert stable.n_success == 1
    assert stable.n_unstable == 1


def test_unconditional_summary_excludes_unsupported_values() -> None:
    result = run_monte_carlo(_lightweight_config())
    unsupported_attempt = replace(result.attempts[1], status="unsupported")
    audited = replace(
        result,
        attempts=(result.attempts[0], unsupported_attempt),
    )

    summary = next(
        row
        for row in audited.summaries()
        if row.metric == "q_f"
        and row.target_type == "population_project"
    )
    attempt_summary = audited.attempt_summaries()[0]

    assert summary.n_attempts == 2
    assert summary.n_estimates == 1
    assert summary.n_unsupported == 1
    assert attempt_summary.n_unsupported == 1
    assert attempt_summary.unsupported_rate == 0.5


def test_configuration_json_round_trip(tmp_path) -> None:
    config = _lightweight_config()
    value = config_to_dict(config)
    reconstructed = config_from_dict(value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    loaded = load_monte_carlo_config(path)

    assert reconstructed == config
    assert loaded == config


def test_full_dgp_estimator_matrix_config_is_declared() -> None:
    config = load_monte_carlo_config(
        "configs/dgp_estimator_matrix_pilot.json"
    )

    assert [scenario.population_kind for scenario in config.scenarios] == [
        "akm",
        "crippa",
        "blm",
        "low_rank",
        "gklp",
    ]
    assert config.estimators.run_fe_kss
    assert config.estimators.run_blm
    assert config.estimators.run_bs20
    assert config.estimators.run_low_rank
    assert config.estimators.blm_worker_types == 2
    assert config.estimators.blm_firm_types == 3
    assert all(
        scenario.blm_worker_types is None
        and scenario.blm_firm_types is None
        for scenario in config.scenarios
    )


def test_seed_groups_use_common_random_numbers() -> None:
    scenarios = tuple(
        ScenarioConfig(
            name=name,
            population_kind="free_factor",
            population_kwargs={
                "n_workers": 16,
                "n_firms": 5,
                "rank": 0,
                "singular_values": (),
                "common_sorting": sorting,
            },
            panel_kwargs={
                "n_periods": 5,
                "redraw_probability": 0.8,
                "error_sd": 0.2,
            },
            true_rank=0,
            plugin_ranks=(0,),
            seed_group=77,
        )
        for name, sorting in (
            ("independent", 0.0),
            ("sorted", 0.5),
        )
    )
    config = MonteCarloConfig(
        scenarios=scenarios,
        replications=1,
        seed=1901,
        estimators=EstimatorConfig(
            run_bic=False,
            run_fe_kss=False,
            run_bs20=False,
            run_blm=False,
            low_rank_n_starts=1,
            low_rank_max_iterations=100,
        ),
    )

    result = run_monte_carlo(config)

    assert len(result.attempts) == 2
    seed_triplets = {
        (
            attempt.population_seed,
            attempt.panel_seed,
            attempt.estimator_seed,
        )
        for attempt in result.attempts
    }
    assert len(seed_triplets) == 1


def test_result_persistence_writes_declared_tables(tmp_path) -> None:
    result = run_monte_carlo(_lightweight_config())

    output = save_monte_carlo_results(result, tmp_path / "results")

    expected = {
        "attempt_summary.csv",
        "attempts.csv",
        "conditional_stable_summary.csv",
        "config.json",
        "metadata.json",
        "records.csv",
        "summary.csv",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert (output / "records.csv").read_text(
        encoding="utf-8"
    ).startswith("scenario,replication,population_seed")
    metadata = json.loads(
        (output / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["record_count"] == len(result.records)
    assert metadata["attempt_count"] == len(result.attempts)
    assert metadata["conditional_stable_summary_count"] == len(
        result.summaries(included_statuses=("success",))
    )
    assert metadata["replication_indices"] == [0, 1]
    assert metadata["config_fingerprint"] == config_fingerprint(
        result.config
    )
    assert load_monte_carlo_results(output) == result


def test_disjoint_shards_merge_to_exact_unsharded_result(tmp_path) -> None:
    config = replace(_lightweight_config(), replications=4)
    full = run_monte_carlo(config)
    first_indices = shard_replication_indices(
        config.replications,
        shard_index=0,
        shard_count=2,
    )
    second_indices = shard_replication_indices(
        config.replications,
        shard_index=1,
        shard_count=2,
    )
    first = run_monte_carlo(
        config,
        replication_indices=first_indices,
    )
    second = run_monte_carlo(
        config,
        replication_indices=second_indices,
    )

    merged = merge_monte_carlo_results((second, first))

    assert first_indices == (0, 2)
    assert second_indices == (1, 3)
    assert merged == full

    first_path = save_monte_carlo_results(first, tmp_path / "first")
    second_path = save_monte_carlo_results(
        second,
        tmp_path / "second",
    )
    merged_path = merge_saved_monte_carlo_results(
        (first_path, second_path),
        tmp_path / "merged",
    )
    assert load_monte_carlo_results(merged_path) == full


def test_merge_rejects_overlap_and_incomplete_coverage() -> None:
    config = replace(_lightweight_config(), replications=3)
    first = run_monte_carlo(
        config,
        replication_indices=(0, 1),
    )
    overlap = run_monte_carlo(
        config,
        replication_indices=(1, 2),
    )

    with pytest.raises(ValueError, match="overlapping"):
        merge_monte_carlo_results((first, overlap))
    with pytest.raises(ValueError, match="missing"):
        merge_monte_carlo_results((first,))

    partial = merge_monte_carlo_results(
        (first,),
        require_complete=False,
    )
    assert partial.replication_indices == (0, 1)
