import json

from loo_sim import (
    EstimatorConfig,
    MonteCarloConfig,
    ScenarioConfig,
    config_from_dict,
    config_to_dict,
    load_monte_carlo_config,
    run_monte_carlo,
    save_monte_carlo_results,
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


def test_configuration_json_round_trip(tmp_path) -> None:
    config = _lightweight_config()
    value = config_to_dict(config)
    reconstructed = config_from_dict(value)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    loaded = load_monte_carlo_config(path)

    assert reconstructed == config
    assert loaded == config


def test_result_persistence_writes_declared_tables(tmp_path) -> None:
    result = run_monte_carlo(_lightweight_config())

    output = save_monte_carlo_results(result, tmp_path / "results")

    expected = {
        "attempt_summary.csv",
        "attempts.csv",
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
