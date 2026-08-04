"""Configuration-driven Monte Carlo experiments for the LOO project.

The runner deliberately treats the current project procedure as a low-rank
plug-in without the unfinished leave-out correction.  It keeps native
procedure targets separate from the project's schedule-based targets so that
estimation error and estimand differences are visible in different rows.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .dgp import (
    GroupedPopulationDGP,
    PopulationDGP,
    generate_akm_population,
    generate_crippa_population,
    generate_gklp_population,
    generate_grouped_population,
    generate_population,
)
from .low_rank import (
    LowRankPluginResult,
    fit_low_rank_plugin,
    select_low_rank_bic,
)
from .panel import PanelData, sample_panel
from .pytwoway_estimators import (
    align_blm_cell_means,
    estimate_blm,
    estimate_bs20,
    estimate_fe_kss,
)
from .targets import (
    ProcedureTargets,
    compute_blm_evaluation_groups,
    compute_blm_grouped_target,
    compute_procedure_targets,
)
from .truth import PopulationTruth, compute_population_truth


PopulationKind = Literal[
    "akm",
    "crippa",
    "blm",
    "low_rank",
    "gklp",
    "free_factor",
    "grouped",
]
AttemptStatus = Literal["success", "unstable", "failure"]


@dataclass(frozen=True)
class ScenarioConfig:
    """One rung of the population and observation-design ladder."""

    name: str
    population_kind: PopulationKind
    population_kwargs: Mapping[str, Any]
    panel_kwargs: Mapping[str, Any]
    true_rank: int
    plugin_ranks: tuple[int, ...]
    seed_group: int | None = None
    blm_worker_types: int | None = None
    blm_firm_types: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Scenario names cannot be empty.")
        population_kinds = {
            "akm",
            "crippa",
            "blm",
            "low_rank",
            "gklp",
            "free_factor",
            "grouped",
        }
        if self.population_kind not in population_kinds:
            raise ValueError(
                "population_kind must identify one of the declared DGPs."
            )
        if self.true_rank < 0:
            raise ValueError("true_rank cannot be negative.")
        if (
            "rank" in self.population_kwargs
            and int(self.population_kwargs["rank"]) != self.true_rank
        ):
            raise ValueError(
                "true_rank must equal population_kwargs['rank']."
            )
        if not self.plugin_ranks:
            raise ValueError("plugin_ranks cannot be empty.")
        if any(rank < 0 for rank in self.plugin_ranks):
            raise ValueError("plugin_ranks cannot contain negative values.")
        if self.seed_group is not None and self.seed_group < 0:
            raise ValueError("seed_group cannot be negative.")
        if "seed" in self.population_kwargs or "seed" in self.panel_kwargs:
            raise ValueError(
                "Scenario kwargs cannot set seeds; the runner assigns them."
            )
        blm_values = (self.blm_worker_types, self.blm_firm_types)
        if any(value is not None for value in blm_values):
            if self.population_kind not in ("grouped", "blm"):
                raise ValueError(
                    "Legacy scenario-level BLM type counts are only valid "
                    "for grouped populations."
                )
            if any(value is None or value < 1 for value in blm_values):
                raise ValueError(
                    "Both positive BLM worker and firm type counts are "
                    "required."
                )
            if (
                "n_worker_types" in self.population_kwargs
                and int(self.population_kwargs["n_worker_types"])
                != self.blm_worker_types
            ):
                raise ValueError(
                    "blm_worker_types must match the grouped DGP."
                )
            if (
                "n_firm_types" in self.population_kwargs
                and int(self.population_kwargs["n_firm_types"])
                != self.blm_firm_types
            ):
                raise ValueError(
                    "blm_firm_types must match the grouped DGP."
                )


@dataclass(frozen=True)
class EstimatorConfig:
    """Estimator switches and numerical settings shared across scenarios."""

    run_low_rank: bool = True
    run_bic: bool = True
    run_fe_kss: bool = True
    run_bs20: bool = True
    run_blm: bool = True
    low_rank_minimum_degree: int | None = None
    low_rank_n_starts: int = 3
    low_rank_tolerance: float = 1e-6
    low_rank_max_iterations: int = 300
    fe_exact: bool = False
    blm_variants: tuple[str, ...] = ("oracle", "estimated")
    blm_n_init: int = 4
    blm_n_best: int = 2
    blm_n_iterations: int = 250
    blm_threshold: float = 1e-6
    blm_cdf_resolution: int = 10
    blm_worker_types: int | None = None
    blm_firm_types: int | None = None

    def __post_init__(self) -> None:
        if self.low_rank_n_starts < 1:
            raise ValueError("low_rank_n_starts must be positive.")
        if (
            self.low_rank_minimum_degree is not None
            and self.low_rank_minimum_degree < 1
        ):
            raise ValueError(
                "low_rank_minimum_degree must be positive when supplied."
            )
        if self.low_rank_tolerance <= 0:
            raise ValueError("low_rank_tolerance must be positive.")
        if self.low_rank_max_iterations < 1:
            raise ValueError("low_rank_max_iterations must be positive.")
        variants = tuple(dict.fromkeys(self.blm_variants))
        if any(value not in ("oracle", "estimated") for value in variants):
            raise ValueError(
                "blm_variants may contain only 'oracle' and 'estimated'."
            )
        object.__setattr__(self, "blm_variants", variants)
        if self.run_blm and not variants:
            raise ValueError(
                "At least one BLM variant is required when run_blm is true."
            )
        if self.blm_n_init < 1:
            raise ValueError("blm_n_init must be positive.")
        if self.blm_n_best < 1 or self.blm_n_best > self.blm_n_init:
            raise ValueError(
                "blm_n_best must lie between one and blm_n_init."
            )
        if self.blm_n_iterations < 1:
            raise ValueError("blm_n_iterations must be positive.")
        if self.blm_threshold <= 0:
            raise ValueError("blm_threshold must be positive.")
        if self.blm_cdf_resolution < 2:
            raise ValueError("blm_cdf_resolution must be at least two.")
        blm_type_counts = (self.blm_worker_types, self.blm_firm_types)
        if any(value is not None for value in blm_type_counts):
            if any(
                value is None or value < 1
                for value in blm_type_counts
            ):
                raise ValueError(
                    "Both positive BLM worker and firm type counts are "
                    "required when either is supplied."
                )


@dataclass(frozen=True)
class MonteCarloConfig:
    """Complete reproducible Monte Carlo specification."""

    scenarios: tuple[ScenarioConfig, ...]
    replications: int = 100
    seed: int = 20260729
    estimators: EstimatorConfig = EstimatorConfig()

    def __post_init__(self) -> None:
        if not self.scenarios:
            raise ValueError("At least one scenario is required.")
        if self.replications < 1:
            raise ValueError("replications must be positive.")
        if self.seed < 0:
            raise ValueError("seed cannot be negative.")
        names = [scenario.name for scenario in self.scenarios]
        if len(names) != len(set(names)):
            raise ValueError("Scenario names must be unique.")


@dataclass(frozen=True)
class MonteCarloRecord:
    """One scalar estimate paired with one declared population target."""

    scenario: str
    replication: int
    population_seed: int
    panel_seed: int
    estimator_seed: int
    estimator: str
    metric: str
    target_type: str
    estimate: float
    target: float
    error: float
    squared_error: float
    n_observations: int
    n_workers: int
    n_firms: int


@dataclass(frozen=True)
class EstimatorAttempt:
    """Execution and stability diagnostic for one estimator replication."""

    scenario: str
    replication: int
    population_seed: int
    panel_seed: int
    estimator_seed: int
    estimator: str
    status: AttemptStatus
    message: str
    n_observations: int
    n_workers: int
    n_firms: int


@dataclass(frozen=True)
class MonteCarloSummary:
    """Bias, dispersion, RMSE, retention, and failure counts."""

    scenario: str
    estimator: str
    metric: str
    target_type: str
    n_attempts: int
    n_estimates: int
    n_success: int
    n_unstable: int
    n_failure: int
    mean_estimate: float
    mean_target: float
    bias: float
    standard_deviation: float
    error_standard_deviation: float
    bias_monte_carlo_se: float
    rmse: float
    mean_observations: float
    mean_workers: float
    mean_firms: float


@dataclass(frozen=True)
class EstimatorAttemptSummary:
    """Estimator-level completion and stability rates."""

    scenario: str
    estimator: str
    n_attempts: int
    n_success: int
    n_unstable: int
    n_failure: int
    success_rate: float
    unstable_rate: float
    failure_rate: float


@dataclass(frozen=True)
class MonteCarloResult:
    """Long-form estimates, attempt diagnostics, and derived summaries."""

    config: MonteCarloConfig
    records: tuple[MonteCarloRecord, ...]
    attempts: tuple[EstimatorAttempt, ...]
    replication_indices: tuple[int, ...]

    def attempt_summaries(self) -> tuple[EstimatorAttemptSummary, ...]:
        """Aggregate attempt status even when every estimate failed."""

        groups: dict[
            tuple[str, str], list[EstimatorAttempt]
        ] = defaultdict(list)
        for attempt in self.attempts:
            groups[(attempt.scenario, attempt.estimator)].append(attempt)

        output: list[EstimatorAttemptSummary] = []
        for (scenario, estimator), attempts in sorted(groups.items()):
            total = len(attempts)
            success = sum(
                attempt.status == "success" for attempt in attempts
            )
            unstable = sum(
                attempt.status == "unstable" for attempt in attempts
            )
            failure = sum(
                attempt.status == "failure" for attempt in attempts
            )
            output.append(
                EstimatorAttemptSummary(
                    scenario=scenario,
                    estimator=estimator,
                    n_attempts=total,
                    n_success=success,
                    n_unstable=unstable,
                    n_failure=failure,
                    success_rate=success / total,
                    unstable_rate=unstable / total,
                    failure_rate=failure / total,
                )
            )
        return tuple(output)

    def summaries(
        self,
        *,
        included_statuses: Iterable[AttemptStatus] | None = None,
    ) -> tuple[MonteCarloSummary, ...]:
        """Aggregate records, optionally restricting returned-value status.

        The default is the unconditional procedure-performance summary and
        includes values returned by unstable fits. Passing
        ``included_statuses=("success",)`` produces the separately labeled
        conditional-on-stability robustness summary. Attempt counts always
        describe the full set of attempts so instability remains visible.
        """

        selected_statuses: frozenset[AttemptStatus] | None
        if included_statuses is None:
            selected_statuses = None
        else:
            selected_statuses = frozenset(included_statuses)
            valid_statuses = {"success", "unstable", "failure"}
            if (
                not selected_statuses
                or not selected_statuses.issubset(valid_statuses)
            ):
                raise ValueError(
                    "included_statuses must contain one or more valid "
                    "attempt statuses."
                )

        attempt_status = {
            (
                attempt.scenario,
                attempt.replication,
                attempt.estimator,
            ): attempt.status
            for attempt in self.attempts
        }

        record_groups: dict[
            tuple[str, str, str, str], list[MonteCarloRecord]
        ] = defaultdict(list)
        for record in self.records:
            key = (
                record.scenario,
                record.estimator,
                record.metric,
                record.target_type,
            )
            records_for_key = record_groups[key]
            if selected_statuses is None or attempt_status[
                (
                    record.scenario,
                    record.replication,
                    record.estimator,
                )
            ] in selected_statuses:
                records_for_key.append(record)

        attempt_groups: dict[
            tuple[str, str], list[EstimatorAttempt]
        ] = defaultdict(list)
        for attempt in self.attempts:
            attempt_groups[
                (attempt.scenario, attempt.estimator)
            ].append(attempt)

        summaries: list[MonteCarloSummary] = []
        for key in sorted(record_groups):
            scenario, estimator, metric, target_type = key
            records = record_groups[key]
            attempts = attempt_groups[(scenario, estimator)]
            estimate = np.asarray(
                [record.estimate for record in records],
                dtype=float,
            )
            target = np.asarray(
                [record.target for record in records],
                dtype=float,
            )
            error = estimate - target
            finite = np.isfinite(estimate) & np.isfinite(target)
            finite_estimate = estimate[finite]
            finite_target = target[finite]
            finite_error = error[finite]
            error_standard_deviation = _sample_sd_or_nan(
                finite_error
            )
            summaries.append(
                MonteCarloSummary(
                    scenario=scenario,
                    estimator=estimator,
                    metric=metric,
                    target_type=target_type,
                    n_attempts=len(attempts),
                    n_estimates=int(np.sum(finite)),
                    n_success=sum(
                        attempt.status == "success"
                        for attempt in attempts
                    ),
                    n_unstable=sum(
                        attempt.status == "unstable"
                        for attempt in attempts
                    ),
                    n_failure=sum(
                        attempt.status == "failure"
                        for attempt in attempts
                    ),
                    mean_estimate=_mean_or_nan(finite_estimate),
                    mean_target=_mean_or_nan(finite_target),
                    bias=_mean_or_nan(finite_error),
                    standard_deviation=_sample_sd_or_nan(
                        finite_estimate
                    ),
                    error_standard_deviation=error_standard_deviation,
                    bias_monte_carlo_se=(
                        error_standard_deviation
                        / np.sqrt(finite_error.size)
                        if finite_error.size
                        else float("nan")
                    ),
                    rmse=_rmse_or_nan(finite_error),
                    mean_observations=_mean_or_nan(
                        np.asarray(
                            [
                                record.n_observations
                                for record in records
                            ],
                            dtype=float,
                        )
                    ),
                    mean_workers=_mean_or_nan(
                        np.asarray(
                            [
                                record.n_workers
                                for record in records
                            ],
                            dtype=float,
                        )
                    ),
                    mean_firms=_mean_or_nan(
                        np.asarray(
                            [
                                record.n_firms
                                for record in records
                            ],
                            dtype=float,
                        )
                    ),
                )
            )
        return tuple(summaries)


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else float("nan")


def _sample_sd_or_nan(values: np.ndarray) -> float:
    return (
        float(np.std(values, ddof=1))
        if values.size >= 2
        else float("nan")
    )


def _rmse_or_nan(errors: np.ndarray) -> float:
    return (
        float(np.sqrt(np.mean(errors**2)))
        if errors.size
        else float("nan")
    )


def _scalar_record(
    *,
    scenario: ScenarioConfig,
    replication: int,
    seeds: tuple[int, int, int],
    estimator: str,
    metric: str,
    target_type: str,
    estimate: float,
    target: float,
    sample: tuple[int, int, int],
) -> MonteCarloRecord:
    population_seed, panel_seed, estimator_seed = seeds
    error = float(estimate - target)
    return MonteCarloRecord(
        scenario=scenario.name,
        replication=replication,
        population_seed=population_seed,
        panel_seed=panel_seed,
        estimator_seed=estimator_seed,
        estimator=estimator,
        metric=metric,
        target_type=target_type,
        estimate=float(estimate),
        target=float(target),
        error=error,
        squared_error=float(error**2),
        n_observations=sample[0],
        n_workers=sample[1],
        n_firms=sample[2],
    )


def _attempt(
    *,
    scenario: ScenarioConfig,
    replication: int,
    seeds: tuple[int, int, int],
    estimator: str,
    status: AttemptStatus,
    message: str,
    sample: tuple[int, int, int],
) -> EstimatorAttempt:
    population_seed, panel_seed, estimator_seed = seeds
    return EstimatorAttempt(
        scenario=scenario.name,
        replication=replication,
        population_seed=population_seed,
        panel_seed=panel_seed,
        estimator_seed=estimator_seed,
        estimator=estimator,
        status=status,
        message=message,
        n_observations=sample[0],
        n_workers=sample[1],
        n_firms=sample[2],
    )


def _project_metrics(truth: PopulationTruth) -> dict[str, float]:
    return {
        "q_f": truth.q_f,
        "h_f": truth.h_f,
        "rho_h": truth.rho_h,
        "c_assign": truth.c_assign,
    }


def _panel_sample(panel: PanelData) -> tuple[int, int, int]:
    return (
        panel.n_observations,
        panel.n_workers,
        panel.n_firms_observed,
    )


def _low_rank_sample(
    estimate: LowRankPluginResult,
) -> tuple[int, int, int]:
    return (
        estimate.sample.observations,
        estimate.sample.workers,
        estimate.sample.firms,
    )


def _append_project_records(
    records: list[MonteCarloRecord],
    *,
    scenario: ScenarioConfig,
    replication: int,
    seeds: tuple[int, int, int],
    estimator: str,
    estimate: PopulationTruth,
    target: PopulationTruth,
    target_type: str,
    sample: tuple[int, int, int],
) -> None:
    estimate_metrics = _project_metrics(estimate)
    target_metrics = _project_metrics(target)
    for metric in estimate_metrics:
        records.append(
            _scalar_record(
                scenario=scenario,
                replication=replication,
                seeds=seeds,
                estimator=estimator,
                metric=metric,
                target_type=target_type,
                estimate=estimate_metrics[metric],
                target=target_metrics[metric],
                sample=sample,
            )
        )


def _run_low_rank(
    *,
    scenario: ScenarioConfig,
    replication: int,
    population: PopulationDGP,
    panel: PanelData,
    targets: ProcedureTargets,
    seed_triplet: tuple[int, int, int],
    config: EstimatorConfig,
    records: list[MonteCarloRecord],
    attempts: list[EstimatorAttempt],
) -> None:
    candidates = tuple(sorted(set(scenario.plugin_ranks)))
    expected = [f"project_plugin_r{rank}" for rank in candidates]
    if config.run_bic:
        expected.append("project_plugin_bic")
    try:
        if len(candidates) > 1 or config.run_bic:
            selection = select_low_rank_bic(
                panel,
                candidate_ranks=candidates,
                minimum_degree=config.low_rank_minimum_degree,
                n_starts=config.low_rank_n_starts,
                tolerance=config.low_rank_tolerance,
                max_iterations=config.low_rank_max_iterations,
                seed=seed_triplet[2],
            )
            estimates = selection.estimates
            selected = selection.selected
        else:
            estimates = (
                fit_low_rank_plugin(
                    panel,
                    rank=candidates[0],
                    minimum_degree=config.low_rank_minimum_degree,
                    n_starts=config.low_rank_n_starts,
                    tolerance=config.low_rank_tolerance,
                    max_iterations=config.low_rank_max_iterations,
                    seed=seed_triplet[2],
                ),
            )
            selected = estimates[0]
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for estimator in expected:
            attempts.append(
                _attempt(
                    scenario=scenario,
                    replication=replication,
                    seeds=seed_triplet,
                    estimator=estimator,
                    status="failure",
                    message=message,
                    sample=_panel_sample(panel),
                )
            )
        return

    for estimate in estimates:
        estimator = f"project_plugin_r{estimate.rank}"
        _record_one_low_rank(
            scenario=scenario,
            replication=replication,
            population=population,
            population_target=targets.project,
            seed_triplet=seed_triplet,
            estimator=estimator,
            estimate=estimate,
            records=records,
            attempts=attempts,
        )

    if config.run_bic:
        estimator = "project_plugin_bic"
        _record_one_low_rank(
            scenario=scenario,
            replication=replication,
            population=population,
            population_target=targets.project,
            seed_triplet=seed_triplet,
            estimator=estimator,
            estimate=selected,
            records=records,
            attempts=attempts,
        )
        sample = _low_rank_sample(selected)
        records.append(
            _scalar_record(
                scenario=scenario,
                replication=replication,
                seeds=seed_triplet,
                estimator=estimator,
                metric="selected_rank",
                target_type="rank_diagnostic",
                estimate=float(selected.rank),
                target=float(scenario.true_rank),
                sample=sample,
            )
        )


def _record_one_low_rank(
    *,
    scenario: ScenarioConfig,
    replication: int,
    population: PopulationDGP,
    population_target: PopulationTruth,
    seed_triplet: tuple[int, int, int],
    estimator: str,
    estimate: LowRankPluginResult,
    records: list[MonteCarloRecord],
    attempts: list[EstimatorAttempt],
) -> None:
    sample = _low_rank_sample(estimate)
    status: AttemptStatus = (
        "success" if estimate.functionally_stable else "unstable"
    )
    message = (
        f"converged={estimate.converged}; "
        f"iterations={estimate.iterations}; "
        f"functionally_stable={estimate.functionally_stable}; "
        f"near_optimal_starts={estimate.near_optimal_starts}; "
        f"q_f_spread={estimate.q_f_spread:.8g}; "
        f"h_f_spread={estimate.h_f_spread:.8g}; "
        f"c_assign_spread={estimate.c_assign_spread:.8g}; "
        f"rectangles={estimate.sample.rectangles}; "
        f"edge_mean_rmse={estimate.edge_mean_rmse:.8g}"
    )
    attempts.append(
        _attempt(
            scenario=scenario,
            replication=replication,
            seeds=seed_triplet,
            estimator=estimator,
            status=status,
            message=message,
            sample=sample,
        )
    )
    _append_project_records(
        records,
        scenario=scenario,
        replication=replication,
        seeds=seed_triplet,
        estimator=estimator,
        estimate=estimate.functionals,
        target=population_target,
        target_type="population_project",
        sample=sample,
    )
    analysis_target = compute_population_truth(
        population.schedule[
            np.ix_(estimate.worker_ids, estimate.firm_ids)
        ],
        estimate.assignment,
    )
    _append_project_records(
        records,
        scenario=scenario,
        replication=replication,
        seeds=seed_triplet,
        estimator=estimator,
        estimate=estimate.functionals,
        target=analysis_target,
        target_type="analysis_sample_project",
        sample=sample,
    )


def _run_fe_kss(
    *,
    scenario: ScenarioConfig,
    replication: int,
    panel: PanelData,
    targets: ProcedureTargets,
    seed_triplet: tuple[int, int, int],
    config: EstimatorConfig,
    records: list[MonteCarloRecord],
    attempts: list[EstimatorAttempt],
) -> None:
    estimator_names = ("akm_fe", "kss_ho", "kss_he")
    try:
        estimate = estimate_fe_kss(
            panel,
            seed=seed_triplet[2],
            exact=config.fe_exact,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for estimator in estimator_names:
            attempts.append(
                _attempt(
                    scenario=scenario,
                    replication=replication,
                    seeds=seed_triplet,
                    estimator=estimator,
                    status="failure",
                    message=message,
                    sample=_panel_sample(panel),
                )
            )
        return

    sample = (
        estimate.sample.observations,
        estimate.sample.workers,
        estimate.sample.firms,
    )
    values = {
        "akm_fe": (
            estimate.var_psi_fe,
            estimate.cov_psi_alpha_fe,
        ),
        "kss_ho": (
            estimate.var_psi_ho,
            estimate.cov_psi_alpha_ho,
        ),
        "kss_he": (
            estimate.var_psi_he,
            estimate.cov_psi_alpha_he,
        ),
    }
    target_pairs = (
        (
            "native_akm",
            targets.akm.firm_variance,
            targets.akm.covariance,
        ),
        (
            "population_project",
            targets.project.q_f,
            targets.project.c_assign,
        ),
    )
    for estimator, (firm_variance, covariance) in values.items():
        attempts.append(
            _attempt(
                scenario=scenario,
                replication=replication,
                seeds=seed_triplet,
                estimator=estimator,
                status="success",
                message="PyTwoWay FE/KSS fit completed",
                sample=sample,
            )
        )
        for target_type, target_variance, target_covariance in target_pairs:
            records.extend(
                [
                    _scalar_record(
                        scenario=scenario,
                        replication=replication,
                        seeds=seed_triplet,
                        estimator=estimator,
                        metric="firm_variance",
                        target_type=target_type,
                        estimate=firm_variance,
                        target=target_variance,
                        sample=sample,
                    ),
                    _scalar_record(
                        scenario=scenario,
                        replication=replication,
                        seeds=seed_triplet,
                        estimator=estimator,
                        metric="worker_firm_covariance",
                        target_type=target_type,
                        estimate=covariance,
                        target=target_covariance,
                        sample=sample,
                    ),
                ]
            )


def _run_bs20(
    *,
    scenario: ScenarioConfig,
    replication: int,
    panel: PanelData,
    targets: ProcedureTargets,
    seed_triplet: tuple[int, int, int],
    records: list[MonteCarloRecord],
    attempts: list[EstimatorAttempt],
) -> None:
    estimator = "bs20"
    try:
        estimate = estimate_bs20(panel)
    except Exception as exc:
        attempts.append(
            _attempt(
                scenario=scenario,
                replication=replication,
                seeds=seed_triplet,
                estimator=estimator,
                status="failure",
                message=f"{type(exc).__name__}: {exc}",
                sample=_panel_sample(panel),
            )
        )
        return

    sample = (
        estimate.sample.observations,
        estimate.sample.workers,
        estimate.sample.firms,
    )
    attempts.append(
        _attempt(
            scenario=scenario,
            replication=replication,
            seeds=seed_triplet,
            estimator=estimator,
            status="success",
            message="PyTwoWay BS20 fit completed",
            sample=sample,
        )
    )
    native = {
        "worker_variance": (
            estimate.var_worker_type,
            targets.project.bs_worker_variance,
        ),
        "firm_variance": (
            estimate.var_firm_type,
            targets.project.bs_firm_variance,
        ),
        "worker_firm_covariance": (
            estimate.covariance,
            targets.project.bs_covariance,
        ),
        "worker_firm_correlation": (
            estimate.correlation,
            targets.project.bs_correlation,
        ),
    }
    for metric, (value, target) in native.items():
        records.append(
            _scalar_record(
                scenario=scenario,
                replication=replication,
                seeds=seed_triplet,
                estimator=estimator,
                metric=metric,
                target_type="native_bs20",
                estimate=value,
                target=target,
                sample=sample,
            )
        )
    records.append(
        _scalar_record(
            scenario=scenario,
            replication=replication,
            seeds=seed_triplet,
            estimator=estimator,
            metric="worker_firm_covariance",
            target_type="population_project",
            estimate=estimate.covariance,
            target=targets.project.c_assign,
            sample=sample,
        )
    )


def _run_blm(
    *,
    scenario: ScenarioConfig,
    replication: int,
    population: PopulationDGP,
    panel: PanelData,
    targets: ProcedureTargets,
    seed_triplet: tuple[int, int, int],
    config: EstimatorConfig,
    records: list[MonteCarloRecord],
    attempts: list[EstimatorAttempt],
) -> None:
    n_worker_types = (
        config.blm_worker_types
        if config.blm_worker_types is not None
        else scenario.blm_worker_types
    )
    n_firm_types = (
        config.blm_firm_types
        if config.blm_firm_types is not None
        else scenario.blm_firm_types
    )
    if n_worker_types is None or n_firm_types is None:
        return
    true_grouped_target = (
        isinstance(population, GroupedPopulationDGP)
        and np.unique(population.worker_groups).size == n_worker_types
        and np.unique(population.firm_groups).size == n_firm_types
    )
    if true_grouped_target:
        worker_groups = population.worker_groups
        firm_groups = population.firm_groups
        grouping_method = "true_simulated_blm_types"
    else:
        evaluation_groups = compute_blm_evaluation_groups(
            population.schedule,
            population.assignment,
            n_worker_types=n_worker_types,
            n_firm_types=n_firm_types,
        )
        worker_groups = evaluation_groups.worker_groups
        firm_groups = evaluation_groups.firm_groups
        grouping_method = evaluation_groups.method
    target = compute_blm_grouped_target(
        population.schedule,
        population.assignment,
        worker_groups,
        firm_groups,
    )
    for variant_index, variant in enumerate(config.blm_variants):
        if variant == "oracle" and not true_grouped_target:
            continue
        estimator = f"blm_{variant}"
        variant_seed = int(
            np.random.SeedSequence(
                [seed_triplet[2], variant_index]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        seeds = (seed_triplet[0], seed_triplet[1], variant_seed)
        try:
            estimate = estimate_blm(
                panel,
                n_worker_types=n_worker_types,
                n_firm_types=n_firm_types,
                firm_groups=(
                    firm_groups
                    if variant == "oracle"
                    else None
                ),
                n_init=config.blm_n_init,
                n_best=config.blm_n_best,
                n_iterations=config.blm_n_iterations,
                threshold=config.blm_threshold,
                cdf_resolution=config.blm_cdf_resolution,
                seed=variant_seed,
            )
            alignment = align_blm_cell_means(
                estimate.stationary_cell_means,
                target.cell_means,
                allow_firm_permutation=variant != "oracle",
            )
            fitted_project = compute_population_truth(
                alignment.aligned_estimate,
                target.group_assignment,
            )
        except Exception as exc:
            attempts.append(
                _attempt(
                    scenario=scenario,
                    replication=replication,
                    seeds=seeds,
                    estimator=estimator,
                    status="failure",
                    message=f"{type(exc).__name__}: {exc}",
                    sample=_panel_sample(panel),
                )
            )
            continue

        sample = (
            estimate.sample.observations,
            estimate.sample.workers,
            estimate.sample.firms,
        )
        stable = (
            estimate.mover_likelihood_monotone
            and estimate.stayer_likelihood_monotone
        )
        attempts.append(
            _attempt(
                scenario=scenario,
                replication=replication,
                seeds=seeds,
                estimator=estimator,
                status="success" if stable else "unstable",
                message=(
                    "BLM likelihood paths passed monotonicity checks; "
                    f"evaluation_groups={grouping_method}"
                    if stable
                    else "BLM returned values but a likelihood path failed "
                    "its monotonicity check; "
                    f"evaluation_groups={grouping_method}"
                ),
                sample=sample,
            )
        )
        for worker_type in range(target.cell_means.shape[0]):
            for firm_type in range(target.cell_means.shape[1]):
                metric = (
                    f"cell_mean_l{worker_type}_k{firm_type}"
                )
                records.append(
                    _scalar_record(
                        scenario=scenario,
                        replication=replication,
                        seeds=seeds,
                        estimator=estimator,
                        metric=metric,
                        target_type=(
                            "native_blm_cell"
                            if true_grouped_target
                            else "blm_projection_cell"
                        ),
                        estimate=alignment.aligned_estimate[
                            worker_type, firm_type
                        ],
                        target=target.cell_means[
                            worker_type, firm_type
                        ],
                        sample=sample,
                    )
                )
        records.append(
            _scalar_record(
                scenario=scenario,
                replication=replication,
                seeds=seeds,
                estimator=estimator,
                metric="cell_mean_rmse",
                target_type="alignment_diagnostic",
                estimate=alignment.rmse,
                target=0.0,
                sample=sample,
            )
        )
        _append_project_records(
            records,
            scenario=scenario,
            replication=replication,
            seeds=seeds,
            estimator=estimator,
            estimate=fitted_project,
            target=target.project_functionals,
            target_type=(
                "grouped_population_project"
                if true_grouped_target
                else "blm_projection_project"
            ),
            sample=sample,
        )
        _append_project_records(
            records,
            scenario=scenario,
            replication=replication,
            seeds=seeds,
            estimator=estimator,
            estimate=fitted_project,
            target=targets.project,
            target_type="population_project",
            sample=sample,
        )


def _replication_seeds(
    master_seed: int,
    scenario_seed_group: int,
    replication: int,
) -> tuple[int, int, int, int, int]:
    state = np.random.SeedSequence(
        [master_seed, scenario_seed_group, replication]
    ).generate_state(5, dtype=np.uint32)
    return tuple(int(value) for value in state)  # type: ignore[return-value]


def _generate_replication(
    scenario: ScenarioConfig,
    *,
    population_seed: int,
    panel_seed: int,
) -> tuple[PopulationDGP, PanelData, ProcedureTargets]:
    population_kwargs = dict(scenario.population_kwargs)
    if scenario.population_kind in ("grouped", "blm"):
        population = generate_grouped_population(
            **population_kwargs,
            seed=population_seed,
        )
    elif scenario.population_kind in ("free_factor", "low_rank"):
        population = generate_population(
            **population_kwargs,
            seed=population_seed,
        )
    elif scenario.population_kind == "akm":
        population = generate_akm_population(
            **population_kwargs,
            seed=population_seed,
        )
    elif scenario.population_kind == "crippa":
        population = generate_crippa_population(
            **population_kwargs,
            seed=population_seed,
        )
    elif scenario.population_kind == "gklp":
        population = generate_gklp_population(
            **population_kwargs,
            seed=population_seed,
        )
    else:  # pragma: no cover - ScenarioConfig validates this branch away.
        raise ValueError(f"Unknown population kind: {scenario.population_kind}.")
    targets = compute_procedure_targets(
        population.schedule,
        population.assignment,
    )
    panel = sample_panel(
        population,
        **dict(scenario.panel_kwargs),
        seed=panel_seed,
    )
    return population, panel, targets


def _normalize_replication_indices(
    config: MonteCarloConfig,
    replication_indices: Iterable[int] | None,
) -> tuple[int, ...]:
    if replication_indices is None:
        return tuple(range(config.replications))
    indices = tuple(sorted(int(value) for value in replication_indices))
    if len(indices) != len(set(indices)):
        raise ValueError("replication_indices cannot contain duplicates.")
    if any(
        value < 0 or value >= config.replications
        for value in indices
    ):
        raise ValueError(
            "Every replication index must lie between zero and "
            "config.replications - 1."
        )
    return indices


def shard_replication_indices(
    replications: int,
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[int, ...]:
    """Assign global replication indices to one deterministic shard."""

    if replications < 1:
        raise ValueError("replications must be positive.")
    if shard_count < 1:
        raise ValueError("shard_count must be positive.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            "shard_index must lie between zero and shard_count - 1."
        )
    return tuple(range(shard_index, replications, shard_count))


def _record_sort_key(
    record: MonteCarloRecord,
) -> tuple[str, int, str, str, str]:
    return (
        record.scenario,
        record.replication,
        record.estimator,
        record.metric,
        record.target_type,
    )


def _attempt_sort_key(
    attempt: EstimatorAttempt,
) -> tuple[str, int, str]:
    return (
        attempt.scenario,
        attempt.replication,
        attempt.estimator,
    )


def run_monte_carlo(
    config: MonteCarloConfig,
    *,
    replication_indices: Iterable[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> MonteCarloResult:
    """Run selected global replications while isolating estimator failures."""

    selected_replications = _normalize_replication_indices(
        config,
        replication_indices,
    )
    records: list[MonteCarloRecord] = []
    attempts: list[EstimatorAttempt] = []
    for scenario_index, scenario in enumerate(config.scenarios):
        scenario_seed_group = (
            scenario_index
            if scenario.seed_group is None
            else scenario.seed_group
        )
        for replication in selected_replications:
            (
                population_seed,
                panel_seed,
                low_rank_seed,
                fe_seed,
                blm_seed,
            ) = _replication_seeds(
                config.seed,
                scenario_seed_group,
                replication,
            )
            if progress is not None:
                progress(
                    f"{scenario.name}: replication "
                    f"{replication + 1}/{config.replications}"
                )
            try:
                population, panel, targets = _generate_replication(
                    scenario,
                    population_seed=population_seed,
                    panel_seed=panel_seed,
                )
            except Exception as exc:
                attempts.append(
                    _attempt(
                        scenario=scenario,
                        replication=replication,
                        seeds=(
                            population_seed,
                            panel_seed,
                            population_seed,
                        ),
                        estimator="dgp_panel",
                        status="failure",
                        message=f"{type(exc).__name__}: {exc}",
                        sample=(0, 0, 0),
                    )
                )
                continue

            if config.estimators.run_low_rank:
                _run_low_rank(
                    scenario=scenario,
                    replication=replication,
                    population=population,
                    panel=panel,
                    targets=targets,
                    seed_triplet=(
                        population_seed,
                        panel_seed,
                        low_rank_seed,
                    ),
                    config=config.estimators,
                    records=records,
                    attempts=attempts,
                )
            if config.estimators.run_fe_kss:
                _run_fe_kss(
                    scenario=scenario,
                    replication=replication,
                    panel=panel,
                    targets=targets,
                    seed_triplet=(
                        population_seed,
                        panel_seed,
                        fe_seed,
                    ),
                    config=config.estimators,
                    records=records,
                    attempts=attempts,
                )
            if config.estimators.run_bs20:
                _run_bs20(
                    scenario=scenario,
                    replication=replication,
                    panel=panel,
                    targets=targets,
                    seed_triplet=(
                        population_seed,
                        panel_seed,
                        fe_seed,
                    ),
                    records=records,
                    attempts=attempts,
                )
            if config.estimators.run_blm:
                _run_blm(
                    scenario=scenario,
                    replication=replication,
                    population=population,
                    panel=panel,
                    targets=targets,
                    seed_triplet=(
                        population_seed,
                        panel_seed,
                        blm_seed,
                    ),
                    config=config.estimators,
                    records=records,
                    attempts=attempts,
                )

    return MonteCarloResult(
        config=config,
        records=tuple(sorted(records, key=_record_sort_key)),
        attempts=tuple(sorted(attempts, key=_attempt_sort_key)),
        replication_indices=selected_replications,
    )


def default_dgp_ladder(
    *,
    replications: int = 100,
    seed: int = 20260729,
) -> MonteCarloConfig:
    """Return the seven-rung baseline ladder described in the game plan."""

    free_panel = {
        "n_periods": 7,
        "redraw_probability": 0.75,
        "error_sd": 0.5,
    }
    free_size = {"n_workers": 80, "n_firms": 10}
    scenarios = (
        ScenarioConfig(
            name="additive_independent",
            population_kind="free_factor",
            population_kwargs={
                **free_size,
                "rank": 0,
                "singular_values": (),
            },
            panel_kwargs=free_panel,
            true_rank=0,
            plugin_ranks=(0, 1),
        ),
        ScenarioConfig(
            name="additive_common_sorting",
            population_kind="free_factor",
            population_kwargs={
                **free_size,
                "rank": 0,
                "singular_values": (),
                "common_sorting": 0.8,
            },
            panel_kwargs=free_panel,
            true_rank=0,
            plugin_ranks=(0, 1),
        ),
        ScenarioConfig(
            name="rank1_independent",
            population_kind="free_factor",
            population_kwargs={
                **free_size,
                "rank": 1,
                "singular_values": (1.0,),
            },
            panel_kwargs=free_panel,
            true_rank=1,
            plugin_ranks=(0, 1),
        ),
        ScenarioConfig(
            name="rank1_common_sorting",
            population_kind="free_factor",
            population_kwargs={
                **free_size,
                "rank": 1,
                "singular_values": (1.0,),
                "common_sorting": 0.8,
            },
            panel_kwargs=free_panel,
            true_rank=1,
            plugin_ranks=(0, 1),
        ),
        ScenarioConfig(
            name="rank1_interaction_sorting",
            population_kind="free_factor",
            population_kwargs={
                **free_size,
                "rank": 1,
                "singular_values": (1.0,),
                "interaction_sorting": 0.4,
            },
            panel_kwargs={**free_panel, "n_periods": 10},
            true_rank=1,
            plugin_ranks=(0, 1),
        ),
        ScenarioConfig(
            name="grouped_blm",
            population_kind="grouped",
            population_kwargs={
                "n_workers": 300,
                "n_firms": 18,
                "n_worker_types": 2,
                "n_firm_types": 3,
                "rank": 1,
                "singular_values": (1.0,),
                "common_sorting": 0.4,
                "interaction_sorting": 0.2,
            },
            panel_kwargs={
                "n_periods": 5,
                "redraw_probability": 0.35,
                "error_sd": 0.5,
            },
            true_rank=1,
            plugin_ranks=(0, 1),
            blm_worker_types=2,
            blm_firm_types=3,
        ),
        ScenarioConfig(
            name="rank2_misspecification",
            population_kind="free_factor",
            population_kwargs={
                **free_size,
                "rank": 2,
                "singular_values": (1.0, 0.5),
                "common_sorting": 0.4,
                "interaction_sorting": 0.4,
            },
            panel_kwargs={**free_panel, "n_periods": 15},
            true_rank=2,
            plugin_ranks=(0, 1, 2),
        ),
    )
    return MonteCarloConfig(
        scenarios=scenarios,
        replications=replications,
        seed=seed,
    )


def config_to_dict(config: MonteCarloConfig) -> dict[str, Any]:
    """Convert a configuration to a JSON-compatible dictionary."""

    value = asdict(config)
    estimator_value = value["estimators"]
    for name in ("blm_worker_types", "blm_firm_types"):
        if estimator_value[name] is None:
            estimator_value.pop(name)
    return value


def config_fingerprint(config: MonteCarloConfig) -> str:
    """Return a stable SHA-256 fingerprint of the resolved configuration."""

    encoded = json.dumps(
        config_to_dict(config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def config_from_dict(value: Mapping[str, Any]) -> MonteCarloConfig:
    """Validate and construct a configuration loaded from JSON."""

    scenario_values = value.get("scenarios")
    if not isinstance(scenario_values, (list, tuple)):
        raise ValueError("Configuration 'scenarios' must be a list.")
    scenarios_list: list[ScenarioConfig] = []
    for item in scenario_values:
        population_kwargs = dict(item.get("population_kwargs", {}))
        if "singular_values" in population_kwargs:
            population_kwargs["singular_values"] = tuple(
                float(value)
                for value in population_kwargs["singular_values"]
            )
        scenarios_list.append(
            ScenarioConfig(
                name=item["name"],
                population_kind=item["population_kind"],
                population_kwargs=population_kwargs,
                panel_kwargs=item.get("panel_kwargs", {}),
                true_rank=int(item["true_rank"]),
                plugin_ranks=tuple(
                    int(rank) for rank in item["plugin_ranks"]
                ),
                seed_group=(
                    None
                    if item.get("seed_group") is None
                    else int(item["seed_group"])
                ),
                blm_worker_types=(
                    None
                    if item.get("blm_worker_types") is None
                    else int(item["blm_worker_types"])
                ),
                blm_firm_types=(
                    None
                    if item.get("blm_firm_types") is None
                    else int(item["blm_firm_types"])
                ),
            )
        )
    scenarios = tuple(scenarios_list)
    estimator_value = dict(value.get("estimators", {}))
    if "blm_variants" in estimator_value:
        estimator_value["blm_variants"] = tuple(
            estimator_value["blm_variants"]
        )
    for name in ("blm_worker_types", "blm_firm_types"):
        if estimator_value.get(name) is not None:
            estimator_value[name] = int(estimator_value[name])
    estimators = EstimatorConfig(**estimator_value)
    return MonteCarloConfig(
        scenarios=scenarios,
        replications=int(value.get("replications", 100)),
        seed=int(value.get("seed", 20260729)),
        estimators=estimators,
    )


def load_monte_carlo_config(path: str | Path) -> MonteCarloConfig:
    """Load and validate a Monte Carlo configuration JSON file."""

    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("The top-level configuration must be an object.")
    return config_from_dict(value)


def _write_dataclass_csv(
    path: Path,
    rows: tuple[Any, ...],
    row_type: type[Any],
) -> None:
    names = [field.name for field in fields(row_type)]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _record_from_csv(row: Mapping[str, str]) -> MonteCarloRecord:
    return MonteCarloRecord(
        scenario=row["scenario"],
        replication=int(row["replication"]),
        population_seed=int(row["population_seed"]),
        panel_seed=int(row["panel_seed"]),
        estimator_seed=int(row["estimator_seed"]),
        estimator=row["estimator"],
        metric=row["metric"],
        target_type=row["target_type"],
        estimate=float(row["estimate"]),
        target=float(row["target"]),
        error=float(row["error"]),
        squared_error=float(row["squared_error"]),
        n_observations=int(row["n_observations"]),
        n_workers=int(row["n_workers"]),
        n_firms=int(row["n_firms"]),
    )


def _attempt_from_csv(row: Mapping[str, str]) -> EstimatorAttempt:
    status = row["status"]
    if status not in ("success", "unstable", "failure"):
        raise ValueError(f"Unknown estimator-attempt status: {status}.")
    return EstimatorAttempt(
        scenario=row["scenario"],
        replication=int(row["replication"]),
        population_seed=int(row["population_seed"]),
        panel_seed=int(row["panel_seed"]),
        estimator_seed=int(row["estimator_seed"]),
        estimator=row["estimator"],
        status=status,
        message=row["message"],
        n_observations=int(row["n_observations"]),
        n_workers=int(row["n_workers"]),
        n_firms=int(row["n_firms"]),
    )


def load_monte_carlo_results(
    input_directory: str | Path,
) -> MonteCarloResult:
    """Load and validate one saved full run or shard."""

    input_path = Path(input_directory)
    config = load_monte_carlo_config(input_path / "config.json")
    with (input_path / "metadata.json").open(
        encoding="utf-8"
    ) as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("Result metadata must be a JSON object.")

    expected_fingerprint = metadata.get("config_fingerprint")
    actual_fingerprint = config_fingerprint(config)
    if (
        expected_fingerprint is not None
        and expected_fingerprint != actual_fingerprint
    ):
        raise ValueError(
            f"Configuration fingerprint mismatch in {input_path}."
        )

    records = tuple(
        _record_from_csv(row)
        for row in _read_csv(input_path / "records.csv")
    )
    attempts = tuple(
        _attempt_from_csv(row)
        for row in _read_csv(input_path / "attempts.csv")
    )
    if int(metadata.get("record_count", -1)) != len(records):
        raise ValueError(f"Record count mismatch in {input_path}.")
    if int(metadata.get("attempt_count", -1)) != len(attempts):
        raise ValueError(f"Attempt count mismatch in {input_path}.")

    saved_indices = metadata.get("replication_indices")
    if saved_indices is None:
        replication_indices = tuple(
            sorted(
                {
                    record.replication for record in records
                }
                | {
                    attempt.replication for attempt in attempts
                }
            )
        )
    else:
        replication_indices = _normalize_replication_indices(
            config,
            (int(value) for value in saved_indices),
        )
    allowed = set(replication_indices)
    if any(record.replication not in allowed for record in records):
        raise ValueError(
            f"A record lies outside the declared shard in {input_path}."
        )
    if any(attempt.replication not in allowed for attempt in attempts):
        raise ValueError(
            f"An attempt lies outside the declared shard in {input_path}."
        )
    return MonteCarloResult(
        config=config,
        records=tuple(sorted(records, key=_record_sort_key)),
        attempts=tuple(sorted(attempts, key=_attempt_sort_key)),
        replication_indices=replication_indices,
    )


def merge_monte_carlo_results(
    results: Iterable[MonteCarloResult],
    *,
    require_complete: bool = True,
) -> MonteCarloResult:
    """Merge disjoint shards after strict configuration validation."""

    result_list = tuple(results)
    if not result_list:
        raise ValueError("At least one result is required for merging.")
    config = result_list[0].config
    fingerprint = config_fingerprint(config)
    scenario_names = {
        scenario.name for scenario in config.scenarios
    }
    replication_indices: set[int] = set()
    records: list[MonteCarloRecord] = []
    attempts: list[EstimatorAttempt] = []
    for result in result_list:
        if config_fingerprint(result.config) != fingerprint:
            raise ValueError(
                "Cannot merge results with different configurations."
            )
        normalized_indices = _normalize_replication_indices(
            config,
            result.replication_indices,
        )
        if normalized_indices != result.replication_indices:
            raise ValueError(
                "Shard replication indices must be sorted and unique."
            )
        shard_indices = set(result.replication_indices)
        overlap = replication_indices & shard_indices
        if overlap:
            raise ValueError(
                "Cannot merge overlapping replication indices: "
                f"{sorted(overlap)}."
            )
        replication_indices.update(shard_indices)
        if any(
            record.replication not in shard_indices
            or record.scenario not in scenario_names
            for record in result.records
        ):
            raise ValueError(
                "A scalar record is inconsistent with its shard or config."
            )
        if any(
            attempt.replication not in shard_indices
            or attempt.scenario not in scenario_names
            for attempt in result.attempts
        ):
            raise ValueError(
                "An estimator attempt is inconsistent with its shard "
                "or config."
            )
        records.extend(result.records)
        attempts.extend(result.attempts)

    expected = set(range(config.replications))
    if require_complete and replication_indices != expected:
        missing = sorted(expected - replication_indices)
        extra = sorted(replication_indices - expected)
        raise ValueError(
            "Merged shards do not cover the full configuration; "
            f"missing={missing}, extra={extra}."
        )

    record_keys = [_record_sort_key(record) for record in records]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("Merged results contain duplicate scalar records.")
    attempt_keys = [_attempt_sort_key(attempt) for attempt in attempts]
    if len(attempt_keys) != len(set(attempt_keys)):
        raise ValueError(
            "Merged results contain duplicate estimator attempts."
        )
    return MonteCarloResult(
        config=config,
        records=tuple(sorted(records, key=_record_sort_key)),
        attempts=tuple(sorted(attempts, key=_attempt_sort_key)),
        replication_indices=tuple(sorted(replication_indices)),
    )


def merge_saved_monte_carlo_results(
    input_directories: Iterable[str | Path],
    output_directory: str | Path,
    *,
    require_complete: bool = True,
) -> Path:
    """Load, merge, validate, and persist a collection of result shards."""

    merged = merge_monte_carlo_results(
        (
            load_monte_carlo_results(path)
            for path in input_directories
        ),
        require_complete=require_complete,
    )
    return save_monte_carlo_results(merged, output_directory)


def save_monte_carlo_results(
    result: MonteCarloResult,
    output_directory: str | Path,
) -> Path:
    """Persist long records, attempts, summaries, config, and metadata."""

    normalized_indices = _normalize_replication_indices(
        result.config,
        result.replication_indices,
    )
    if normalized_indices != result.replication_indices:
        raise ValueError(
            "result.replication_indices must be sorted and unique."
        )
    allowed = set(normalized_indices)
    if any(record.replication not in allowed for record in result.records):
        raise ValueError("A result record lies outside its declared shard.")
    if any(attempt.replication not in allowed for attempt in result.attempts):
        raise ValueError("A result attempt lies outside its declared shard.")

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summaries = result.summaries()
    stable_summaries = result.summaries(
        included_statuses=("success",)
    )
    attempt_summaries = result.attempt_summaries()
    _write_dataclass_csv(
        output / "records.csv",
        result.records,
        MonteCarloRecord,
    )
    _write_dataclass_csv(
        output / "attempts.csv",
        result.attempts,
        EstimatorAttempt,
    )
    _write_dataclass_csv(
        output / "attempt_summary.csv",
        attempt_summaries,
        EstimatorAttemptSummary,
    )
    _write_dataclass_csv(
        output / "summary.csv",
        summaries,
        MonteCarloSummary,
    )
    _write_dataclass_csv(
        output / "conditional_stable_summary.csv",
        stable_summaries,
        MonteCarloSummary,
    )
    with (output / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            config_to_dict(result.config),
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")
    metadata = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": config_fingerprint(result.config),
        "replication_indices": list(result.replication_indices),
        "is_complete": result.replication_indices
        == tuple(range(result.config.replications)),
        "record_count": len(result.records),
        "attempt_count": len(result.attempts),
        "attempt_summary_count": len(attempt_summaries),
        "summary_count": len(summaries),
        "conditional_stable_summary_count": len(stable_summaries),
    }
    with (output / "metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return output
