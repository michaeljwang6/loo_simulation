"""Apply the audited BLM support rule to an existing deterministic run.

This migration does not refit any estimator.  It reconstructs each simulated
panel from its saved seeds, repeats the original BLM cleaning and clustering,
marks samples without complete firm-class support as ``unsupported``, and
removes their previously returned BLM values from performance summaries.  It
also adds KSS's model-implied interaction zeros and gives the BS20 comparison
with project assignment covariance an explicit cross-target label.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from loo_sim import load_monte_carlo_results, save_monte_carlo_results
from loo_sim.monte_carlo import (
    EstimatorAttempt,
    MonteCarloRecord,
    MonteCarloResult,
    _attempt_sort_key,
    _generate_replication,
    _record_sort_key,
    _replication_seeds,
)
from loo_sim.pytwoway_estimators import BLMSupportError, prepare_blm_data


KSS_ESTIMATORS = ("akm_fe", "kss_ho", "kss_he")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _imposed_zero_record(
    *,
    attempt: EstimatorAttempt,
    metric: str,
    target: float,
) -> MonteCarloRecord:
    return MonteCarloRecord(
        scenario=attempt.scenario,
        replication=attempt.replication,
        population_seed=attempt.population_seed,
        panel_seed=attempt.panel_seed,
        estimator_seed=attempt.estimator_seed,
        estimator=attempt.estimator,
        metric=metric,
        target_type="population_project",
        estimate=0.0,
        target=float(target),
        error=float(-target),
        squared_error=float(target**2),
        n_observations=attempt.n_observations,
        n_workers=attempt.n_workers,
        n_firms=attempt.n_firms,
    )


def reclassify(result: MonteCarloResult) -> MonteCarloResult:
    """Return the support-audited result without rerunning fitted models."""

    config = result.config
    attempt_index = {
        (attempt.scenario, attempt.replication, attempt.estimator): attempt
        for attempt in result.attempts
    }
    updated_attempts = dict(attempt_index)
    unsupported_keys: set[tuple[str, int, str]] = set()
    project_targets: dict[tuple[str, int], object] = {}

    for scenario_index, scenario in enumerate(config.scenarios):
        seed_group = (
            scenario_index if scenario.seed_group is None else scenario.seed_group
        )
        n_worker_types = (
            config.estimators.blm_worker_types
            if config.estimators.blm_worker_types is not None
            else scenario.blm_worker_types
        )
        n_firm_types = (
            config.estimators.blm_firm_types
            if config.estimators.blm_firm_types is not None
            else scenario.blm_firm_types
        )
        for replication in result.replication_indices:
            population_seed, panel_seed, *_ = _replication_seeds(
                config.seed,
                seed_group,
                replication,
            )
            _, panel, targets = _generate_replication(
                scenario,
                population_seed=population_seed,
                panel_seed=panel_seed,
            )
            project_targets[(scenario.name, replication)] = targets.project
            if n_worker_types is None or n_firm_types is None:
                continue
            key = (scenario.name, replication, "blm_estimated")
            attempt = attempt_index.get(key)
            if attempt is None:
                continue
            try:
                prepared = prepare_blm_data(
                    panel,
                    n_firm_types=n_firm_types,
                    firm_groups=None,
                    periods=None,
                    cdf_resolution=config.estimators.blm_cdf_resolution,
                    seed=attempt.estimator_seed,
                )
            except BLMSupportError as exc:
                unsupported_keys.add(key)
                updated_attempts[key] = replace(
                    attempt,
                    status="unsupported",
                    message=str(exc),
                    n_observations=exc.sample.observations,
                    n_workers=exc.sample.workers,
                    n_firms=exc.sample.firms,
                )
            else:
                if not prepared.support.complete:
                    raise RuntimeError("BLM support audit returned an invalid pass.")

    records: list[MonteCarloRecord] = []
    for record in result.records:
        key = (record.scenario, record.replication, record.estimator)
        if key in unsupported_keys:
            continue
        if (
            record.estimator == "bs20"
            and record.metric == "worker_firm_covariance"
            and record.target_type == "population_project"
        ):
            record = replace(
                record,
                target_type="cross_target_project_c_assign",
            )
        records.append(record)

    existing = {
        (
            record.scenario,
            record.replication,
            record.estimator,
            record.metric,
            record.target_type,
        )
        for record in records
    }
    for key, attempt in updated_attempts.items():
        if attempt.estimator not in KSS_ESTIMATORS or attempt.status != "success":
            continue
        target = project_targets[(attempt.scenario, attempt.replication)]
        for metric, value in (("h_f", target.h_f), ("rho_h", target.rho_h)):
            record_key = (*key, metric, "population_project")
            if record_key in existing:
                continue
            records.append(
                _imposed_zero_record(
                    attempt=attempt,
                    metric=metric,
                    target=float(value),
                )
            )

    return replace(
        result,
        records=tuple(sorted(records, key=_record_sort_key)),
        attempts=tuple(sorted(updated_attempts.values(), key=_attempt_sort_key)),
    )


def main() -> None:
    args = _arguments()
    source = load_monte_carlo_results(args.input)
    corrected = reclassify(source)
    output = save_monte_carlo_results(corrected, args.output)
    metadata_path = output / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "derived_from": str(args.input.resolve()),
            "support_audit": (
                "Reconstructed panels and BLM estimated-group clustering from "
                "saved seeds; excluded samples missing a stayer class or mover "
                "class-pair; no estimator was refitted."
            ),
        }
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved support-audited results to {output.resolve()}.")


if __name__ == "__main__":
    main()
