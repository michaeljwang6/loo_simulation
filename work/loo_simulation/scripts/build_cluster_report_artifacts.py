"""Build tables and figures for the completed cluster production run."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from loo_sim import config_fingerprint, load_monte_carlo_results
from loo_sim.matrix_reporting import (
    COMMON_SPECS,
    ESTIMANDS,
    PROCEDURES,
    PROCEDURE_LABELS,
    SCENARIOS,
    SCENARIO_LABELS,
    _common_rows,
    _latex_escape,
    _load_plotting,
    _plot_common_rmse,
    _plot_project_tail_risk,
    _plot_rank_selection,
    _plot_status,
    _plot_target_gaps,
    _rank_rows,
    _status_rows,
    _summary,
    _summary_index,
    _validate_result,
    _write_csv,
    _write_latex,
)


EXPECTED_FINGERPRINT = (
    "e075e50934629f7ae20971667f117c418f3628bf576653f079602a07eb76a735"
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/dgp_estimator_matrix_cluster_v2/merged"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dgp_estimator_matrix_cluster_v2"),
    )
    return parser.parse_args()


def _target_rows(result: Any) -> list[dict[str, Any]]:
    summaries = _summary_index(result.summaries())
    rows: list[dict[str, Any]] = []
    base_specs = (
        (
            "kss_he",
            "q_f",
            "firm_variance",
            "native_akm",
            "population_project",
        ),
        (
            "kss_he",
            "c_assign",
            "worker_firm_covariance",
            "native_akm",
            "population_project",
        ),
        (
            "bs20",
            "c_assign",
            "worker_firm_covariance",
            "native_bs20",
            "cross_target_project_c_assign",
        ),
    )
    for scenario in SCENARIOS:
        for estimator, estimand, metric, native_type, project_type in base_specs:
            native = _summary(
                summaries, scenario, estimator, metric, native_type
            )
            project = _summary(
                summaries, scenario, estimator, metric, project_type
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "estimator": estimator,
                    "procedure_label": PROCEDURE_LABELS[estimator],
                    "estimand": estimand,
                    "native_target_type": native_type,
                    "native_target": native.mean_target,
                    "project_target": project.mean_target,
                    "target_gap": native.mean_target - project.mean_target,
                    "native_bias": native.bias,
                    "native_rmse": native.rmse,
                    "project_bias": project.bias,
                    "project_rmse": project.rmse,
                }
            )

        native_type = (
            "grouped_population_project"
            if scenario == "blm_grouped_dgp"
            else "blm_projection_project"
        )
        for estimand in ESTIMANDS:
            native = _summary(
                summaries,
                scenario,
                "blm_estimated",
                estimand,
                native_type,
            )
            project = _summary(
                summaries,
                scenario,
                "blm_estimated",
                estimand,
                "population_project",
            )
            if native.n_estimates != result.config.replications:
                raise ValueError(
                    "Production target contrasts require all BLM estimates."
                )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "estimator": "blm_estimated",
                    "procedure_label": PROCEDURE_LABELS["blm_estimated"],
                    "estimand": estimand,
                    "native_target_type": native_type,
                    "native_target": native.mean_target,
                    "project_target": project.mean_target,
                    "target_gap": native.mean_target - project.mean_target,
                    "native_bias": native.bias,
                    "native_rmse": native.rmse,
                    "project_bias": project.bias,
                    "project_rmse": project.rmse,
                }
            )
    return rows


def _format_rmse(value: float, *, structural_zero: bool = False) -> str:
    if structural_zero and abs(value) < 1e-8:
        return "0*"
    if abs(value) < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _project_truth_rows(result: Any) -> list[dict[str, Any]]:
    summaries = _summary_index(result.summaries())
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        row: dict[str, Any] = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
        }
        for estimand in ESTIMANDS:
            summary = _summary(
                summaries,
                scenario,
                "project_plugin_bic",
                estimand,
                "population_project",
            )
            row[estimand] = summary.mean_target
        rows.append(row)
    return rows


def _bs_native_rows(result: Any) -> list[dict[str, Any]]:
    summaries = _summary_index(result.summaries())
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        correlation = _summary(
            summaries,
            scenario,
            "bs20",
            "worker_firm_correlation",
            "native_bs20",
        )
        covariance = _summary(
            summaries,
            scenario,
            "bs20",
            "worker_firm_covariance",
            "native_bs20",
        )
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "rho_bs_target": correlation.mean_target,
                "rho_bs_bias": correlation.bias,
                "rho_bs_bias_mcse": correlation.bias_monte_carlo_se,
                "rho_bs_rmse": correlation.rmse,
                "c_bs_target": covariance.mean_target,
                "c_bs_bias": covariance.bias,
                "c_bs_bias_mcse": covariance.bias_monte_carlo_se,
                "c_bs_rmse": covariance.rmse,
            }
        )
    return rows


def _write_benchmark_table(
    output: Path,
    common_rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
) -> Path:
    common = {
        (row["scenario"], row["estimator"], row["estimand"]): row
        for row in common_rows
    }
    status = {
        (row["scenario"], row["estimator"]): row for row in status_rows
    }
    specifications = [
        ("KSS-HE", "akm_dgp", "kss_he", False),
        ("BLM", "blm_grouped_dgp", "blm_estimated", False),
        ("Project", "akm_dgp", "project_plugin_bic", False),
        ("Project", "crippa_tukey_dgp", "project_plugin_bic", False),
        ("Project, pass only", "crippa_tukey_dgp", "project_plugin_bic", True),
        ("Project", "blm_grouped_dgp", "project_plugin_bic", False),
        ("Project, pass only", "blm_grouped_dgp", "project_plugin_bic", True),
        ("Project", "low_rank_factor_dgp", "project_plugin_bic", False),
        ("Project, pass only", "low_rank_factor_dgp", "project_plugin_bic", True),
        ("Project", "gklp_perfect_information_dgp", "project_plugin_bic", False),
    ]
    rows: list[tuple[str, ...]] = []
    for label, scenario, estimator, stable_only in specifications:
        values = [common[(scenario, estimator, metric)] for metric in ESTIMANDS]
        pass_count = status[(scenario, estimator)]["n_success"]
        rmse_key = "stable_rmse" if stable_only else "rmse"
        rows.append(
            (
                _latex_escape(label),
                _latex_escape(SCENARIO_LABELS[scenario]),
                str(pass_count) if not stable_only else str(pass_count),
                *(
                    _format_rmse(
                        float(value[rmse_key]),
                        structural_zero=(
                            scenario == "akm_dgp"
                            and metric in ("h_f", "rho_h")
                        ),
                    )
                    for value, metric in zip(values, ESTIMANDS)
                ),
            )
        )
    path = output / "tables" / "benchmark_accuracy.tex"
    _write_latex(
        path,
        columns="llrrrrr",
        header=(
            "Procedure",
            "DGP",
            "Pass",
            "$Q_F$",
            "$H_F$",
            "$\\rho_H$",
            "$C^w_{\\mathrm{assign}}$",
        ),
        rows=rows,
        notes=(
            "Entries are RMSE against the full population-project truth. "
            "Pass-only rows condition on the numerical diagnostic and are "
            "not unconditional performance. 0* denotes a structural zero."
        ),
    )
    return path


def _attempt_diagnostic_rows(result: Any) -> list[dict[str, Any]]:
    patterns = {
        name: re.compile(rf"(?:^|; ){name}=([^;]+)")
        for name in (
            "worker_design_condition_max",
            "firm_design_condition_max",
            "worker_factor_norm_max",
            "firm_factor_norm_max",
        )
    }
    values: dict[tuple[str, str], list[dict[str, float]]] = {}
    for attempt in result.attempts:
        if attempt.estimator != "project_plugin_bic":
            continue
        parsed: dict[str, float] = {}
        for name, pattern in patterns.items():
            match = pattern.search(attempt.message)
            parsed[name] = float(match.group(1)) if match else float("nan")
        values.setdefault((attempt.scenario, attempt.status), []).append(parsed)

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for status in ("success", "unstable"):
            group = values.get((scenario, status), [])
            if not group:
                continue
            local_condition = np.asarray(
                [
                    max(
                        item["worker_design_condition_max"],
                        item["firm_design_condition_max"],
                    )
                    for item in group
                ]
            )
            factor_norm = np.asarray(
                [
                    max(
                        item["worker_factor_norm_max"],
                        item["firm_factor_norm_max"],
                    )
                    for item in group
                ]
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "status": status,
                    "count": len(group),
                    "median_max_local_condition": float(
                        np.median(local_condition)
                    ),
                    "p90_max_local_condition": float(
                        np.quantile(local_condition, 0.9)
                    ),
                    "median_max_factor_norm": float(np.median(factor_norm)),
                    "p90_max_factor_norm": float(
                        np.quantile(factor_norm, 0.9)
                    ),
                }
            )
    return rows


def main() -> None:
    args = _arguments()
    matplotlib_cache = Path("tmp/matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache.resolve()))
    result = load_monte_carlo_results(args.input)
    _validate_result(result)
    actual = config_fingerprint(result.config)
    if actual != EXPECTED_FINGERPRINT:
        raise ValueError(
            f"Unexpected production configuration fingerprint: {actual}."
        )

    output = args.output
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    status_rows = _status_rows(result)
    common_rows = _common_rows(result)
    target_rows = _target_rows(result)
    rank_rows = _rank_rows(result)
    diagnostic_rows = _attempt_diagnostic_rows(result)
    truth_rows = _project_truth_rows(result)
    bs_rows = _bs_native_rows(result)

    for name, rows in (
        ("status_rates.csv", status_rows),
        ("common_target_performance.csv", common_rows),
        ("native_project_target_contrasts.csv", target_rows),
        ("rank_selection.csv", rank_rows),
        ("project_numerical_diagnostics.csv", diagnostic_rows),
        ("project_truths.csv", truth_rows),
        ("bs20_native_accuracy.csv", bs_rows),
    ):
        _write_csv(table_dir / name, rows)

    status_index = {
        (row["scenario"], row["estimator"]): row for row in status_rows
    }
    _write_latex(
        table_dir / "status_rates.tex",
        columns="lrrrr",
        header=("DGP", "KSS", "BLM", "BS20", "Project"),
        rows=[
            (
                _latex_escape(SCENARIO_LABELS[scenario]),
                *(
                    f"{status_index[(scenario, estimator)]['n_success']}/"
                    f"{status_index[(scenario, estimator)]['n_unstable']}/"
                    f"{status_index[(scenario, estimator)]['n_unsupported']}/"
                    f"{status_index[(scenario, estimator)]['n_failure']}"
                    for estimator in PROCEDURES
                ),
            )
            for scenario in SCENARIOS
        ],
        notes=(
            "Cells report pass or completion / warning / unsupported / "
            "failure out of 100."
        ),
    )
    _write_latex(
        table_dir / "rank_selection.tex",
        columns="lrrrrr",
        header=("DGP", "True", "Select 0", "Select 1", "Select 2", "Pass"),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                str(row["true_rank"]),
                str(row["selected_rank_0"]),
                str(row["selected_rank_1"]),
                str(row["selected_rank_2"]),
                f"{100 * float(row['stable_rate']):.0f}\\%",
            )
            for row in rank_rows
        ],
        notes=(
            "BIC rank counts; pass is the separate numerical-diagnostic rate."
        ),
    )
    _write_latex(
        table_dir / "project_truths.tex",
        columns="lrrrr",
        header=(
            "DGP",
            "$Q_F$",
            "$H_F$",
            "$\\rho_H$",
            "$C^w_{\\mathrm{assign}}$",
        ),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                f"{float(row['q_f']):.4f}",
                f"{float(row['h_f']):.4f}",
                f"{float(row['rho_h']):.4f}",
                f"{float(row['c_assign']):.4f}",
            )
            for row in truth_rows
        ],
        notes=(
            "Entries are means of exact finite-population truths across the "
            "100 independently generated populations."
        ),
    )
    _write_latex(
        table_dir / "bs20_native_accuracy.tex",
        columns="lrrrrrr",
        header=(
            "DGP",
            "$\\rho_{\\mathrm{BS}}$ truth",
            "Bias",
            "RMSE",
            "$C_{\\mathrm{BS}}$ truth",
            "Bias",
            "RMSE",
        ),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                f"{float(row['rho_bs_target']):.4f}",
                f"{float(row['rho_bs_bias']):.4f}",
                f"{float(row['rho_bs_rmse']):.4f}",
                f"{float(row['c_bs_target']):.4f}",
                f"{float(row['c_bs_bias']):.4f}",
                f"{float(row['c_bs_rmse']):.4f}",
            )
            for row in bs_rows
        ],
        notes=(
            "BS20 is evaluated against its native targets. It is not scored "
            "as an estimator of the proposal's assignment contribution."
        ),
    )
    _write_benchmark_table(output, common_rows, status_rows)

    plt, LogNorm, Rectangle, _ = _load_plotting()
    files: list[Path] = []
    files.extend(_plot_status(plt, Rectangle, figure_dir, status_rows))
    files.extend(_plot_common_rmse(plt, LogNorm, figure_dir, common_rows))
    files.extend(_plot_target_gaps(plt, figure_dir, target_rows))
    files.extend(_plot_rank_selection(plt, Rectangle, figure_dir, rank_rows))
    files.extend(_plot_project_tail_risk(plt, figure_dir, common_rows))
    plt.close("all")

    print(
        f"Built production tables and {len(files)} figure files under "
        f"{output.resolve()}."
    )


if __name__ == "__main__":
    main()
