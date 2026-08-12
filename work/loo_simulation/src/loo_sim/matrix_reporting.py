"""Tables, figures, and narrative for the five-by-four simulation matrix."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .dgp import GroupedPopulationDGP
from .monte_carlo import (
    EstimatorAttemptSummary,
    MonteCarloResult,
    MonteCarloSummary,
    _generate_replication,
    _replication_seeds,
    config_fingerprint,
)
from .targets import compute_blm_evaluation_groups, compute_blm_grouped_target


SCENARIOS = (
    "akm_dgp",
    "crippa_tukey_dgp",
    "blm_grouped_dgp",
    "low_rank_factor_dgp",
    "gklp_perfect_information_dgp",
)

SCENARIO_LABELS = {
    "akm_dgp": "AKM",
    "crippa_tukey_dgp": "Crippa/Tukey",
    "blm_grouped_dgp": "BLM types",
    "low_rank_factor_dgp": "Low-rank factors",
    "gklp_perfect_information_dgp": "GKLP",
}

PROCEDURES = (
    "kss_he",
    "blm_estimated",
    "bs20",
    "project_plugin_bic",
)

PROCEDURE_LABELS = {
    "kss_he": "KSS-HE",
    "blm_estimated": "BLM",
    "bs20": "BS20",
    "project_plugin_bic": "Project plug-in",
}

ESTIMANDS = ("q_f", "h_f", "rho_h", "c_assign")

ESTIMAND_LABELS = {
    "q_f": r"$Q_F$",
    "h_f": r"$H_F$",
    "rho_h": r"$\rho_H$",
    "c_assign": r"$C_{\mathrm{assign}}$",
}

COMMON_SPECS: Mapping[str, Mapping[str, tuple[str, str]]] = {
    "q_f": {
        "kss_he": ("firm_variance", "population_project"),
        "blm_estimated": ("q_f", "population_project"),
        "project_plugin_bic": ("q_f", "population_project"),
    },
    "h_f": {
        "kss_he": ("h_f", "population_project"),
        "blm_estimated": ("h_f", "population_project"),
        "project_plugin_bic": ("h_f", "population_project"),
    },
    "rho_h": {
        "kss_he": ("rho_h", "population_project"),
        "blm_estimated": ("rho_h", "population_project"),
        "project_plugin_bic": ("rho_h", "population_project"),
    },
    "c_assign": {
        "kss_he": ("worker_firm_covariance", "population_project"),
        "blm_estimated": ("c_assign", "population_project"),
        "project_plugin_bic": ("c_assign", "population_project"),
    },
}

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
RED = "#B91C1C"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def _summary_index(
    summaries: Iterable[MonteCarloSummary],
) -> dict[tuple[str, str, str, str], MonteCarloSummary]:
    return {
        (row.scenario, row.estimator, row.metric, row.target_type): row
        for row in summaries
    }


def _summary(
    index: Mapping[tuple[str, str, str, str], MonteCarloSummary],
    scenario: str,
    estimator: str,
    metric: str,
    target_type: str,
) -> MonteCarloSummary:
    return index[(scenario, estimator, metric, target_type)]


def _validate_result(result: MonteCarloResult) -> None:
    if result.replication_indices != tuple(range(result.config.replications)):
        raise ValueError("Matrix reporting requires a complete merged result.")
    observed = {scenario.name for scenario in result.config.scenarios}
    missing = set(SCENARIOS) - observed
    if missing:
        raise ValueError(
            "Matrix result is missing scenarios: "
            + ", ".join(sorted(missing))
        )


def _status_rows(result: MonteCarloResult) -> list[dict[str, Any]]:
    index = {
        (row.scenario, row.estimator): row
        for row in result.attempt_summaries()
    }
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for estimator in PROCEDURES:
            value = index[(scenario, estimator)]
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "estimator": estimator,
                    "procedure_label": PROCEDURE_LABELS[estimator],
                    "n_attempts": value.n_attempts,
                    "n_success": value.n_success,
                    "n_unstable": value.n_unstable,
                    "n_unsupported": value.n_unsupported,
                    "n_failure": value.n_failure,
                    "success_rate": value.success_rate,
                    "unstable_rate": value.unstable_rate,
                    "unsupported_rate": value.unsupported_rate,
                    "failure_rate": value.failure_rate,
                }
            )
    return rows


def _common_rows(
    result: MonteCarloResult,
) -> list[dict[str, Any]]:
    returned = _summary_index(result.summaries())
    stable = _summary_index(
        result.summaries(included_statuses=("success",))
    )
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for estimand in ESTIMANDS:
            for estimator, (metric, target_type) in COMMON_SPECS[
                estimand
            ].items():
                full = _summary(
                    returned,
                    scenario,
                    estimator,
                    metric,
                    target_type,
                )
                conditional = _summary(
                    stable,
                    scenario,
                    estimator,
                    metric,
                    target_type,
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "estimand": estimand,
                        "estimand_label": ESTIMAND_LABELS[estimand],
                        "estimator": estimator,
                        "procedure_label": PROCEDURE_LABELS[estimator],
                        "n_attempts": full.n_attempts,
                        "n_returned": full.n_estimates,
                        "n_stable": conditional.n_estimates,
                        "n_unstable": full.n_unstable,
                        "n_unsupported": full.n_unsupported,
                        "n_failure": full.n_failure,
                        "mean_target": full.mean_target,
                        "bias": full.bias,
                        "bias_mcse": full.bias_monte_carlo_se,
                        "rmse": full.rmse,
                        "stable_bias": conditional.bias,
                        "stable_bias_mcse": (
                            conditional.bias_monte_carlo_se
                        ),
                        "stable_rmse": conditional.rmse,
                    }
                )
    return rows


def _target_contrast_rows(
    result: MonteCarloResult,
) -> list[dict[str, Any]]:
    summaries = _summary_index(result.summaries())
    specs = (
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
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for estimator, estimand, metric, native_type, project_type in specs:
            native = _summary(
                summaries,
                scenario,
                estimator,
                metric,
                native_type,
            )
            project = _summary(
                summaries,
                scenario,
                estimator,
                metric,
                project_type,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "estimator": estimator,
                    "procedure_label": PROCEDURE_LABELS[estimator],
                    "estimand": estimand,
                    "estimand_label": ESTIMAND_LABELS[estimand],
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
    rows.extend(_blm_target_contrast_rows(result, summaries))
    return rows


def _blm_target_contrast_rows(
    result: MonteCarloResult,
    summaries: Mapping[tuple[str, str, str, str], MonteCarloSummary],
) -> list[dict[str, Any]]:
    """Compare the BLM grouped projection with the full project population.

    The grouped target is reconstructed for every population replication, so
    this estimand comparison is not conditioned on whether the BLM optimizer
    returned an estimate.
    """

    estimator_config = result.config.estimators
    target_values: dict[tuple[str, str], list[tuple[float, float]]] = {
        (scenario, estimand): []
        for scenario in SCENARIOS
        for estimand in ESTIMANDS
    }
    native_types: dict[str, str] = {}

    for scenario_index, scenario in enumerate(result.config.scenarios):
        if scenario.name not in SCENARIOS:
            continue
        scenario_seed_group = (
            scenario_index
            if scenario.seed_group is None
            else scenario.seed_group
        )
        n_worker_types = (
            estimator_config.blm_worker_types
            if estimator_config.blm_worker_types is not None
            else scenario.blm_worker_types
        )
        n_firm_types = (
            estimator_config.blm_firm_types
            if estimator_config.blm_firm_types is not None
            else scenario.blm_firm_types
        )
        if n_worker_types is None or n_firm_types is None:
            continue

        for replication in range(result.config.replications):
            population_seed, panel_seed, *_ = _replication_seeds(
                result.config.seed,
                scenario_seed_group,
                replication,
            )
            population, _, targets = _generate_replication(
                scenario,
                population_seed=population_seed,
                panel_seed=panel_seed,
            )
            true_grouped_target = (
                isinstance(population, GroupedPopulationDGP)
                and np.unique(population.worker_groups).size
                == n_worker_types
                and np.unique(population.firm_groups).size == n_firm_types
            )
            if true_grouped_target:
                worker_groups = population.worker_groups
                firm_groups = population.firm_groups
                native_type = "grouped_population_project"
            else:
                evaluation_groups = compute_blm_evaluation_groups(
                    population.schedule,
                    population.assignment,
                    n_worker_types=n_worker_types,
                    n_firm_types=n_firm_types,
                )
                worker_groups = evaluation_groups.worker_groups
                firm_groups = evaluation_groups.firm_groups
                native_type = "blm_projection_project"
            native_types[scenario.name] = native_type
            grouped = compute_blm_grouped_target(
                population.schedule,
                population.assignment,
                worker_groups,
                firm_groups,
            )
            for estimand in ESTIMANDS:
                target_values[(scenario.name, estimand)].append(
                    (
                        float(getattr(grouped.project_functionals, estimand)),
                        float(getattr(targets.project, estimand)),
                    )
                )

    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        native_type = native_types[scenario]
        for estimand in ESTIMANDS:
            values = np.asarray(target_values[(scenario, estimand)], dtype=float)
            native_summary = _summary(
                summaries,
                scenario,
                "blm_estimated",
                estimand,
                native_type,
            )
            project_summary = _summary(
                summaries,
                scenario,
                "blm_estimated",
                estimand,
                "population_project",
            )
            native_target = float(np.mean(values[:, 0]))
            project_target = float(np.mean(values[:, 1]))
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "estimator": "blm_estimated",
                    "procedure_label": PROCEDURE_LABELS["blm_estimated"],
                    "estimand": estimand,
                    "estimand_label": ESTIMAND_LABELS[estimand],
                    "native_target_type": native_type,
                    "native_target": native_target,
                    "project_target": project_target,
                    "target_gap": native_target - project_target,
                    "native_bias": native_summary.bias,
                    "native_rmse": native_summary.rmse,
                    "project_bias": project_summary.bias,
                    "project_rmse": project_summary.rmse,
                }
            )
    return rows


def _rank_rows(result: MonteCarloResult) -> list[dict[str, Any]]:
    true_rank = {
        scenario.name: scenario.true_rank
        for scenario in result.config.scenarios
    }
    attempts = {
        (row.scenario, row.estimator): row
        for row in result.attempt_summaries()
    }
    selections: dict[str, list[int]] = {name: [] for name in SCENARIOS}
    for record in result.records:
        if (
            record.estimator == "project_plugin_bic"
            and record.metric == "selected_rank"
            and record.target_type == "rank_diagnostic"
        ):
            selections[record.scenario].append(int(round(record.estimate)))
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        counts = Counter(selections[scenario])
        attempt = attempts[(scenario, "project_plugin_bic")]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "true_rank": true_rank[scenario],
                "selected_rank_0": counts[0],
                "selected_rank_1": counts[1],
                "selected_rank_2": counts[2],
                "correct_rank_rate": (
                    counts[true_rank[scenario]] / len(selections[scenario])
                ),
                "stable_rate": attempt.success_rate,
                "unstable_rate": attempt.unstable_rate,
                "failure_rate": attempt.failure_rate,
            }
        )
    return rows


def _failure_rows(result: MonteCarloResult) -> list[dict[str, Any]]:
    counter: Counter[tuple[str, str, str]] = Counter()
    for attempt in result.attempts:
        if attempt.estimator != "blm_estimated" or attempt.status not in (
            "unsupported",
            "failure",
        ):
            continue
        if attempt.status == "unsupported":
            start = attempt.message.find("BLM_SUPPORT[")
            end = attempt.message.find("]", start)
            reason = (
                attempt.message[start + len("BLM_SUPPORT[") : end]
                if start >= 0 and end > start
                else "unsupported_unknown"
            )
        else:
            reason = "optimizer_or_software_failure"
        counter[(attempt.scenario, attempt.status, reason)] += 1
    rows: list[dict[str, Any]] = []
    for (scenario, status, reason), count in sorted(counter.items()):
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "status": status,
                "reason": reason,
                "count": count,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        value = value.replace(old, new)
    return value


def _write_latex(
    path: Path,
    *,
    columns: str,
    header: Sequence[str],
    rows: Sequence[Sequence[str]],
    notes: str,
) -> None:
    lines = [
        "% Generated by scripts/report_dgp_estimator_matrix.py",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [r"\bottomrule", r"\end{tabular}", "", "% " + notes, ""]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_plotting() -> tuple[Any, Any, Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "figure.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    return plt, LogNorm, Rectangle, matplotlib


def _save_figure(
    fig: Any,
    output: Path,
    stem: str,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    png = output / f"{stem}.png"
    pdf = output / f"{stem}.pdf"
    fig.savefig(png, dpi=240, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    return [png, pdf]


def _plot_status(
    plt: Any,
    Rectangle: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    index = {(row["scenario"], row["estimator"]): row for row in rows}
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    colors = (GREEN, ORANGE, PURPLE, RED)
    labels = (
        "Passed/completed",
        "Returned with warning",
        "Unsupported sample",
        "Estimator failure",
    )
    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, estimator in enumerate(PROCEDURES):
            row = index[(scenario, estimator)]
            rates = (
                float(row["success_rate"]),
                float(row["unstable_rate"]),
                float(row["unsupported_rate"]),
                float(row["failure_rate"]),
            )
            left = column_index - 0.46
            for rate, color in zip(rates, colors):
                ax.add_patch(
                    Rectangle(
                        (left, row_index - 0.32),
                        0.92 * rate,
                        0.64,
                        facecolor=color,
                        edgecolor="none",
                    )
                )
                left += 0.92 * rate
            ax.add_patch(
                Rectangle(
                    (column_index - 0.46, row_index - 0.32),
                    0.92,
                    0.64,
                    facecolor="none",
                    edgecolor="#374151",
                    linewidth=0.5,
                )
            )
            ax.text(
                column_index,
                row_index,
                f"{row['n_success']}/{row['n_unstable']}/"
                f"{row['n_unsupported']}/{row['n_failure']}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox={
                    "boxstyle": "round,pad=0.13",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )
    ax.set_xlim(-0.55, len(PROCEDURES) - 0.45)
    ax.set_ylim(len(SCENARIOS) - 0.45, -0.55)
    ax.set_xticks(
        range(len(PROCEDURES)),
        [PROCEDURE_LABELS[value] for value in PROCEDURES],
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(
        range(len(SCENARIOS)),
        [SCENARIO_LABELS[value] for value in SCENARIOS],
    )
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none")
        for color in colors
    ]
    ax.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    ax.set_title(
        "Estimator status (pass / warning / unsupported / failure)",
        pad=34,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    return _save_figure(fig, output, "status-matrix")


def _plot_common_rmse(
    plt: Any,
    LogNorm: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    index = {
        (row["scenario"], row["estimator"], row["estimand"]): row
        for row in rows
    }
    # A few unstable fits have RMSE above one million, while exact rank-zero
    # identities are numerically near 1e-30. Clipping only the color mapping
    # keeps the economically relevant 0.01-to-1000 range legible; every cell
    # still prints its uncapped RMSE.
    norm = LogNorm(vmin=1e-2, vmax=1e3, clip=True)
    fig = plt.figure(figsize=(9.2, 7.1))
    grid = fig.add_gridspec(
        2,
        3,
        width_ratios=(1, 1, 0.055),
        wspace=0.2,
        hspace=0.3,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])],
            [fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])],
        ]
    )
    colorbar_axis = fig.add_subplot(grid[:, 2])
    image = None
    for facet_index, (ax, estimand) in enumerate(
        zip(axes.ravel(), ESTIMANDS)
    ):
        values = np.full((len(SCENARIOS), len(PROCEDURES)), np.nan)
        for i, scenario in enumerate(SCENARIOS):
            for j, estimator in enumerate(PROCEDURES):
                row = index.get((scenario, estimator, estimand))
                if row is not None:
                    values[i, j] = float(row["rmse"])
        masked = np.ma.masked_invalid(values)
        image = ax.imshow(masked, cmap="viridis", norm=norm, aspect="auto")
        for i in range(len(SCENARIOS)):
            for j in range(len(PROCEDURES)):
                value = values[i, j]
                if not np.isfinite(value):
                    ax.text(j, i, "N/A", ha="center", va="center", color=GRAY)
                    continue
                clipped = float(np.clip(value, norm.vmin, norm.vmax))
                position = (
                    np.log(clipped) - np.log(norm.vmin)
                ) / (np.log(norm.vmax) - np.log(norm.vmin))
                color = "white" if position < 0.58 else "black"
                structural_zero = (
                    SCENARIOS[i] == "akm_dgp"
                    and PROCEDURES[j] in ("kss_he", "project_plugin_bic")
                    and estimand in ("h_f", "rho_h")
                    and value < 1e-8
                )
                if structural_zero:
                    label = "0*"
                else:
                    label = "0" if value < 1e-8 else f"{value:.3g}"
                ax.text(j, i, label, ha="center", va="center", color=color, fontsize=7.5)
        ax.set_title(ESTIMAND_LABELS[estimand])
        ax.set_xticks(
            range(len(PROCEDURES)),
            [PROCEDURE_LABELS[value] for value in PROCEDURES],
            rotation=28,
            ha="right",
        )
        ax.set_yticks(
            range(len(SCENARIOS)),
            [SCENARIO_LABELS[value] for value in SCENARIOS],
        )
        if facet_index % 2 == 1:
            ax.tick_params(labelleft=False)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    assert image is not None
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("RMSE among admissible returned estimates (log scale)")
    fig.suptitle("Accuracy relative to the common population-project truth")
    fig.subplots_adjust(left=0.18, right=0.93, bottom=0.12, top=0.91)
    return _save_figure(fig, output, "common-target-rmse")


def _plot_target_gaps(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    specs = (
        ("kss_he", "q_f", "KSS native firm variance\nminus " + r"$Q_F$"),
        ("kss_he", "c_assign", "KSS native covariance\nminus " + r"$C_{\mathrm{assign}}$"),
        ("bs20", "c_assign", "BS20 native covariance\nminus " + r"$C_{\mathrm{assign}}$"),
    )
    index = {
        (row["scenario"], row["estimator"], row["estimand"]): row
        for row in rows
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 4.5), sharey=True)
    for ax, (estimator, estimand, title) in zip(axes, specs):
        values = np.asarray(
            [
                index[(scenario, estimator, estimand)]["target_gap"]
                for scenario in SCENARIOS
            ],
            dtype=float,
        )
        y = np.arange(len(SCENARIOS))
        ax.axvline(0, color=LIGHT_GRAY, linewidth=1)
        ax.scatter(values, y, color=BLUE if estimator == "kss_he" else PURPLE, s=30, zorder=3)
        span = max(float(np.ptp(values)), 0.08)
        for value, position in zip(values, y):
            direction = 1 if value >= 0 else -1
            ax.text(
                value + direction * 0.025 * span,
                position - 0.12,
                f"{value:+.2f}",
                ha="left" if direction > 0 else "right",
                va="bottom",
                fontsize=7,
            )
        ax.set_title(title)
        ax.set_xlabel("Target difference")
        ax.set_yticks(y, [SCENARIO_LABELS[value] for value in SCENARIOS])
        ax.invert_yaxis()
        ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
        ax.margins(x=0.12)
    fig.suptitle("Native procedure targets need not equal project estimands")
    fig.tight_layout()
    fig.subplots_adjust(top=0.82, wspace=0.2)
    return _save_figure(fig, output, "native-project-target-gaps")


def _plot_rank_selection(
    plt: Any,
    Rectangle: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    values = np.asarray(
        [
            [row[f"selected_rank_{rank}"] for rank in range(3)]
            for row in rows
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    image = ax.imshow(values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, row in enumerate(rows):
        for rank in range(3):
            count = int(values[i, rank])
            color = "white" if count >= 55 else "black"
            ax.text(rank, i, f"{count}%", ha="center", va="center", color=color)
        true_rank = int(row["true_rank"])
        ax.add_patch(
            Rectangle(
                (true_rank - 0.46, i - 0.43),
                0.92,
                0.86,
                facecolor="none",
                edgecolor=ORANGE,
                linewidth=2.0,
            )
        )
        ax.text(
            2.65,
            i,
            f"pass {100 * float(row['stable_rate']):.0f}%",
            va="center",
            fontsize=8,
        )
    ax.set_xlim(-0.5, 3.45)
    ax.set_xticks(range(3), ("Selected 0", "Selected 1", "Selected 2"))
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=8)
    ax.set_yticks(range(len(SCENARIOS)), [row["scenario_label"] for row in rows])
    ax.tick_params(axis="y", length=0)
    ax.set_title("BIC rank selection; orange box marks the true rank", pad=32)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Share of replications (%)")
    fig.tight_layout()
    return _save_figure(fig, output, "rank-selection")


def _plot_project_tail_risk(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    nonadditive = SCENARIOS[1:]
    index = {
        (row["scenario"], row["estimator"], row["estimand"]): row
        for row in rows
    }
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.8), sharex=True)
    x = np.arange(len(nonadditive))
    for ax, estimand in zip(axes.ravel(), ESTIMANDS):
        full = np.asarray(
            [
                index[(scenario, "project_plugin_bic", estimand)]["rmse"]
                for scenario in nonadditive
            ],
            dtype=float,
        )
        stable = np.asarray(
            [
                index[(scenario, "project_plugin_bic", estimand)]["stable_rmse"]
                for scenario in nonadditive
            ],
            dtype=float,
        )
        ax.plot(x, full, color=ORANGE, marker="o", label="All returned")
        ax.plot(
            x,
            stable,
            color=BLUE,
            marker="s",
            label="Diagnostic pass only",
        )
        ax.set_yscale("log")
        ax.set_title(ESTIMAND_LABELS[estimand])
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.6)
        ax.set_xticks(
            x,
            [SCENARIO_LABELS[value] for value in nonadditive],
            rotation=24,
            ha="right",
        )
        ax.set_ylabel("RMSE (log scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(
        "Project plug-in: diagnostic warnings identify large tail risk",
        y=0.995,
    )
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.16, top=0.87, hspace=0.34, wspace=0.24)
    return _save_figure(fig, output, "project-returned-vs-stable-rmse")


def _write_tables(
    output: Path,
    *,
    status_rows: Sequence[Mapping[str, Any]],
    common_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    table_dir = output / "tables"
    files: list[Path] = []
    for name, rows in (
        ("status_rates.csv", status_rows),
        ("common_target_performance.csv", common_rows),
        ("native_project_target_contrasts.csv", target_rows),
        ("rank_selection.csv", rank_rows),
        ("blm_failure_reasons.csv", failure_rows),
    ):
        path = table_dir / name
        _write_csv(path, rows)
        files.append(path)

    status_tex = table_dir / "status_rates.tex"
    status_index = {
        (row["scenario"], row["estimator"]): row for row in status_rows
    }
    _write_latex(
        status_tex,
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
            "Cells report diagnostic pass or completion / returned with "
            "warning / unsupported sample / estimator failure out of 100."
        ),
    )
    files.append(status_tex)

    rank_tex = table_dir / "rank_selection.tex"
    _write_latex(
        rank_tex,
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
            "Project plug-in BIC selections; the diagnostic-pass rate is "
            "reported separately."
        ),
    )
    files.append(rank_tex)
    return files


def _fmt(value: float, *, signed: bool = False) -> str:
    prefix = "+" if signed else ""
    absolute = abs(value)
    if absolute >= 100:
        precision = 1
    elif absolute >= 10:
        precision = 2
    elif absolute >= 0.01:
        precision = 3
    else:
        precision = 4
    return f"{value:{prefix}.{precision}f}"


def _status_cell(row: Mapping[str, Any]) -> str:
    return (
        f"{row['n_success']} / {row['n_unstable']} / "
        f"{row['n_unsupported']} / {row['n_failure']}"
    )


def _write_markdown(
    path: Path,
    *,
    result: MonteCarloResult,
    output: Path,
    status_rows: Sequence[Mapping[str, Any]],
    common_rows: Sequence[Mapping[str, Any]],
    target_rows: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    failure_rows: Sequence[Mapping[str, Any]],
) -> None:
    status_index = {
        (row["scenario"], row["estimator"]): row for row in status_rows
    }
    common_index = {
        (row["scenario"], row["estimator"], row["estimand"]): row
        for row in common_rows
    }
    target_index = {
        (row["scenario"], row["estimator"], row["estimand"]): row
        for row in target_rows
    }
    figure = output / "figures"
    relative = lambda name: (figure / name).relative_to(path.parent).as_posix()

    kss_q = common_index[("akm_dgp", "kss_he", "q_f")]
    kss_c = common_index[("akm_dgp", "kss_he", "c_assign")]
    blm_q = common_index[("blm_grouped_dgp", "blm_estimated", "q_f")]
    blm_h = common_index[("blm_grouped_dgp", "blm_estimated", "h_f")]
    blm_c = common_index[("blm_grouped_dgp", "blm_estimated", "c_assign")]
    project_akm_h = common_index[("akm_dgp", "project_plugin_bic", "h_f")]
    project_akm_rho = common_index[
        ("akm_dgp", "project_plugin_bic", "rho_h")
    ]
    blm_unsupported = sum(
        int(row["count"])
        for row in failure_rows
        if row["status"] == "unsupported"
    )
    blm_failures = sum(
        int(row["count"])
        for row in failure_rows
        if row["status"] == "failure"
    )
    blm_returned = int(blm_q["n_returned"])

    status_table = [
        "| DGP | KSS | BLM | BS20 | Project plug-in |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        status_table.append(
            "| "
            + SCENARIO_LABELS[scenario]
            + " | "
            + " | ".join(
                _status_cell(status_index[(scenario, estimator)])
                for estimator in PROCEDURES
            )
            + " |"
        )

    rank_table = [
        "| DGP | True rank | Selected ranks (0 / 1 / 2) | Diagnostic pass |",
        "|---|---:|---:|---:|",
    ]
    for row in rank_rows:
        rank_table.append(
            f"| {row['scenario_label']} | {row['true_rank']} | "
            f"{row['selected_rank_0']} / {row['selected_rank_1']} / "
            f"{row['selected_rank_2']} | "
            f"{100 * float(row['stable_rate']):.0f}% |"
        )

    gap_table = [
        "| DGP | KSS $Q_F$ gap | KSS covariance gap | BS20 covariance gap |",
        "|---|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        gap_table.append(
            f"| {SCENARIO_LABELS[scenario]} | "
            f"{_fmt(float(target_index[(scenario, 'kss_he', 'q_f')]['target_gap']), signed=True)} | "
            f"{_fmt(float(target_index[(scenario, 'kss_he', 'c_assign')]['target_gap']), signed=True)} | "
            f"{_fmt(float(target_index[(scenario, 'bs20', 'c_assign')]['target_gap']), signed=True)} |"
        )

    blm_gap_table = [
        "| DGP | BLM $Q_F$ gap | BLM $H_F$ gap | BLM $\\rho_H$ gap | BLM covariance gap |",
        "|---|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        blm_gap_table.append(
            f"| {SCENARIO_LABELS[scenario]} | "
            f"{_fmt(float(target_index[(scenario, 'blm_estimated', 'q_f')]['target_gap']), signed=True)} | "
            f"{_fmt(float(target_index[(scenario, 'blm_estimated', 'h_f')]['target_gap']), signed=True)} | "
            f"{_fmt(float(target_index[(scenario, 'blm_estimated', 'rho_h')]['target_gap']), signed=True)} | "
            f"{_fmt(float(target_index[(scenario, 'blm_estimated', 'c_assign')]['target_gap']), signed=True)} |"
        )

    reason_labels = {
        "no_mover_events": "No mover events",
        "no_stayer_events": "No stayer events",
        "missing_stayer_classes": "At least one stayer firm class absent",
        "missing_mover_pairs": "At least one mover class-pair absent",
        "missing_stayer_classes_and_mover_pairs": (
            "Both stayer classes and mover class-pairs absent"
        ),
        "optimizer_or_software_failure": "Estimator/software failure",
    }
    blm_reason_table = [
        "| DGP | BLM status | Exact reason | Count |",
        "|---|---|---|---:|",
    ]
    for row in failure_rows:
        blm_reason_table.append(
            f"| {row['scenario_label']} | {row['status']} | "
            f"{reason_labels.get(str(row['reason']), str(row['reason']))} | "
            f"{row['count']} |"
        )

    lines = [
        "# DGP-by-estimator Monte Carlo: methods and results",
        "",
        "This report asks a deliberately symmetric question: what happens when each of four procedures is applied to data from each of five worker-firm wage models? The answer is not a single winner. KSS and BLM perform well when their own structures are correct, while nonadditivity, rank selection, sparse support, and differences in estimands explain the cross-model reversals.",
        "",
        "Every numerical statement below is derived from the support-audited merged output with configuration fingerprint `"
        + config_fingerprint(result.config)
        + "`. The support audit reconstructs the original panels and BLM clustering from their saved seeds; it does not refit a successful model. Equations attributed to papers are cited. Every other equation is labeled as a definition or derived below.",
        "",
        "## 1. Question and design",
        "",
        "The design is fixed before estimation. Each DGP has 100 replications, and every procedure is applied to every DGP.",
        "",
        "| Common parameter | Value |",
        "|---|---:|",
        "| Workers / firms / periods | 300 / 18 / 10 |",
        "| Observations before cleaning | 3,000 |",
        "| Grand mean of systematic wages | 0 |",
        "| Firm-redraw probability after period 1 | 0.40 |",
        "| Wage disturbance | independent $N(0,0.5^2)$ |",
        "| Worker and firm population marginals | uniform |",
        "| Monte Carlo replications per DGP | 100 |",
        "| Master random seed | 20260804 |",
        "",
        "All Gaussian coordinates are redrawn in every replication. Main effects are standardized to finite-population mean zero and variance one. Factor columns are centered and orthonormalized under the uniform population weights, so the declared singular values hold exactly in each realized population.",
        "",
        "| DGP | Latent-variable construction | Interaction parameters |",
        "|---|---|---|",
        "| AKM | Independent raw worker and firm $N(0,1)$ effects | Rank 0; $m_{ij}=\\alpha_i+\\psi_j$ |",
        "| Crippa/Tukey | Independent raw $\\alpha_i,\\psi_j\\sim N(0,1)$ | Rank 1; $h_{ij}=0.75\\alpha_i\\psi_j$; singular value 0.75 |",
        "| BLM types | Two worker types with 150 workers each and three firm classes with 6 firms each; raw type effects and factors are $N(0,1)$ | Rank 1; singular value 1 |",
        "| Low-rank factors | Raw worker factor correlation with $\\alpha_i$ is 0.35; raw firm factor correlation with $\\psi_j$ is 0.25; innovations are Gaussian | Rank 2; singular values $(1,0.5)$ |",
        "| GKLP | Raw $(Z_i,\\eta_i)$ correlation 0.35; raw $(c_j,b_j)$ correlation 0.25 | Rank 1; $\\sigma_e=0.5$ and $m_{ij}=Z_i+c_j+b_j\\eta_i+\\frac{1}{2}b_j^2\\sigma_e^2$ |",
        "",
        "The Crippa row uses the Tukey surface in Crippa (2025, Section 2.2). The GKLP row uses the residualized perfect-information wage expression in Gibbons, Katz, Lemieux, and Parent (2002, 2005). The BLM row is discrete; the low-rank and GKLP rows use continuous Gaussian coordinates.",
        "",
        "Observed matches are drawn from the balanced assignment law",
        "",
        "$$",
        "P_{ij}=a_i b_j\\exp\\{0.5\\,g_i f_j+0.3\\,h_{ij}/\\operatorname{sd}(h)\\}.",
        "$$",
        "",
        "Here $g_i$ and $f_j$ are the DGP's worker and firm main components. For AKM, $h_{ij}=0$ and the interaction-sorting term is omitted. The balancing constants $a_i$ and $b_j$ restore uniform worker and firm marginals. A worker's first firm is drawn from the worker-specific conditional assignment distribution. In later periods, the worker redraws from that same distribution with probability 0.40 and otherwise stays; a redraw may select the same firm.",
        "",
        "## 2. Estimands and estimators",
        "",
        "| Symbol | Exact definition | Interpretation |",
        "|---|---|---|",
        "| $P_{ij}$ | observed population match probability | Assignment law used to sample firms |",
        "| $p_i,q_j$ | marginals of $P_{ij}$ | Worker and firm population weights |",
        "| $m_{ij}$ | $E[Y_{ijt}\\mid i,j]$ | Complete systematic wage schedule |",
        "| $h_{ij}$ | interaction in $m_{ij}=\\mu+a_i+b_j+h_{ij}$ under weights $p_iq_j$ | Nonadditive match component |",
        "| $Q_F$ | $E_{pq}[(m_{ij}-E_q[m_{ij}\\mid i])^2]$ | Firm-side schedule variation |",
        "| $H_F$ | $2E_{pq}[h_{ij}^2]$ | Twice the interaction variance |",
        "| $\\rho_H$ | $H_F/(2Q_F)$ when $Q_F>0$ | Interaction share of firm-side variation |",
        "| $C_{\\mathrm{assign}}$ | $\\{\\operatorname{Var}_P(m)-\\operatorname{Var}_{pq}(m)\\}/2$ | Variance contribution of observed assignment |",
        "",
        "The project procedure is the current low-rank plug-in with BIC rank selection, without the unfinished LOO correction. KSS estimates additive-projection variance components. BLM estimates a discrete worker-type by firm-class wage surface. BS20 estimates moments of worker and firm wage types. Because these objects are not identical under nonadditivity, Sections 4.3 and 4.4 distinguish accuracy for a procedure's native target from accuracy for the common project target.",
        "",
        "| Procedure | Settings used in every DGP |",
        "|---|---|",
        "| KSS-HE | PyTwoWay heteroskedastic correction with fe_exact=False; approximate rather than exact trace and leverage calculations |",
        "| BLM | Estimated firm groups; 2 worker types; 3 firm classes; CDF resolution 10; 4 initializations; retain best 2; at most 250 iterations; threshold $10^{-6}$ |",
        "| BS20 | Weighted PyTwoWay estimator after strongly connected, spell-level, no-return cleaning |",
        "| Project plug-in | 3 starts for positive rank; tolerance $10^{-6}$; at most 300 iterations; common support core requiring degree 3 for candidate set $\\{0,1\\}$ and degree 4 for $\\{0,1,2\\}$ |",
        "",
        "The BIC candidates are $\\{0,1\\}$ for AKM and grouped BLM and $\\{0,1,2\\}$ for Crippa, continuous low-rank, and GKLP. These choices matter because a candidate set that excludes the true rank cannot recover it.",
        "",
        "## 3. Evaluation rules",
        "",
        "The report uses four execution statuses: **pass or completed**, **returned with warning**, **unsupported sample**, and **estimator failure**. They do not have the same strength across procedures:",
        "",
        "| Procedure | Pass or completed | Returned with warning | Unsupported sample |",
        "|---|---|---|---|",
        "| KSS-HE and BS20 | PyTwoWay returned an estimate | Not used | Not used |",
        "| BLM | Every stayer class and mover class-pair is observed, and both likelihood paths avoid a one-step decline larger than $10^{-4}$ | Complete support, but a likelihood path violates that monotonicity check | At least one of the 3 stayer classes or 9 mover class-pairs is absent after cleaning and clustering |",
        "| Project plug-in | The selected fit converges; for positive rank, at least 2 of 3 starts have objective and functional spreads within the exact tolerances stated below | A finite fit fails at least one of those diagnostics | Not used |",
        "",
        "An **unsupported sample** is rejected before BLM fitting, whereas an **estimator failure** means an admissible sample reached the estimator but no value was returned. The $10^{-4}$ likelihood tolerance and the project multi-start thresholds are engineering warning rules, not statistical critical values. A pass therefore does not establish correct specification, rank, or low bias. RMSE includes every value returned from an admissible sample; the pass-only RMSE is a separately labeled sensitivity calculation.",
        "",
        "BLM preparation is: drop workers who return to a previous firm; collapse spells; estimate three firm clusters from empirical wage CDFs using K-means; form mover and stayer event-study samples; check all required observed cells; run four mover starts; retain the best two; select by connectedness; and finally fit the stayer model. The support check is necessary because PyTwoWay 0.3.21 forms probability arrays of size $2\\times 3^2$ for movers and $2\\times3$ for stayers. A missing firm-class cell otherwise becomes a label-dependent reshape error or a zero probability row.",
        "",
        "For continuous DGPs, BLM receives no oracle type labels. Product-marginal wage means define fixed equal-count reference groups only for scoring its fitted cell table. The BLM functionals are also compared with the full project truth, so discretization error remains part of the comparison.",
        "",
        "## 4. Results",
        "",
        "### 4.1 Completion and numerical warnings",
        "",
        "Each cell reports pass or completed / returned with warning / unsupported / failed. For KSS and BS20, the first number means completion only. The project fit passes its convergence and multi-start checks in every additive and Crippa replication, but in only 6% of grouped-BLM replications. Crippa shows why this is only a numerical check: all project fits pass even though BIC selects the wrong rank in every replication.",
        "",
        *status_table,
        "",
        f"![Figure 1. Pass or completion, returned-with-warning, unsupported samples, and estimator failures. KSS and BS20 use completion only.]({relative('status-matrix.png')})",
        "",
        f"The corrected classification finds {blm_unsupported} unsupported BLM samples and {blm_failures} estimator failures among 500 attempts. The exact reasons are:",
        "",
        *blm_reason_table,
        "",
        "The high unsupported rate follows from the observation design. As an approximation, the probability of remaining at the same firm in one later period is $0.6+0.4/18=0.622$. Remaining at the same firm through all nine later periods therefore has probability $0.622^9=0.014$, or about four workers out of 300. After returners are dropped, those few stayers must cover all three firm classes. This calculation is derived from the configured redraw rule; the exact audit uses the realized cleaned samples.",
        "",
        "### 4.2 Correctly specified benchmarks",
        "",
        "The expected benchmarks work. Under the additive AKM DGP, KSS-HE estimates $Q_F$ with bias "
        + _fmt(float(kss_q["bias"]), signed=True)
        + " and RMSE "
        + _fmt(float(kss_q["rmse"]))
        + "; its assignment-covariance bias is "
        + _fmt(float(kss_c["bias"]), signed=True)
        + f". Under the grouped BLM DGP, the {blm_returned} admissible returned BLM fits have biases "
        + _fmt(float(blm_q["bias"]), signed=True)
        + " for $Q_F$, "
        + _fmt(float(blm_h["bias"]), signed=True)
        + " for $H_F$, and "
        + _fmt(float(blm_c["bias"]), signed=True)
        + " for $C_{\\mathrm{assign}}$. These conditional results describe the supported subsample, not all 100 replications; feasibility is therefore part of BLM's performance rather than a footnote.",
        "",
        "### 4.3 Cross-DGP accuracy",
        "",
        "Figure 2 compares each reported or model-implied output with the common project truth, using admissible returned estimates only. `N/A` means the procedure has no defensible mapping to that project object; it does not mean zero or failure.",
        "",
        "| Procedure | $Q_F$ | $H_F$ | $\\rho_H$ | $C_{\\mathrm{assign}}$ |",
        "|---|---:|---:|---:|---:|",
        "| KSS-HE | Estimated | 0, imposed | 0, imposed when $Q_F>0$ | Estimated additive covariance |",
        "| BLM | Reported | Reported | Reported | Reported |",
        "| BS20 | N/A | N/A | N/A | N/A; native covariance shown in Section 4.4 |",
        "| Project plug-in | Reported | Reported | Reported | Reported |",
        "",
        "KSS now appears under $H_F$ and $\\rho_H$ as a structural benchmark. Its fitted schedule is additive, so $h_{ij}=0$, $H_F=0$, and $\\rho_H=0$ whenever fitted $Q_F>0$. These are restrictions imposed by the model, not interaction estimates learned from the data. BS20 is no longer placed under $C_{\\mathrm{assign}}$: it estimates $\\operatorname{Cov}_P(\\lambda_i,\\mu_j)$, a different object defined in Section 4.4.",
        "",
        "The `0*` cells require a different explanation. Under the AKM DGP, the project truth has $h_{ij}=0$, hence $H_F=\\rho_H=0$. KSS imposes the same zeros, and BIC selects rank zero in all 100 project replications. The project raw RMSEs are "
        + f"{float(project_akm_h['rmse']):.2e} and {float(project_akm_rho['rmse']):.2e}; "
        + "these are floating-point residue displayed as structural zeros. They test whether rank selection creates a false interaction, not ordinary precision in estimating a nonzero quantity. The log color scale is used because positive-rank warning cases generate errors above one million in some cells.",
        "",
        f"![Figure 2. RMSE against the common project truth among admissible returned estimates. N/A means no defensible mapping; 0* marks AKM structural zeros.]({relative('common-target-rmse.png')})",
        "",
        "KSS remains accurate for its additive target, but its error relative to project $Q_F$ expands when nonadditivity changes the target. BLM is highly accurate on the grouped DGP when it returns. The project procedure performs well for GKLP conditional on a diagnostic pass, but performs poorly for Crippa because BIC always removes the true rank-one interaction. Under the rank-two DGP, BIC always selects rank one and some returned fits have very large functional errors.",
        "",
        "### 4.4 Estimand differences: native versus project targets",
        "",
        "Section 4.3 measures each reported output against the project truth. That error combines two distinct components. For procedure $p$ with native target $\\theta_p$ and project target $\\theta_{\\mathrm{proj}}$:",
        "",
        "$$",
        "\\widehat{\\theta}_p-\\theta_{\\mathrm{proj}}=(\\widehat{\\theta}_p-\\theta_p)+(\\theta_p-\\theta_{\\mathrm{proj}}).",
        "$$",
        "",
        "The first term is estimation error for the procedure's own object. The second is an estimand difference. This section reports the second term, averaged over all 100 simulated populations; it contains no estimator sampling error.",
        "",
        "For KSS firm variance, the sign under independent assignment follows directly from the product-weighted ANOVA. Since $m_{ij}=\\mu+a_i+b_j+h_{ij}$ has zero weighted margins,",
        "",
        "$$",
        "Q_F=\\operatorname{Var}_q(b_j)+E_{pq}[h_{ij}^2]=\\operatorname{Var}_q(b_j)+H_F/2.",
        "$$",
        "",
        "Under independent assignment, the population additive projection recovers $b_j$. Therefore KSS native firm variance minus project $Q_F$ equals $-H_F/2\\leq0$. Under sorted assignment, the additive projection can absorb part of $h_{ij}$, so the negative sign is no longer a theorem; the plotted magnitudes then depend on the DGP.",
        "",
        *gap_table,
        "",
        f"![Figure 3. Population estimand differences, not estimator bias: native procedure target minus project target.]({relative('native-project-target-gaps.png')})",
        "",
        "For example, under Crippa the mean KSS native firm-variance target is "
        + _fmt(float(target_index[("crippa_tukey_dgp", "kss_he", "q_f")]["native_target"]))
        + ", while project $Q_F$ is "
        + _fmt(float(target_index[("crippa_tukey_dgp", "kss_he", "q_f")]["project_target"]))
        + ". The target gap is "
        + _fmt(float(target_index[("crippa_tukey_dgp", "kss_he", "q_f")]["target_gap"]), signed=True)
        + ". KSS's native-target bias is only "
        + _fmt(float(target_index[("crippa_tukey_dgp", "kss_he", "q_f")]["native_bias"]), signed=True)
        + ", but its RMSE against project $Q_F$ is "
        + _fmt(float(target_index[("crippa_tukey_dgp", "kss_he", "q_f")]["project_rmse"]))
        + ". Much of the common-target error is therefore disagreement about the object, not failure to estimate the KSS object.",
        "",
        "For BS20, define the native wage types $\\lambda_i=E_P[m_{ij}\\mid i]$ and $\\mu_j=E_P[m_{ij}\\mid j]$. BS20 targets $C_{\\mathrm{BS}}=\\operatorname{Cov}_P(\\lambda_i,\\mu_j)$; the project instead targets $C_{\\mathrm{assign}}=\\{\\operatorname{Var}_P(m)-\\operatorname{Var}_{pq}(m)\\}/2$. The simulation reads PyTwoWay's `cov(lambda, mu)` directly and compares its population target with $C_{\\mathrm{assign}}$ only in this estimand-gap section. No algebraic reconstruction of a BS20 value is used.",
        "",
        "The positive BS20 gap has a derivation in the additive case. If $m_{ij}=a_i+b_j$, let $Tb(i)=E[b_J\\mid I=i]$ and $T^*a(j)=E[a_I\\mid J=j]$. Expanding the two conditional wage types yields",
        "",
        "$$",
        "C_{\\mathrm{BS}}-C_{\\mathrm{assign}}=\\lVert Tb\\rVert^2+\\lVert T^*a\\rVert^2+\\langle Tb,TT^*a\\rangle\\geq0.",
        "$$",
        "",
        "The inequality follows from Cauchy--Schwarz and the contraction property of conditional expectation. For a general nonadditive schedule there is no universal sign, so the positive gaps outside AKM are features of the simulated sorting laws rather than a theorem.",
        "",
        "BLM creates an additional estimand difference when a continuous wage schedule is reduced to two worker groups by three firm groups. The table below reports grouped-schedule functional minus full-schedule project functional. The grouped BLM DGP uses its true simulated labels; all other rows use deterministic equal-count groups based on product-marginal wage means.",
        "",
        *blm_gap_table,
        "",
        "The project plug-in has no analogous row because its native target is the project target by definition. These population gaps are why native-target performance and common-target performance must be reported separately in the grand comparison.",
        "",
        "### 4.5 Rank selection",
        "",
        "After removing additive worker and firm effects, rank is the number of independent interaction dimensions in",
        "",
        "$$",
        "m_{ij}=\\alpha_i+\\psi_j+\\sum_{\\ell=1}^{r}\\lambda_{\\ell}U_{i\\ell}V_{j\\ell}.",
        "$$",
        "",
        "Rank zero is AKM: workers have no firm-specific comparative advantage. Rank one allows one comparative-advantage dimension; rank two allows two. The estimator does not observe $r$, so it fits every candidate on the same retained sample and selects the smallest observation-level BIC:",
        "",
        "$$",
        "\\operatorname{BIC}(r)=n\\log(\\operatorname{SSE}_r/n)+\\operatorname{df}(r)\\log n,",
        "$$",
        "",
        "where $\\operatorname{df}(r)=N+J-1+r(N+J-2-r)$. The first term rewards fit; the second penalizes the additional worker and firm factor coordinates. This BIC is an exploratory simulation rule, not a derived part of the unfinished LOO theory.",
        "",
        *rank_table,
        "",
        f"![Figure 4. Each cell is the percentage of 100 replications selecting that rank. The orange outline marks the true rank; the right label is the separate diagnostic-pass rate.]({relative('rank-selection.png')})",
        "",
        "A dark cell inside the orange outline means correct selection. BIC is correct in 100% of AKM and GKLP replications and 98% of grouped-BLM replications. It fails systematically in the two most informative continuous tests: Crippa is assigned rank zero in all 100 replications, and the rank-two DGP is assigned rank one in all 100. In Crippa, the fit improvement from rank one never overcomes the BIC penalty. In the rank-two DGP, the selection pattern is consistent with retaining the stronger singular-value-1 dimension while discarding the weaker singular-value-0.5 dimension.",
        "",
        "Rank selection and numerical warnings answer different questions. Crippa passes the project numerical checks in 100% of replications while choosing the wrong rank in 100%. On the grouped-BLM DGP, the project selector chooses the correct rank in 98% but passes its multi-start diagnostic in only 6%. Selecting too low a rank forces real nonadditivity to zero; selecting too high a rank can fit noise and destabilize schedule completion. A future LOO correction cannot repair either kind of rank mistake, so rank choice must be solved before interpreting LOO bias correction as the main remaining problem.",
        "",
        "### 4.6 Numerical warnings and tail risk",
        "",
        "Figure 5 compares RMSE across all returned project-BIC values with RMSE conditional on passing the convergence and multi-start checks in Section 3. Large gaps mean that warning cases dominate squared error. The pass-only line is a useful sensitivity calculation, but it conditions on an outcome of estimation and therefore cannot replace unconditional procedure performance. The thresholds are heuristic, so the figure should be read as evidence of tail-risk concentration rather than a formal good-fit/bad-fit classification.",
        "",
        f"![Figure 5. Project plug-in RMSE among all returned fits and conditional on a diagnostic pass.]({relative('project-returned-vs-stable-rmse.png')})",
        "",
        "## 5. Interpretation",
        "",
        "First, the original all-AKM comparison was uninformative because it rewarded additive estimators by construction. The full matrix now exposes both model misspecification and estimand differences.",
        "",
        "Second, correct specification is visible but not sufficient. KSS is accurate under AKM, and returned BLM fits are accurate under the grouped BLM DGP. BLM nevertheless needs enough mover and stayer support after cleaning.",
        "",
        "Third, the current project plug-in's central weakness is rank selection and positive-rank numerical reproducibility, not merely small-sample bias. The Crippa and rank-two results show that BIC can erase or truncate economically meaningful nonadditivity. Rank choice and multi-start agreement must be addressed before a leave-out correction can solve bias.",
        "",
        "Fourth, no single RMSE table can honestly compare all procedures on all quantities. N/A cells, structural zeros, and native-target gaps have different meanings. KSS and BS20 have native targets that differ from the schedule-based objects; BLM adds discretization error outside a genuinely grouped DGP; and only BLM and the project procedure yield all four project functionals. The report therefore keeps completion, numerical warnings, native-target accuracy, estimand differences, and common-target accuracy conceptually separate.",
        "",
        "## References",
        "",
        "Bonhomme, S., Lamadon, T., and Manresa, E. (2019). A Distributional Framework for Matched Employer-Employee Data. *Econometrica*, 87(3), 699-739. [doi:10.3982/ECTA15722](https://doi.org/10.3982/ECTA15722).",
        "",
        "Borovickova, K., and Shimer, R. (2017; February 2020 manuscript version). High Wage Workers Work for High Wage Firms. NBER Working Paper 24074. [doi:10.3386/w24074](https://doi.org/10.3386/w24074).",
        "",
        "Crippa, F. (2025). Identification, Estimation, and Inference in Two-Sided Interaction Models. Manuscript supplied to the project, Section 2.2.",
        "",
        "Gibbons, R., Katz, L. F., Lemieux, T., and Parent, D. (2002). Comparative Advantage, Learning, and Sectoral Wage Determination. NBER Working Paper 8889. Published in *Journal of Labor Economics* 23(4), 681-723 (2005). [doi:10.3386/w8889](https://doi.org/10.3386/w8889).",
        "",
        "Kline, P., Saggio, R., and Solvsten, M. (2020). Leave-Out Estimation of Variance Components. *Econometrica*, 88(5), 1859-1898. [doi:10.3982/ECTA16410](https://doi.org/10.3982/ECTA16410).",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_matrix_report(
    result: MonteCarloResult,
    output_directory: str | Path,
    markdown_path: str | Path,
) -> tuple[Path, ...]:
    """Generate the archived small-design report source bundle.

    The prose below contains fixed dimensions and historical interpretation.
    Reject other configurations rather than silently attaching that prose to
    the corrected cluster results.
    """

    archived_fingerprint = (
        "3e8ab3b57d35190989c47117f139b414b9f656db8a933d6c5d1df6d2aa5fbcc4"
    )
    actual_fingerprint = config_fingerprint(result.config)
    if actual_fingerprint != archived_fingerprint:
        raise ValueError(
            "This report generator is specific to the archived small "
            "design. It cannot describe a different configuration without "
            "silently using wrong sample sizes, BLM preparation, and "
            "interpretation. Build the corrected cluster report only after "
            "the 50 shards are merged."
        )

    _validate_result(result)
    output = Path(output_directory)
    markdown = Path(markdown_path)
    output.mkdir(parents=True, exist_ok=True)
    status_rows = _status_rows(result)
    common_rows = _common_rows(result)
    target_rows = _target_contrast_rows(result)
    rank_rows = _rank_rows(result)
    failure_rows = _failure_rows(result)

    files = _write_tables(
        output,
        status_rows=status_rows,
        common_rows=common_rows,
        target_rows=target_rows,
        rank_rows=rank_rows,
        failure_rows=failure_rows,
    )
    plt, LogNorm, Rectangle, _ = _load_plotting()
    figure_dir = output / "figures"
    files.extend(_plot_status(plt, Rectangle, figure_dir, status_rows))
    files.extend(_plot_common_rmse(plt, LogNorm, figure_dir, common_rows))
    files.extend(_plot_target_gaps(plt, figure_dir, target_rows))
    files.extend(_plot_rank_selection(plt, Rectangle, figure_dir, rank_rows))
    files.extend(_plot_project_tail_risk(plt, figure_dir, common_rows))
    plt.close("all")

    _write_markdown(
        markdown,
        result=result,
        output=output,
        status_rows=status_rows,
        common_rows=common_rows,
        target_rows=target_rows,
        rank_rows=rank_rows,
        failure_rows=failure_rows,
    )
    files.append(markdown)

    report_root = markdown.resolve().parent
    metadata = {
        "config_fingerprint": config_fingerprint(result.config),
        "replications": result.config.replications,
        "record_count": len(result.records),
        "attempt_count": len(result.attempts),
        "generated_files": [
            (
                path.resolve().relative_to(report_root).as_posix()
                if path.resolve().is_relative_to(report_root)
                else str(path.resolve())
            )
            for path in sorted(files)
        ],
    }
    metadata_path = output / "report_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files.append(metadata_path)
    return tuple(files)
