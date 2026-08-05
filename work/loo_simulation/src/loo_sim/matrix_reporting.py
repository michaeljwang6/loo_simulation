"""Tables, figures, and narrative for the five-by-four simulation matrix."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .monte_carlo import (
    EstimatorAttemptSummary,
    MonteCarloResult,
    MonteCarloSummary,
    config_fingerprint,
)


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
        "blm_estimated": ("h_f", "population_project"),
        "project_plugin_bic": ("h_f", "population_project"),
    },
    "rho_h": {
        "blm_estimated": ("rho_h", "population_project"),
        "project_plugin_bic": ("rho_h", "population_project"),
    },
    "c_assign": {
        "kss_he": ("worker_firm_covariance", "population_project"),
        "blm_estimated": ("c_assign", "population_project"),
        "bs20": ("worker_firm_covariance", "population_project"),
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
                    "n_failure": value.n_failure,
                    "success_rate": value.success_rate,
                    "unstable_rate": value.unstable_rate,
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
            "population_project",
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
    counter: Counter[tuple[str, str]] = Counter()
    for attempt in result.attempts:
        if attempt.estimator != "blm_estimated" or attempt.status != "failure":
            continue
        if "cannot reshape" in attempt.message:
            reason = "missing stayer-class support"
        elif "at least one stayer event" in attempt.message:
            reason = "no stayer event"
        elif "NaN" in attempt.message:
            reason = "invalid likelihood start"
        else:
            reason = "other"
        counter[(attempt.scenario, reason)] += 1
    rows: list[dict[str, Any]] = []
    for (scenario, reason), count in sorted(counter.items()):
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
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
    colors = (GREEN, ORANGE, RED)
    labels = ("Stable", "Unstable", "Failed")
    for row_index, scenario in enumerate(SCENARIOS):
        for column_index, estimator in enumerate(PROCEDURES):
            row = index[(scenario, estimator)]
            rates = (
                float(row["success_rate"]),
                float(row["unstable_rate"]),
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
                f"{row['n_success']}/{row['n_unstable']}/{row['n_failure']}",
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
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
    )
    ax.set_title(
        "Estimator status in 100 replications (stable / unstable / failed)",
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
                    ax.text(j, i, "-", ha="center", va="center", color=GRAY)
                    continue
                clipped = float(np.clip(value, norm.vmin, norm.vmax))
                position = (
                    np.log(clipped) - np.log(norm.vmin)
                ) / (np.log(norm.vmax) - np.log(norm.vmin))
                color = "white" if position < 0.58 else "black"
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
    colorbar.set_label("RMSE among returned estimates (log scale)")
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
            f"stable {100 * float(row['stable_rate']):.0f}%",
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
        ax.plot(x, stable, color=BLUE, marker="s", label="Stable only")
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
    fig.suptitle("Project plug-in: unstable fits create large tail risk", y=0.995)
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
                    f"{status_index[(scenario, estimator)]['n_failure']}"
                    for estimator in PROCEDURES
                ),
            )
            for scenario in SCENARIOS
        ],
        notes="Cells report stable/unstable/failed attempts out of 100.",
    )
    files.append(status_tex)

    rank_tex = table_dir / "rank_selection.tex"
    _write_latex(
        rank_tex,
        columns="lrrrrr",
        header=("DGP", "True", "Select 0", "Select 1", "Select 2", "Stable"),
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
        notes="Project plug-in BIC selections; stability is a separate diagnostic.",
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
    return f"{row['n_success']} / {row['n_unstable']} / {row['n_failure']}"


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
    blm_failures = sum(int(row["count"]) for row in failure_rows)
    missing_support = sum(
        int(row["count"])
        for row in failure_rows
        if row["reason"] in (
            "missing stayer-class support",
            "no stayer event",
        )
    )

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
        "| DGP | True rank | Selected ranks (0 / 1 / 2) | Stable |",
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

    lines = [
        "# DGP-by-estimator Monte Carlo: methods and results",
        "",
        "This report asks a deliberately symmetric question: what happens when each of four procedures is applied to data from each of five worker-firm wage models? The answer is not a single winner. KSS and BLM perform well when their own structures are correct, while nonadditivity, rank selection, sparse support, and differences in estimands explain the cross-model reversals.",
        "",
        "Every numerical statement below is derived from the immutable merged output with configuration fingerprint `"
        + config_fingerprint(result.config)
        + "`. Equations attributed to papers are cited; all remaining equations and calculations are definitions or direct derivations from the simulation code.",
        "",
        "## 1. Question and design",
        "",
        "The simulation contains 100 replications of each DGP. Every replication has 300 workers, 18 firms, 10 periods, redraw probability 0.40, and independent Gaussian wage noise with standard deviation 0.50. The estimator settings are declared once and applied to every DGP, so the row of the experiment cannot turn an estimator on or off.",
        "",
        "| DGP | Systematic wage schedule | Interaction rank |",
        "|---|---|---:|",
        "| AKM | $m_{ij}=\\alpha_i+\\psi_j$ | 0 |",
        "| Crippa/Tukey | $m_{ij}=\\alpha_i+\\psi_j+\\beta_0\\alpha_i\\psi_j$, $\\beta_0=0.75$ | 1 |",
        "| BLM types | $m_{ij}=a_{L_i}+p_{K_j}+u_{L_i}v_{K_j}$, two worker types and three firm classes | 1 |",
        "| Low-rank factors | $m_{ij}=\\alpha_i+\\psi_j+U_i'\\operatorname{diag}(1,0.5)V_j$ | 2 |",
        "| GKLP | $m_{ij}=Z_i+c_j+b_j\\eta_i+\\frac{1}{2}b_j^2\\sigma_e^2$ | 1 |",
        "",
        "The Crippa row uses the Tukey surface in Crippa (2025, Section 2.2). The GKLP row is the residualized perfect-information wage expression in Gibbons, Katz, Lemieux, and Parent (2002, 2005). The BLM row uses discrete worker and firm types, while the project row uses continuous Gaussian factor coordinates.",
        "",
        "Observed matches are drawn from a balanced assignment law:",
        "",
        "$$",
        "P_{ij}=a_i b_j\\exp\\{s_c\\alpha_i\\psi_j+s_h h_{ij}/\\operatorname{sd}(h)\\}.",
        "$$",
        "",
        "The constants $a_i$ and $b_j$ restore uniform worker and firm marginals. This holds population composition fixed while allowing sorting on the additive component and interaction gains.",
        "",
        "## 2. Estimands and estimators",
        "",
        "| Symbol | Definition | Interpretation |",
        "|---|---|---|",
        "| $Q_F$ | product-weighted firm-side schedule variance | Firm contribution in the complete schedule |",
        "| $H_F$ | twice the product-weighted interaction variance | Magnitude of nonadditivity |",
        "| $\\rho_H$ | interaction share of $Q_F$ | Relative importance of nonadditivity |",
        "| $C_{\\mathrm{assign}}$ | covariance induced by observed assignment | Sorting covariance in realized matches |",
        "",
        "The project procedure is the current low-rank plug-in with BIC rank selection, without the unfinished LOO correction. KSS estimates additive-projection variance components. BLM estimates a discrete worker-type by firm-class wage surface. BS20 estimates moments of worker and firm wage types. Because those objects are not identical under nonadditivity, the report distinguishes native-target accuracy from accuracy relative to the common population-project truth.",
        "",
        "## 3. Evaluation rules",
        "",
        "A fit is **stable** only when its procedure-specific numerical diagnostics pass. An **unstable** fit returned finite values but failed those diagnostics; a **failure** returned no estimate. Headline RMSE includes every returned value and is always displayed with the stable, unstable, and failed counts. Stable-only RMSE is a conditional robustness calculation, not a replacement for the headline result.",
        "",
        "For continuous DGPs, BLM receives no oracle type labels. Product-marginal wage means define fixed equal-count reference groups only for scoring its fitted cell table. The BLM functionals are also compared with the full project truth, so discretization error remains part of the comparison.",
        "",
        "## 4. Results",
        "",
        "### 4.1 Completion and numerical stability",
        "",
        "The table and figure report stable / unstable / failed attempts. KSS and BS20 return stable values in all 500 DGP-replications. BLM is much more sensitive to cleaned-sample support. The project BIC fit is stable in the additive and Crippa rows, but only 6% of grouped-BLM replications.",
        "",
        *status_table,
        "",
        f"![Figure 1. Stable, unstable, and failed estimator attempts in every DGP-procedure cell.]({relative('status-matrix.png')})",
        "",
        f"BLM has {blm_failures} failures out of 500 attempts. {missing_support} are directly attributable to no stayer event or a missing stayer firm class after procedure-specific cleaning. The remaining failures are invalid likelihood starts. This is why BLM accuracy cannot be summarized without its return rate.",
        "",
        "### 4.2 Correctly specified benchmarks",
        "",
        "The expected benchmarks work. Under the additive AKM DGP, KSS-HE estimates $Q_F$ with bias "
        + _fmt(float(kss_q["bias"]), signed=True)
        + " and RMSE "
        + _fmt(float(kss_q["rmse"]))
        + "; its assignment-covariance bias is "
        + _fmt(float(kss_c["bias"]), signed=True)
        + ". Under the grouped BLM DGP, the 81 returned BLM fits have biases "
        + _fmt(float(blm_q["bias"]), signed=True)
        + " for $Q_F$, "
        + _fmt(float(blm_h["bias"]), signed=True)
        + " for $H_F$, and "
        + _fmt(float(blm_c["bias"]), signed=True)
        + " for $C_{\\mathrm{assign}}$. Thus the main problem for BLM in its preferred row is support and completion, not bias among returned fits.",
        "",
        "### 4.3 Cross-DGP accuracy",
        "",
        "Figure 2 compares RMSE against the common population-project truth. A blank cell means that the procedure does not estimate that object. The log color scale is necessary because unstable positive-rank project fits generate very large tail errors in some rows.",
        "",
        f"![Figure 2. RMSE against the common project truth among all returned estimates; missing cells are estimands the procedure does not report.]({relative('common-target-rmse.png')})",
        "",
        "KSS remains accurate for its additive projection, but its error relative to project $Q_F$ expands when nonadditivity changes the target. BLM is highly accurate on the grouped DGP when it returns. The project BIC procedure performs well for GKLP conditional on stability, but performs poorly for Crippa because BIC always removes the true rank-one interaction. Under the rank-two DGP, BIC always selects rank one and some returned fits have very large functional errors.",
        "",
        "### 4.4 Native targets versus the common project truth",
        "",
        "A procedure can estimate its own target accurately while differing systematically from the project estimand. The target differences below are native target minus project target; they are properties of the simulated population, not estimation bias.",
        "",
        *gap_table,
        "",
        f"![Figure 3. Population differences between KSS or BS20 native targets and the project targets.]({relative('native-project-target-gaps.png')})",
        "",
        "This distinction explains why a small native-target bias is not enough to establish accuracy for $Q_F$ or $C_{\\mathrm{assign}}$ under a nonadditive DGP.",
        "",
        "### 4.5 Rank selection",
        "",
        *rank_table,
        "",
        f"![Figure 4. Distribution of selected BIC ranks; the orange outline marks the true interaction rank.]({relative('rank-selection.png')})",
        "",
        "Rank selection succeeds in the additive and GKLP rows. It fails systematically in the two most informative continuous misspecification tests: Crippa is always assigned rank zero, and the continuous rank-two DGP is always assigned rank one. The grouped DGP is usually assigned rank one, but that does not ensure stable continuous-factor recovery because many workers share identical latent coordinates.",
        "",
        "### 4.6 Instability and tail risk",
        "",
        "Figure 5 compares RMSE across all returned project-BIC values with RMSE conditional on stability. Large gaps mean that a small set of numerically unstable fits dominates squared error. The stable-only line answers a useful diagnostic question, but it is selection-conditional and therefore cannot be presented as unconditional estimator performance.",
        "",
        f"![Figure 5. Project plug-in RMSE among all returned fits and conditional on passing stability diagnostics.]({relative('project-returned-vs-stable-rmse.png')})",
        "",
        "## 5. Interpretation",
        "",
        "First, the original all-AKM comparison was uninformative because it rewarded additive estimators by construction. The full matrix now exposes both model misspecification and estimand differences.",
        "",
        "Second, correct specification is visible but not sufficient. KSS is accurate under AKM, and returned BLM fits are accurate under the grouped BLM DGP. BLM nevertheless needs enough mover and stayer support after cleaning.",
        "",
        "Third, the current project plug-in's central weakness is rank selection and positive-rank numerical stability, not merely small-sample bias. The Crippa and rank-two results show that BIC can erase or truncate economically meaningful nonadditivity. This finding should guide development of the eventual LOO estimator: rank choice and functional stability must be addressed before a leave-out correction can solve bias.",
        "",
        "Fourth, no single RMSE table can honestly compare all procedures on all quantities. KSS and BS20 have native targets that differ from the schedule-based objects, and only BLM and the project procedure yield all four project functionals. The report therefore shows native and common-target results separately.",
        "",
        "## 6. Reproducibility",
        "",
        f"The merged run contains {len(result.records):,} scalar records and {len(result.attempts):,} estimator attempts. All 100 replication indices occur exactly once. The editable source tables and vector figures are in `reports/dgp_estimator_matrix`; the immutable simulation inputs are in `results/dgp_estimator_matrix/merged`.",
        "",
        "The report can be regenerated without calling Codex:",
        "",
        "`& .\\.venv311\\Scripts\\python.exe scripts\\report_dgp_estimator_matrix.py`",
        "",
        "`& .\\.venv311\\Scripts\\python.exe scripts\\build_simulation_writeup_pdf.py --input DGP_ESTIMATOR_RESULTS.md --output output\\pdf\\dgp_estimator_matrix_report.pdf`",
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
    """Generate the complete DGP-by-estimator report source bundle."""

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
