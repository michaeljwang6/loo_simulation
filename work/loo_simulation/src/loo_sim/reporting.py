"""Paper-ready tables and figures for the production Monte Carlo."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .monte_carlo import (
    MonteCarloResult,
    MonteCarloSummary,
    config_fingerprint,
)


SCENARIO_LABELS = {
    "additive_independent": "Additive, independent",
    "additive_common_sorting": "Additive, common sorting",
    "rank1_independent": "Rank 1, independent",
    "rank1_common_sorting": "Rank 1, common sorting",
    "rank1_interaction_sorting": "Rank 1, interaction sorting",
    "grouped_blm": "Grouped BLM",
    "rank2_misspecification": "Rank 2",
}

ESTIMATOR_LABELS = {
    "project_plugin_bic": "Low-rank plug-in, no LOO (BIC)",
    "akm_fe": "AKM plug-in",
    "kss_ho": "KSS-HO",
    "kss_he": "KSS-HE",
    "blm_oracle": "BLM, oracle groups",
    "blm_estimated": "BLM, estimated groups",
    "bs20": "BS20",
}

METRIC_LABELS = {
    "q_f": r"$Q_F$",
    "h_f": r"$H_F$",
    "c_assign": r"$C_{\mathrm{assign}}$",
    "rho_h": r"$\rho_H$",
}

ADDITIVE_SCENARIOS = (
    "additive_independent",
    "additive_common_sorting",
)

NONADDITIVE_SCENARIOS = (
    "rank1_independent",
    "rank1_common_sorting",
    "rank1_interaction_sorting",
    "rank2_misspecification",
    "grouped_blm",
)

ALL_SCENARIOS = ADDITIVE_SCENARIOS + NONADDITIVE_SCENARIOS

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#6B7280"
LIGHT_GRAY = "#D1D5DB"


def _summary_index(
    summaries: Iterable[MonteCarloSummary],
) -> dict[tuple[str, str, str, str], MonteCarloSummary]:
    return {
        (
            row.scenario,
            row.estimator,
            row.metric,
            row.target_type,
        ): row
        for row in summaries
    }


def _summary(
    index: Mapping[
        tuple[str, str, str, str],
        MonteCarloSummary,
    ],
    scenario: str,
    estimator: str,
    metric: str,
    target_type: str,
) -> MonteCarloSummary:
    return index[(scenario, estimator, metric, target_type)]


def _rank_selection_rows(
    result: MonteCarloResult,
) -> list[dict[str, Any]]:
    scenario_config = {
        scenario.name: scenario for scenario in result.config.scenarios
    }
    attempt_summary = {
        (row.scenario, row.estimator): row
        for row in result.attempt_summaries()
    }
    selections: dict[str, list[int]] = {}
    for record in result.records:
        if (
            record.estimator == "project_plugin_bic"
            and record.metric == "selected_rank"
            and record.target_type == "rank_diagnostic"
        ):
            selections.setdefault(record.scenario, []).append(
                int(round(record.estimate))
            )

    rows: list[dict[str, Any]] = []
    for scenario in ALL_SCENARIOS:
        values = selections[scenario]
        true_rank = scenario_config[scenario].true_rank
        counts = Counter(values)
        distribution = "; ".join(
            f"{rank}: {count}"
            for rank, count in sorted(counts.items())
        )
        attempts = attempt_summary[(scenario, "project_plugin_bic")]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "true_rank": true_rank,
                "selection_distribution": distribution,
                "correct_rank_rate": float(
                    np.mean(np.asarray(values) == true_rank)
                ),
                "bic_unstable_rate": attempts.unstable_rate,
                "bic_failure_rate": attempts.failure_rate,
            }
        )
    return rows


def _additive_rows(
    unconditional: Mapping[
        tuple[str, str, str, str],
        MonteCarloSummary,
    ],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in ADDITIVE_SCENARIOS:
        for estimator in (
            "project_plugin_bic",
            "akm_fe",
            "kss_ho",
            "kss_he",
        ):
            if estimator == "project_plugin_bic":
                firm_metric = "q_f"
                covariance_metric = "c_assign"
                target_type = "population_project"
            else:
                firm_metric = "firm_variance"
                covariance_metric = "worker_firm_covariance"
                target_type = "native_akm"
            firm = _summary(
                unconditional,
                scenario,
                estimator,
                firm_metric,
                target_type,
            )
            covariance = _summary(
                unconditional,
                scenario,
                estimator,
                covariance_metric,
                target_type,
            )
            rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "estimator": estimator,
                    "estimator_label": ESTIMATOR_LABELS[estimator],
                    "firm_bias": firm.bias,
                    "firm_bias_mcse": firm.bias_monte_carlo_se,
                    "firm_rmse": firm.rmse,
                    "covariance_bias": covariance.bias,
                    "covariance_bias_mcse": (
                        covariance.bias_monte_carlo_se
                    ),
                    "covariance_rmse": covariance.rmse,
                }
            )
    return rows


def _nonadditive_rows(
    unconditional: Mapping[
        tuple[str, str, str, str],
        MonteCarloSummary,
    ],
    stable: Mapping[
        tuple[str, str, str, str],
        MonteCarloSummary,
    ],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in NONADDITIVE_SCENARIOS:
        stable_q = _summary(
            stable,
            scenario,
            "project_plugin_bic",
            "q_f",
            "population_project",
        )
        native_firm = _summary(
            unconditional,
            scenario,
            "akm_fe",
            "firm_variance",
            "native_akm",
        )
        project_firm = _summary(
            unconditional,
            scenario,
            "akm_fe",
            "firm_variance",
            "population_project",
        )
        kss_he = _summary(
            unconditional,
            scenario,
            "kss_he",
            "firm_variance",
            "native_akm",
        )
        row: dict[str, Any] = {
            "scenario": scenario,
            "scenario_label": SCENARIO_LABELS[scenario],
            "unstable_rate": stable_q.n_unstable / stable_q.n_attempts,
            "stable_n": stable_q.n_estimates,
            "akm_target_gap": (
                native_firm.mean_target - project_firm.mean_target
            ),
            "kss_he_native_bias": kss_he.bias,
            "kss_he_native_bias_mcse": kss_he.bias_monte_carlo_se,
        }
        for metric in ("q_f", "h_f", "c_assign", "rho_h"):
            unconditional_row = _summary(
                unconditional,
                scenario,
                "project_plugin_bic",
                metric,
                "population_project",
            )
            stable_row = _summary(
                stable,
                scenario,
                "project_plugin_bic",
                metric,
                "population_project",
            )
            row[f"{metric}_unconditional_rmse"] = (
                unconditional_row.rmse
            )
            row[f"{metric}_stable_rmse"] = stable_row.rmse
            row[f"{metric}_stable_bias"] = stable_row.bias
            row[f"{metric}_stable_bias_mcse"] = (
                stable_row.bias_monte_carlo_se
            )
        rows.append(row)
    return rows


def _grouped_blm_rows(
    stable: Mapping[
        tuple[str, str, str, str],
        MonteCarloSummary,
    ],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for estimator in (
        "blm_oracle",
        "blm_estimated",
        "project_plugin_bic",
    ):
        target_type = (
            "grouped_population_project"
            if estimator.startswith("blm_")
            else "population_project"
        )
        q_f = _summary(
            stable,
            "grouped_blm",
            estimator,
            "q_f",
            target_type,
        )
        row: dict[str, Any] = {
            "estimator": estimator,
            "estimator_label": ESTIMATOR_LABELS[estimator],
            "stable_n": q_f.n_estimates,
            "unstable_rate": q_f.n_unstable / q_f.n_attempts,
        }
        if estimator.startswith("blm_"):
            cell = _summary(
                stable,
                "grouped_blm",
                estimator,
                "cell_mean_rmse",
                "alignment_diagnostic",
            )
            row["mean_cell_rmse"] = cell.mean_estimate
        else:
            row["mean_cell_rmse"] = float("nan")
        for metric in ("q_f", "h_f", "c_assign", "rho_h"):
            metric_row = _summary(
                stable,
                "grouped_blm",
                estimator,
                metric,
                target_type,
            )
            row[f"{metric}_rmse"] = metric_row.rmse
            row[f"{metric}_bias"] = metric_row.bias
            row[f"{metric}_bias_mcse"] = (
                metric_row.bias_monte_carlo_se
            )
        rows.append(row)
    return rows


def _bs20_rows(
    result: MonteCarloResult,
    unconditional: Mapping[
        tuple[str, str, str, str],
        MonteCarloSummary,
    ],
) -> list[dict[str, Any]]:
    attempts: dict[str, list[Any]] = {}
    for attempt in result.attempts:
        if attempt.estimator == "bs20":
            attempts.setdefault(attempt.scenario, []).append(attempt)
    rows: list[dict[str, Any]] = []
    for scenario in ALL_SCENARIOS:
        correlation = _summary(
            unconditional,
            scenario,
            "bs20",
            "worker_firm_correlation",
            "native_bs20",
        )
        sample = attempts[scenario]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "mean_observations": float(
                    np.mean(
                        [attempt.n_observations for attempt in sample]
                    )
                ),
                "mean_workers": float(
                    np.mean([attempt.n_workers for attempt in sample])
                ),
                "mean_firms": float(
                    np.mean([attempt.n_firms for attempt in sample])
                ),
                "correlation_bias": correlation.bias,
                "correlation_bias_mcse": (
                    correlation.bias_monte_carlo_se
                ),
                "correlation_rmse": correlation.rmse,
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
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
        "% Generated by scripts/report_results.py",
        r"\begin{tabular}{" + columns + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            "% " + notes,
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _format_bias(row: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{row[prefix + '_bias']:+.3f} "
        f"({row[prefix + '_bias_mcse']:.3f})"
    )


def _write_tables(
    output: Path,
    *,
    rank_rows: Sequence[Mapping[str, Any]],
    additive_rows: Sequence[Mapping[str, Any]],
    nonadditive_rows: Sequence[Mapping[str, Any]],
    grouped_rows: Sequence[Mapping[str, Any]],
    bs20_rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    table_dir = output / "tables"
    files: list[Path] = []
    table_specs = (
        ("stability_rank_selection.csv", rank_rows),
        ("additive_comparison.csv", additive_rows),
        ("nonadditive_comparison.csv", nonadditive_rows),
        ("grouped_blm_comparison.csv", grouped_rows),
        ("bs20_native.csv", bs20_rows),
    )
    for name, rows in table_specs:
        path = table_dir / name
        _write_csv(path, rows)
        files.append(path)

    rank_tex = table_dir / "stability_rank_selection.tex"
    _write_latex(
        rank_tex,
        columns="lrrrr",
        header=(
            "Scenario",
            "True rank",
            "Selected rank (count)",
            "Correct",
            "Unstable",
        ),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                str(row["true_rank"]),
                _latex_escape(str(row["selection_distribution"])),
                f"{100 * row['correct_rank_rate']:.0f}\\%",
                f"{100 * row['bic_unstable_rate']:.0f}\\%",
            )
            for row in rank_rows
        ],
        notes=(
            "Correct rank and instability refer to the exploratory "
            "low-rank plug-in without LOO correction (BIC)."
        ),
    )
    files.append(rank_tex)

    additive_tex = table_dir / "additive_comparison.tex"
    _write_latex(
        additive_tex,
        columns="llrrrr",
        header=(
            "Scenario",
            "Procedure",
            r"$Q_F$ / firm bias (MCSE)",
            "RMSE",
            r"$C_{\rm assign}$ / covariance bias (MCSE)",
            "RMSE",
        ),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                _latex_escape(str(row["estimator_label"])),
                _format_bias(row, "firm"),
                f"{row['firm_rmse']:.3f}",
                _format_bias(row, "covariance"),
                f"{row['covariance_rmse']:.3f}",
            )
            for row in additive_rows
        ],
        notes=(
            "The additive designs equate the native AKM moments with the "
            "project moments. MCSE is the Monte Carlo standard error of bias."
        ),
    )
    files.append(additive_tex)

    nonadditive_tex = table_dir / "nonadditive_comparison.tex"
    _write_latex(
        nonadditive_tex,
        columns="lrrrrrrr",
        header=(
            "Scenario",
            "Unstable",
            "Stable $N$",
            r"$Q_F$",
            r"$H_F$",
            r"$C_{\rm assign}$",
            r"$\rho_H$",
            "AKM gap",
        ),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                f"{100 * row['unstable_rate']:.0f}\\%",
                str(row["stable_n"]),
                f"{row['q_f_stable_rmse']:.3f}",
                f"{row['h_f_stable_rmse']:.3f}",
                f"{row['c_assign_stable_rmse']:.3f}",
                f"{row['rho_h_stable_rmse']:.3f}",
                f"{row['akm_target_gap']:+.3f}",
            )
            for row in nonadditive_rows
        ],
        notes=(
            "Project columns report stable-only RMSE. AKM gap is the native "
            "AKM firm-variance target minus project Q_F."
        ),
    )
    files.append(nonadditive_tex)

    grouped_tex = table_dir / "grouped_blm_comparison.tex"
    _write_latex(
        grouped_tex,
        columns="lrrrrrr",
        header=(
            "Procedure",
            "Stable $N$",
            "Cell RMSE",
            r"$Q_F$",
            r"$H_F$",
            r"$C_{\rm assign}$",
            r"$\rho_H$",
        ),
        rows=[
            (
                _latex_escape(str(row["estimator_label"])),
                str(row["stable_n"]),
                (
                    "--"
                    if not np.isfinite(row["mean_cell_rmse"])
                    else f"{row['mean_cell_rmse']:.3f}"
                ),
                f"{row['q_f_rmse']:.3f}",
                f"{row['h_f_rmse']:.3f}",
                f"{row['c_assign_rmse']:.3f}",
                f"{row['rho_h_rmse']:.3f}",
            )
            for row in grouped_rows
        ],
        notes="All reported RMSE values are conditional on a stable fit.",
    )
    files.append(grouped_tex)

    bs20_tex = table_dir / "bs20_native.tex"
    _write_latex(
        bs20_tex,
        columns="lrrrr",
        header=(
            "Scenario",
            "Mean observations",
            "Mean workers",
            "Correlation bias (MCSE)",
            "RMSE",
        ),
        rows=[
            (
                _latex_escape(str(row["scenario_label"])),
                f"{row['mean_observations']:.1f}",
                f"{row['mean_workers']:.1f}",
                (
                    f"{row['correlation_bias']:+.3f} "
                    f"({row['correlation_bias_mcse']:.3f})"
                ),
                f"{row['correlation_rmse']:.3f}",
            )
            for row in bs20_rows
        ],
        notes=(
            "Bias and RMSE refer to the native BS20 worker-firm "
            "correlation target."
        ),
    )
    files.append(bs20_tex)
    return files


def _load_plotting() -> tuple[Any, Any]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Reporting figures require the optional 'report' dependencies. "
            "Install the project with `pip install -e .[report]`."
        ) from exc
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    return plt, None


def _save_figure(plt: Any, fig: Any, output: Path, stem: str) -> list[Path]:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    png = figure_dir / f"{stem}.png"
    pdf = figure_dir / f"{stem}.pdf"
    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={"Software": "loo_sim.reporting"},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.05,
        metadata={
            "Creator": "loo_sim.reporting",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(fig)
    return [png, pdf]


def _plot_stability(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    labels = [str(row["scenario_label"]) for row in rows]
    unstable = np.asarray(
        [100 * row["bic_unstable_rate"] for row in rows]
    )
    correct = np.asarray(
        [100 * row["correct_rank_rate"] for row in rows]
    )
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.barh(
        y,
        unstable,
        height=0.52,
        color=ORANGE,
        alpha=0.84,
        label="Unstable BIC attempts",
    )
    ax.scatter(
        correct,
        y,
        marker="D",
        s=30,
        color=BLUE,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        label="Correct-rank selection",
    )
    for index, (unstable_value, correct_value) in enumerate(
        zip(unstable, correct)
    ):
        if unstable_value > 0:
            ax.text(
                unstable_value + 1.4,
                index,
                f"{unstable_value:.0f}%",
                va="center",
                color=ORANGE,
            )
        ax.text(
            correct_value - 1.5,
            index - 0.23,
            f"{correct_value:.0f}%",
            va="center",
            ha="right",
            color=BLUE,
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 106)
    ax.set_xlabel("Percent of 100 replications")
    ax.set_title("Rank selection does not guarantee functional stability")
    ax.axvline(100, color=LIGHT_GRAY, linewidth=0.8, zorder=0)
    ax.legend(
        frameon=False,
        loc="center",
        bbox_to_anchor=(0.55, 0.53),
    )
    fig.tight_layout()
    return _save_figure(
        plt,
        fig,
        output,
        "stability-rank-selection",
    )


def _plot_additive_bias(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    estimator_order = (
        "project_plugin_bic",
        "akm_fe",
        "kss_ho",
        "kss_he",
    )
    colors = {
        "project_plugin_bic": ORANGE,
        "akm_fe": GRAY,
        "kss_ho": GREEN,
        "kss_he": BLUE,
    }
    markers = {
        "project_plugin_bic": "s",
        "akm_fe": "o",
        "kss_ho": "^",
        "kss_he": "D",
    }
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 5.0),
        sharey=True,
    )
    for row_index, scenario in enumerate(ADDITIVE_SCENARIOS):
        scenario_rows = {
            str(row["estimator"]): row
            for row in rows
            if row["scenario"] == scenario
        }
        for column_index, (
            prefix,
            title,
        ) in enumerate(
            (
                ("firm", r"$Q_F$ / firm variance"),
                (
                    "covariance",
                    r"$C_{\mathrm{assign}}$ / covariance",
                ),
            )
        ):
            ax = axes[row_index, column_index]
            y = np.arange(len(estimator_order))
            for index, estimator in enumerate(estimator_order):
                row = scenario_rows[estimator]
                estimate = float(row[f"{prefix}_bias"])
                error = 1.96 * float(row[f"{prefix}_bias_mcse"])
                ax.errorbar(
                    estimate,
                    index,
                    xerr=error,
                    fmt=markers[estimator],
                    color=colors[estimator],
                    markersize=5,
                    capsize=2.5,
                    linewidth=1.2,
                )
            ax.axvline(0, color=LIGHT_GRAY, linewidth=1)
            ax.set_yticks(
                y,
                [ESTIMATOR_LABELS[item] for item in estimator_order],
            )
            if row_index == 0:
                ax.set_title(title)
            if column_index == 0:
                ax.set_ylabel(SCENARIO_LABELS[scenario])
            if row_index == 1:
                ax.set_xlabel("Bias with 95% Monte Carlo interval")
    axes[0, 0].invert_yaxis()
    fig.suptitle(
        "Additive designs: common targets, different finite-sample behavior",
        y=1.01,
    )
    fig.tight_layout()
    return _save_figure(plt, fig, output, "additive-bias")


def _plot_nonadditive_rmse(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.1, 6.0),
        sharey=True,
    )
    labels = [str(row["scenario_label"]) for row in rows]
    y = np.arange(len(rows))
    for ax, metric in zip(
        axes.flat,
        ("q_f", "h_f", "c_assign", "rho_h"),
    ):
        stable = np.asarray(
            [row[f"{metric}_stable_rmse"] for row in rows],
            dtype=float,
        )
        unconditional = np.asarray(
            [row[f"{metric}_unconditional_rmse"] for row in rows],
            dtype=float,
        )
        for index in range(len(rows)):
            ax.plot(
                [stable[index], unconditional[index]],
                [index, index],
                color=LIGHT_GRAY,
                linewidth=1,
                zorder=1,
            )
        ax.scatter(
            stable,
            y,
            marker="o",
            s=25,
            color=BLUE,
            label="Stable-only",
            zorder=3,
        )
        ax.scatter(
            unconditional,
            y,
            marker="^",
            s=30,
            color=ORANGE,
            label="Unconditional",
            zorder=3,
        )
        ax.set_xscale("log")
        ax.set_yticks(y, labels)
        ax.set_title(METRIC_LABELS[metric])
        ax.set_xlabel("RMSE (log scale)")
        if metric == "rho_h":
            ax.set_xticks((0.03, 0.1, 0.3), ("0.03", "0.10", "0.30"))
            ax.minorticks_off()
    axes[0, 0].invert_yaxis()
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
    )
    fig.suptitle(
        "Nonadditive low-rank plug-in (no LOO): "
        "instability drives tail risk",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save_figure(
        plt,
        fig,
        output,
        "nonadditive-unconditional-vs-stable-rmse",
    )


def _plot_estimand_gap(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    labels = [str(row["scenario_label"]) for row in rows]
    gaps = np.asarray([row["akm_target_gap"] for row in rows])
    bias = np.asarray([row["kss_he_native_bias"] for row in rows])
    error = 1.96 * np.asarray(
        [row["kss_he_native_bias_mcse"] for row in rows]
    )
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.1, 4.0))
    ax.barh(
        y,
        gaps,
        height=0.5,
        color=BLUE,
        alpha=0.82,
        label=r"Native AKM target minus project $Q_F$",
    )
    ax.errorbar(
        bias,
        y,
        xerr=error,
        fmt="D",
        markersize=5,
        color=ORANGE,
        capsize=2.5,
        linewidth=1.2,
        label="KSS-HE bias for native target",
    )
    for index, gap in enumerate(gaps):
        ax.text(
            gap + 0.025,
            index,
            f"{gap:.2f}",
            va="center",
            ha="left",
            color="white",
        )
    ax.axvline(0, color=LIGHT_GRAY, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Firm-variance units")
    ax.set_title(
        r"KSS estimates its native target; it differs from project $Q_F$"
    )
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    return _save_figure(plt, fig, output, "akm-project-estimand-gap")


def _plot_grouped_blm(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    metrics = ("q_f", "h_f", "c_assign", "rho_h")
    x = np.arange(len(metrics))
    style = {
        "blm_oracle": (BLUE, "o"),
        "blm_estimated": (GREEN, "s"),
        "project_plugin_bic": (ORANGE, "^"),
    }
    fig, ax = plt.subplots(figsize=(7.1, 3.9))
    for row in rows:
        estimator = str(row["estimator"])
        color, marker = style[estimator]
        values = np.asarray(
            [row[f"{metric}_rmse"] for metric in metrics]
        )
        ax.plot(
            x,
            values,
            color=color,
            marker=marker,
            markersize=5,
            linewidth=1.4,
            label=(
                f"{row['estimator_label']} "
                f"(stable {row['stable_n']}/100)"
            ),
        )
    ax.set_yscale("log")
    ax.set_ylim(0.009, 20)
    ax.set_xticks(x, [METRIC_LABELS[metric] for metric in metrics])
    ax.set_ylabel("RMSE (log scale)")
    ax.set_title("Grouped DGP favors the grouped BLM procedure")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    return _save_figure(plt, fig, output, "grouped-blm-comparison")


def _plot_bs20(
    plt: Any,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
) -> list[Path]:
    labels = [str(row["scenario_label"]) for row in rows]
    bias = np.asarray([row["correlation_bias"] for row in rows])
    error = 1.96 * np.asarray(
        [row["correlation_bias_mcse"] for row in rows]
    )
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.1, 3.8))
    ax.errorbar(
        bias,
        y,
        xerr=error,
        fmt="o",
        color=PURPLE,
        markersize=5,
        capsize=2.5,
        linewidth=1.2,
    )
    ax.axvline(0, color=LIGHT_GRAY, linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Bias with 95% Monte Carlo interval")
    ax.set_title("BS20 native worker-firm correlation is downward biased")
    fig.tight_layout()
    return _save_figure(plt, fig, output, "bs20-native-correlation-bias")


def build_production_report(
    result: MonteCarloResult,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    """Generate deterministic CSV/LaTeX tables and PNG/PDF figures."""

    if result.replication_indices != tuple(
        range(result.config.replications)
    ):
        raise ValueError("Production reporting requires a complete result.")
    missing_scenarios = set(ALL_SCENARIOS) - {
        scenario.name for scenario in result.config.scenarios
    }
    if missing_scenarios:
        raise ValueError(
            "Production result is missing required scenarios: "
            + ", ".join(sorted(missing_scenarios))
        )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    unconditional = _summary_index(result.summaries())
    stable = _summary_index(
        result.summaries(included_statuses=("success",))
    )
    rank_rows = _rank_selection_rows(result)
    additive_rows = _additive_rows(unconditional)
    nonadditive_rows = _nonadditive_rows(
        unconditional,
        stable,
    )
    grouped_rows = _grouped_blm_rows(stable)
    bs20_rows = _bs20_rows(result, unconditional)

    files = _write_tables(
        output,
        rank_rows=rank_rows,
        additive_rows=additive_rows,
        nonadditive_rows=nonadditive_rows,
        grouped_rows=grouped_rows,
        bs20_rows=bs20_rows,
    )
    plt, _ = _load_plotting()
    files.extend(_plot_stability(plt, output, rank_rows))
    files.extend(_plot_additive_bias(plt, output, additive_rows))
    files.extend(
        _plot_nonadditive_rmse(
            plt,
            output,
            nonadditive_rows,
        )
    )
    files.extend(_plot_estimand_gap(plt, output, nonadditive_rows))
    files.extend(_plot_grouped_blm(plt, output, grouped_rows))
    files.extend(_plot_bs20(plt, output, bs20_rows))

    metadata = {
        "config_fingerprint": config_fingerprint(result.config),
        "replications": result.config.replications,
        "record_count": len(result.records),
        "attempt_count": len(result.attempts),
        "generated_files": [
            str(path.relative_to(output)).replace("\\", "/")
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
