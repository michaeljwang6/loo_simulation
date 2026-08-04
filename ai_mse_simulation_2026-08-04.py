"""Monte Carlo check of the held-out empirical-Bayes MSE expansion.

Model (proposal notation):
    theta_j ~ N(mu, sigma_theta^2)
    S_j = a + beta * theta_j + nu_j
    theta_hat_j^v = theta_j + eta_j

The program estimates the shared parameters from m randomized validation
cells, predicts an unvalidated cell from its score, and compares its MSE with
the oracle floor omega^2 = Var(theta_j | S_j).

Only NumPy and Pillow are required.  The default run begins with 2,000 Monte
Carlo repetitions and increases the count only if the relative Monte Carlo
standard error of the excess-risk estimates exceeds three percent.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


M_GRID = np.array([10, 20, 40, 80, 160, 320, 640], dtype=int)
RATE_GRID_MIN = 40
EPSILON = 1.0e-10


@dataclass(frozen=True)
class Parameters:
    mu: float = 30.0
    sigma_theta: float = 10.0
    beta: float = 0.4
    rho: float = 0.44
    validation_sd: float = 3.0

    @property
    def theta_var(self) -> float:
        return self.sigma_theta**2

    @property
    def a(self) -> float:
        # Centers the raw score at the true mean while compressing deviations.
        return (1.0 - self.beta) * self.mu

    @property
    def score_noise_var(self) -> float:
        signal_var = self.beta**2 * self.theta_var
        return signal_var * (self.rho ** (-2.0) - 1.0)

    @property
    def validation_var(self) -> float:
        return self.validation_sd**2

    @property
    def score_mean(self) -> float:
        return self.a + self.beta * self.mu

    @property
    def score_var(self) -> float:
        return self.beta**2 * self.theta_var + self.score_noise_var

    @property
    def oracle_slope(self) -> float:
        return self.beta * self.theta_var / self.score_var

    @property
    def omega2(self) -> float:
        return self.theta_var * self.score_noise_var / self.score_var


def _sample_variance(centered: np.ndarray, m: int) -> np.ndarray:
    return np.sum(centered * centered, axis=1) / (m - 1)


def simulate_scenario(
    params: Parameters,
    reps: int,
    seed: int,
    finite_atlas: bool,
    atlas_cells: int = 1000,
) -> list[dict[str, float]]:
    """Simulate nested randomized anchors and return summaries for every m."""

    rng = np.random.default_rng(seed)
    # Keep the array shape fixed across sensitivity scenarios so the same seed
    # supplies common random numbers and differences are not simulation noise.
    columns = atlas_cells

    theta = rng.normal(
        params.mu, params.sigma_theta, size=(reps, columns)
    )
    score = (
        params.a
        + params.beta * theta
        + rng.normal(
            0.0, math.sqrt(params.score_noise_var), size=(reps, columns)
        )
    )
    validation = theta[:, : M_GRID.max()] + rng.normal(
        0.0,
        params.validation_sd,
        size=(reps, int(M_GRID.max())),
    )

    rows: list[dict[str, float]] = []

    for m in M_GRID:
        score_v = score[:, :m]
        truth_v = validation[:, :m]

        score_bar = np.mean(score_v, axis=1)
        truth_bar = np.mean(truth_v, axis=1)
        score_centered = score_v - score_bar[:, None]
        truth_centered = truth_v - truth_bar[:, None]

        score_sample_var = _sample_variance(score_centered, int(m))
        validation_sample_var = _sample_variance(truth_centered, int(m))
        sample_cov = np.sum(
            score_centered * truth_centered, axis=1
        ) / (m - 1)

        theta_var_raw = validation_sample_var - params.validation_var
        theta_var_hat = np.maximum(theta_var_raw, EPSILON)
        beta_raw = sample_cov / theta_var_hat
        beta_hat = np.clip(beta_raw, 0.0, 1.0)
        score_noise_raw = score_sample_var - beta_hat**2 * theta_var_hat
        score_noise_hat = np.maximum(score_noise_raw, EPSILON)

        constrained = (
            (theta_var_raw <= 0.0)
            | (beta_raw <= 0.0)
            | (beta_raw >= 1.0)
            | (score_noise_raw <= 0.0)
        )

        fitted_score_var = beta_hat**2 * theta_var_hat + score_noise_hat
        fitted_slope = beta_hat * theta_var_hat / fitted_score_var

        # Exact risk over a new held-out cell, conditional on each fitted rule.
        # It removes test-sample noise without replacing the simulated anchor.
        mean_error = (
            truth_bar
            + fitted_slope * (params.score_mean - score_bar)
            - params.mu
        )
        slope_error = fitted_slope - params.oracle_slope
        excess_risk = mean_error**2 + slope_error**2 * params.score_var
        total_risk = params.omega2 + excess_risk

        finite_mse_mean = float("nan")
        finite_mse_mcse = float("nan")
        if finite_atlas:
            heldout_theta = theta[:, m:atlas_cells]
            heldout_score = score[:, m:atlas_cells]
            prediction = truth_bar[:, None] + fitted_slope[:, None] * (
                heldout_score - score_bar[:, None]
            )
            realized_mse = np.mean(
                (prediction - heldout_theta) ** 2, axis=1
            )
            finite_mse_mean = float(np.mean(realized_mse))
            finite_mse_mcse = float(
                np.std(realized_mse, ddof=1) / math.sqrt(reps)
            )

        excess_mean = float(np.mean(excess_risk))
        excess_mcse = float(
            np.std(excess_risk, ddof=1) / math.sqrt(reps)
        )
        total_mean = float(np.mean(total_risk))

        rows.append(
            {
                "m": int(m),
                "repetitions": reps,
                "omega2": params.omega2,
                "omega_rmse": math.sqrt(params.omega2),
                "excess_mse": excess_mean,
                "excess_mse_mcse": excess_mcse,
                "total_mse": total_mean,
                "total_rmse": math.sqrt(total_mean),
                "floor_share": params.omega2 / total_mean,
                "excess_to_floor": excess_mean / params.omega2,
                "finite_atlas_mse": finite_mse_mean,
                "finite_atlas_mse_mcse": finite_mse_mcse,
                "constraint_rate": float(np.mean(constrained)),
            }
        )

        del score_v, truth_v, score_centered, truth_centered

    return rows


def rate_diagnostics(rows: list[dict[str, float]]) -> dict[str, float]:
    rate_rows = [row for row in rows if row["m"] >= RATE_GRID_MIN]
    m = np.array([row["m"] for row in rate_rows], dtype=float)
    excess = np.array([row["excess_mse"] for row in rate_rows])
    slope, intercept = np.polyfit(np.log(m), np.log(excess), 1)
    inverse_m = 1.0 / m
    lambda_hat = float(
        np.sum(inverse_m * excess) / np.sum(inverse_m * inverse_m)
    )
    relative_mcse = np.array(
        [row["excess_mse_mcse"] / row["excess_mse"] for row in rate_rows]
    )
    return {
        "log_log_slope": float(slope),
        "log_log_intercept": float(intercept),
        "lambda_hat": lambda_hat,
        "max_relative_mcse": float(relative_mcse.max()),
    }


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\segoeuib.ttf", "DejaVuSans-Bold.ttf"]
        if bold
        else [r"C:\Windows\Fonts\segoeui.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _map_linear(value: float, low: float, high: float, start: int, end: int) -> int:
    return int(round(start + (value - low) * (end - start) / (high - low)))


def _panel_axes(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    y_label: str,
    y_ticks: list[float],
    y_format,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    plot_left, plot_top = left + 72, top + 44
    plot_right, plot_bottom = right - 22, bottom - 54
    draw.text((left + 8, top + 4), title, fill="#172033", font=_font(22, True))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#536174", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#536174", width=2)
    y_min, y_max = min(y_ticks), max(y_ticks)
    for tick in y_ticks:
        y = _map_linear(tick, y_min, y_max, plot_bottom, plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#dde3ea", width=1)
        label = y_format(tick)
        bbox = draw.textbbox((0, 0), label, font=_font(14))
        draw.text(
            (plot_left - 9 - (bbox[2] - bbox[0]), y - 8),
            label,
            fill="#4c596b",
            font=_font(14),
        )
    draw.text((left + 4, plot_top - 26), y_label, fill="#4c596b", font=_font(14))
    return plot_left, plot_top, plot_right, plot_bottom


def write_primary_plot(
    path: Path, rows: list[dict[str, float]], diagnostics: dict[str, float]
) -> None:
    width, height = 1600, 560
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text(
        (28, 12),
        "Held-out MSE as the randomized validation anchor grows",
        fill="#101827",
        font=_font(28, True),
    )
    draw.text(
        (28, 48),
        "Proposal calibration: corr(score, truth) = 0.44; beta = 0.4; validation SE = 3 pp",
        fill="#526173",
        font=_font(16),
    )

    panels = [(20, 82, 530, 540), (545, 82, 1055, 540), (1070, 82, 1580, 540)]
    m = np.array([row["m"] for row in rows], dtype=float)
    log_m = np.log2(m)
    x_min, x_max = float(log_m.min()), float(log_m.max())
    x_labels = [10, 20, 40, 80, 160, 320, 640]
    navy, orange, green = "#2355a5", "#c45a16", "#2a7a55"

    # Panel 1: total risk and its floor.
    p = _panel_axes(
        draw,
        panels[0],
        "Total MSE approaches the floor",
        "MSE (pp squared)",
        [75, 80, 85, 90, 95, 100, 105],
        lambda value: f"{value:.0f}",
    )
    pl, pt, pr, pb = p
    for value in x_labels:
        x = _map_linear(math.log2(value), x_min, x_max, pl, pr)
        draw.text((x - 12, pb + 10), str(value), fill="#4c596b", font=_font(13))
    floor_y = _map_linear(rows[0]["omega2"], 75, 105, pb, pt)
    draw.line((pl, floor_y, pr, floor_y), fill=orange, width=3)
    draw.text((pr - 118, floor_y - 25), "omega^2 = 80.64", fill=orange, font=_font(14, True))
    points = []
    for row in rows:
        x = _map_linear(math.log2(row["m"]), x_min, x_max, pl, pr)
        y = _map_linear(row["total_mse"], 75, 105, pb, pt)
        points.append((x, y))
    draw.line(points, fill=navy, width=4)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=navy)
    draw.text((pl + 4, pb + 34), "validated cells m", fill="#4c596b", font=_font(14))

    # Panel 2: excess risk on log-log axes.
    log_excess = np.log10([row["excess_mse"] for row in rows])
    y_low, y_high = -0.7, 1.5
    p = _panel_axes(
        draw,
        panels[1],
        "The shrinking term is proportional to 1/m",
        "Excess MSE (log scale)",
        [y_low, 0.0, 0.7, y_high],
        lambda value: f"{10**value:.1f}",
    )
    pl, pt, pr, pb = p
    for value in x_labels:
        x = _map_linear(math.log2(value), x_min, x_max, pl, pr)
        draw.text((x - 12, pb + 10), str(value), fill="#4c596b", font=_font(13))
    points = []
    fit_points = []
    for row, log_y in zip(rows, log_excess):
        x = _map_linear(math.log2(row["m"]), x_min, x_max, pl, pr)
        y = _map_linear(float(log_y), y_low, y_high, pb, pt)
        points.append((x, y))
        fitted = diagnostics["lambda_hat"] / row["m"]
        fy = _map_linear(math.log10(fitted), y_low, y_high, pb, pt)
        fit_points.append((x, fy))
    draw.line(fit_points, fill=orange, width=3)
    draw.line(points, fill=navy, width=4)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=navy)
    draw.text(
        (pl + 8, pt + 7),
        f"fitted log-log slope = {diagnostics['log_log_slope']:.3f}",
        fill="#172033",
        font=_font(14, True),
    )
    draw.text((pl + 4, pb + 34), "validated cells m", fill="#4c596b", font=_font(14))

    # Panel 3: share of total risk due to the floor.
    p = _panel_axes(
        draw,
        panels[2],
        "The floor already dominates at m = 40",
        "Floor share of MSE",
        [75, 80, 85, 90, 95, 100],
        lambda value: f"{value:.0f}%",
    )
    pl, pt, pr, pb = p
    for value in x_labels:
        x = _map_linear(math.log2(value), x_min, x_max, pl, pr)
        draw.text((x - 12, pb + 10), str(value), fill="#4c596b", font=_font(13))
    points = []
    for row in rows:
        x = _map_linear(math.log2(row["m"]), x_min, x_max, pl, pr)
        y = _map_linear(100.0 * row["floor_share"], 75, 100, pb, pt)
        points.append((x, y))
    draw.line(points, fill=green, width=4)
    for x, y in points:
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=green)
    row_40 = next(row for row in rows if row["m"] == 40)
    x_40 = _map_linear(math.log2(40), x_min, x_max, pl, pr)
    y_40 = _map_linear(100.0 * row_40["floor_share"], 75, 100, pb, pt)
    draw.text(
        (x_40 + 10, y_40 - 25),
        f"{100.0 * row_40['floor_share']:.1f}%",
        fill=green,
        font=_font(15, True),
    )
    draw.text((pl + 4, pb + 34), "validated cells m", fill="#4c596b", font=_font(14))

    image.save(path)


def write_summary(
    path: Path,
    params: Parameters,
    rows: list[dict[str, float]],
    diagnostics: dict[str, float],
    sensitivity: list[dict[str, float]],
    elapsed: float,
) -> None:
    row_40 = next(row for row in rows if row["m"] == 40)
    row_320 = next(row for row in rows if row["m"] == 320)
    finite_gap = max(
        abs(row["finite_atlas_mse"] - row["total_mse"])
        for row in rows
    )
    ten_percent_m = math.ceil(
        diagnostics["lambda_hat"] / (0.10 * params.omega2)
    )
    scaled_excess = ", ".join(
        f"{int(row['m'])}: {row['m'] * row['excess_mse']:.1f}"
        for row in rows
        if row["m"] >= RATE_GRID_MIN
    )
    lines = [
        "# Simulation of the MSE floor",
        "",
        "## Calibration",
        "",
        f"- mu = {params.mu:.0f} percentage points",
        f"- sigma_theta = {params.sigma_theta:.0f} percentage points",
        f"- beta = {params.beta:.1f}",
        f"- corr(S_j, theta_j) = {params.rho:.2f}",
        f"- validation standard error = {params.validation_sd:.0f} percentage points",
        f"- omega^2 = {params.omega2:.2f} squared percentage points",
        f"- floor RMSE = {math.sqrt(params.omega2):.2f} percentage points",
        "",
        "## Main findings",
        "",
        f"The fitted log-log slope of excess MSE on m is {diagnostics['log_log_slope']:.3f}. "
        "The theoretical 1/m rate corresponds to -1.",
        "",
        f"The values of m times excess MSE are {scaled_excess}. Their stability "
        "is the direct numerical check of the 1/m approximation.",
        "",
        f"At m = 40, total MSE is {row_40['total_mse']:.2f}. The floor is "
        f"{100.0 * row_40['floor_share']:.1f}% of that total, and the shrinking "
        f"term is {row_40['excess_mse']:.2f}, or "
        f"{100.0 * row_40['excess_to_floor']:.1f}% of the floor.",
        "",
        f"At m = 320, the shrinking term is {row_320['excess_mse']:.3f}, while "
        f"omega^2 remains {params.omega2:.2f}. The floor is "
        f"{100.0 * row_320['floor_share']:.1f}% of total MSE.",
        "",
        f"The fitted Lambda/m approximation puts the point at which the shrinking "
        f"term is below 10% of the floor at approximately m = {ten_percent_m}.",
        "",
        f"The largest relative Monte Carlo standard error among m >= 40 is "
        f"{100.0 * diagnostics['max_relative_mcse']:.2f}%. The largest difference "
        f"between analytically integrated held-out risk and the simulated "
        f"J = 1,000 atlas risk is {finite_gap:.3f} MSE units.",
        "",
        "## Sensitivity at m = 40",
        "",
        "| Score correlation | Validation SE | omega^2 | Excess MSE | Floor share |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| {row['rho']:.2f} | {row['validation_sd']:.0f} | "
            f"{row['omega2']:.2f} | {row['excess_mse']:.2f} | "
            f"{100.0 * row['floor_share']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "Changing validation precision changes the shrinking term but not the "
            "oracle floor. Changing score quality changes the floor itself.",
            "",
            "## Computation",
            "",
            f"The run used {rows[0]['repetitions']:,} Monte Carlo repetitions and "
            f"finished locally in {elapsed:.1f} seconds. The structural moment "
            "estimator is projected onto sigma_theta^2 >= 0, 0 <= beta <= 1, "
            "and sigma_S^2 >= 0. The projection rate is reported in the CSV; it "
            "is relevant mainly at the smallest validation budgets and vanishes "
            "over the asymptotic range used for the rate check.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-reps", type=int, default=2000)
    parser.add_argument("--max-reps", type=int, default=10000)
    parser.add_argument("--target-relative-mcse", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    params = Parameters()
    reps = args.initial_reps

    while True:
        primary_rows = simulate_scenario(
            params, reps, args.seed, finite_atlas=True
        )
        diagnostics = rate_diagnostics(primary_rows)
        if (
            diagnostics["max_relative_mcse"] <= args.target_relative_mcse
            or reps >= args.max_reps
        ):
            break
        reps = min(2 * reps, args.max_reps)

    scenario_specs = [
        (0.30, 3.0),
        (0.44, 3.0),
        (0.70, 3.0),
        (0.44, 1.0),
        (0.44, 6.0),
    ]
    sensitivity_rows: list[dict[str, float]] = []
    all_sensitivity_rows: list[dict[str, float]] = []
    for rho, validation_sd in scenario_specs:
        scenario = Parameters(rho=rho, validation_sd=validation_sd)
        scenario_rows = simulate_scenario(
            scenario,
            reps,
            args.seed,
            finite_atlas=False,
        )
        for row in scenario_rows:
            all_sensitivity_rows.append(
                {"rho": rho, "validation_sd": validation_sd, **row}
            )
        row_40 = next(row for row in scenario_rows if row["m"] == 40)
        sensitivity_rows.append(
            {"rho": rho, "validation_sd": validation_sd, **row_40}
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_csv = output_dir / "ai_mse_simulation_results_2026-08-04.csv"
    sensitivity_csv = output_dir / "ai_mse_simulation_sensitivity_2026-08-04.csv"
    plot_path = output_dir / "ai_mse_simulation_primary_2026-08-04.png"
    summary_path = output_dir / "ai_mse_simulation_summary_2026-08-04.md"

    write_csv(primary_csv, primary_rows)
    write_csv(sensitivity_csv, all_sensitivity_rows)
    write_primary_plot(plot_path, primary_rows, diagnostics)
    elapsed = time.perf_counter() - started
    write_summary(
        summary_path,
        params,
        primary_rows,
        diagnostics,
        sensitivity_rows,
        elapsed,
    )

    print(f"repetitions={reps}")
    print(f"max_relative_mcse={diagnostics['max_relative_mcse']:.6f}")
    print(f"log_log_slope={diagnostics['log_log_slope']:.6f}")
    print(f"omega2={params.omega2:.6f}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(summary_path)


if __name__ == "__main__":
    main()
