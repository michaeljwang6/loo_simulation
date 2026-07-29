"""Simulation tools for the LOO worker--firm wage project."""

from .dgp import (
    GroupedPopulationDGP,
    PopulationDGP,
    generate_grouped_population,
    generate_population,
)
from .low_rank import (
    BICRankSelectionResult,
    LowRankAnalysisSample,
    LowRankPluginResult,
    fit_low_rank_plugin,
    select_low_rank_bic,
)
from .monte_carlo import (
    EstimatorAttempt,
    EstimatorAttemptSummary,
    EstimatorConfig,
    MonteCarloConfig,
    MonteCarloRecord,
    MonteCarloResult,
    MonteCarloSummary,
    ScenarioConfig,
    config_from_dict,
    config_to_dict,
    default_dgp_ladder,
    load_monte_carlo_config,
    run_monte_carlo,
    save_monte_carlo_results,
)
from .panel import PanelData, sample_panel
from .targets import (
    AKMPopulationTarget,
    BLMGroupedPopulationTarget,
    ProcedureTargets,
    compute_akm_population_target,
    compute_blm_grouped_target,
    compute_procedure_targets,
)
from .truth import PopulationTruth, compute_population_truth

__all__ = [
    "AKMPopulationTarget",
    "BICRankSelectionResult",
    "BLMGroupedPopulationTarget",
    "EstimatorAttempt",
    "EstimatorAttemptSummary",
    "EstimatorConfig",
    "GroupedPopulationDGP",
    "LowRankAnalysisSample",
    "LowRankPluginResult",
    "MonteCarloConfig",
    "MonteCarloRecord",
    "MonteCarloResult",
    "MonteCarloSummary",
    "PanelData",
    "PopulationDGP",
    "PopulationTruth",
    "ProcedureTargets",
    "ScenarioConfig",
    "compute_akm_population_target",
    "compute_blm_grouped_target",
    "compute_population_truth",
    "compute_procedure_targets",
    "config_from_dict",
    "config_to_dict",
    "default_dgp_ladder",
    "fit_low_rank_plugin",
    "generate_grouped_population",
    "generate_population",
    "load_monte_carlo_config",
    "run_monte_carlo",
    "sample_panel",
    "save_monte_carlo_results",
    "select_low_rank_bic",
]
