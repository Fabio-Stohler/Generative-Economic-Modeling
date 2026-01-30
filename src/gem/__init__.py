"""GEM package: Brock–Mirman surrogates, data utilities, and plotting."""

from .model import BMModel
from .surrogates import Surrogate
from .data import (
    ar1_lognormal_draws,
    calculate_EEE_BM,
    calculate_EEE_BM_Ana,
    construct_single_xy,
    simulate_and_save,
    simulate_BM,
    standardized_moments,
    tensor_to_dataframe,
    load_pickle,
    save_pickle,
)
from .core import Parameters, Ranges, State, Shocks
from .plotting import plots
from .data import helpers, io_utils
from .plotting import (
    plot_naive_comparison,
    plot_surrogate_validation,
    plot_error_histogram,
    plot_error_histogram_three,
    plot_training_validation_loss,
    plot_euler_error_histogram,
)

__all__ = [
    "BMModel",
    "Surrogate",
    "Parameters",
    "Ranges",
    "State",
    "Shocks",
    "ar1_lognormal_draws",
    "calculate_EEE_BM",
    "calculate_EEE_BM_Ana",
    "construct_single_xy",
    "simulate_and_save",
    "simulate_BM",
    "standardized_moments",
    "tensor_to_dataframe",
    "load_pickle",
    "save_pickle",
    "plot_naive_comparison",
    "plot_surrogate_validation",
    "plot_error_histogram",
    "plot_error_histogram_three",
    "plot_training_validation_loss",
    "plot_euler_error_histogram",
    "plots",
    "helpers",
    "io_utils",
]
