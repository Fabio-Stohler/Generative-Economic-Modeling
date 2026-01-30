"""Plotting utilities for surrogate diagnostics and errors."""

from .plots import (
    plot_naive_comparison,
    plot_surrogate_validation,
    plot_error_histogram,
    plot_error_histogram_three,
    plot_training_validation_loss,
    plot_euler_error_histogram,
)

__all__ = [
    "plot_naive_comparison",
    "plot_surrogate_validation",
    "plot_error_histogram",
    "plot_error_histogram_three",
    "plot_training_validation_loss",
    "plot_euler_error_histogram",
]
