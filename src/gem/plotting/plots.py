"""Plotting utilities used to visualize surrogate performance and errors."""

# Third-party dependencies
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator
import numpy as np
import torch


# %% Helper function to set up subplots
# %%
def filter_variables(y, y_pred=None, y_naive=None, y_labels=None, plot_vars=None):
    """
    Filters and validates variables based on plot_vars and y_labels.

    Args:
        y: Ground truth tensor.
        y_pred: Predicted tensor (optional).
        y_naive: Naive predictions tensor (optional).
        y_labels: List of variable labels.
        plot_vars: List of variable names to include.

    Returns:
        Filtered y, y_pred, y_naive, and y_labels.
    """
    if y_labels is None:
        raise ValueError("y_labels must be provided to map variables.")

    if plot_vars is not None:
        indices = [y_labels.index(label) for label in plot_vars if label in y_labels]
        if len(indices) != len(plot_vars):
            missing_vars = set(plot_vars) - set(y_labels)
            raise ValueError(
                f"The following variables in plot_vars are missing from y_labels: {missing_vars}"
            )
    else:
        indices = list(range(len(y_labels)))

    # Filter tensors and labels
    y = y[..., indices]
    y_labels = [y_labels[i] for i in indices]
    if y_pred is not None:
        y_pred = y_pred[..., indices]
    if y_naive is not None:
        y_naive = y_naive[..., indices]

    return y, y_pred, y_naive, y_labels


def setup_subplots(n_vars, ncol, sub_figsize=(10 / 3, 3)):
    """
    Sets up subplots for plotting.

    Args:
        n_vars: Number of variables to plot.
        ncol: Number of columns in the subplot grid.
        sub_figsize: Size of each subplot.

    Returns:
        fig, axes: Matplotlib figure and axes.
    """
    nrow = int(np.ceil(n_vars / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(ncol * sub_figsize[0], nrow * sub_figsize[1])
    )
    axes = np.atleast_1d(axes).flatten()

    # Hide empty subplots
    for i in range(n_vars, len(axes)):
        axes[i].set_visible(False)

    return fig, axes


# %% Check if the NN prediction is better than the naive prediction
def plot_naive_comparison(
    surrogate,
    x_validation=None,
    y_validation=None,
    y_labels=None,
    plot_vars=None,
    save_path=None,
    save_name="naive_full_model",
    suptitle=None,
    ncol=3,
):
    """Compare surrogate predictions to a naive baseline and plot MSE bars."""
    # Determine which data to use for evaluation.
    if x_validation is None:
        x = surrogate.data_validation["x"]
    else:
        x = x_validation

    if y_validation is None:
        y = surrogate.data_validation["y"]
    else:
        y = y_validation

    # Generate predictions and naive baseline.
    y_pred = surrogate.network(x).detach()
    y_naive = x[:, : y.size(-1)]

    # Filter and validate variables
    y, y_pred, y_naive, y_labels = filter_variables(y, y_pred, y_naive, y_labels, plot_vars)

    # Errors
    error_nn = [torch.mean((y[:, i] - y_pred[:, i]) ** 2) for i in range(y.size(-1))]
    error_naive = [torch.mean((y[:, i] - y_naive[:, i]) ** 2) for i in range(y.size(-1))]

    # Print errors
    for i, label in enumerate(y_labels):
        print(f"Variable {label}:")
        print(f"Error NN {label}: {error_nn[i]:.8f}")
        print(f"Error Naive {label}: {error_naive[i]:.8f}")
        print(f"Improvement, absolute {label}: {error_naive[i] - error_nn[i]:.8f}")
        print(f"Improvement, relative {label}: {(error_naive[i] - error_nn[i]) / error_naive[i]:.8f}")

    # Plot errors
    fig, axes = setup_subplots(len(error_nn), ncol)
    for i, ax in enumerate(axes[: len(error_nn)]):
        ax.bar(["Neural Network", "Naive"], [error_nn[i], error_naive[i]])
        ax.set_ylabel("MSE")
        ax.set_yscale("log")
        ax.grid(alpha=0.5)
        ax.set_title(y_labels[i])

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path + save_name)


# %% Check if the benchmark model is able to predict the data
def plot_surrogate_validation(
    surrogate_LKC,
    x_validation=None,
    y_validation=None,
    y_labels=None,
    plot_vars=None,
    save_path=None,
    save_name="validation_full_model",
    suptitle=None,
    ncol=3,
):
    """Scatter-plot predicted vs. actual values for validation data."""
    # Determine which data to use for evaluation.
    if x_validation is None:
        x = surrogate_LKC.data_validation["x"]
    else:
        x = x_validation

    if y_validation is None:
        y = surrogate_LKC.data_validation["y"]
    else:
        y = y_validation

    # Surrogate predictions.
    y_pred = surrogate_LKC.network(x).detach()

    # Filter and validate variables
    y, y_pred, _, y_labels = filter_variables(y, y_pred, None, y_labels, plot_vars)

    # Plot predictions
    fig, axes = setup_subplots(len(y_labels), ncol)
    extra_space = 0.01
    for i, ax in enumerate(axes[: len(y_labels)]):
        ax.scatter(y[:, i], y_pred[:, i], s=5, alpha=0.25)
        ax.set_ylim([y[:, i].min() - extra_space, y[:, i].max() + extra_space])
        ax.set_xlim([y[:, i].min() - extra_space, y[:, i].max() + extra_space])
        ax.plot(
            [y[:, i].min() - extra_space, y[:, i].max() + extra_space],
            [y[:, i].min() - extra_space, y[:, i].max() + extra_space],
            color="red",
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.grid(alpha=0.5)
        ax.set_title(y_labels[i])

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path + save_name)


# %% Plot error histograms for two surrogates
def plot_error_histogram(
    surrogate1,
    surrogate2,
    x_validation=None,
    y_validation=None,
    y_labels=None,
    plot_vars=None,
    save_path=None,
    save_name="error_histogram",
    suptitle=None,
    x_range=[-0.025, 0.025],
    ncol=3,
):
    """Plot relative error histograms for two surrogates on the same axes."""
    if x_validation is None:
        x = surrogate1.data_validation["x"]
    else:
        x = x_validation

    if y_validation is None:
        y = surrogate1.data_validation["y"]
    else:
        y = y_validation

    y_pred1 = surrogate1.network(x).detach()
    y_pred2 = surrogate2.network(x).detach()

    # Filter and validate variables
    y, y_pred1, y_pred2, y_labels = filter_variables(y, y_pred1, y_pred2, y_labels, plot_vars)

    # Relative errors
    error_nn1 = [(y[:, i] - y_pred1[:, i]) / y[:, i] for i in range(y.size(-1))]
    error_nn2 = [(y[:, i] - y_pred2[:, i]) / y[:, i] for i in range(y.size(-1))]

    # Plot histograms
    fig, axes = setup_subplots(len(y_labels), ncol)
    for i, ax in enumerate(axes[: len(y_labels)]):
        ax.hist(error_nn1[i].numpy(), bins=100, alpha=0.75, label="Surrogate", color="C0", range=x_range)
        ax.hist(error_nn2[i].numpy(), bins=100, alpha=0.75, label="Baseline", color="C1", range=x_range)
        ax.grid(alpha=0.5)
        ax.set_xlabel("Relative error")
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        # x-axis in percent with one decimal place
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.set_title(y_labels[i])

    axes[0].legend()
    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path + save_name)


# %% plotting the training loss
def plot_training_validation_loss(loss_dict, label, smoothing_window=100, ylim=None, selected_keys=None, save_path=None, save_name=None, suffix="", locators=True):
    """Plot smoothed training and validation losses over epochs."""
    iterations = loss_dict["iteration"]

    if selected_keys is None:
        selected_keys = [k for k in loss_dict.keys() if k != "iteration"]

    smoothed_losses = {loss_name: np.convolve(loss_dict[loss_name], np.ones(smoothing_window), "valid") / smoothing_window for loss_name in selected_keys}
    smoothed_iterations = iterations[smoothing_window - 1 :]

    # colors
    colors = ["#0072BD", "#D95319"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for i, (loss_name, smoothed_loss) in enumerate(smoothed_losses.items()):
        axes[i].plot(smoothed_iterations, smoothed_loss, label=loss_name.replace("_", " ").capitalize(), color=colors[i])
        axes[i].set_yscale("log")

        # introduce six ticks for each dataseries on the y-axis
        if locators == True:
            axes[i].yaxis.set_major_locator(plt.MaxNLocator(8))
            axes[i].yaxis.set_minor_locator(AutoMinorLocator(2))

        # set labels
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Mean Squared Error")
        if ylim:
            axes[i].set_ylim(ylim)

        if locators == True:
            axes[i].grid(True, alpha=0.5, which="both")
        else:
            axes[i].grid(True, alpha=0.5)
        axes[i].legend()

    if save_path is not None and save_name is not None:
        full_path = f"{save_path}{save_name}"
        if suffix:
            full_path = f"{full_path}{suffix}"
        plt.tight_layout()
        plt.savefig(full_path)

    return fig, axes


# %% Plot error histograms for three surrogates
def plot_error_histogram_three(
    surrogate1,
    surrogate2,
    surrogate3,
    x_validation=None,
    y_validation=None,
    y_labels=None,
    plot_vars=None,
    save_path=None,
    save_name="error_histogram",
    suptitle=None,
    x_range=[-0.025, 0.025],
    Latex_label=None,
    maxpercent=1.0,
    percentsteps=0.2,
    labels=["Incomplete", "Complete", "Generative"],
    ncol=3,
    **kwargs,
):
    """Plot relative error histograms for three surrogates."""
    if x_validation is None:
        x = surrogate1.data_validation["x"]
    else:
        x = x_validation

    if y_validation is None:
        y = surrogate1.data_validation["y"]
    else:
        y = y_validation

    y_pred1 = surrogate1.network(x).detach()
    y_pred2 = surrogate2.network(x).detach()
    y_pred3 = surrogate3.network(x).detach()

    # Filter and validate variables
    _, y_pred1, _, _ = filter_variables(y, y_pred1, y_pred2, y_labels, plot_vars)
    y, y_pred2, y_pred3, y_labels = filter_variables(y, y_pred2, y_pred3, y_labels, plot_vars)

    # Relative errors
    error_nn1 = [(y[:, i] - y_pred1[:, i]) / y[:, i] for i in range(y.size(-1))]
    error_nn2 = [(y[:, i] - y_pred2[:, i]) / y[:, i] for i in range(y.size(-1))]
    error_nn3 = [(y[:, i] - y_pred3[:, i]) / y[:, i] for i in range(y.size(-1))]

    # Plot histograms
    fig, axes = setup_subplots(len(y_labels), ncol, **kwargs)
    for i, ax in enumerate(axes[: len(y_labels)]):
        # Plot all three histograms
        bars1 = ax.hist(error_nn1[i].numpy(), bins=50, alpha=0.75, color="#5A7F20", edgecolor="black", range=x_range, label=labels[0])  # Incomplete
        bars2 = ax.hist(error_nn2[i].numpy(), bins=50, alpha=0.75, color="#D95319", edgecolor="black", range=x_range, label=labels[1])  # Complete
        bars3 = ax.hist(error_nn3[i].numpy(), bins=50, alpha=0.75, color="#0072BD", edgecolor="black", range=x_range, label=labels[2])  # Generative

        ax.grid(alpha=0.5)
        ax.set_xlabel("Relative error")
        ax.set_ylabel("Percentage")
        ax.set_yticks(np.arange(0, maxpercent, percentsteps) * len(y_pred1))
        ax.set_yticklabels([f"{int(y*100)}%" for y in np.arange(0, maxpercent, percentsteps)])
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1%}"))
        ax.set_title(y_labels[i] if Latex_label is None else Latex_label[i])

        # Convert y-axis to percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/len(y_pred1)*100:.0f}%"))

    # Custom legend: Generative, Complete, Incomplete
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#0072BD", ec="black", label="Generative"),
        plt.Rectangle((0, 0), 1, 1, fc="#D95319", ec="black", label="Complete"),
        plt.Rectangle((0, 0), 1, 1, fc="#5A7F20", ec="black", label="Incomplete"),
    ]
    ax.legend(handles=handles)

    if suptitle:
        fig.suptitle(suptitle)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path + save_name)


def plot_euler_error_histogram(
    euler_errors_complete,
    euler_errors_generative,
    save_path,
    save_name,
    number_bins=50,
    num_locators_x=10,
    num_locators_y=10,
):
    """Plot histogram of Euler equation errors for two models (log scale)."""
    # colors to use for plotting
    colors = ["#D95319", "#0072BD"]

    # the log-data
    log_complete = np.log10(euler_errors_complete)
    log_generative = np.log10(euler_errors_generative)

    # Determine the overall data range
    min_val = min(log_complete.min(), log_generative.min())
    max_val = max(log_complete.max(), log_generative.max())

    # create common bins
    common_bins = np.linspace(min_val, max_val, number_bins + 1)

    # Plot histograms
    plt.figure(figsize=(6, 4))
    plt.hist(log_complete, bins=common_bins, alpha=0.7, color=colors[0], edgecolor="black", label="Complete Model")
    plt.hist(log_generative, bins=common_bins, alpha=0.7, color=colors[1], edgecolor="black", label="Generative Model")
    plt.title("Histogram of the Euler Equation Error")
    plt.xlabel("Logarithm of Euler Equation Error")
    plt.ylabel("Density")
    plt.grid(alpha=0.5)
    # plt.gca().set_yticks(np.arange(0, maxpercent, percentsteps) * len(euler_errors_complete))
    # plt.gca().set_yticklabels([f"{int(y*100)}%" for y in np.arange(0, maxpercent, percentsteps)])
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(num_locators_x))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}"))

    # Convert y-axis to percentage
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(num_locators_y))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y/len(euler_errors_complete)*100:.0f}%"))

    # Custom legend: Generative, Complete, Incomplete
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="#0072BD", ec="black", label="Generative"),
        plt.Rectangle((0, 0), 1, 1, fc="#D95319", ec="black", label="Complete"),
    ]
    plt.legend(handles=handles)

    plt.tight_layout()
    plt.savefig(f"{save_path}/{save_name}.png", dpi=300)
    plt.show()
