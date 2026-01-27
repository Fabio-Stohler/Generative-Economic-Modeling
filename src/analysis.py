# %%
import torch
import math
from pathlib import Path
from copy import deepcopy
import pickle
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, NullLocator
import numpy as np
import pandas as pd
from tqdm import trange
import scipy.stats

# Helper functions
from helpers import ergodic_sigma

# Surrogate
from surrogate import Surrogate

# Plotting functions
import importlib
import plots

importlib.reload(plots)
from plots import plot_surrogate_validation, plot_naive_comparison, plot_error_histogram

# %% Check if CUDA is available
Force_CPU = True

device = "cuda" if torch.cuda.is_available() else "cpu"
if Force_CPU:
    device = "cpu"
    print("Forcing CPU usage.")

# %% Constants and lists
labels_outputs = ["K", "C", "L"]  # this has to correspond to output vector
plot_outputs = ["K", "C", "L"]  # Exclude "δ"
plot_ncol = 3

# retrain the full surrogate model
retrain = False

# Option to rerun all experiments or only load and analyze the models
rerun = False

# Truncation length for dataset
truncate_length = 10000  # Set to None to use the full dataset

# %%
# Fix seed for torch and numpy
torch.manual_seed(42)
np.random.seed(42)

# Structures to hold the elements of the model
from structures import Parameters, Ranges, Shocks, State

# Helper functions
from helpers import ergodic_sigma

# Surrogate
from surrogate import Surrogate

# BM Model
from BM import BMModel

# Custom distribution for state variable experiments
from distributions import CustomCategorical
import pandas as pd

# %% Matplotlib settings
# Increase the font size
plt.rcParams.update({"font.size": 13})


# %% Functions for all analysis steps
# function to simulate the model for different parameter sets and save the data
def simulate_and_save(
    par,
    pars,
    distr_ranges,
    model_save_path,
    model_names_list,
    draws=100,
    steps=120,
    burn=0,
    labnorm=True,
):

    # For entries in distr_ranges create a model with the name from model_names_list, simulate the data, save the data
    for i, distr_range in enumerate(distr_ranges):
        model = BMModel(par, pars, distr_range)

        K = 25
        for j in range(K):
            model.simulate_dataset(draw=draws, steps=steps, burn=burn, labnorm=labnorm)
            # Check for NaN values in the dataset
            if (
                torch.isnan(model.dataset["x"]).any()
                or torch.isnan(model.dataset["y"]).any()
            ):
                print(f"NaN values in the dataset for model {model_names_list[i]}")
            else:
                print(f"No NaN values in the dataset for model {model_names_list[i]}")
                break

        if j == K - 1:
            raise ValueError(
                f"Could not simulate the dataset for model {model_names_list[i]}"
            )

        # Save the model
        model.save(model_save_path, model_names_list[i])

    return None


# simulate data
# Settings for the experiment in general
experiment_settings = {
    "draws": 1,
    "steps": 10001,
    "burn": 0,
    "labnorm": True,
    "normalize_input": True,
    "scale_output": False,
    "model_save_path": "../bld/data/BM/",
    "fig_save_path": "../bld/figures/BM/",
    "nn_save_path": "../bld/models/BM/",
}

# Specify the parameters, ranges and distributions for the experiment
# Parameters
par = {
    "alpha": 0.33,
    "beta": 0.96,
    "omega": 1.0,
    "gamma": 5.0,
    "rho_a": 0.9,
    "rho_z": 0.9,
    "rho_mu": 0.0,
    "sigma_a": 0.05,
    "sigma_z": 0.05,
    "sigma_mu": 0.05,
    "Abar": 1.0,
    "Zbar": 1.0,
    "mubar": 1.0,
    "tau_L": 0.0,
    "tau_K": 0.0,
    "tau_C": 0.0,
    "kss": 1.0,
    "tau_switch": 0.0,
}

# distr_ranges for the simulation
# Distributions
distr_F = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}
distr_AB = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0e-14),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}
distr_AC = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0e-14),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}
distr_BC = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0e-14),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}
distr_A = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0e-14),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0e-14),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}
distr_B = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0e-14),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0e-14),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}
distr_C = {
    "epsilon_a": torch.distributions.Normal(0.0, 1.0e-14),
    "epsilon_z": torch.distributions.Normal(0.0, 1.0e-14),
    "epsilon_mu": torch.distributions.Normal(0.0, 1.0),
    "tau": torch.distributions.Uniform(0.0, 1.0),
}

distr_ranges = [distr_F, distr_AB, distr_AC, distr_BC, distr_A, distr_B, distr_C]

# simulate the models
if rerun:
    simulate_and_save(
        par,
        {},
        distr_ranges,
        experiment_settings["model_save_path"],
        ["F", "AB", "AC", "BC", "A", "B", "C"],
        draws=experiment_settings["draws"],
        steps=experiment_settings["steps"],
        burn=experiment_settings["burn"],
        labnorm=experiment_settings["labnorm"],
    )

# load the data into a tensor
data_all = BMModel.load(experiment_settings["model_save_path"] + "F.pkl").dataset
data_AB = BMModel.load(experiment_settings["model_save_path"] + "AB.pkl").dataset
data_AC = BMModel.load(experiment_settings["model_save_path"] + "AC.pkl").dataset
data_BC = BMModel.load(experiment_settings["model_save_path"] + "BC.pkl").dataset
data_A = BMModel.load(experiment_settings["model_save_path"] + "A.pkl").dataset
data_B = BMModel.load(experiment_settings["model_save_path"] + "B.pkl").dataset
data_C = BMModel.load(experiment_settings["model_save_path"] + "C.pkl").dataset


#  %% Join the first and second dimension for "x" and "y" in each dataset
for data in [data_all, data_AB, data_AC, data_BC, data_A, data_B, data_C]:
    for key in ["x", "y"]:
        if key in data and data[key] is not None and data[key].ndim >= 2:
            shape = data[key].shape
            data[key] = (
                data[key].reshape(-1, *shape[2:])
                if data[key].ndim > 2
                else data[key].reshape(-1, shape[-1])
            )


# save tensors as dataframes
def tensor_to_dataframe(tensor, columns):
    df = pd.DataFrame(tensor.numpy(), columns=columns)
    return df


# extract the first four variables of x, and the last two variables of y, and drop the last row
data_all_df = pd.concat(
    [
        tensor_to_dataframe(data_all["x"][:, :4], ["K", "A", "Z", "mu"]),
        tensor_to_dataframe(data_all["y"][:, -2:], ["C", "L"]),
    ],
    axis=1,
).iloc[:-1]
data_AB_df = pd.concat(
    [
        tensor_to_dataframe(data_AB["x"][:, :4], ["K", "A", "Z", "mu"]),
        tensor_to_dataframe(data_AB["y"][:, -2:], ["C", "L"]),
    ],
    axis=1,
).iloc[:-1]
data_AC_df = pd.concat(
    [
        tensor_to_dataframe(data_AC["x"][:, :4], ["K", "A", "Z", "mu"]),
        tensor_to_dataframe(data_AC["y"][:, -2:], ["C", "L"]),
    ],
    axis=1,
).iloc[:-1]
data_BC_df = pd.concat(
    [
        tensor_to_dataframe(data_BC["x"][:, :4], ["K", "A", "Z", "mu"]),
        tensor_to_dataframe(data_BC["y"][:, -2:], ["C", "L"]),
    ],
    axis=1,
).iloc[:-1]
data_C_df = pd.concat(
    [
        tensor_to_dataframe(data_C["x"][:, :4], ["K", "A", "Z", "mu"]),
        tensor_to_dataframe(data_C["y"][:, -2:], ["C", "L"]),
    ],
    axis=1,
).iloc[:-1]


# %%
# for each entry in exo_shocks_BM calculate the autocorrelation
def estimate_ar1_coefficients(df, cols):
    rho = {}
    a = {}
    for c in cols:
        z = np.log(df[c].values)
        x = z[:-1]
        y = z[1:]
        A = np.vstack([x, np.ones_like(x)]).T
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        rho[c] = float(beta[0])
        a[c] = float(beta[1])
    return rho, a


# %% Helper to construct a dataset
def construct_single_xy(
    data,
    states=None,
    controls=None,
    exo_states=None,
    eps=None,
    active_eps=None,
    return_torch=True,
    extract_eps=False,
    verbose=True,
):
    if states is None:
        states = ["K"]
    if controls is None:
        controls = ["C", "L"]
    if exo_states is None:
        exo_states = ["A", "Z", "mu"]
    if eps is None:
        eps = ["ϵ_a", "ϵ_z", "ϵ_mu"]
    if active_eps is None:
        active_eps = ["ϵ_a", "ϵ_z", "ϵ_mu"]

    # Back out eps from the states
    if extract_eps:

        # Drop the eps columns if they exist
        for ep in eps:
            if ep in data.columns:
                data = data.drop(columns=[ep])

        # Estimate the AR coefficients from the shock data
        rho, a = estimate_ar1_coefficients(data, states)

        # Get the innovations for the states
        data_eps = pd.DataFrame()
        for shock in states:
            data_eps[f"ϵ_{shock.lower()}"] = np.log(data[shock]) - (
                a[shock] + rho[shock] * np.log(data[shock].shift(1))
            )

        # Append it to the data
        data = pd.concat([data, data_eps], axis=1)

    # If eps is not in the active_eps, multiply it by zero
    # NOTE: The data currently has eps with mean and std different form zero even for the partial datasets
    for ep in eps:
        if ep not in active_eps:
            data[ep] = 0

    # Construct x = [states(t), exo_states(t)]
    x_states = data[states][:-1].reset_index(drop=True)
    x_exo_states = data[exo_states][:-1].reset_index(drop=True)
    x = pd.concat([x_states, x_exo_states], axis=1)

    # construct y = [states(t+1), controls(t)]
    y_controls = data[controls][:-1].reset_index(drop=True)
    y_states = data[states][1:].reset_index(drop=True)
    y = pd.concat([y_states, y_controls], axis=1)

    if return_torch:
        x = torch.tensor(x.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.float32)

    # Print summary for the process
    if verbose:
        print("=" * 50)

        if extract_eps:
            print("AR coefficients:")
            for shock in states:
                print(f"{shock}: rho = {rho[shock]:.4f}, a = {a[shock]:.4f}")

        # Summary statistics for the states
        for ep in eps:
            print(f"Summary statistics for {ep}:")
            if ep in data.columns:
                print(f"Mean of {ep}: {data[ep].mean():.4f}")
                print(f"Std of {ep}: {data[ep].std():.4f}")

        print("-" * 50)

    return {"x": x, "y": y}


# %% Construct datasets for all models
partial_datasets = {}
print("Constructind dataset from all data")
partial_datasets["all"] = construct_single_xy(
    data_all_df, active_eps=["ϵ_a", "ϵ_z", "ϵ_mu"]
)

print("Constructing dataset from tfp_zeta data")
partial_datasets["AB"] = construct_single_xy(data_AB_df, active_eps=["ϵ_a", "ϵ_z"])

print("Constructing dataset from tfp_delta data")
partial_datasets["AC"] = construct_single_xy(data_AC_df, active_eps=["ϵ_a", "ϵ_mu"])

print("Constructing dataset from zeta_delta data")
partial_datasets["BC"] = construct_single_xy(data_BC_df, active_eps=["ϵ_z", "ϵ_mu"])

print("Constructing dataset from delta data")
partial_datasets["C"] = construct_single_xy(data_C_df, active_eps=["ϵ_mu"])


# %% Combine the datasets
combinations = {
    "F": ["all"],
    "ABC": ["AB", "AC", "BC"],
    "C": ["C"],
}

# %% Combine, shuffle, and truncate the datasets
datasets = {}
for key, combination in combinations.items():
    temp_x_list = []
    temp_y_list = []

    for name in combination:
        # Shuffle each dataset individually
        indices = torch.randperm(partial_datasets[name]["x"].shape[0])
        shuffled_x = partial_datasets[name]["x"][indices]
        shuffled_y = partial_datasets[name]["y"][indices]

        # Truncate each dataset proportionally if truncation is enabled
        if truncate_length is not None:
            proportion = truncate_length // len(combination)
            shuffled_x = shuffled_x[:proportion]
            shuffled_y = shuffled_y[:proportion]

        temp_x_list.append(shuffled_x)
        temp_y_list.append(shuffled_y)

    # Concatenate the shuffled and truncated datasets
    temp_x = torch.cat(temp_x_list, dim=0)
    temp_y = torch.cat(temp_y_list, dim=0)

    # Perform a final shuffle to mix the datasets
    indices = torch.randperm(temp_x.shape[0])
    temp_x = temp_x[indices]
    temp_y = temp_y[indices]

    datasets[key] = {"x": temp_x, "y": temp_y}

# %% Specify the neural network settings

# Settings for the neural network
nn_settings = {
    "hidden": 128,
    "layers": 5,
    "normalize_input": True,
    "scale_output": True,
}
# Settings for the training of the neural network
training_settings = {
    "F": {
        "batch": 100,
        "epochs": 1000,
        "lr": 1e-3,
        "eta_min": 1e-10,
        "validation_share": 0.8,
        "print_after": 100,
    },
    "ABC": {
        "batch": 100,
        "epochs": 1000,
        "lr": 1e-3,
        "eta_min": 1e-10,
        "validation_share": 0.8,
        "print_after": 100,
    },
    "C": {
        "batch": 100,
        "epochs": 1000,
        "lr": 1e-3,
        "eta_min": 1e-10,
        "validation_share": 0.8,
        "print_after": 100,
    },
}


# %% Training or loading the surrogate models
models_path = "../bld/models/BM/"
Path(models_path).mkdir(parents=True, exist_ok=True)
surrogates = {}

# Depending on the specified option, either retrain the models or load the results
if retrain:
    # Train the surrogate models
    for key, data in datasets.items():
        surrogate = Surrogate(data=data)
        surrogate.make_network(**nn_settings)
        surrogate.train(**training_settings[key], device=device, shuffle_data=True)
        surrogate.save(f"{models_path}surrogate_{key}")
        surrogates[key] = surrogate
else:
    # Load the surrogate models
    for key in combinations.keys():
        surrogate = Surrogate()
        surrogate.load_attributes(f"{models_path}surrogate_{key}/surrogate.pkl")
        surrogates[key] = surrogate


# %% Move the surrogate models to CPU for plotting
for key, surrogate in surrogates.items():
    surrogate.network.to("cpu")
    surrogate.data_validation["x"] = surrogate.data_validation["x"].to("cpu")
    surrogate.data_validation["y"] = surrogate.data_validation["y"].to("cpu")
    surrogate.data_train["x"] = surrogate.data_train["x"].to("cpu")
    surrogate.data_train["y"] = surrogate.data_train["y"].to("cpu")

# %% Figures
figures_path = "../bld/figures/BM/"
figure_suffix = ""
Path(figures_path).mkdir(parents=True, exist_ok=True)


# %%
surrogate_labels = {
    "F": "Full",
    "ABC": "Glued",
    "C": "Mu only",
}
smoothing_window = 10

# plot the training and the validation loss for all but the true surrogate
for key, surrogate in [(k, v) for k, v in surrogates.items() if k != "True"]:
    loss_dict = surrogate.loss_dict
    save_name = f"loss_curves_{key}{figure_suffix}"
    fig, ax = plots.plot_training_validation_loss(
        loss_dict,
        label=surrogate_labels[key],
        ylim=min(np.minimum(loss_dict["training_loss"], loss_dict["validation_loss"])),
        smoothing_window=smoothing_window,
        selected_keys=[
            "training_loss",
            "validation_loss",
        ],
        save_path=figures_path,
        save_name=save_name,
        locators=False,
    )
    plt.show()


# %%
# Example with three surrogates
plots.plot_error_histogram_three(
    surrogate1=surrogates["C"],
    surrogate2=surrogates["F"],
    surrogate3=surrogates["ABC"],
    x_validation=surrogates["F"].data_validation["x"],
    y_validation=surrogates["F"].data_validation["y"],
    y_labels=["K", "C", "L"],
    plot_vars=["K"],
    save_name="error_histogram_three",
    save_path="../bld/figures/BM/",
    ncol=1,
    sub_figsize=(6, 4),
    x_range=(-0.1, 0.1),
    maxpercent=0.5,
    percentsteps=0.1,
    Latex_label=[
        r"Histogram of the Approximation Error of $K_t$",
    ],
)

# %% example with the other two variables
plots.plot_error_histogram_three(
    surrogate1=surrogates["C"],
    surrogate2=surrogates["F"],
    surrogate3=surrogates["ABC"],
    x_validation=surrogates["F"].data_validation["x"],
    y_validation=surrogates["F"].data_validation["y"],
    y_labels=["K", "C", "L"],
    plot_vars=["C", "L"],
    save_name="error_histogram_three_other_variables",
    save_path="../bld/figures/BM/",
    ncol=2,
    sub_figsize=(6, 4),
    x_range=(-0.75, 0.75),
    maxpercent=0.5,
    percentsteps=0.1,
    Latex_label=[r"$C_t$", r"$L_t$"],
)

plots.plot_error_histogram_three(
    surrogate1=surrogates["C"],
    surrogate2=surrogates["F"],
    surrogate3=surrogates["ABC"],
    x_validation=surrogates["F"].data_validation["x"],
    y_validation=surrogates["F"].data_validation["y"],
    y_labels=["K", "C", "L"],
    plot_vars=["C", "L"],
    save_name="error_histogram_three_other_variables_zoomed",
    save_path="../bld/figures/BM/",
    ncol=2,
    sub_figsize=(6, 4),
    x_range=(-0.003, 0.003),
    maxpercent=0.5,
    percentsteps=0.1,
    Latex_label=[r"$C_t$", r"$L_t$"],
)


# %% calculate the static Euler equation error
# Function to draw Monte Carlo samples
def ar1_lognormal_draws(states, rhos, stds, shocks, log_means):
    # states, rhos, stds, log_means are length-3 tensors on the same device
    # shocks is (N,3)
    # log s_{t+1} = (1−rho) log_mean + rho log s_t + σ ε
    # level s_{t+1} = exp(log s_{t+1} − 0.5 σ^2) to keep the mean at exp(log_mean)
    loc = (1 - rhos) * log_means + rhos * torch.log(states)
    samples = torch.exp(loc + shocks * stds - 0.5 * stds**2)
    return samples


# extracting the shocks from the full dataset
exo_shocks_BM = data_all["x"][:, 1:4]

# extract the shocks into a dataframe
shocks_df = pd.DataFrame(exo_shocks_BM.cpu().numpy(), columns=["A", "Z", "mu"])
rhos = torch.tensor([0.9, 0.9, 0.0], device=device, dtype=torch.float32)
log_means = torch.log(
    torch.tensor(
        [par["Abar"], par["Zbar"], par["mubar"]], device=device, dtype=torch.float32
    )
)
stds = torch.tensor([0.05, 0.05, 0.05], device=device)
log_means[2] = stds[2] ** 2  # set mean of mu to 1

# fix the seed for reproducibility
torch.manual_seed(420)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(420)

# number of Monte Carlo simulations
N_monte_carlo = 5000


# simulate the shocks
eps = torch.randn(N_monte_carlo // 2, 3, device=device)

# mirror the same shocks again (antithetic)
eps = torch.cat((eps, -eps), dim=0)


# %%setup of the shocks and states
predict = ["Kp", "C", "L"]
states = ["K", "A", "Z", "mu"]

alpha = par["alpha"]
beta = par["beta"]
gamma = par["gamma"]
omega = par["omega"]


def L(mu, mubar, alpha, beta, gamma, omega):
    num = mubar * (1 - alpha)
    den = mu * omega * (mubar - alpha * beta)
    return (num / den) ** (1 / gamma)


def Y(K, A, Z, L, alpha):
    return A * K**alpha * (Z * L) ** (1 - alpha)


def C(Y, alpha, beta, mubar):
    return (1 - alpha * beta / mubar) * Y


def Kp(Y, alpha, beta, mubar):
    return alpha * beta * Y / mubar


# function to calculate the EEE given a state vector and a surrogate model
def calculate_EEE_BM_Ana(states, future_exo_states, par):
    Kt, At, Zt, mut = states
    Lt = L(mut, par["mubar"], par["alpha"], par["beta"], par["gamma"], par["omega"])
    Yt = Y(Kt, At, Zt, Lt, par["alpha"])
    Kpt = Kp(Yt, par["alpha"], par["beta"], par["mubar"])
    Ct = C(Yt, par["alpha"], par["beta"], par["mubar"])

    Apt, Zpt, mutp = (
        future_exo_states[:, 0],
        future_exo_states[:, 1],
        future_exo_states[:, 2],
    )
    Lpt = L(mutp, par["mubar"], par["alpha"], par["beta"], par["gamma"], par["omega"])
    Ypt = Y(Kpt, Apt, Zpt, Lpt, par["alpha"])
    Cpt = C(Ypt, par["alpha"], par["beta"], par["mubar"])

    # Gross return on capital
    R = (
        par["alpha"]
        * Apt
        / mutp
        * (Kpt / (Zpt * Lpt)).clamp_min(1e-12) ** (par["alpha"] - 1)
    )

    err = par["beta"] * (R / Cpt) / (1.0 / Ct) - 1.0
    return err.mean().abs().item()


# function to calculate the EEE given a state vector and a surrogate model
def calculate_EEE_BM(states, surrogate, future_exo_states, par):
    model = surrogate.network
    model.eval()
    with torch.no_grad():
        x = states.repeat(future_exo_states.size(0), 1)  # (N,4)
        pred = model(x)
        Kp = pred[:, 0]
        C = pred[:, 1].clamp_min(1e-10)

        xp = torch.cat([Kp.unsqueeze(1), future_exo_states], dim=1)
        predp = model(xp)
        Cp = predp[:, 1].clamp_min(1e-10)
        Lp = predp[:, 2].clamp_min(1e-10)

        # Map exogenous states
        Ap = future_exo_states[:, 0]
        Zp = future_exo_states[:, 1]
        mup = future_exo_states[:, 2].clamp_min(1e-10)

        # Gross return on capital
        R = (
            par["alpha"]
            * Ap
            / mup
            * (Kp / (Zp * Lp)).clamp_min(1e-12) ** (par["alpha"] - 1)
        )

        err = par["beta"] * (R / Cp) / (1.0 / C) - 1.0
        return err.mean().abs().item()


# transform the parameters to tensors
device = eps.device
dtype = eps.dtype
means = torch.as_tensor(data_all["x"][0, 1:4], device=device, dtype=dtype)
rhos_t = torch.as_tensor(rhos, device=device, dtype=dtype)
stds_t = torch.as_tensor(stds, device=device, dtype=dtype)

# static evaluation of the Euler Equation Error at the mean state
EEE_static = {}
EEE_analytical = {}
print("Static Euler Equation Error at the mean state:")
for name in ["F", "ABC"]:
    todays_state = data_all["x"][6, :4].to(device)
    future_exo_states = ar1_lognormal_draws(
        states=todays_state[1:], rhos=rhos, stds=stds, shocks=eps, log_means=log_means
    )  # A, Z, mu today
    EEE_analytical[name] = calculate_EEE_BM_Ana(todays_state, future_exo_states, par)
    EEE_static[name] = calculate_EEE_BM(
        todays_state, surrogates[name], future_exo_states, par
    )
    print(f"BM EEE of {name} at mean state: {EEE_static[name]:.5f}")
    print(f"Analytical EEE of {name} at mean state: {EEE_analytical[name]:.5f}")


# %% function that simulates the economy one period ahead
def simulate_BM(states, surrogate, shocks_prime):
    x = torch.ones(1, 4, device=device) * states

    # determine next periods capital stock
    Kp = torch.ones(1, device=device) * surrogate.network(x)[:, 0]

    # combine Kp with the exo_shocks
    xp = torch.cat((Kp.unsqueeze(1), shocks_prime.unsqueeze(0)), dim=1)
    return xp


# %% simulate the economy forward for N_sim periods and calculate the EEE for each period
N_sim = 1000  # lower number than in the paper, but for faster runtime

# simulating forward
print(f"Dynamic Euler Equation Error for {N_sim} periods:")
EEE = {}

# simulate the economy forward
for name in ["F", "ABC"]:
    # initialize the state with the mean state from the full dataset
    todays_state = data_all["x"][0, :4].to(device)

    # initialize lists to store the results
    EEE[name] = []

    # get the surrogate model
    surrogate = surrogates[name]

    # simulating the economy forward for N_sim periods
    for t in trange(N_sim - 1):
        # simulate the shocks
        shocks = torch.randn(N_monte_carlo // 2, 3, device=device)

        # mirror the same shocks again (antithetic)
        shocks = torch.cat((shocks, -shocks), dim=0)

        # draw the shocks for the next period to calculate the EEE
        exo_shocks = ar1_lognormal_draws(
            states=todays_state[1:],
            rhos=rhos,
            stds=stds,
            shocks=eps,
            log_means=log_means,
        )

        # calculate the EEE for the current period
        EEE_t = calculate_EEE_BM(todays_state, surrogates[name], exo_shocks, par)

        # store the EEE in a list
        EEE[name].append(EEE_t)

        # simulate the economy one period ahead
        next_state = simulate_BM(
            todays_state, surrogates[name], exo_shocks_BM[t + 1, :]
        ).squeeze()

        # update the todays_state variable
        todays_state = next_state.detach().squeeze().clone()

    # print the average EEE over the simulated periods
    print(f"Average BM EEE of {name} over {N_sim} periods: {np.mean(EEE[name]):.5f}%")


# %% plot the EEE over time
importlib.reload(plots)
plots.plot_euler_error_histogram(
    EEE["F"],
    EEE["ABC"],
    save_path="../bld/figures/BM/",
    save_name="dynamic_euler_error_histogram",
    number_bins=50,
    num_locators_x=5,
    num_locators_y=5,
)


# %% simulate the model forward using the exogenous processes of the simulations
Capital = {}
# simulate the economy forward
for name, data in [
    ("F", data_all["x"][:, :4].to(device)),
    ("ABC", data_all["x"][:, :4].to(device)),
    ("C", data_C["x"][:, :4].to(device)),
]:
    # initialize the state with the mean state from the full dataset
    todays_state = data[0, :4].to(device)

    # load the surrogate model
    surrogate = surrogates[name]

    # initialize lists to store the results
    Capital[name] = []

    # simulating the economy forward for N_sim periods
    for t in trange(N_sim - 1):
        # store the capital in a list
        Capital[name].append(todays_state[0].item())

        # simulate the economy one period ahead
        next_state = simulate_BM(
            todays_state, surrogates[name], data[t + 1, 1:4]
        ).squeeze()

        # update the todays_state variable
        todays_state = next_state.detach().squeeze().clone()


# %% calculate the first four standardized moments of the capital stock distribution
# function to calculate the standardized moments
def standardized_moments(data):
    mean = np.mean(data)
    std = np.std(data)
    skewness = scipy.stats.skew(data)
    kurtosis = scipy.stats.kurtosis(data)
    return mean, std, skewness, kurtosis


# Print standardized moments as a formatted table
moment_names = ["Mean", "Std", "Skewness", "Kurtosis"]
results = []
for name in ["F", "ABC", "C"]:
    moments = standardized_moments(Capital[name])
    results.append([name] + [f"{m:.4f}" for m in moments])

df_moments = pd.DataFrame(results, columns=["Model"] + moment_names)
print(df_moments.to_string(index=False))

# %%
