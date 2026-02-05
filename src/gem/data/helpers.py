"""Shared helper utilities for simulation, analysis, and diagnostics."""

from __future__ import annotations

# Third-party dependencies
import numpy as np
import pandas as pd
import scipy.stats
import torch


# %% Function to count parameters of a neural network and visualize the architecture
def count_parameters(model):
    """Print the total and trainable parameter counts for a PyTorch model."""
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


# %% Function to calculate the ergodic standard deviation of the AR(1) shock process
def ergodic_sigma(rho, sigma):
    """Compute the ergodic standard deviation for an AR(1) process."""
    return (sigma) / (1.0 - rho**2) ** 0.5


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
    """Simulate datasets for each distribution range and persist dataset pickles."""
    from ..model.bm import BMModel
    from .io_utils import save_pickle

    # For entries in distr_ranges create a model with the name from model_names_list, simulate the data, save the data
    for i, distr_range in enumerate(distr_ranges):
        model = BMModel(par, pars, distr_range)

        # Retry simulation if NaNs appear (numerical issues can arise for some draws).
        K = 25
        dataset = None
        for j in range(K):
            dataset, _ = model.simulate_dataset(
                draw=draws, steps=steps, burn=burn, labnorm=labnorm, store=True
            )
            # Check for NaN values in the dataset
            if torch.isnan(dataset["x"]).any() or torch.isnan(dataset["y"]).any():
                print(f"NaN values in the dataset for model {model_names_list[i]}")
            else:
                print(f"No NaN values in the dataset for model {model_names_list[i]}")
                break

        if j == K - 1:
            raise ValueError(
                f"Could not simulate the dataset for model {model_names_list[i]}"
            )

        # Save just the dataset for portability
        save_pickle(dataset, model_save_path, model_names_list[i])

    return None


def tensor_to_dataframe(tensor, columns):
    """Convert a torch tensor to a pandas DataFrame with named columns."""
    return pd.DataFrame(tensor.numpy(), columns=columns)


def estimate_ar1_coefficients(df, cols):
    """Estimate AR(1) coefficients for log processes in a dataframe."""
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


def construct_single_xy(
    data,
    states=None,
    controls=None,
    exo_states=None,
    eps=None,
    active_eps=None,
    return_torch=True,
    extract_eps=False,
    verbose=False,
):
    """Construct state/control tensors (x, y) from a dataframe slice."""
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

    # Back out shocks from exogenous states when requested.
    if extract_eps:
        # Drop the eps columns if they exist
        for ep in eps:
            if ep in data.columns:
                data = data.drop(columns=[ep])

        # Estimate the AR coefficients from the shock data
        shock_states = exo_states if exo_states is not None else states
        rho, a = estimate_ar1_coefficients(data, shock_states)

        # Get the innovations for the states
        data_eps = pd.DataFrame()
        for shock in shock_states:
            data_eps[f"ϵ_{shock.lower()}"] = np.log(data[shock]) - (
                a[shock] + rho[shock] * np.log(data[shock].shift(1))
            )

        # Append it to the data
        data = pd.concat([data, data_eps], axis=1)

    # Zero out inactive shock innovations so partial datasets match the intended model.
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
            for shock in shock_states:
                print(f"{shock}: rho = {rho[shock]:.4f}, a = {a[shock]:.4f}")

        # Summary statistics for the states
        for ep in eps:
            if ep not in active_eps:
                continue
            print(f"Summary statistics for {ep}:")
            if ep in data.columns:
                print(f"Mean of {ep}: {data[ep].mean():.4f}")
                print(f"Std of {ep}: {data[ep].std():.4f}")
            else:
                print(f"{ep} not found in data; enable extract_eps to compute it.")

        print("-" * 50)

    return {"x": x, "y": y}


def ar1_lognormal_draws(states, rhos, stds, shocks, log_means):
    """Draw lognormal AR(1) samples given current states and shocks."""
    # Uses a log-AR(1) transition with a mean correction so E[exp(log s)] matches exp(log_mean).
    # states, rhos, stds, log_means are length-3 tensors on the same device
    # shocks is (N,3)
    # log s_{t+1} = (1−rho) log_mean + rho log s_t + σ ε
    # level s_{t+1} = exp(log s_{t+1} − 0.5 σ^2) to keep the mean at exp(log_mean)
    loc = (1 - rhos) * log_means + rhos * torch.log(states)
    samples = torch.exp(loc + shocks * stds - 0.5 * stds**2)
    return samples


def L(mu, mubar, alpha, beta, gamma, omega):
    """Closed-form labor policy function."""
    num = mubar * (1 - alpha)
    den = mu * omega * (mubar - alpha * beta)
    return (num / den) ** (1 / gamma)


def Y(K, A, Z, L_value, alpha):
    """Cobb-Douglas production function."""
    return A * K**alpha * (Z * L_value) ** (1 - alpha)


def C(Y_value, alpha, beta, mubar):
    """Consumption policy from output and parameters."""
    return (1 - alpha * beta / mubar) * Y_value


def Kp(Y_value, alpha, beta, mubar):
    """Capital accumulation policy from output and parameters."""
    return alpha * beta * Y_value / mubar


def calculate_EEE_BM_Ana(states, future_exo_states, par):
    """Compute analytical Euler equation error using closed-form policies."""
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


def calculate_EEE_BM(states, surrogate, future_exo_states, par):
    """Compute Euler equation error using surrogate model predictions."""
    model = surrogate.network
    model.eval()
    with torch.no_grad():
        x = states.repeat(future_exo_states.size(0), 1)  # (N,4)
        pred = model(x)
        Kp_value = pred[:, 0]
        C_value = pred[:, 1].clamp_min(1e-10)

        xp = torch.cat([Kp_value.unsqueeze(1), future_exo_states], dim=1)
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
            * (Kp_value / (Zp * Lp)).clamp_min(1e-12) ** (par["alpha"] - 1)
        )

        err = par["beta"] * (R / Cp) / (1.0 / C_value) - 1.0
        return err.mean().abs().item()


def simulate_BM(states, surrogate, shocks_prime, device):
    """Advance the BM state one step using the surrogate model."""
    x = torch.ones(1, 4, device=device) * states

    # determine next periods capital stock
    Kp_value = torch.ones(1, device=device) * surrogate.network(x)[:, 0]

    # combine Kp with the exo_shocks
    xp = torch.cat((Kp_value.unsqueeze(1), shocks_prime.unsqueeze(0)), dim=1)
    return xp


def standardized_moments(data):
    """Compute mean, std, skewness, and kurtosis for a series."""
    mean = np.mean(data)
    std = np.std(data)
    skewness = scipy.stats.skew(data)
    kurtosis = scipy.stats.kurtosis(data)
    return mean, std, skewness, kurtosis
