"""Brock–Mirman analytical model implementation and simulation utilities.

Main entry point: :class:`BMModel` for simulating datasets and policy functions
used in surrogate training.
"""

# %% [markdown]
# # Code for "Generative Economic Modeling"
#
# This code solves and simulates the analytical version of the Brock-Mirman model.
# For convenience, the model is written in the form of a class, which can be instantiated with the desired parameters.
# The class either draws random parameters from a normal distribution or uses the parameters provided by the user.
# The results of the simulation are stored in the class and saved via pickle.

# %%
# Third-party dependencies
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from tqdm import trange

# Local imports
from ..data.helpers import ergodic_sigma
from ..structures import Parameters, Ranges, Shocks, State


class BMModel(object):
    """
    Class that implements the analytical Brock-Mirman model.
    """

    def __init__(self, parameters, ranges, shocks) -> None:
        # Initialize parameter/range/shock containers used throughout the model.
        self.range = Ranges(parameters, ranges)
        self.shock = Shocks(shocks)
        self.par = Parameters(parameters)
        self.par_draw = None
        self.ss = None
        self.state = None
        self.loss_dict = None
        self.training_conf = None
        self.dataset = None
        self.dataset_keys = None

    def to(self, device):
        # Move internal tensors to the requested device.
        self.par.to(device)
        self.par_draw.to(device)
        self.ss.to(device)
        self.state.to(device)

    def labor(self, state, par=None):
        """
        Labor policy based on equation (10) in the paper.
        """

        if par is None:
            par = self.par_draw

        # Return labor
        return (
            par.mubar
            * (1 - state.tau_L)
            * (1 - par.alpha)
            / (
                torch.exp(state.mu)
                * par.omega
                * (1 + par.tau_C)
                * (par.mubar - (1 - par.tau_K) * par.alpha * par.beta)
            )
        ) ** (1 / (1 + par.gamma))

    def output(self, state, par=None):
        """
        Function that calcualtes output based on a Cobb-Douglas production function.
        """

        if par is None:
            par = self.par_draw

        # Return output
        return (
            torch.exp(state.A)
            * state.capital**par.alpha
            * (torch.exp(state.Z) * self.labor(state, par)) ** (1 - par.alpha)
        )

    def consumption(self, state, par=None):
        """
        Function that calculates the policy function for consumption.
        """

        if par is None:
            par = self.par_draw

        # Calculate output
        Y = self.output(state)

        # Calculate consumption
        return (1 - par.alpha * par.beta / par.mubar * (1 - par.tau_K)) * Y

    def capital(self, state, par=None):
        """
        Function that calculates the policy function for capital.
        """

        if par is None:
            par = self.par_draw

        # Calculate output
        Y = self.output(state)

        # Calculate capital
        return par.alpha * par.beta / par.mubar * (1 - par.tau_K) * Y

    def steady_state(self, par=None, labnorm=True):
        """
        Compute the steady state of the Brock-Mirman model, such that L = 1/3.
        """
        if par is None:
            par = self.par_draw

        # solving the solution for labor supply in proposition 1 for omega
        # We use the mean of tau_L to calculate the SS value of omega
        if labnorm:
            L = 1
            omega = (
                (1 - par.alpha)
                * (1 - self.shock["tau"].mean)
                / L ** (1 + par.gamma)
                / (1 + par.tau_C)
                / (par.mubar - par.alpha * par.beta * (1 - par.tau_K))
            )
        else:
            omega = par.omega

        # solving for the steady state value of capital
        kss = (
            (par.alpha * par.beta / par.mubar * (1 - par.tau_K) * par.Abar)
            ** (1 / (1 - par.alpha))
            * self.labor(self.state, par)
            * par.Zbar
        )

        # setting the calibrated value of omega and the capital in ss
        self.par.omega = omega
        self.par.kss = kss
        self.par_draw.omega = omega

        # Return the new parameter structure with the updated omega
        return Parameters({"omega": omega, "kss": kss})

    def initialize_state(self, par=None, multiplier=1.0, device="cpu", labnorm=True):
        """
        Prepare the initial state of the model, with epsilon = 0 and K at steady state
        """
        if par is None:
            par = self.par_draw

        # Draw initial value for log A = A
        A = torch.log(par.Abar) * torch.ones((1,), device=device)

        # Draw initial value for log Z = Z
        Z = torch.log(par.Zbar) * torch.ones((1,), device=device)

        # Draw initial value for log mu = mu
        mu = torch.log(par.mubar) * torch.ones((1,), device=device)

        # Draw initial value for tau as mean of uniform distribution
        tau = self.shock["tau"].mean * par.tau_switch

        # assign the stocks above
        self.state = State({"A": A, "Z": Z, "mu": mu, "tau_L": tau})

        # Compute steady state
        ss = self.steady_state(par, labnorm)

        # overwrite the capital stock
        self.state["capital"] = ss.kss

        return State({"A": A, "Z": Z, "mu": mu, "capital": ss.kss, "tau_L": tau})

    def initialize_state_stochastic(
        self, par=None, multiplier=1.0, device="cpu", labnorm=True
    ):
        """
        Prepare the initial state of the model, with a random realization of epsilon and the captial stock.
        """
        if par is None:
            par = self.par_draw

        # Compute steady state
        ss = self.steady_state(par, labnorm)

        # Ergodic standard deviation of epsilon for a and z
        rho_a = par.rho_a
        sigma_a = par.sigma_a
        ergodic_a = ergodic_sigma(rho_a, sigma_a)
        rho_z = par.rho_z
        sigma_z = par.sigma_z
        ergodic_z = ergodic_sigma(rho_z, sigma_z)
        rho_mu = par.rho_mu
        sigma_mu = par.sigma_mu
        ergodic_mu = ergodic_sigma(rho_mu, sigma_mu)

        # Draw initial value for log A from ergodic distribution
        A = torch.randn((1,), device=device) * ergodic_a * multiplier + torch.log(
            par.Abar
        )
        Z = torch.randn((1,), device=device) * ergodic_z * multiplier + torch.log(
            par.Zbar
        )
        mu = torch.randn((1,), device=device) * ergodic_mu * multiplier + torch.log(
            par.mubar
        )

        # Draw initial value for tau
        tau = self.shock.sample((1,), device=device)["tau"] * par.tau_switch

        # Draw multiplier for capital stock
        factor = torch.empty((1,), device=device).uniform_(0.5, 1.5)

        return State(
            {"A": A, "Z": Z, "mu": mu, "capital": ss.kss * factor, "tau_L": tau}
        )

    def draw_parameters(self, shape, device="cpu"):
        return self.range.sample(shape, device=device)

    def draw_shocks(self, shape, antithetic=False, device="cpu"):
        return self.shock.sample(shape, antithetic, device=device)

    def policy(self, state=None, par=None):
        """Return policy functions (labor, consumption, next-period capital)."""
        if state is None:
            state = self.state
        if par is None:
            par = self.par_draw

        # Calculate labor
        L = self.labor(state, par)

        # Calculate output
        Y = self.output(state, par)

        # Calculate consumption
        C = self.consumption(state, par)

        # Calculate capital
        Kprime = self.capital(state, par)

        return L, C, Kprime

    @torch.no_grad()
    def step(self, e):
        """Advance the state one period given a shock draw."""
        par = self.par_draw
        state = self.state

        # Log-AR(1) processes for technology and preference shocks.
        A_next = (
            (1 - par.rho_a) * (torch.log(par.Abar) - par.sigma_a**2 / 2)
            + par.rho_a * state.A
            + e.epsilon_a * par.sigma_a
        )
        Z_next = (
            (1 - par.rho_z) * (torch.log(par.Zbar) - par.sigma_z**2 / 2)
            + par.rho_z * state.Z
            + e.epsilon_z * par.sigma_z
        )
        mu_next = (
            (torch.log(par.mubar)) + 0.5 * par.sigma_mu**2 + e.epsilon_mu * par.sigma_mu
        )
        K_next = self.capital(state, par)
        tau_next = e.tau * par.tau_switch

        return State(
            {
                "A": A_next,
                "Z": Z_next,
                "mu": mu_next,
                "capital": K_next,
                "tau_L": tau_next,
            }
        )

    def steps(self, device, steps):
        """Iterate the model forward for a number of steps (used for burn-in)."""
        for _ in range(steps):
            e = self.draw_shocks((1,), device=device)
            self.state = self.step(e)

    @torch.no_grad()
    def sim_step(self, par=None):
        """Return the simulated observable variables for the current state."""
        if par is None:
            par = self.par_draw
        A = torch.exp(self.state.A)
        Z = torch.exp(self.state.Z)
        mu = torch.exp(self.state.mu)
        tau_L = self.state.tau_L
        L, C, Kprime = self.policy(self.state, par)
        return {
            "A": A,
            "Z": Z,
            "mu": mu,
            "K": self.state.capital,
            "tau_L": tau_L,
            "C": C,
            "L": L,
            "Kp": Kprime,
        }

    @torch.no_grad()
    def simulate(
        self, par=None, burn=0, steps=120, device="cpu", seed=None, labnorm=True
    ):
        """Simulate a single trajectory of length `steps` and return states/controls/shocks."""
        # Manual seed
        if seed is not None:
            torch.manual_seed(seed)

        # Set parameters and dimensions
        if par is None:
            self.par_draw = self.par.expand((1,))
        else:
            self.par_draw = par.expand((1,))

        # Change the device of the model
        # self.to(device)

        # Initialize
        self.state = self.initialize_state(device=device, labnorm=labnorm)
        self.ss = self.steady_state(labnorm=labnorm)

        # update omega in the parameters
        self.par_draw.omega = self.ss.omega

        # Burn-in
        self.steps(device=device, steps=burn)

        # Simulate
        controls = {"Kp": [], "C": [], "L": []}
        states = {"K": [], "A": [], "Z": [], "mu": [], "tau_L": []}
        shocks = {"epsilon_a": [], "epsilon_z": [], "epsilon_mu": [], "tau": []}
        for _ in range(steps):
            out = self.sim_step()

            # Store results
            for key, value in out.items():
                if key in controls.keys():
                    controls[key].append(value.squeeze(-1))
                if key in states.keys():
                    states[key].append(value.squeeze(-1))

            # Update state
            e = self.draw_shocks((1,), device=device)
            self.state = self.step(e)

            # Store shocks
            # rather than storing the shocks, we store the scaled shocks
            for key, value in e.items():
                if key == "epsilon_mu":  # very quick fix to avoid scaling the mu shocks
                    shocks[key].append(
                        (self.par_draw["sigma" + key[-3:]] * value).squeeze(-1)
                    )
                elif (
                    key != "tau" and key != "mu"
                ):  # very quick fix to avoid scaling the tau shocks
                    shocks[key].append(
                        (self.par_draw["sigma" + key[-2:]] * value).squeeze(-1)
                    )
                else:
                    shocks[key].append(value.squeeze(-1))

        # Stack results and shocks
        for key, value in states.items():
            states[key] = torch.stack(value, dim=-1)

        for key, value in controls.items():
            controls[key] = torch.stack(value, dim=-1)

        for key, value in shocks.items():
            shocks[key] = torch.stack(value, dim=-1)

        return states, controls, shocks

    def simulate_dataset(
        self,
        par=None,
        draw=100,
        burn=0,
        steps=120,
        device="cpu",
        seed=None,
        labnorm=True,
        store: bool = True,
    ):
        """
        Simulate the model for multiple parameter draws.

        Returns (dataset, keys). Optionally stores on the instance for compatibility.
        """
        x = []
        y = []
        for _ in trange(draw):
            # Draw parameter vector
            par = self.draw_parameters(shape=(1,), device=device)

            # Simulate
            states, controls, shocks = self.simulate(
                par, burn, steps, device, seed, labnorm
            )

            # Convert par, results, shocks to tensors and stack
            v_par = torch.cat([value.unsqueeze(0) for value in par.values()], dim=-1).expand(
                steps, -1
            )
            v_states = torch.stack([value for value in states.values()], dim=-1)
            v_controls = torch.stack([value for value in controls.values()], dim=-1)
            v_shocks = torch.stack([value for value in shocks.values()], dim=-1)

            # Stack results: x uses states/parameters at t, y uses controls at t.
            # Drop the last state so x and y align.
            temp_x = torch.cat([v_states, v_par], dim=-1)[:-1, :]
            temp_y = torch.cat([v_controls], dim=-1)

            x.append(temp_x)
            y.append(temp_y)

        # Stack over draws
        x = torch.stack(x, dim=0)
        y = torch.stack(y, dim=0)

        # Keys
        k_par = par.keys()
        k_state = list(states.keys())
        k_cont = list(controls.keys())
        k_shocks = list(shocks.keys())
        keys = k_state + k_cont + k_par + k_shocks

        dataset = {"x": x, "y": y}

        if store:
            self.dataset = dataset
            self.dataset_keys = keys

        return dataset, keys


# %% testing the code
if __name__ == "__main__":
    # Dict with economic parameters
    par = {
        "alpha": 0.33,
        "beta": 0.96,
        "omega": 1.0,
        "gamma": 5.0,
        "rho_a": 0.9,
        "rho_z": 0.9,
        "rho_mu": 0.9,
        "sigma_a": 0.05,
        "sigma_z": 0.05,
        "sigma_mu": 0.05,
        "Abar": 5.73,
        "Zbar": 1.0,
        "mubar": 1.5,
        "tau_L": 0.0,
        "tau_K": 0.0,
        "tau_C": 0.0,
        "kss": 1.0,
        "tau_switch": 0.0,
    }

    # dict with parameter ranges to potentially sample from
    par_ranges = {}

    # dict for the distribution of the shock process
    distr = {
        "epsilon_a": torch.distributions.Normal(0.0, 1.0),
        "epsilon_z": torch.distributions.Normal(0.0, 1.0),
        "epsilon_mu": torch.distributions.Normal(0.0, 1.0),
        "tau": torch.distributions.Uniform(0.0, 1.0),
    }

    # set a seed for reproducibility
    torch.manual_seed(42)

    # Create model
    model = BMModel(par, par_ranges, distr)
    model.par_draw = model.draw_parameters((1,))

    # simulate the model
    states, controls, shocks = model.simulate(burn=0, steps=100, labnorm=True)

    # plot the simulated series for capital and labor
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(states["K"].squeeze().cpu().numpy(), label="Capital")
    axs[0].set_title("Capital over time")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Capital")
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[0].xaxis.set_minor_locator(AutoMinorLocator())
    axs[0].yaxis.set_minor_locator(AutoMinorLocator())
    axs[1].plot(controls["L"].squeeze().cpu().numpy(), label="Labor", color="green")
    axs[1].set_title("Labor over time")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Labor")
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.5)
    axs[1].xaxis.set_minor_locator(AutoMinorLocator())
    axs[1].yaxis.set_minor_locator(AutoMinorLocator())
    plt.tight_layout()
    plt.show()


# %%
