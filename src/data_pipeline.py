"""Dataset assembly utilities for the Brock–Mirman model."""

# Standard library
from typing import Optional

# Third-party dependencies
import torch
from tqdm import trange


def simulate_dataset(
    model,
    par=None,
    draw: int = 100,
    burn: int = 0,
    steps: int = 120,
    device: str = "cpu",
    seed: Optional[int] = None,
    labnorm: bool = True,
):
    """
    Simulate a BMModel across multiple parameter draws and store the dataset on the model.

    Mirrors the original BMModel.simulate_dataset implementation; behavior unchanged.
    """
    x = []
    y = []
    for _ in trange(draw):
        # Draw parameter vector
        par = model.draw_parameters(shape=(1,), device=device)

        # Simulate
        states, controls, shocks = model.simulate(
            par, burn, steps, device, seed, labnorm
        )

        # Convert par, results, shocks to tensors and stack
        v_par = torch.cat([value.unsqueeze(0) for value in par.values()], dim=-1).expand(
            steps, -1
        )
        v_states = torch.stack([value for _, value in states.items()], dim=-1)
        v_controls = torch.stack([value for _, value in controls.items()], dim=-1)
        v_shocks = torch.stack([value for _, value in shocks.items()], dim=-1)

        # Stack results
        temp_x = torch.cat([v_states, v_par], dim=-1)[:-1, :]
        temp_y = torch.cat([v_controls], dim=-1)

        # Append to list
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
    k_all = k_state + k_cont + k_par + k_shocks

    # Persist on the model for backward compatibility
    model.dataset = {"x": x, "y": y}
    model.dataset_keys = k_all

    return model.dataset
