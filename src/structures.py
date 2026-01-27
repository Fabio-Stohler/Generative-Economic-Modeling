"""Typed containers for parameters, states, and shocks used throughout the model."""

# Standard library
from typing import Dict

# Third-party dependencies
import torch
from torch.distributions import constraints


# %% Base class for the elements of the model
class Element(object):
    """Base container that exposes dict-like access to tensor attributes."""
    def __init__(self, dict: Dict) -> None:
        # Store provided values as attributes, normalizing floats to tensors.
        for key, value in dict.items():
            if isinstance(value, float):
                value = torch.tensor([value])
            setattr(self, key, value)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        output = ""
        for key, value in self.__dict__.items():
            output += f"{key}: {value}\n"
        return output

    def update(self, dict: Dict):
        for key, value in dict.items():
            setattr(self, key, value)

    def get(self, key):
        return getattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def items(self):
        return list(self.__dict__.items())

    def cat(self):
        """Concatenate all stored tensors along the last dimension."""
        return torch.cat(self.values(), dim=-1)

    def to(self, device):
        """Move all tensor attributes to a device."""
        for key, value in self.__dict__.items():
            setattr(self, key, value.to(device))
        return self

    def requires_grad(self, requires_grad=True):
        """Set requires_grad on all tensor attributes."""
        for value in self.__dict__.values():
            value.requires_grad = requires_grad
        return self

    def zero_grad(self):
        """Clear gradients for all tensor attributes."""
        for value in self.__dict__.values():
            value.grad = None
        return self


# %% Class for the parameters
class Parameters(Element):
    """Parameter container with a convenience expand helper."""
    def expand(self, shape):
        return Parameters({key: value.expand(shape) for key, value in self.__dict__.items()})


# %% Identity class for fixed parameters
class Identity(object):
    """Fixed (degenerate) distribution wrapper for deterministic parameters."""
    def __init__(self, value) -> None:
        self.value = torch.tensor(value)

    def __str__(self):
        return f"Identity({str(self.value.item())})"

    def sample(self, shape):
        """Return a broadcasted tensor of the fixed value."""
        return self.value.expand(shape)

    @property
    def support(self):
        """Return a nominal support interval (used by range utilities)."""
        return constraints.interval(torch.tensor(-1.0), torch.tensor(1.0))

    @property
    def low(self):
        return self.support.lower_bound

    @property
    def high(self):
        return self.support.upper_bound


# %% Class for the priors
class Ranges(Element):
    """Parameter ranges/prior distributions used to draw model parameters."""
    def __init__(self, par_dict: Dict, priors_dict: Dict) -> None:
        # Replace parameters with prior distributions when available.
        for key, value in par_dict.items():
            if key in priors_dict:
                value = priors_dict[key]
            else:
                value = Identity(value)
            setattr(self, key, value)

    def cat(self):
        pass

    def limits(self):
        """Return support limits for each parameter."""
        return {key: value.support for key, value in self.__dict__.items()}

    def low_tensor(self):
        """Collect lower bounds for all parameters."""
        return torch.tensor([value.low for value in self.__dict__.values()])

    def high_tensor(self):
        """Collect upper bounds for all parameters."""
        return torch.tensor([value.high for value in self.__dict__.values()])

    def sample(self, shape, device="cpu"):
        """Sample a Parameters object from the stored distributions."""
        par_draw = {
            key: value.sample(shape).to(device) for key, value in self.__dict__.items()
        }
        return Parameters(par_draw)


# %% Class for the state
class State(Element):
    """State container (inherits all behavior from Element)."""
    pass


# %% Class for the shocks
class Shocks(Element):
    """Shock container with sampling utilities (including antithetics)."""
    def sample(self, shape, antithetic=False, device="cpu"):
        # Generate shocks, optionally with antithetic pairs for variance reduction.
        if antithetic:
            shape_antithetic = (shape[0] // 2, *shape[1:])
            shock_draw = {}
            for key, value in self.__dict__.items():
                sample = value.sample(shape_antithetic) - value.mean
                shock_draw[key] = torch.cat(
                    [value.mean + sample, value.mean - sample], dim=0
                ).to(device)
        else:
            shock_draw = {}
            for key, value in self.__dict__.items():
                shock_draw[key] = value.sample(shape).to(device)
        return Element(shock_draw)
