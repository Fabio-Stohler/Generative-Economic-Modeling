"""Lightweight containers for parameters, states, and shocks."""

# Standard library
from typing import Dict

# Third-party dependencies
import torch
from torch.distributions import constraints


class Element:
    """Minimal dot-access container backed by torch tensors."""

    def __init__(self, mapping: Dict) -> None:
        for key, value in mapping.items():
            if isinstance(value, float):
                value = torch.tensor([value])
            setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        return list(self.__dict__.keys())

    def values(self):
        return list(self.__dict__.values())

    def items(self):
        return list(self.__dict__.items())

    def to(self, device):
        for key, value in self.__dict__.items():
            setattr(self, key, value.to(device))
        return self


class Parameters(Element):
    """Parameter container with expand helper."""

    def expand(self, shape):
        return Parameters({k: v.expand(shape) for k, v in self.__dict__.items()})


class Identity:
    """Fixed (degenerate) distribution wrapper for deterministic parameters."""

    def __init__(self, value) -> None:
        self.value = torch.tensor(value)

    def sample(self, shape):
        return self.value.expand(shape)

    @property
    def support(self):
        return constraints.interval(torch.tensor(-1.0), torch.tensor(1.0))

    @property
    def low(self):
        return self.support.lower_bound

    @property
    def high(self):
        return self.support.upper_bound


class Ranges(Element):
    """Parameter ranges/prior distributions used to draw model parameters."""

    def __init__(self, par_dict: Dict, priors_dict: Dict) -> None:
        mapping = {}
        for key, value in par_dict.items():
            mapping[key] = priors_dict.get(key, Identity(value))
        super().__init__(mapping)

    def sample(self, shape, device="cpu"):
        par_draw = {k: v.sample(shape).to(device) for k, v in self.__dict__.items()}
        return Parameters(par_draw)


class State(Element):
    """State container."""


class Shocks(Element):
    """Shock container with sampling utilities (including antithetics)."""

    def sample(self, shape, antithetic=False, device="cpu"):
        if antithetic:
            # Use antithetic pairs to reduce Monte Carlo variance.
            shape_antithetic = (shape[0] // 2, *shape[1:])
            shock_draw = {}
            for key, value in self.__dict__.items():
                sample = value.sample(shape_antithetic) - value.mean
                shock_draw[key] = torch.cat(
                    [value.mean + sample, value.mean - sample], dim=0
                ).to(device)
        else:
            shock_draw = {key: value.sample(shape).to(device) for key, value in self.__dict__.items()}
        return Element(shock_draw)
