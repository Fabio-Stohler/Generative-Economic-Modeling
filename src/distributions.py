"""Custom distribution helpers used by simulation/analysis code."""

# Third-party dependencies
import torch


class CustomCategorical(torch.distributions.Categorical):
    """Categorical distribution with optional deterministic cycling behavior."""

    def __init__(self, values, probs=None, deterministic=True):
        # If values is not a tensor, convert it to a tensor
        if not torch.is_tensor(values):
            values = torch.tensor(values)

        if probs is None:
            probs = torch.ones(len(values)) / len(values)

        if not torch.is_tensor(probs):
            probs = torch.tensor(probs)

        # Initialize the parent distribution with probability weights.
        super().__init__(probs=probs)
        self.values = values
        self.deterministic = deterministic

        if self.deterministic:
            self._generate_deterministic_sequence()

    def _generate_deterministic_sequence(self):
        """Create a repeatable sequence used by deterministic sampling."""
        self.deterministic_sequence = self.values.tolist()
        self.current_index = 0

    def sample(self, sample_shape=torch.Size()):
        """Sample values; cycle deterministically when enabled."""
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        if self.deterministic:
            num_samples = sample_shape.numel()
            samples = []
            for _ in range(num_samples):
                samples.append(self.deterministic_sequence[self.current_index])
                self.current_index = (self.current_index + 1) % len(
                    self.deterministic_sequence
                )
            return torch.tensor(samples).reshape(sample_shape)
        else:
            sample_index = super().sample(sample_shape)
            return self.values[sample_index].reshape(sample_shape)

    @property
    def mean(self):
        # Expected value under the categorical distribution.
        return torch.sum(self.probs * self.values)

    @property
    def variance(self):
        # Variance computed from probabilities and values.
        mean = self.mean
        return torch.sum(self.probs * (self.values - mean) ** 2)

    @property
    def stddev(self):
        # Standard deviation (sqrt of variance).
        return self.variance.sqrt()
