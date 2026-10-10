from abc import ABC, abstractmethod
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset


class ArtificialDataGenerator(ABC):
    """
    Class for artificial data generation, that can be passed to a dataloader.
    However, dataloaders should disable batching as this handles batching on its own.
    Children should use super().__init__(batch_size).
    """

    @property
    @abstractmethod
    def truth_length(self) -> int:
        """Return L, the number of frames in a truth tensor."""
        ...

    @property
    @abstractmethod
    def feature_shape(self) -> tuple:
        """Return the shape of a single feature. Empty tuple means no features."""
        ...

    @abstractmethod
    def sample_n(self, n: int) -> tuple[Tensor, Tensor]:
        """The sampling function to override"""
        ...

    def visualize_truths(
        self, samples: int = 50, title=None, distribution=False, show=True, **kwargs
    ):
        """Visualize samples of the truth. Useful if the truths are mostly fixed in time."""
        _, sample_truths = self.sample_n(samples)
        if title is not None:
            plt.title(title)
        if distribution:
            sample_truths = sample_truths.mean(dim=0, dtype=torch.float)[None, :]
        plt.imshow(sample_truths, aspect="auto", **kwargs)
        if distribution:
            plt.colorbar()
        if show:
            plt.show()

    def visualize_single(self):
        x, y = self.sample_n(1)
        x = x.squeeze(0)
        y = y.squeeze(0)
        f_shape = self.feature_shape

        if len(f_shape) == 2:
            pass
        elif len(f_shape) == 1 and f_shape[0] > 0:
            x = x.unsqueeze(1)
        else:
            raise ValueError(f"Cannot visualize features with shape {f_shape}")
        plt.subplot(2, 1, 1)
        plt.imshow(x.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.title("Features")
        plt.show()

        plt.subplot(2, 1, 2)
        plt.imshow(y.unsqueeze(1).detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.title("Truths")
        plt.show()


class ArtificialDataset(IterableDataset):
    """
    A way to turn an ArtificialDataGenerator into a dataset that can be handed off to a dataloader.
    If batch_size=0, the iterator will return items without batching, so it can be used with dataloader batching.
    """

    def __init__(self, adg: ArtificialDataGenerator, batch_size: int = 1):
        self.adg = adg
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        if self.batch_size > 0:
            return self.adg.sample_n(n=self.batch_size)
        elif self.batch_size == 0:
            x, y = self.adg.sample_n(1)
            return x.unsqueeze(0), y.unsqueeze(0)


class BeatMaskGenerator(ArtificialDataGenerator):
    def __init__(
        self,
        mask,
    ) -> None:
        assert mask.ndim == 1
        self._length = len(mask)
        self.mask = mask

    @property
    def truth_length(self) -> int:
        return self._length

    @property
    def feature_shape(self) -> tuple:
        return (0,)

    def sample_n(self, n: int) -> tuple[Tensor, Tensor]:
        return (
            torch.empty(n, 0),
            torch.ones(size=(n, 1)) * self.mask,
        )

    @classmethod
    def from_grid(
        cls,
        length: int = 50,
        samples_per_beat: int = 10,
        offset: int = 5,
    ):
        mask = torch.zeros(length)
        beat_locations = torch.arange(offset, length, samples_per_beat)
        mask[beat_locations] = 1

        return cls(mask=mask)


def create_tempo_change_generator(fpb1, fpb2, min_changeover, length, offset=5):
    first_part = torch.arange(offset, min_changeover, fpb1)
    second_part = torch.arange(first_part[-1] + fpb1, length, fpb2)
    total_mask = torch.zeros(length)
    total_mask[first_part] = 1
    total_mask[second_part] = 1

    return BeatMaskGenerator(total_mask)


class NoisyGenerator(ArtificialDataGenerator):
    def __init__(
        self,
        base: ArtificialDataGenerator,
        false_negative_rate: float = 0.01,
        false_positive_rate: float = 0.01,
    ) -> None:
        self.base = base

        self.true_positive_rate = 1 - false_negative_rate
        self.false_positive_rate = false_positive_rate

    @property
    def truth_length(self) -> int:
        return self.base.truth_length

    @property
    def feature_shape(self) -> tuple:
        return self.base.feature_shape

    def sample_n(self, n: int) -> tuple[Tensor, Tensor]:
        x, y = self.base.sample_n(n)
        mask = self.true_positive_rate * y + self.false_positive_rate * (1 - y)
        res = torch.rand_like(y)

        return (x, (res / mask <= 1).int())


class WeightedRandomGenerator(ArtificialDataGenerator):
    def __init__(
        self,
        generators: list[ArtificialDataGenerator],
        weights: list[float] | np.ndarray | None,
    ) -> None:
        self.length = generators[0].truth_length
        if not all(g.truth_length == self.length for g in generators[1:]):
            raise ValueError("Cannot combine sample generators of different lengths")

        self.shape = generators[0].feature_shape
        if not all(g.feature_shape == self.shape for g in generators[1:]):
            raise ValueError(
                "Cannot combine sample generators of different feature shapes"
            )

        if weights is not None:
            if len(weights) != len(generators):
                raise ValueError("Generator list and weights must have same lengths")
            weights = np.array(weights)
            weights = weights / np.sum(weights, dtype=float)

        self.generators = generators
        self.weights = Tensor(weights)

    @property
    def truth_length(self) -> int:
        return self.length

    @property
    def feature_shape(self) -> tuple:
        return self.shape

    def sample_n(self, n: int) -> tuple[Tensor, Tensor]:
        ids = torch.multinomial(self.weights, n, replacement=True)
        n_of_each = torch.bincount(ids, minlength=len(self.weights)).int()

        feats = torch.zeros(size=(n, *self.shape))
        res = torch.zeros(size=(n, self.length))

        current_low = 0
        for i, n_of_specific in enumerate(n_of_each):
            xi, yi = self.generators[i].sample_n(int(n_of_specific))
            feats[current_low : current_low + n_of_specific] = xi
            res[current_low : current_low + n_of_specific] = yi
            current_low = current_low + n_of_specific

        # Shuffle here? Should never matter
        return feats, res


class EasyTempoGenerator(ArtificialDataGenerator):
    def __init__(
        self,
        T: int,
        freq: float,
        log_freq_std: float = 0.2,
        random_phase: bool = True,
    ) -> None:
        super().__init__()
        self.shape = (T, 2)
        self.length = T
        self.freq = freq
        self.log_freq_std = log_freq_std
        self.random_phase = random_phase

    @property
    def feature_shape(self) -> tuple:
        return self.shape

    @property
    def truth_length(self) -> int:
        return self.length

    def sample_n(self, n):
        phases = torch.rand(n)
        freqs = torch.exp(torch.randn(n) * self.log_freq_std) * self.freq

        x = torch.arange(self.length)[None, ...] * freqs[..., None] + phases[..., None]
        # x : (B, T)
        x *= 2 * torch.pi

        features = torch.empty(size=(n, self.length, 2))
        cos_features = torch.cos(x)
        features[..., 0] = cos_features
        features[..., 1] = torch.sin(x)

        spread = torch.max_pool1d(cos_features, 3, 1, 1)
        mask = (cos_features == spread) & (cos_features > 0.8)
        return features, mask


class AmbiguousFeatureGenerator(ArtificialDataGenerator):
    def __init__(
        self,
        T: int,
        freq: float,
        double_time_probability=0.1,
        log_freq_std: float = 0.2,
        random_phase: bool = True,
    ) -> None:
        super().__init__()
        self.shape = (T, 2)
        self.length = T
        self.freq = freq
        self.double_time_probability = double_time_probability
        self.log_freq_std = log_freq_std
        self.random_phase = random_phase

    @property
    def feature_shape(self) -> tuple:
        return self.shape

    @property
    def truth_length(self) -> int:
        return self.length

    def sample_n(self, n):
        phases = torch.rand(n)
        freqs = torch.exp(torch.randn(n) * self.log_freq_std) * self.freq

        x = torch.arange(self.length)[None, ...] * freqs[..., None] + phases[..., None]
        # x : (B, T)
        x *= 2 * torch.pi
        beat_activation_factor = 1 + (torch.rand(n) < self.double_time_probability)
        # beat_activation_factor : (B)

        features = torch.empty(size=(n, self.length, 2))
        features[..., 0] = torch.cos(x)
        features[..., 1] = torch.sin(x)

        activation_features = torch.cos(x * beat_activation_factor[:, None])

        spread = torch.max_pool1d(activation_features, 3, 1, 1)
        edge_mask = torch.zeros_like(activation_features)
        edge_mask[:, 0] = 1
        edge_mask[:, -1] = 1
        mask = (activation_features == spread) & (~edge_mask.bool())
        return features, mask


class NoisyDataGenerator(ArtificialDataGenerator):
    def __init__(
        self,
        base_generator: ArtificialDataGenerator,
        feature_noise=0,
        truth_false_positive=0,
        truth_false_negative=0,
    ):
        super().__init__()
        self.base_generator = base_generator
        self.feature_noise = feature_noise
        self.truth_false_positive = truth_false_positive
        self.truth_false_negative = truth_false_negative
