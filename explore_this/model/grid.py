import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn

from explore_this.batchable_dataclass import BatchableDataclass


@dataclass
class GridOutput(BatchableDataclass):
    # B x T
    activation: torch.Tensor

    # B x W x C x 3 (w is window, c is bin)
    features: torch.Tensor

    # B x T (cos, sin)
    mask: torch.Tensor

    # B x T x 2 (cos, sin)
    waves: torch.Tensor


# TODO breaks / width adjustment / window overlap
class WindowedGrid(nn.Module):
    """
    A module that transforms windowed latent tempo-phase representations into a time-dense activation.

    Representations for one datapoint have the shape (W, C, 3) with W being the number of windows,
    and C being the number of bins.

    if x is the input:

    - x[...,0] represent the bin probabilities.
    - x[...,1] represent the bin adjustment, i.e. where in the bin the frequency is.
    - x[...,2] represent phases, with a period of 1.

    For every window and bin, an activation wave is calculated based on the frequency and phase.
    The activation wave is a cosine wave transformed to [0,1] and then squared.
    For each window the total activation wave is calculated as the weighted sum according to the probabilities.
    The activation is differentiable and should be used to calculate the loss.

    A max-bin activation is also calculated in the same way, except that instead of weighted sum it only cares about the most probable bin in each window.
    This max-bin activation is then used to create a binary grid-mask. The grid-mask is not differentiable.
    """

    def __init__(
        self,
        window_size: int,
        min_freq: float,
        bins_per_octave: int = 3,
        octaves: int = 4,
        gradient_spread=0.2,
        sharp_activations=False,
        half_crossfade_frames=0,
        activation_cutoff: int | None = None,
        wave_smoothing_sigma=3,
        wave_smoothing_kernel_size=15,
    ) -> None:

        super().__init__()
        self.window_size = window_size
        self.min_freq = min_freq

        self.bins_per_octave = bins_per_octave
        self.octaves = octaves
        self.bin_size = 2 ** (1 / bins_per_octave)

        self.n_bins = bins_per_octave * octaves
        bin_minimums = min_freq * 2 ** (
            torch.arange(self.n_bins) / self.bins_per_octave
        )
        # self.register_buffer("bin_minimums", bin_minimums)
        self.register_buffer("bin_minimums", bin_minimums, persistent=False)
        self.max_pool = nn.MaxPool1d(9, 1, 4)
        self.activation_cutoff = activation_cutoff
        self.gradient_spread = gradient_spread
        self.sharp_activations = sharp_activations
        self.half_crossfade_frames = half_crossfade_frames

        assert wave_smoothing_kernel_size % 2 == 1

        kernel = _make_gaussian_kernel(
            2, wave_smoothing_kernel_size, wave_smoothing_sigma
        )

        self.register_buffer("wave_smoothing_kernel", kernel, persistent=False)
        self.wave_smoothing_padding = wave_smoothing_kernel_size // 2

    def input_shape(self, windows):
        return (windows, self.n_bins, 3)

    @staticmethod
    def probabilities(x):
        # ..., 3 -> ...
        return torch.softmax(x[..., 0], dim=-1)

    @staticmethod
    def log_probabilities(x):
        # ..., 3 -> ...
        return torch.log_softmax(x[..., 0], dim=-1)

    def gradient_spread_weighted_sum(self, values, weights, spread):
        # Expects values to be on the form (..., C, t)
        spread_weights = (1 - spread) * weights + spread / self.n_bins

        activation = torch.sum(values * weights[..., None], dim=-2)
        spread_activation = torch.sum(values * spread_weights[..., None], dim=-2)

        return spread_activation + (activation - spread_activation).detach()

    def adjustments(self, x):
        # ..., 3 -> ...
        return 1 + torch.sigmoid(x[..., 1]) * (self.bin_size - 1)

    @staticmethod
    def phases(x):
        # ..., 3 -> ...
        return x[..., 2]

    def bin_frequencies(self, x):
        """Frequencies in peak/frame"""
        # ..., 3 -> ...
        return self.bin_minimums * self.adjustments(x)

    def _compute_mask_and_waves(
        self,
        frame_phase: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes the (B, W, C, t) frame_phase tensor and (B, W, C) probabilities.

        Returns (B, T) mask and (B, T, 2) waves.
        """

        device = frame_phase.device

        # ml_bin : (B, W)
        ml_bin = probabilities.argmax(dim=-1)

        b = torch.arange(ml_bin.shape[0], device=device)[:, None]  # (B,1)
        w = torch.arange(ml_bin.shape[1], device=device)[None, :]  # (1,W)

        ml_bin_phase = frame_phase[b, w, ml_bin, :]
        cos_waves = torch.cos(ml_bin_phase)
        cos_waves = crossfade_windows(cos_waves, 0)

        sin_waves = torch.sin(ml_bin_phase)
        sin_waves = crossfade_windows(sin_waves, 0)

        # B x 2 x T
        waves = torch.stack([cos_waves, sin_waves], dim=-2)

        waves = torch.nn.functional.conv1d(
            waves,
            self.wave_smoothing_kernel,
            padding=self.wave_smoothing_padding,
            groups=2,
        )
        waves = waves.transpose(-1, -2)

        ml_bin_cosine = waves[..., 0]
        mask = ml_bin_cosine == self.max_pool(ml_bin_cosine)
        if self.activation_cutoff is not None:
            mask = mask & (ml_bin_cosine > self.activation_cutoff)

        mask[:, 0] = 0
        mask[:, -1] = 0

        return mask.detach(), waves.detach()

    def forward(self, x: torch.Tensor, spread_gradient: bool = True) -> GridOutput:
        # Assume features x is of shape (B, W, C, 3) and return something of shape (B, T)
        # where W is the number of windows and C is self.n_bins
        device = x.device

        expanded_window_size = self.window_size + self.half_crossfade_frames * 2

        with torch.autocast(device_type=x.device.type, enabled=False):
            frame_phase = (
                (
                    (
                        torch.arange(expanded_window_size, device=device)
                        - self.half_crossfade_frames
                    )
                    * self.bin_frequencies(x)[..., None]
                    + self.phases(x)[..., None]
                )
                * 2
                * torch.pi
            )
            mask, waves = self._compute_mask_and_waves(
                frame_phase[
                    ...,
                    self.half_crossfade_frames : (-self.half_crossfade_frames or None),
                ],
                self.probabilities(x),
            )

        # y : (B, W, C, t_1)
        y = torch.cos(frame_phase)
        y = (y + 1) / 2  # Transform y to [0, 1]
        if self.sharp_activations:
            y = y**2
        # y : (B, W, C, t_1)

        if spread_gradient and torch.is_grad_enabled():
            activation = self.gradient_spread_weighted_sum(
                y, self.probabilities(x), self.gradient_spread
            )
        else:
            activation = torch.sum(y * self.probabilities(x)[..., None], dim=-2)

        # activation : (B, W, t_1)

        activation = crossfade_windows(activation, self.half_crossfade_frames)
        # activation : (B, T)

        return GridOutput(
            activation=activation,
            features=x,
            mask=mask,
            waves=waves,
        )

    def advance(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, W, C, 3)
        y = x.clone()
        # y[..., 2] = x[..., 2]
        y[..., 2] = x[..., 2] + self.bin_frequencies(x) * self.window_size
        return y


class RecallLoss(nn.Module):
    """
    Calculate the (slightly offset) mean of the activation function, evaluated at the mask target.
    """

    def __init__(self, downbeat_weight=1.0, exponent=1) -> None:
        super().__init__()
        self.downbeat_weight = downbeat_weight
        self.exponent = exponent

    def forward(self, x, target, target_downbeat=None):
        # x, target : (B, T)
        # divide by (1 + sum) to avoid NaNs when there are no targets.
        # can't do early return because of batching
        # could perhaps do a is_zero mask and then a torch.where to avoid NaNs?

        mask = target
        if target_downbeat is not None:
            weights = target + target_downbeat * (self.downbeat_weight - 1)
        else:
            weights = target

        return torch.mean(
            (((1 - x) * mask) ** self.exponent * weights).sum(dim=-1)
            / (1 + weights.sum(dim=-1))
        )


class FrequencyLoss(nn.Module):
    def __init__(
        self,
        grid_model: WindowedGrid,
        scale: float = 1e-4,
    ) -> None:
        super().__init__()
        self.grid_model = grid_model
        self.scale = scale

    def forward(self, x: torch.Tensor):
        probs = self.grid_model.probabilities(x)
        freqs = self.grid_model.bin_frequencies(x)
        return torch.mean(self.scale * probs * freqs)


class WaveMeanLoss(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor):
        # x : (B, T, 2)
        return torch.mean(torch.mean(x[..., 0], dim=-1) ** 2) * self.scale


class LowProbLoss(nn.Module):
    def __init__(self, grid_model: WindowedGrid, scale) -> None:
        super().__init__()
        self.grid_model = grid_model
        self.scale = scale

    def forward(self, x: torch.Tensor):
        """x is a feature tensor"""
        # x : (..., C, 3) -> ()
        return self.scale * (1 / self.grid_model.probabilities(x)).mean()


# TODO improve
class WindowedGridRegularizationLoss(nn.Module):
    """
    A metric of regularization loss for WindowedGrid features.
    The regularization tries to keep the guesses consistent over time.
    It does that by comparing every window to what it expected based on the previous window.
    The discrepancy between these are calculated, and the total/mean for all discrepancies is reported as loss.

    The complicated part is computing the discrepancy between two windows.

    """

    def __init__(
        self,
        grid_model: WindowedGrid,
        scale=1e-1,
        frequency_log_step: float = math.log(1.05),
        phase_factor: float = 1,
        max_pointwise_distance=2,
        exponent: float | int = 1,
        sinkhorn_eps=1e-4,
        sinkhorn_reg=1,
        sinkhorn_max_iter=10,
        probability_spread=0.1,
    ) -> None:
        super().__init__()
        self.frequency_log_step = frequency_log_step
        self.phase_factor = phase_factor
        self.scale = scale
        self.grid_model = grid_model
        self.exponent = exponent
        self.max_pointwise_distance = max_pointwise_distance
        self.sinkhorn = SinkhornDistance(sinkhorn_eps, sinkhorn_reg, sinkhorn_max_iter)
        self.probability_spread = probability_spread

    def tempo_phase_distance(self, f1, p1, f2, p2) -> torch.Tensor:
        """
        Shape (...) -> (...)
        """
        # TODO improve this
        window_size = self.grid_model.window_size

        f_diff = (
            (torch.log(f2) - torch.log(f1)) / (window_size * self.frequency_log_step)
        ).abs()

        p_diff = (
            (
                (1 - torch.cos(2 * torch.pi * (p2 - p1))) ** 2
                + torch.sin(2 * torch.pi * (p2 - p1)) ** 2
            )
            * self.phase_factor
            / (f1 * window_size)
        )

        return f_diff + p_diff

    def distance_matrix(self, x1, x2) -> torch.Tensor:
        # TODO think about this distance, it should use logarithmic scaling of frequency and frequency-based scaling for phase.
        # Simplest solution is euclidian distance
        f1 = self.grid_model.bin_frequencies(x1)
        p1 = self.grid_model.phases(x1)

        f2 = self.grid_model.bin_frequencies(x2)
        p2 = self.grid_model.phases(x2)

        distance_matrix = self.tempo_phase_distance(
            f1[..., :, None],
            p1[..., :, None],
            f2[..., None, :],
            p2[..., None, :],
        )

        # Make sure that the distance is clamped, while keeping linear behaviour at low distances
        distance_matrix = (
            (torch.sigmoid(distance_matrix / self.max_pointwise_distance) - 0.5)
            * 2
            * self.max_pointwise_distance
        )
        return distance_matrix

    def feature_distance(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        extra_uniform: None | float = None,
    ) -> torch.Tensor:
        """
        (..., C, 3) -> (...)
        Calculate a measure of distance between two single-window latent representations x1 and x2.
        x1 and x2 should be of shape (..., C, 3)

        Feature distance is based on the Sinkhorn algorithm to solve an optimal transport problem.
        """
        # First, distance matrix for each pair of points in x1 and x2
        distance_matrix = self.distance_matrix(x1, x2)

        prob1 = self.grid_model.probabilities(x1)
        prob2 = self.grid_model.probabilities(x2)

        if self.training and extra_uniform is not None and torch.is_grad_enabled():

            def add_uniform(p):
                return p * (1 - extra_uniform) + extra_uniform / p.shape[-1]

            prob1 = add_uniform(prob1)
            prob2 = add_uniform(prob2)

        _, cost = self.sinkhorn(distance_matrix, prob1, prob2)

        return cost

    def forward(
        self,
        x,
        reduce="mean",
        extra_uniform: None | float = None,
    ):
        """
        shape (B, W, C, 3) -> ()
        """
        if x.shape[-3] == 1:
            if reduce == "mean":
                return torch.zeros(1, device=x.device, dtype=x.dtype)
            elif reduce is None:
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        y = self.grid_model.advance(x[..., :-1, :, :])
        window_distances = self.feature_distance(
            y,
            x[..., 1:, :, :],
            extra_uniform=extra_uniform or self.probability_spread,
        )  # (B, W-1)

        if reduce == "mean":
            return self.scale * torch.mean(window_distances**self.exponent)
        elif reduce is None:
            return self.scale * window_distances**self.exponent


# Taken from https://chulminator.github.io/posts/An-introduction-to-the-Sinkhorn-Algorithm-efficient-optimal-transport-solutions-Copy/
# Author: Chulminator
# Adapted to handle multi-dimensional stuff
# TODO fix this
class SinkhornDistance(nn.Module):
    def __init__(self, eps, reg, max_iter):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reg = reg

    def forward(self, cost_matrix, source, target):
        # source: (..., N)
        # target: (..., K)
        # cost_matrix: (, ..., K, N)
        M = torch.exp(-cost_matrix / self.reg)
        u = torch.ones_like(source)  # size of u: N
        v = torch.ones_like(target)  # size of v: K

        err = 1
        ii = 0

        P = torch.diag_embed(u) @ M @ torch.diag_embed(v)
        P_prev = torch.clone(P)
        for ii in range(self.max_iter):
            ii += 1
            u = source / (M @ v.unsqueeze(-1)).squeeze(-1)
            v = target / (M.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1)
            P = torch.diag_embed(u) @ M @ torch.diag_embed(v)

            err = torch.linalg.norm(P_prev - P, ord="fro", dim=(-1, -2))
            if torch.all(err < self.eps):
                break
            P_prev = torch.clone(P)
        min_cost = torch.sum(P * cost_matrix, dim=(-1, -2))
        return P, min_cost


def crossfade_windows(x, edge_influence):
    """
    A replacement of
        x = rearrange(x, "b w t -> b (w t)")
    but with some crossfade_between windows.
    This also means that the windows in x are expected to be of size t_w + 2 * overlap_frames, to

    Is this actually useful at all? Probably not? But good tech to have
    """

    if edge_influence == 0:
        return rearrange(x, "b w t -> b (w t)")

    x = x.clone()
    B, W, t_w = x.shape

    fade_duration = 2 * edge_influence

    fadein = torch.ones(size=(t_w,), device=x.device)
    fadein[:fade_duration] = torch.arange(fade_duration) / fade_duration

    fadeout = torch.ones(size=(t_w,), device=x.device)
    fadeout[-fade_duration:] = 1 - (torch.arange(fade_duration)) / fade_duration

    x[..., 1:, :] = x[..., 1:, :] * fadein
    x[..., :-1, :] = x[..., :-1, :] * fadeout

    pure_size = t_w - 2 * fade_duration

    res = torch.zeros(
        size=(B, W * (pure_size + fade_duration) + fade_duration), device=x.device
    )

    for i in range(W):
        start_frame = i * (t_w - fade_duration)
        res[..., start_frame : start_frame + t_w] += x[..., i, :]

    res = res[..., edge_influence:-edge_influence]
    return res


def _make_gaussian_kernel(channels, kernel_size, sigma):

    # build gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # shape → (C, 1, K)
    kernel = kernel.view(1, 1, -1).repeat(channels, 1, 1)
    return kernel
