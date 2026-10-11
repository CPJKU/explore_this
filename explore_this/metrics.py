import warnings

import mir_eval.beat
import numpy as np
import torch
from torch import Tensor


def mask_recall(prediction: Tensor, target: Tensor, frame_leniency: int = 3) -> Tensor:
    """
    Assume prediction and reference are (..., T) masks.
    """

    kernel = torch.ones(1, 1, frame_leniency * 2 + 1, device=prediction.device).float()
    pred_conv = torch.nn.functional.conv1d(
        prediction.unsqueeze(-2).float(),  # (..., 1, T)
        kernel,  # (1,1,7)
        padding=frame_leniency,  # keep same length
    )

    pred_conv = pred_conv.squeeze(-2)  # back to (..., T)

    overlap = (target.float() * (pred_conv == 1)).sum(dim=-1)
    target_count = target.sum(dim=-1)

    recall = torch.where(
        target_count > 0, overlap / target_count, target_count
    )  # Trick to not have to init extra 0-tensor, utilize the fact that targets will be 0 there!

    return recall


def f_measure(
    prediction: Tensor,
    target: Tensor,
    frame_leniency: int = 3,
    beta=1.0,
) -> Tensor:
    """
    Assume prediction and reference are (..., T) masks.
    Return (...) of floats
    """
    recall = mask_recall(prediction, target, frame_leniency)
    precision = mask_recall(target, prediction, frame_leniency)

    res_defined = (1 + beta**2) * precision * recall / ((beta**2) * precision + recall)

    res = torch.where(
        (precision > 0) & (recall > 0), res_defined, precision
    )  # Precision is equiavlent to 0 for last tensor argument

    return res


def _mask_to_times(mask: torch.Tensor, fps=50) -> np.ndarray | list:
    """Helper function for metrics that need a list of times instead of a mask"""
    if mask.ndim > 1:
        return [_mask_to_times(m, fps) for m in mask]

    times = torch.nonzero(mask, as_tuple=True)[0] / fps
    return times.cpu().numpy()


def continuity(prediction: torch.Tensor, target: torch.Tensor) -> tuple:
    """
    Assume prediction and target are (B, T) masks.
    Returns:
    tuple
    (CMLc, CMLt, AMLc, AMLt)
    of floats if without batching, otherwise tuple of lists.
    """
    prediction_times = _mask_to_times(prediction)
    target_times = _mask_to_times(target)

    if prediction.ndim == 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = mir_eval.beat.continuity(target_times, prediction_times)
        return res

    res = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p, t in zip(prediction_times, target_times):
            res.append(mir_eval.beat.continuity(t, p))

    # List of tuples -> tuple of lists
    res = tuple(map(list, zip(*res)))

    return res


def mask_mir_eval_f_measure(
    prediction: torch.Tensor, target: torch.Tensor, fps=50
) -> float | list[float]:
    """
    Implemented for comparison purposes, to ensure consistensy with my own f-measure algorithm.

    Assume prediction and target are (T) or (B, T) masks.
    """
    prediction_times = _mask_to_times(prediction, fps)
    target_times = _mask_to_times(target, fps)

    if prediction.ndim == 1:
        return mir_eval.beat.f_measure(target_times, prediction_times)

    return [
        mir_eval.beat.f_measure(t, p) for p, t in zip(prediction_times, target_times)
    ]
