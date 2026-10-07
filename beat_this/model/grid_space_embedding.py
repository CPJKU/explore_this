import torch

"""
A file for creating a canonical mapping between integers and integer-pairs (called states).
ORDERING: (meter, phase)
0: 1.1 
1: 2.1 
2: 2.2 
3: 3.1 
4: 3.2 
5: 3.3
"""


def _C(x: int | torch.Tensor) -> int | torch.Tensor:
    return x * (x + 1) // 2


def _C_inv_floored(x: torch.Tensor) -> torch.Tensor:
    return (-0.5 + (2 * x + 0.25) ** 0.5).int()


def get_state(index: torch.Tensor) -> torch.Tensor:
    # Assume x is integer tensor of shape P
    meter = _C_inv_floored(index)  # 0 indexed meter
    base_size = _C(meter)
    phase = index - base_size
    return torch.stack([meter + 1, phase + 1], dim=-1)


def get_index(state: torch.Tensor) -> torch.Tensor:
    # Left inverse of get_index
    # Undefined behaviour for impossible values for x
    # Assume x is integer tensor of shape P x 2
    meters = state[..., 0] - 1
    phases = state[..., 1] - 1
    return _C(meters) + phases


def space_size(max_meter: int) -> int:
    return _C(max_meter)  # type: ignore


def advance_state(x: torch.Tensor) -> torch.Tensor:
    # TODO think about if this needs to be detached as well
    y = x.clone()
    y[..., 1] = (x[..., 1] % x[..., 0]) + 1
    return y


def advance_index(x: torch.Tensor) -> torch.Tensor:
    return get_index(advance_state(get_state(x)))


def is_hit(x: torch.Tensor) -> torch.Tensor:
    return get_state(x)[..., 1] == 1
