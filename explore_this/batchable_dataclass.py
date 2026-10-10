from abc import ABC
from dataclasses import fields

import torch


class BatchableDataclass(ABC):
    """
    Helper class to enable dataclasses with tensors to be nicely batchable and shape manipulated.
    Can only be inherited by dataclasses.
    """

    def _apply_to_tensors(self, fn):
        data = {}

        for f in fields(self):  # type: ignore
            v = getattr(self, f.name)

            if isinstance(v, torch.Tensor):
                data[f.name] = fn(v)
            else:
                data[f.name] = v

        return type(self)(**data)

    def squeeze(self, dim: int):
        return self._apply_to_tensors(lambda t: t.squeeze(dim))

    def unsqueeze(self, dim: int):
        return self._apply_to_tensors(lambda t: t.unsqueeze(dim))

    def __getitem__(self, key):
        return self._apply_to_tensors(lambda t: t[key].unsqueeze(0))

    def to(self, *args, **kwargs):
        return self._apply_to_tensors(lambda t: t.to(*args, **kwargs))

    def hasnan(self):
        for f in fields(self):  # type: ignore
            v = getattr(self, f.name)
            if isinstance(v, torch.Tensor) and torch.any(torch.isnan(v)):
                return True

        return False
