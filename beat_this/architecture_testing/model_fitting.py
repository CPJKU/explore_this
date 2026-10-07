from typing import Callable
from typing import LiteralString

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from beat_this.architecture_testing.artificial_data_generation import ArtificialDataset


class DirectFeatureModel(nn.Module):
    def __init__(self, features, forward_model) -> None:
        super().__init__()
        self.features = nn.Parameter(features.clone().detach())
        self.forward_model = forward_model

    def forward(self, x):
        B = x.shape[0]
        return self.forward_model(
            self.features.unsqueeze(0).expand((B, *self.features.shape))
        )


def fitModel(
    model: nn.Module,
    dataset: ArtificialDataset,
    loss: nn.Module,
    batches: int = 100,
    optimizer_args: dict | None = None,
    device="cpu",
):
    if optimizer_args is None:
        optimizer_args = {}
    model = model.to(device=device)  # type: ignore
    loss = loss.to(device=device)  # type: ignore

    optimizer = torch.optim.AdamW(model.parameters(), **optimizer_args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, batches)
    loss_record = []

    data_iter = iter(dataset)

    iterator = tqdm(range(batches))

    try:
        for _ in iterator:
            x, y = next(data_iter)
            x = x.to(device)
            y = y.to(device)

            l = loss(
                model.forward(x),
                y,
            )

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            scheduler.step()

            loss_record.append(l.detach().item())
            running_avg_loss = np.mean(loss_record[-50:])
            iterator.set_postfix({"loss": f"{running_avg_loss:.3f}"})
    except KeyboardInterrupt:
        print("Aborting training early")
    return loss_record
