"""The policy network: a small ResNet with a convolutional from-to head."""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from chess_trainer.agents.encoding import N_PLANES, N_PROMO


class Output(NamedTuple):
    from_to: torch.Tensor  # (N, 4096), index = to_square * 64 + from_square
    promo: torch.Tensor  # (N, 32)
    value: torch.Tensor | None  # (N,) in [-1, 1], expected result for the side to move
    features: torch.Tensor  # (N, channels) pooled trunk features, used for retrieval


class Block(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        return F.relu(x + self.bn2(self.conv2(y)))


class PolicyNet(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 64, value_head: bool = True) -> None:
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels, "value_head": value_head}
        self.stem = nn.Sequential(
            nn.Conv2d(N_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        # One output plane per target square; each cell is a from-square.
        self.from_to = nn.Conv2d(channels, 64, 1)
        self.promo = nn.Linear(channels, N_PROMO)
        self.value = (
            nn.Sequential(
                nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, 1), nn.Tanh()
            )
            if value_head
            else None
        )

    def forward(self, x: torch.Tensor) -> Output:
        hidden = self.body(self.stem(x))
        features = hidden.mean(dim=(2, 3))
        value = self.value(features).squeeze(1) if self.value is not None else None
        return Output(self.from_to(hidden).flatten(1), self.promo(features), value, features)


def legal_logits(
    from_to: torch.Tensor, promo: torch.Tensor, legal_ft: torch.Tensor, legal_promo: torch.Tensor
) -> torch.Tensor:
    """Logit of each legal move: (N, L), with -inf where the move list is padded (-1).

    `legal_ft` and `legal_promo` hold each position's legal moves as index pairs."""
    logits = from_to.gather(1, legal_ft.clamp(min=0))
    promo_term = promo.gather(1, legal_promo.clamp(min=0))
    logits = logits + promo_term * (legal_promo >= 0)
    return logits.masked_fill(legal_ft < 0, float("-inf"))
