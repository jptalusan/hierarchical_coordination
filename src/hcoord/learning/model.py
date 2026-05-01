"""Tiny MLP scorer for (vehicle, request) candidates.

Two-headed: feasibility logit + cost regression. The dispatcher uses the
combined score (predicted-feasible + predicted-cheap) to pick the argmin
vehicle without exhaustive (p, q) search across the whole fleet.

The architecture is deliberately small (3-layer MLP, ~64-dim hidden) — the
input is 21 standardized scalars; we don't need depth, we need a clean
ranking signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ScorerConfig:
    n_features: int
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    dropout: float = 0.0


class InsertionScorer(nn.Module):
    """Two-headed MLP: feasibility logit + log-cost regression."""

    def __init__(self, cfg: ScorerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        layers: list[nn.Module] = [nn.Linear(cfg.n_features, cfg.hidden_dim), nn.ReLU()]
        for _ in range(cfg.n_hidden_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
        self.trunk = nn.Sequential(*layers)
        self.feasibility_head = nn.Linear(cfg.hidden_dim, 1)
        # Predict log1p(cost). Cost is non-negative; log scaling matches the
        # heavy-ish tail seen in collected data (cost range 2-89 minutes).
        self.cost_head = nn.Linear(cfg.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (feasibility_logit, log1p_cost) per row."""
        h = self.trunk(x)
        return self.feasibility_head(h).squeeze(-1), self.cost_head(h).squeeze(-1)

    @torch.no_grad()
    def score(self, x: torch.Tensor, infeasibility_penalty: float = 1e6) -> torch.Tensor:
        """Lower-is-better score: predicted cost, penalized when predicted infeasible.

        `infeasibility_penalty` is added when the feasibility logit is < 0
        (i.e., probability < 0.5). The dispatcher picks `argmin(score)`.
        """
        feas_logit, log1p_cost = self.forward(x)
        cost_pred = torch.expm1(log1p_cost).clamp_min(0.0)
        penalty = (feas_logit < 0).float() * infeasibility_penalty
        return cost_pred + penalty
