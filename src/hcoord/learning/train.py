"""Training loop + per-decision evaluation for the LLP scorer.

Loss: weighted sum of BCE-with-logits on feasibility + MSE on log1p(cost)
restricted to feasible rows. We log per-row metrics each epoch (feasibility
AUC-proxy via accuracy, cost MAE on feasible rows) and report the
per-decision top-1/top-3 vehicle-match accuracy on the test split — the
metric that actually matters for the dispatcher plug-in.

`train_scorer(...)` returns the trained model, the standardizer, and an
eval dict. `save_checkpoint` / `load_checkpoint` persist {model_state,
standardizer_arrays, config}.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hcoord.learning.dataset import PreparedSplit, Standardizer
from hcoord.learning.model import InsertionScorer, ScorerConfig


@dataclass
class TrainConfig:
    n_epochs: int = 30
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    feasibility_weight: float = 1.0
    cost_weight: float = 1.0
    seed: int = 0


def _make_loader(split: PreparedSplit, batch_size: int, shuffle: bool) -> DataLoader:
    X = torch.from_numpy(split.X)
    feas = torch.from_numpy(split.feasible.astype(np.float32))
    log1p_cost = torch.from_numpy(np.log1p(split.cost).astype(np.float32))
    return DataLoader(
        TensorDataset(X, feas, log1p_cost),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def _epoch_metrics(model: InsertionScorer, split: PreparedSplit) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(split.X)
        feas_logit, log1p_cost_pred = model(X)
        feas_pred = (feas_logit > 0).numpy()
        feas_acc = float((feas_pred == split.feasible).mean())
        feasible_idx = np.where(split.feasible)[0]
        if len(feasible_idx) > 0:
            cost_pred = torch.expm1(log1p_cost_pred).clamp_min(0.0).numpy()
            cost_mae = float(
                np.mean(np.abs(cost_pred[feasible_idx] - split.cost[feasible_idx]))
            )
        else:
            cost_mae = float("nan")
    return {"feasibility_acc": feas_acc, "cost_mae_feasible": cost_mae}


def _per_decision_metrics(
    model: InsertionScorer,
    split: PreparedSplit,
    k_values: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Per-decision quality metrics that measure what the dispatcher cares about.

    Top-k: does the model's top-k vehicle ranking contain the heuristic's
    argmin? With 60-120 vehicles per decision this is a strict metric.

    Mean / median regret: the actual figure of merit. For each decision, take
    the model's argmin (over predicted-feasible vehicles, by predicted cost)
    and look up that vehicle's TRUE cost. Compare to the optimal vehicle's
    true cost. The dispatcher's downstream behavior depends on this gap, not
    on exact rank match — many vehicles can be near-tied.

    `model_pick_infeasible_rate`: how often the model picks a vehicle that
    turns out to be infeasible. The plug-in (step 3) fall-back behavior
    should account for this.

    Decisions where no vehicle is feasible are excluded — no defensible
    argmin exists.
    """
    model.eval()
    with torch.no_grad():
        X = torch.from_numpy(split.X)
        scores = model.score(X).numpy()

    keys = np.array([f"{r}_{d}" for r, d in zip(split.run_id, split.decision_id)])
    unique_keys, group_idx = np.unique(keys, return_inverse=True)

    topk_hits: dict[int, list[bool]] = {k: [] for k in k_values}
    regrets: list[float] = []
    infeas_picks = 0
    n_decisions = 0

    for gi in range(len(unique_keys)):
        idx = np.where(group_idx == gi)[0]
        feas_in_group = split.feasible[idx]
        if not feas_in_group.any():
            continue
        n_decisions += 1
        feasible_local = idx[feas_in_group]
        true_argmin = feasible_local[np.argmin(split.cost[feasible_local])]
        true_min_cost = float(split.cost[true_argmin])
        order = idx[np.argsort(scores[idx])]
        model_pick = order[0]
        for k in k_values:
            topk_hits[k].append(true_argmin in order[:k])

        if not split.feasible[model_pick]:
            infeas_picks += 1
            # Fall back to the model's best feasible pick for regret.
            feasible_in_order = [i for i in order if split.feasible[i]]
            if not feasible_in_order:
                continue
            model_pick = feasible_in_order[0]
        regrets.append(float(split.cost[model_pick]) - true_min_cost)

    out: dict[str, float] = {}
    for k in k_values:
        out[f"top{k}_acc"] = (
            float(np.mean(topk_hits[k])) if topk_hits[k] else float("nan")
        )
    out["mean_regret_min"] = float(np.mean(regrets)) if regrets else float("nan")
    out["median_regret_min"] = float(np.median(regrets)) if regrets else float("nan")
    out["p95_regret_min"] = (
        float(np.quantile(regrets, 0.95)) if regrets else float("nan")
    )
    out["model_pick_infeasible_rate"] = (
        infeas_picks / n_decisions if n_decisions > 0 else float("nan")
    )
    return out


def train_scorer(
    train_split: PreparedSplit,
    test_split: PreparedSplit,
    *,
    train_cfg: TrainConfig | None = None,
    model_cfg: ScorerConfig | None = None,
    log_every: int = 5,
) -> tuple[InsertionScorer, dict[str, Any]]:
    """Train an InsertionScorer, return (model, eval_dict)."""
    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ScorerConfig(n_features=train_split.X.shape[1])
    if model_cfg.n_features != train_split.X.shape[1]:
        raise ValueError(
            f"model_cfg.n_features={model_cfg.n_features} but data has "
            f"{train_split.X.shape[1]} features"
        )

    torch.manual_seed(train_cfg.seed)
    model = InsertionScorer(model_cfg)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    feas_loss = nn.BCEWithLogitsLoss()
    cost_loss = nn.MSELoss()

    loader = _make_loader(train_split, train_cfg.batch_size, shuffle=True)
    history: list[dict[str, Any]] = []
    for epoch in range(train_cfg.n_epochs):
        model.train()
        epoch_total = 0.0
        epoch_feas = 0.0
        epoch_cost = 0.0
        for X, feas, log1p_cost in loader:
            optimizer.zero_grad()
            feas_logit, cost_pred = model(X)
            l_feas = feas_loss(feas_logit, feas)
            mask = feas > 0.5
            if mask.any():
                l_cost = cost_loss(cost_pred[mask], log1p_cost[mask])
            else:
                l_cost = torch.zeros(())
            loss = train_cfg.feasibility_weight * l_feas + train_cfg.cost_weight * l_cost
            loss.backward()
            optimizer.step()
            epoch_total += float(loss.detach()) * X.shape[0]
            epoch_feas += float(l_feas.detach()) * X.shape[0]
            epoch_cost += float(l_cost.detach()) * X.shape[0]

        n = len(loader.dataset)
        train_metrics = _epoch_metrics(model, train_split)
        test_metrics = _epoch_metrics(model, test_split)
        rec = {
            "epoch": epoch + 1,
            "loss": epoch_total / n,
            "feas_loss": epoch_feas / n,
            "cost_loss": epoch_cost / n,
            "train_feas_acc": train_metrics["feasibility_acc"],
            "train_cost_mae": train_metrics["cost_mae_feasible"],
            "test_feas_acc": test_metrics["feasibility_acc"],
            "test_cost_mae": test_metrics["cost_mae_feasible"],
        }
        history.append(rec)
        if (epoch + 1) % log_every == 0 or epoch == 0:
            print(
                f"epoch {epoch + 1:3d}/{train_cfg.n_epochs} | "
                f"loss {rec['loss']:.4f} (feas {rec['feas_loss']:.4f}, "
                f"cost {rec['cost_loss']:.4f}) | "
                f"test feas_acc {rec['test_feas_acc']:.3f}, "
                f"cost MAE {rec['test_cost_mae']:.2f} min",
                flush=True,
            )

    per_dec = _per_decision_metrics(model, test_split)
    final_metrics = {
        **_epoch_metrics(model, test_split),
        **per_dec,
        "n_train_rows": int(len(train_split.X)),
        "n_test_rows": int(len(test_split.X)),
    }
    return model, {"history": history, "final": final_metrics,
                   "train_cfg": asdict(train_cfg),
                   "model_cfg": asdict(model_cfg)}


def save_checkpoint(
    path: str | Path,
    model: InsertionScorer,
    standardizer: Standardizer,
    eval_dict: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": asdict(model.cfg),
            "feature_names": list(standardizer.feature_names),
            "standardizer_mean": standardizer.mean.tolist(),
            "standardizer_std": standardizer.std.tolist(),
            "eval": eval_dict,
        },
        path,
    )
    eval_path = path.with_suffix(".eval.json")
    with eval_path.open("w") as f:
        json.dump(eval_dict, f, indent=2, default=float)


def load_checkpoint(path: str | Path) -> tuple[InsertionScorer, Standardizer, dict[str, Any]]:
    blob = torch.load(path, weights_only=False)
    cfg = ScorerConfig(**blob["model_cfg"])
    model = InsertionScorer(cfg)
    model.load_state_dict(blob["model_state_dict"])
    model.eval()
    standardizer = Standardizer(
        mean=np.asarray(blob["standardizer_mean"], dtype=np.float32),
        std=np.asarray(blob["standardizer_std"], dtype=np.float32),
        feature_names=tuple(blob["feature_names"]),
    )
    return model, standardizer, blob["eval"]
