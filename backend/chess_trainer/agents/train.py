"""Train the policy network.

uv run python -m chess_trainer.agents.train --name resnet_main

Loss = weighted cross-entropy against the master move over the legal moves
       + 0.25 * MSE between the value head and the game result (when the head is on).
The best checkpoint by val top-1 is written to data/checkpoints/<name>.pt.
"""

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from chess_trainer import config
from chess_trainer.agents.dataset import PositionArrays, batch_tensors, build_arrays
from chess_trainer.agents.model import PolicyNet, legal_logits

VALUE_LOSS_WEIGHT = 0.25


def _run(
    model: PolicyNet,
    arrays: PositionArrays,
    order: np.ndarray,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    """One pass. Returns (mean loss per position, top-1 accuracy in percent)."""
    training = optimizer is not None
    model.train(training)
    total_loss, hits, seen = 0.0, 0, 0
    n_batches = (len(order) + batch_size - 1) // batch_size
    for step in range(n_batches):
        batch = batch_tensors(arrays, order[step * batch_size : (step + 1) * batch_size])
        with torch.set_grad_enabled(training):
            out = model(batch["x"])
            logits = legal_logits(out.from_to, out.promo, batch["legal_ft"], batch["legal_promo"])
            cross_entropy = F.cross_entropy(logits, batch["target"], reduction="none")
            loss = (cross_entropy * batch["weight"]).sum() / batch["weight"].sum()
            if out.value is not None:
                loss = loss + VALUE_LOSS_WEIGHT * F.mse_loss(out.value, batch["value"])
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        size = len(batch["target"])
        total_loss += loss.item() * size
        hits += (logits.argmax(dim=1) == batch["target"]).sum().item()
        seen += size
        if progress and (step + 1) % 250 == 0:
            progress(f"  step {step + 1:,}/{n_batches:,}  loss {total_loss / seen:.4f}")
    return total_loss / seen, 100 * hits / seen


def train(
    positions: pd.DataFrame,
    name: str,
    out_dir: Path | None = None,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    result_weighted: bool = True,
    value_head: bool = True,
    patience: int = 2,
    seed: int = config.SEED,
    threads: int | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    out_dir = Path(out_dir or config.CHECKPOINT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if threads:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    started = time.perf_counter()
    train_arrays = build_arrays(positions[positions["split"] == "train"], result_weighted)
    val_arrays = build_arrays(positions[positions["split"] == "val"], result_weighted)
    log(
        f"{name}: {len(train_arrays):,} train and {len(val_arrays):,} val positions encoded "
        f"in {time.perf_counter() - started:.0f}s; result_weighted={result_weighted}, "
        f"value_head={value_head}, seed={seed}, threads={torch.get_num_threads()}"
    )

    model = PolicyNet(value_head=value_head)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (len(train_arrays) + batch_size - 1) // batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * steps_per_epoch)
    val_order = np.arange(len(val_arrays))

    history: dict = {
        "name": name,
        "seed": seed,
        "git_sha": config.git_sha(),
        "params": sum(p.numel() for p in model.parameters()),
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "result_weighted": result_weighted,
            "value_head": value_head,
            "patience": patience,
        },
        "epochs": [],
        "best_epoch": 0,
        "best_val_top1": -1.0,
        "stopped_early": False,
    }
    for epoch in range(1, epochs + 1):
        tick = time.perf_counter()
        order = rng.permutation(len(train_arrays))
        train_loss, train_top1 = _run(
            model, train_arrays, order, batch_size, optimizer, scheduler, progress=log
        )
        val_loss, val_top1 = _run(model, val_arrays, val_order, 1024)
        minutes = (time.perf_counter() - tick) / 60
        history["epochs"].append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "train_top1": round(train_top1, 2),
                "val_loss": round(val_loss, 4),
                "val_top1": round(val_top1, 2),
                "minutes": round(minutes, 2),
            }
        )
        improved = val_top1 > history["best_val_top1"]
        log(
            f"epoch {epoch}/{epochs}  train loss {train_loss:.4f}  val loss {val_loss:.4f}  "
            f"val top-1 {val_top1:.2f}%  ({minutes:.1f} min){'  *best*' if improved else ''}"
        )
        if improved:
            history["best_epoch"], history["best_val_top1"] = epoch, round(val_top1, 2)
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model.config,
                    "epoch": epoch,
                    "val_top1": round(val_top1, 2),
                    "history": history,
                },
                out_dir / f"{name}.pt",
            )
        elif epoch - history["best_epoch"] >= patience:
            history["stopped_early"] = True
            log(f"early stop: no val top-1 improvement for {patience} epochs")
            break
    (out_dir / f"{name}.history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    log(f"best: epoch {history['best_epoch']}, val top-1 {history['best_val_top1']:.2f}%")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the policy network.")
    parser.add_argument("--name", default="resnet_main")
    parser.add_argument("--data", type=Path, default=config.PROCESSED_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--unweighted", action="store_true", help="ablation: no result weighting")
    parser.add_argument("--no-value-head", action="store_true", help="ablation: no value head")
    args = parser.parse_args()

    positions = pd.read_parquet(args.data / "positions.parquet")
    train(
        positions,
        name=args.name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        result_weighted=not args.unweighted,
        value_head=not args.no_value_head,
        patience=args.patience,
        seed=args.seed,
        threads=args.threads,
        log=lambda line: print(line, flush=True),
    )


if __name__ == "__main__":
    main()
