from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .dataset import AcousticH5Dataset
from .models import build_model


@dataclass
class TrainConfig:
    model_name: str
    in_channels: int
    out_channels: int
    base_channels: int
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    lr_patience: int
    lr_factor: float
    early_stop_patience: int
    device: str
    train_h5: str
    val_h5: str
    input_repr: str          # "complex" or "magnitude"
    log_dir: str
    save_best_only: bool = True


def train_model(cfg: TrainConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Datasets / loaders
    train_ds = AcousticH5Dataset(cfg.train_h5, input_repr=cfg.input_repr)
    val_ds = AcousticH5Dataset(cfg.val_h5, input_repr=cfg.input_repr)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model
    model = build_model(
        name=cfg.model_name,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        base_channels=cfg.base_channels,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience, verbose=True
    )

    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = log_dir / "best.ckpt"

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [train]"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_ds)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.num_epochs} [val]"):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                val_loss_sum += loss.item() * x.size(0)

        val_loss = val_loss_sum / len(val_ds)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        dt = time.time() - t0
        print(
            f"Epoch {epoch}: train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}, time={dt:.1f}s"
        )

        # ранняя остановка
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if cfg.save_best_only:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.__dict__,
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    best_ckpt_path,
                )
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.early_stop_patience:
            print("Early stopping triggered.")
            break

    # если не save_best_only — сохрани последнюю модель
    if not cfg.save_best_only:
        last_ckpt_path = log_dir / "last.ckpt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "epoch": epoch,
                "val_loss": val_loss,
            },
            last_ckpt_path,
        )

    train_ds.close()
    val_ds.close()

    return {"history": history, "best_val_loss": best_val_loss}
