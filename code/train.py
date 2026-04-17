"""Training function for epoch loops, validation, checkpoint saving, and metric logging."""

import json
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from config import TaskConfig
from utils import EpochTimer, PeakMemoryTracker


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scheduler,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)  # clear before forward pass
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    return correct / total


def train(
    model: nn.Module,
    task_config: TaskConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    learning_rate: float,
    output_dir: str,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    no_decay_terms = ("bias", "LayerNorm.weight")
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(term in name for term in no_decay_terms):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )

    total_steps = len(train_loader) * task_config.num_epochs
    warmup_steps = int(0.06 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.to(device)

    best_val_acc = 0.0
    best_epoch = 0
    epoch_logs = []

    with PeakMemoryTracker() as mem:
        for epoch in range(1, task_config.num_epochs + 1):
            with EpochTimer() as timer:
                train_loss = train_one_epoch(
                    model, train_loader, optimizer, scheduler, device
                )
                val_acc = evaluate(model, val_loader, device)

            log = {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "epoch_time_sec": round(timer.elapsed, 2),
            }
            epoch_logs.append(log)
            print(
                f"Epoch {epoch}/{task_config.num_epochs} | "
                f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
                f"time={timer.elapsed:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # Query nvidia-smi for peak GPU memory (works on T4 and other GPUs where
    # torch.cuda.max_memory_allocated() returns 0 due to unified memory)
    gpu_memory_used_mb = 0
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if smi.returncode == 0:
            gpu_memory_used_mb = int(smi.stdout.strip().split("\n")[0])
    except Exception:
        pass

    results = {
        "best_val_acc": round(best_val_acc, 4),
        "best_epoch": best_epoch,
        "peak_memory_mb": round(mem.peak_memory_mb, 1),
        "gpu_memory_used_mb": gpu_memory_used_mb,
        "epoch_logs": epoch_logs,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
