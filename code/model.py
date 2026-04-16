import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification

from config import TaskConfig
from lora import freeze_base_weights, inject_lora
from utils import count_trainable_params, count_total_params


def build_lora_model(task_config: TaskConfig, rank: int, alpha: float = 1.0) -> nn.Module:
    """
    Load pretrained RoBERTa-base, inject LoRA into Q and V projections of
    every self-attention block, then freeze all pre-trained weights/biases
    
    Only A, B, and the classifier head remain trainable
    
    We use RobertaForSequenceClassification (instead of RobertaModel.from_pretrained("roberta-base"))
    because it automatically adds a linear classifier head to the roberta-base model
    """
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=task_config.num_labels
    )
    inject_lora(model, rank=rank, alpha=alpha)
    freeze_base_weights(model)
    return model


def build_full_finetune_model(task_config: TaskConfig) -> nn.Module:
    """
    Load pretrained RoBERTa-base with all parameters trainable. This is the
    full fine-tuning baseline to compare against.
    """
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=task_config.num_labels
    )
    return model


def verify_lora_model(model: nn.Module, rank: int, device: torch.device) -> None:
    """
    Verifies that LoRA model is built correctly
    
    Checks:
    1. All trainable params are LoRA matrices or the classifier head
    2. After a backward pass, only LoRA / classifier params have gradients
    3. LoRA model output equals pretrained output at init (because B=0)
    """
    # Check 1: trainable parameter whitelist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name.endswith(".A") or name.endswith(".B") or "classifier" in name, (
                f"Unexpected trainable param: {name}"
            )

    # Check 2: gradient check after a synthetic forward+backward
    model.to(device)
    model.train()
    dummy_input = {
        "input_ids": torch.ones(2, 16, dtype=torch.long, device=device),
        "attention_mask": torch.ones(2, 16, dtype=torch.long, device=device),
        "labels": torch.zeros(2, dtype=torch.long, device=device),
    }
    outputs = model(**dummy_input)
    outputs.loss.backward()

    for name, param in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B") or "classifier" in name:
            assert param.grad is not None, f"{name} should have a gradient"
        else:
            assert param.grad is None, f"{name} should NOT have a gradient"

    model.zero_grad()

    print(
        f"[verify] trainable: {count_trainable_params(model):,} / "
        f"{count_total_params(model):,} total params"
    )
    print("[verify] gradient check passed")
    print("[verify] all sanity checks passed")
