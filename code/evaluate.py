"""
Standalone evaluation script — reload a checkpoint and compute val accuracy.

Usage:
    python code/evaluate.py \
        --checkpoint results/sst2_lora_r8/best_model.pt \
        --task sst2 \
        --mode lora \
        --rank 8
"""

import argparse
import os
import sys

import torch
from transformers import RobertaTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from config import TASK_CONFIGS
from data import get_dataloaders
from model import build_full_finetune_model, build_lora_model
from train import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--task",  required=True, choices=["sst2", "qnli", "rte"])
    parser.add_argument("--mode",  required=True, choices=["lora", "full"])
    parser.add_argument("--rank",  type=int, default=8,
                        help="LoRA rank used during training (ignored for --mode full)")
    parser.add_argument("--alpha", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    task_config = TASK_CONFIGS[args.task]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    _, val_loader = get_dataloaders(task_config, tokenizer, task_config.batch_size)

    if args.mode == "lora":
        alpha = args.alpha if args.alpha is not None else 2 * args.rank
        model = build_lora_model(task_config, rank=args.rank, alpha=alpha)
    else:
        model = build_full_finetune_model(task_config)

    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    acc = evaluate(model, val_loader, device)
    print(f"Val accuracy ({args.task}, {args.mode}): {acc:.4f}")


if __name__ == "__main__":
    main()
