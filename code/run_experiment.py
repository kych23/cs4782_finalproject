"""
Single-run entry point.

Example usage:
    # LoRA run on SST-2 with rank 8
    python code/run_experiment.py --task sst2 --mode lora --rank 8 --seed 42

    # Full fine-tuning baseline on RTE
    python code/run_experiment.py --task rte --mode full --seed 42
"""

import argparse
import json
import os
import sys

import torch
from transformers import RobertaTokenizer

# Allow imports from code/ when called from the repo root
sys.path.insert(0, os.path.dirname(__file__))

from config import TASK_CONFIGS
from data import get_dataloaders
from model import build_full_finetune_model, build_lora_model, verify_lora_model
from train import train
from utils import count_total_params, count_trainable_params, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA / Full fine-tuning experiment")
    parser.add_argument("--task",   required=True, choices=["sst2", "qnli", "rte"])
    parser.add_argument("--mode",   required=True, choices=["lora", "full"])
    parser.add_argument("--rank",   type=int, default=8,
                        help="LoRA rank r (ignored for --mode full)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="LoRA alpha (default: 2 * rank)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for results.json and best_model.pt. "
                             "Defaults to results/{task}_{mode}_r{rank}/")
    parser.add_argument("--verify", action="store_true",
                        help="Run LoRA sanity checks before training")
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

    if args.output_dir is None:
        if args.mode == "lora":
            args.output_dir = f"results/{args.task}_lora_r{args.rank}"
        else:
            args.output_dir = f"results/{args.task}_full"

    set_seed(args.seed)

    print(f"Task: {args.task} | Mode: {args.mode} | Device: {device}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    if args.mode == "lora":
        alpha = args.alpha if args.alpha is not None else 2 * args.rank
        model = build_lora_model(task_config, rank=args.rank, alpha=alpha)
        if args.verify:
            verify_lora_model(model, rank=args.rank, device=device)
            model.zero_grad()
        learning_rate = task_config.lora_lr
    else:
        model = build_full_finetune_model(task_config)
        learning_rate = task_config.full_ft_lr

    trainable = count_trainable_params(model)
    total = count_total_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    train_loader, val_loader = get_dataloaders(task_config, tokenizer, task_config.batch_size)

    results = train(
        model=model,
        task_config=task_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        output_dir=args.output_dir,
    )

    # Augment results with run metadata and save
    metadata = {
        "task": args.task,
        "mode": args.mode,
        "rank": args.rank if args.mode == "lora" else None,
        "alpha": (args.alpha if args.alpha is not None else 2 * args.rank)
                 if args.mode == "lora" else None,
        "seed": args.seed,
        "trainable_params": trainable,
        "total_params": total,
    }
    metadata.update(results)

    out_path = os.path.join(args.output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nBest val accuracy: {results['best_val_acc']:.4f}")
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
