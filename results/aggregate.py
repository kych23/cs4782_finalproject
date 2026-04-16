"""
Aggregate all results JSONs into a summary table.

Usage (from repo root):
    python results/aggregate.py
"""

import glob
import json
import os


def main():
    pattern = os.path.join(os.path.dirname(__file__), "**", "results.json")
    paths = sorted(glob.glob(pattern, recursive=True))

    if not paths:
        print("No results.json files found. Run experiments first.")
        return

    rows = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
        rows.append({
            "task":       d.get("task", "?"),
            "mode":       d.get("mode", "?"),
            "rank":       str(d.get("rank", "—")) if d.get("rank") else "—",
            "val_acc":    f"{d.get('best_val_acc', 0):.4f}",
            "trainable":  f"{d.get('trainable_params', 0):,}",
            "total":      f"{d.get('total_params', 0):,}",
            "time_ep":    f"{sum(e['epoch_time_sec'] for e in d.get('epoch_logs', [])) / max(len(d.get('epoch_logs', [1])), 1):.1f}s",
            "peak_gpu":   f"{d.get('peak_gpu_mb', 0):.0f} MB",
        })

    # Sort: task, then full before lora, then rank ascending
    def sort_key(r):
        rank_val = int(r["rank"]) if r["rank"] != "—" else -1
        return (r["task"], 0 if r["mode"] == "full" else 1, rank_val)

    rows.sort(key=sort_key)

    # Print table
    header = ["Task", "Mode", "Rank", "Val Acc", "Trainable Params", "Total Params", "Time/Epoch", "Peak GPU"]
    col_keys = ["task", "mode", "rank", "val_acc", "trainable", "total", "time_ep", "peak_gpu"]
    widths = [max(len(header[i]), max(len(r[k]) for r in rows))
              for i, k in enumerate(col_keys)]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in widths) + " |"

    print(sep)
    print(fmt.format(*header))
    print(sep)
    for r in rows:
        print(fmt.format(*[r[k] for k in col_keys]))
    print(sep)


if __name__ == "__main__":
    main()
