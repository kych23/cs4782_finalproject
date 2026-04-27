import json
import os
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)

tasks = ["sst2", "qnli", "rte"]
titles = ["SST-2", "QNLI", "RTE"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Validation Accuracy Over Epochs: Full Fine-Tuning vs LoRA (r=8)", fontsize=13)

for ax, task, title in zip(axes, tasks, titles):
    with open(os.path.join(BASE, f"{task}_full/results.json")) as f:
        full_data = json.load(f)
    with open(os.path.join(BASE, f"{task}_lora_r8/results.json")) as f:
        lora_data = json.load(f)

    full_epochs = [e["epoch"] for e in full_data["epoch_logs"]]
    full_acc = [e["val_accuracy"] * 100 for e in full_data["epoch_logs"]]
    lora_epochs = [e["epoch"] for e in lora_data["epoch_logs"]]
    lora_acc = [e["val_accuracy"] * 100 for e in lora_data["epoch_logs"]]

    ax.plot(full_epochs, full_acc, label="Full Fine-Tuning", color="#4C72B0", linewidth=1.5)
    ax.plot(lora_epochs, lora_acc, label="LoRA (r=8)", color="#DD8452", linewidth=1.5)

    best_full_ep = full_data["best_epoch"]
    best_full_acc = full_data["best_val_acc"] * 100
    best_lora_ep = lora_data["best_epoch"]
    best_lora_acc = lora_data["best_val_acc"] * 100

    ax.scatter([best_full_ep], [best_full_acc], color="#4C72B0", zorder=5, s=60)
    ax.scatter([best_lora_ep], [best_lora_acc], color="#DD8452", zorder=5, s=60)

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Accuracy (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(BASE, "val_accuracy_curves.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
