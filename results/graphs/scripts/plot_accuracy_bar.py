import json
import os
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(__file__)

tasks = ["sst2", "qnli", "rte"]
labels = ["SST-2", "QNLI", "RTE"]

full_acc = []
lora_acc = []

for task in tasks:
    with open(os.path.join(BASE, f"{task}_full/results.json")) as f:
        full_acc.append(json.load(f)["best_val_acc"] * 100)
    with open(os.path.join(BASE, f"{task}_lora_r8/results.json")) as f:
        lora_acc.append(json.load(f)["best_val_acc"] * 100)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_full = ax.bar(x - width / 2, full_acc, width, label="Full Fine-Tuning", color="#4C72B0")
bars_lora = ax.bar(x + width / 2, lora_acc, width, label="LoRA (r=8)", color="#DD8452")

ax.set_ylabel("Validation Accuracy (%)")
ax.set_title("Best Validation Accuracy: Full Fine-Tuning vs LoRA (r=8)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(50, 100)

for bar in bars_full:
    ax.annotate(f"{bar.get_height():.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
for bar in bars_lora:
    ax.annotate(f"{bar.get_height():.2f}%",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

plt.tight_layout()
out = os.path.join(BASE, "accuracy_bar.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
