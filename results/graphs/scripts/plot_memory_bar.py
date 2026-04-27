import json
import os
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(__file__)

tasks = ["sst2", "qnli", "rte"]
labels = ["SST-2", "QNLI", "RTE"]

full_mem = []
lora_mem = []

for task in tasks:
    with open(os.path.join(BASE, f"{task}_full/results.json")) as f:
        full_mem.append(json.load(f)["peak_memory_mb"] / 1024)
    with open(os.path.join(BASE, f"{task}_lora_r8/results.json")) as f:
        lora_mem.append(json.load(f)["peak_memory_mb"] / 1024)

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_full = ax.bar(x - width / 2, full_mem, width, label="Full Fine-Tuning", color="#4C72B0")
bars_lora = ax.bar(x + width / 2, lora_mem, width, label="LoRA (r=8)", color="#DD8452")

ax.set_ylabel("Peak GPU Memory (GB)")
ax.set_title("Peak GPU Memory Usage: Full Fine-Tuning vs LoRA (r=8)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 14)

for bar in bars_full:
    ax.annotate(f"{bar.get_height():.1f} GB",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
for bar in bars_lora:
    ax.annotate(f"{bar.get_height():.1f} GB",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

plt.tight_layout()
out = os.path.join(BASE, "memory_bar.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
