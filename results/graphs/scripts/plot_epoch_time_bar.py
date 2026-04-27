import json
import os
import numpy as np
import matplotlib.pyplot as plt

BASE = os.path.dirname(__file__)

tasks = ["sst2", "qnli", "rte"]
labels = ["SST-2", "QNLI", "RTE"]

full_time = []
lora_time = []

for task in tasks:
    with open(os.path.join(BASE, f"{task}_full/results.json")) as f:
        logs = json.load(f)["epoch_logs"]
        full_time.append(np.mean([e["epoch_time_sec"] for e in logs]))
    with open(os.path.join(BASE, f"{task}_lora_r8/results.json")) as f:
        logs = json.load(f)["epoch_logs"]
        lora_time.append(np.mean([e["epoch_time_sec"] for e in logs]))

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
bars_full = ax.bar(x - width / 2, full_time, width, label="Full Fine-Tuning", color="#4C72B0")
bars_lora = ax.bar(x + width / 2, lora_time, width, label="LoRA (r=8)", color="#DD8452")

ax.set_ylabel("Average Epoch Time (seconds)")
ax.set_title("Average Epoch Training Time: Full Fine-Tuning vs LoRA (r=8)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

for bar in bars_full:
    ax.annotate(f"{bar.get_height():.0f}s",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
for bar in bars_lora:
    ax.annotate(f"{bar.get_height():.0f}s",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

plt.tight_layout()
out = os.path.join(BASE, "epoch_time_bar.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
