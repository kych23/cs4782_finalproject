import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = os.path.dirname(__file__)

with open(os.path.join(BASE, "sst2_full/results.json")) as f:
    full_params = json.load(f)["trainable_params"]
with open(os.path.join(BASE, "sst2_lora_r8/results.json")) as f:
    lora_params = json.load(f)["trainable_params"]

labels = ["Full Fine-Tuning", "LoRA (r=8)"]
values = [full_params, lora_params]
colors = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(labels, values, color=colors, height=0.4)
ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters (log scale)")
ax.set_title("Trainable Parameter Count: Full Fine-Tuning vs LoRA (r=8)")

ax.xaxis.set_major_formatter(ticker.FuncFormatter(
    lambda x, _: f"{x/1e6:.0f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
))

for bar, val in zip(bars, values):
    label = f"{val/1e6:.2f}M" if val >= 1e6 else f"{val/1e3:.1f}K"
    ax.text(val * 1.3, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=11, fontweight="bold")

ax.set_xlim(1e4, 1e9)
plt.tight_layout()
out = os.path.join(BASE, "trainable_params.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
