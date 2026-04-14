# CS 4782 Final Project: LoRA: Low-Rank Adaptation of Large Language Models Re-Implementation

## 1) Introduction

This repository contains our CS 4782 final project, which attempts to re-implement key results from **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021).  
The paper's main contribution is a parameter-efficient fine-tuning method that injects low-rank trainable matrices into frozen pretrained model weights.

## 2) Chosen Result

We aim to reproduce the core LoRA claim that low-rank adaptation can match or approach full fine-tuning quality with significantly fewer trainable parameters.  
Primary target from the paper: parameter-efficiency/performance comparison reported in the LoRA experimental results (see paper tables/figures; specify exact table/figure here).

## 3) GitHub Contents

This repository is organized as follows (course-required scaffold):

- `README.md`: project snapshot and usage instructions.
- `code/`: re-implementation scripts, configs, and utilities.
- `data/`: dataset files or instructions for dataset download/preprocessing.
- `results/`: logs, figures, tables, and model outputs.
- `poster/`: final poster PDF used for in-class presentation.
- `report/`: final 2-page report PDF.
- `LICENSE`: project license.
- `.gitignore`: ignored files and directories.

## 4) Re-implementation Details

Our implementation follows LoRA by freezing the base model and training low-rank adapter parameters in selected linear layers.  
Core components include model fine-tuning code, dataset preprocessing, evaluation scripts, and metric tracking for comparison with reported paper results.

## 5) Reproduction Steps

To reproduce our re-implementation locally:

1. Clone this repository and create a Python environment (recommended: Python 3.10+).
2. Install dependencies from `code/requirements.txt` (or equivalent environment file).
3. Prepare datasets using instructions in `data/` (download + preprocessing).
4. Run training/evaluation scripts in `code/` with the provided configuration files.
5. Compare generated outputs in `results/` against target paper metrics.

Example command flow (update with your final scripts):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
python code/train.py --config code/configs/lora.yaml
python code/eval.py --checkpoint results/checkpoints/best.pt
```

Compute requirements: GPU recommended for training; CPU may be sufficient for smaller-scale sanity checks.

## 6) Results / Insights

Expected output is a reproduction-oriented comparison between LoRA and baseline fine-tuning in terms of task performance and trainable parameter count.  
Our final tables/plots and analysis are provided in `results/`, including notes on any discrepancies from the original paper.

## 7) Conclusion

This project evaluates whether LoRA's parameter-efficient adaptation behavior can be reproduced in our implementation setting.  
Key takeaways include empirical performance, resource trade-offs, and practical lessons from re-implementation.

## 8) References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., Wang, W., and Chen, W. (2021). _LoRA: Low-Rank Adaptation of Large Language Models_. arXiv:2106.09685.
- Add any additional tools, libraries, datasets, and implementation references used in this project.

## 9) Acknowledgements

This work was completed as part of Cornell Tech CS 4782 coursework.  
We thank the course staff and peers for feedback during the project, poster session, and final report review process.
