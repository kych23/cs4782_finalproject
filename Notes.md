### Paper

https://arxiv.org/abs/2106.09685

### 2-page report summary

https://docs.google.com/document/d/1KpMVNg_HShuOFuITlZTrP5CjQd3VAM5w01bWWs8ZUYg/edit?usp=sharing

### Project Selection Submission

https://docs.google.com/document/d/1qfKQviobjutXVH3Scg7LJzD9_0iAjYfAdWCnrzCRPuo/edit?tab=t.0

### Notes from Ed Discussions

#### What you must implement yourself

- The LoRA module must be coded from scratch — you cannot call HuggingFace's PEFT/LoRA library or any existing LoRA implementation. This was explicitly cited as a bad example by staff.
- You should reproduce the paper from scratch mostly. Using an existing GitHub repo as your base is discouraged unless you get explicit instructor/TA sign-off first.
- You are implementing the _core feature_ of the paper, which for LoRA means the low-rank decomposition and its injection into attention layers — that is the non-negotiable from-scratch component.

#### What you are allowed to use

- Pre-trained model checkpoints (e.g., loading RoBERTa-base from HuggingFace) are fine, since fine-tuning a pretrained model is the entire point.
- Standard PyTorch primitives and libraries are fine, as long as their use does not make the LoRA implementation itself trivial.
- Existing GLUE benchmark loading utilities are fine — you are not expected to re-implement data pipelines.

#### Scope of experiments

- Exact numerical match to the paper is not required — getting close is sufficient.
- You do not need to replicate every experiment; select those that best demonstrate the core contribution.
- Your proposed scope of SST-2, QNLI, and RTE is well-aligned with this.
- Exploring additional hyperparameters or rank variations (your planned `r ∈ {4, 8, 16}` sweep) is explicitly welcomed.

#### What we used AI for

- config.py
- utils.py
- verify_lora_model in model.py

### order of running

RTE --> SST-2 --> QNLI

### Extra

- Our results for RTE are not as high as the paper because the paper pre-trained on MNLI benchmark first.
  - "Following Liu et al. (2019), we initialize the LoRA modules to our best MNLI checkpoint when adapting to MRPC, RTE, and STS-B, instead of the usual initialization"
  - We started pre-training from random Gaussian initialization

- RTE LoRA took 4 hours to train based on original hyperparameters from paper --> might need to lower number of epochs

Context

     Re-implement LoRA (Low-Rank Adaptation, arxiv 2106.09685) from scratch using PyTorch + HuggingFace RoBERTa-base. Compare LoRA against full fine-tuning on SST-2, QNLI, and RTE (GLUE). No PEFT/LoRA libraries allowed. Code goes in an empty code/ directory.

     ---
     File Structure

     code/
     ├── config.py          # TaskConfig dataclass + TASK_CONFIGS dict
     ├── utils.py           # set_seed, param counters, EpochTimer, PeakMemoryTracker
     ├── lora.py            # LoRALinear, inject_lora(), freeze_base_weights()
     ├── model.py           # build_lora_model(), build_full_finetune_model()
     ├── data.py            # get_dataloaders() — task-agnostic GLUE loading
     ├── train.py           # train_one_epoch(), evaluate(), train()
     ├── run_experiment.py  # CLI entry: single (task, mode, rank) run
     ├── run_sweep.py       # iterates all 12 combinations, calls run_experiment.py
     └── evaluate.py        # reload checkpoint + re-evaluate (standalone)

     Key Design Rationale (for report)

     - B=0 init: Ensures training starts at the exact pretrained model output — identical to full fine-tuning's starting point. Critical for fair comparison.
     - alpha/r scaling: Keeps the magnitude of the LoRA update stable across different ranks.
     - Inject via attribute replacement: Transparent to HuggingFace internals; RobertaSelfAttention.forward() calls self.query(x) dynamically so swapping the attribute is sufficient.
     - Freeze-all-then-unfreeze: Safer and more auditable than trying to enumerate what to freeze.
