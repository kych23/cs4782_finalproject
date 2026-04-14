### 2-page report summary

https://docs.google.com/document/d/1KpMVNg_HShuOFuITlZTrP5CjQd3VAM5w01bWWs8ZUYg/edit?usp=sharing

### Notes from Ed Discussions

#### What you must implement yourself

- The LoRA module must be coded from scratch — you cannot call HuggingFace's PEFT/LoRA library or any existing LoRA implementation. This was explicitly cited as a bad example by staff.
- You should reproduce the paper from scratch mostly. Using an existing GitHub repo as your base is discouraged unless you get explicit instructor/TA sign-off first.
- You are implementing the *core feature* of the paper, which for LoRA means the low-rank decomposition and its injection into attention layers — that is the non-negotiable from-scratch component.

#### What you are allowed to use

- Pre-trained model checkpoints (e.g., loading RoBERTa-base from HuggingFace) are fine, since fine-tuning a pretrained model is the entire point.
- Standard PyTorch primitives and libraries are fine, as long as their use does not make the LoRA implementation itself trivial.
- Existing GLUE benchmark loading utilities are fine — you are not expected to re-implement data pipelines.

#### Scope of experiments

- Exact numerical match to the paper is not required — getting close is sufficient.
- You do not need to replicate every experiment; select those that best demonstrate the core contribution.
- Your proposed scope of SST-2, QNLI, and RTE is well-aligned with this.
- Exploring additional hyperparameters or rank variations (your planned `r ∈ {4, 8, 16}` sweep) is explicitly welcomed.
