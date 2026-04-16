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

     Dependency order (no circular imports):
     config.py → utils.py → lora.py → model.py → data.py → train.py → run_experiment.py ← run_sweep.py

     ---
     Step-by-Step Implementation

     Step 1 — config.py

     Define TaskConfig dataclass with fields:
     - task_name, num_labels, text_fields: tuple[str,...]
     - max_seq_len (128 for SST-2/RTE, 256 for QNLI)
     - num_epochs (3 for SST-2/QNLI, 10 for RTE — small dataset)
     - batch_size=16, learning_rate (2e-4 LoRA, 2e-5 full FT)

     TASK_CONFIGS = {
         "sst2": TaskConfig("sst2", 2, ("sentence",),           max_seq_len=128, num_epochs=3,  ...),
         "qnli": TaskConfig("qnli", 2, ("question", "sentence"), max_seq_len=256, num_epochs=3,  ...),
         "rte":  TaskConfig("rte",  2, ("sentence1","sentence2"),max_seq_len=128, num_epochs=10, ...),
     }

     Step 2 — utils.py

     - set_seed(seed) — torch, numpy, random, cuda
     - count_trainable_params(model) / count_total_params(model)
     - EpochTimer — context manager, records wall-clock time
     - PeakMemoryTracker — wraps torch.cuda.reset_peak_memory_stats() / torch.cuda.max_memory_allocated(); no-ops if no CUDA

     Step 3 — lora.py (core from-scratch component)

     LoRALinear(nn.Module)
     class LoRALinear(nn.Module):
         def __init__(self, original_linear, r, lora_alpha):
             # Copy weight/bias as frozen Parameters (requires_grad=False)
             self.weight = nn.Parameter(original_linear.weight.data.clone(), requires_grad=False)
             self.bias   = nn.Parameter(original_linear.bias.data.clone(),   requires_grad=False)
             # A: (r, in_features) — Kaiming uniform init to break symmetry
             # B: (out_features, r) — zero init so delta_W=0 at start (critical for stability)
             self.lora_A = nn.Parameter(torch.empty(r, in_features))
             self.lora_B = nn.Parameter(torch.zeros(out_features, r))
             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
             self.scaling = lora_alpha / r

         def forward(self, x):
             base_out = F.linear(x, self.weight, self.bias)   # frozen
             lora_out = (x @ self.lora_A.T) @ self.lora_B.T  # trainable
             return base_out + self.scaling * lora_out

     inject_lora(model, r, lora_alpha)
     for name, module in model.named_modules():
         if name.endswith("attention.self"):   # matches all 12 layers
             module.query = LoRALinear(module.query, r, lora_alpha)
             module.value = LoRALinear(module.value, r, lora_alpha)
     Works because RobertaSelfAttention.forward() calls self.query(x) dynamically — no HuggingFace source edits needed.

     freeze_base_weights(model)
     # 1. Freeze everything
     for p in model.parameters(): p.requires_grad = False
     # 2. Unfreeze LoRA matrices
     for n, p in model.named_parameters():
         if "lora_A" in n or "lora_B" in n: p.requires_grad = True
     # 3. Unfreeze classifier head
     for p in model.classifier.parameters(): p.requires_grad = True

     Step 4 — model.py

     def build_lora_model(task_config, r, lora_alpha=1.0):
         model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=...)
         inject_lora(model, r=r, lora_alpha=lora_alpha)
         freeze_base_weights(model)
         return model

     def build_full_finetune_model(task_config):
         return RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=...)

     Step 5 — data.py

     def get_dataloaders(task_config, tokenizer, batch_size):
         raw = datasets.load_dataset("glue", task_config.task_name)
         # tokenize using task_config.text_fields (single or pair)
         # rename "label" -> "labels", set_format("torch")
         # return (train_loader, val_loader)
     Zero task-specific branching in calling code — all variation lives in TaskConfig.

     Step 6 — train.py

     - Optimizer: AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=..., weight_decay=0.01)
     - Scheduler: transformers.get_linear_schedule_with_warmup, 6% warmup steps (matches paper)
     - Gradient clipping: clip_grad_norm_(model.parameters(), 1.0)
     - Per-epoch log: {epoch, train_loss, val_accuracy, epoch_time_sec, peak_gpu_mb}
     - Save best checkpoint by val accuracy; dump full log to JSON at end

     Step 7 — run_experiment.py

     CLI: --task, --mode (lora|full), --rank, --lora_alpha, --seed, --output_dir

     Sequence: set_seed → load tokenizer → build model → print param count → get dataloaders → train → save results.json

     Results JSON fields: task, mode, rank, seed, best_val_acc, trainable_params, total_params, epoch_times[], peak_gpu_mb

     Step 8 — run_sweep.py

     Iterates itertools.product(["sst2","qnli","rte"], ["full","lora"]) × ranks [4,8,16] for LoRA → 12 total runs. Each writes to results/{task}_{mode}_r{rank}/. Default lora_alpha = 2 * r.

     Step 9 — evaluate.py

     Standalone: --checkpoint_path, --task → reload model → run evaluate() → print accuracy. Useful for re-eval without retraining.

     ---
     Verification / Sanity Checks

     Run these after build_lora_model() before any training:

     1. Param count assertion (r=8 example):
     expected = 12 * 2 * (768*r + r*768) + classifier_params
     assert count_trainable_params(model) == expected

     2. Gradient whitelist check (after one forward+backward on fake batch):
     for n, p in model.named_parameters():
         if "lora_A" in n or "lora_B" in n or "classifier" in n:
             assert p.grad is not None
         else:
             assert p.grad is None

     3. Zero-init output equivalence (before any training step):
     # LoRA model must produce identical outputs to pretrained model at init
     # because lora_B=0 → lora_out=0 → LoRALinear(x) == original_linear(x)
     assert torch.allclose(out_original, out_lora, atol=1e-5)

     4. End-to-end sanity — full fine-tuning and LoRA should both show decreasing loss in epoch 1.

     ---
     Results Aggregation

     After all runs, a short results/aggregate.py reads all JSONs and prints a table:

     ┌───────┬──────┬──────┬──────────┬──────────────────┬────────────┬─────────────┐
     │ Task  │ Mode │ Rank │ Accuracy │ Trainable Params │ Time/epoch │ Peak GPU MB │
     ├───────┼──────┼──────┼──────────┼──────────────────┼────────────┼─────────────┤
     │ SST-2 │ Full │ —    │ ...      │ 125M             │ ...        │ ...         │
     ├───────┼──────┼──────┼──────────┼──────────────────┼────────────┼─────────────┤
     │ SST-2 │ LoRA │ 4    │ ...      │ ~300K            │ ...        │ ...         │
     ├───────┼──────┼──────┼──────────┼──────────────────┼────────────┼─────────────┤
     │ ...   │ ...  │ ...  │ ...      │ ...              │ ...        │ ...         │
     └───────┴──────┴──────┴──────────┴──────────────────┴────────────┴─────────────┘

     ---
     Key Design Rationale (for report)

     - B=0 init: Ensures training starts at the exact pretrained model output — identical to full fine-tuning's starting point. Critical for fair comparison.
     - alpha/r scaling: Keeps the magnitude of the LoRA update stable across different ranks.
     - Inject via attribute replacement: Transparent to HuggingFace internals; RobertaSelfAttention.forward() calls self.query(x) dynamically so swapping the attribute is sufficient.
     - Freeze-all-then-unfreeze: Safer and more auditable than trying to enumerate what to freeze.
