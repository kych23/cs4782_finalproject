### Paper

https://arxiv.org/abs/2106.09685

### 2-page report summary

https://docs.google.com/document/d/1KpMVNg_HShuOFuITlZTrP5CjQd3VAM5w01bWWs8ZUYg/edit?usp=sharing

### Project Selection Submission

https://docs.google.com/document/d/1qfKQviobjutXVH3Scg7LJzD9_0iAjYfAdWCnrzCRPuo/edit?tab=t.0

#### What we used AI for

- config.py
- utils.py
- verify_lora_model in model.py
- MIT License

### order of running

RTE --> SST-2 --> QNLI

### Extra

- Our results for RTE are not as high as the paper because the paper pre-trained on MNLI benchmark first.
  - "Following Liu et al. (2019), we initialize the LoRA modules to our best MNLI checkpoint when adapting to MRPC, RTE, and STS-B, instead of the usual initialization"
  - We started pre-training from random Gaussian initialization instead of MNLI checkpoitn
- RTE LoRA took 4 hours to train based on original hyperparameters from paper --> lowered number of epochs for QNLI and SST2 because 1 epoch took around 30 min - 1 hr
- Used L4 GPU for RTE LoRA and Fine-Tuning
- Used A100 High-Ram for SST-2 and QNLI for LoRA and Fine-Tuning
- Remove the sweep test across different ranks because we ran out of compute credits and would take too long
- used Claude Code to create sanity checks such as verify_lora_model
