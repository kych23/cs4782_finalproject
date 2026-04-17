"""Configuration file to define the three different benchmark tasks and their hyperparameters"""

from dataclasses import dataclass


@dataclass
class TaskConfig:
    task_name: str
    num_labels: int
    text_fields: tuple
    max_seq_len: int
    num_epochs: int
    batch_size: int
    lora_lr: float
    full_ft_lr: float


TASK_CONFIGS = {
    "sst2": TaskConfig(
        task_name="sst2",
        num_labels=2,
        text_fields=("sentence",),
        max_seq_len=512,
        num_epochs=10, # adjusted from 60 to 10 to reduce training time
        batch_size=16,
        lora_lr=5e-4,
        full_ft_lr=2e-5,
    ),
    "qnli": TaskConfig(
        task_name="qnli",
        num_labels=2,
        text_fields=("question", "sentence"),
        max_seq_len=512,
        num_epochs=10, # adjusted from 25 to 10 to reduce training time
        batch_size=32,
        lora_lr=4e-4,
        full_ft_lr=2e-5,
    ),
    # NOTE: Our RTE results will be slightly below the paper's reported 86.6%.
    # Section D.1 states: "we initialize the LoRA modules to our best MNLI checkpoint
    # when adapting to MRPC, RTE, and STS-B, instead of the usual initialization."
    # The paper first trains LoRA on MNLI, then uses those trained A/B matrices as the
    # starting point for RTE fine-tuning. We initialize from scratch (A~N(0,1), B=0)
    # for simplicity. This warm-start from MNLI is the primary reason for any gap
    # between our RTE accuracy and the paper's number.
    "rte": TaskConfig(
        task_name="rte",
        num_labels=2,
        text_fields=("sentence1", "sentence2"),
        max_seq_len=512,
        num_epochs=80,
        batch_size=32,
        lora_lr=5e-4,
        full_ft_lr=2e-5,
    ),
}
