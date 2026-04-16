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
        max_seq_len=128,
        num_epochs=3,
        batch_size=16,
        lora_lr=2e-4,
        full_ft_lr=2e-5,
    ),
    "qnli": TaskConfig(
        task_name="qnli",
        num_labels=2,
        text_fields=("question", "sentence"),
        max_seq_len=256,
        num_epochs=3,
        batch_size=16,
        lora_lr=2e-4,
        full_ft_lr=2e-5,
    ),
    "rte": TaskConfig(
        task_name="rte",
        num_labels=2,
        text_fields=("sentence1", "sentence2"),
        max_seq_len=128,
        num_epochs=10,
        batch_size=16,
        lora_lr=2e-4,
        full_ft_lr=2e-5,
    ),
}
