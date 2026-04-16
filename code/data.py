from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

from config import TaskConfig


def get_dataloaders(
    task_config: TaskConfig,
    tokenizer: RobertaTokenizer,
    batch_size: int,
) -> tuple:
    """
    Load a GLUE task, tokenize, and return (train_loader, val_loader)

    All task-specific variation (field names, max length) is captured in TaskConfig
    """
    raw = load_dataset("glue", task_config.task_name)

    def tokenize(batch):
        fields = task_config.text_fields
        if len(fields) == 1:
            return tokenizer(
                batch[fields[0]],
                truncation=True,
                max_length=task_config.max_seq_len,
                padding="max_length",
            )
        else:
            return tokenizer(
                batch[fields[0]],
                batch[fields[1]],
                truncation=True,
                max_length=task_config.max_seq_len,
                padding="max_length",
            )

    tokenized = raw.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    train_loader = DataLoader(
        tokenized["train"], batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        tokenized["validation"], batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader
