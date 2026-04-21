# Datasets

All datasets are downloaded automatically by HuggingFace `datasets` at runtime — no manual download is required.

The following GLUE tasks are used:

| Task  | Description                    | Split used         |
| ----- | ------------------------------ | ------------------ |
| SST-2 | Sentiment classification       | train / validation |
| QNLI  | Question-answering NLI         | train / validation |
| RTE   | Recognizing textual entailment | train / validation |

Datasets are loaded in `code/data.py` via:

```python
from datasets import load_dataset
load_dataset("glue", "<task_name>")
```
