import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Replaces an nn.Linear with a frozen base weight plus a trainable low-rank update
    
    The paper writes the forward pass as h = W_0 @ x + B @ A @ x, where x is a
    column vector of shape (features, 1). PyTorch uses the opposite convention where
    inputs are row vectors of shape (batch, features), and nn.Linear computes x @ W^T rather than W @ x.
    We adjust the forward pass to match PyTorch's conventions:
        output = x @ W_0^T + bias_0 + scaling * (x @ A^T) @ B^T

    where W_0 and bias_0 are frozen, and A (rank x in) and B (out x rank) are trainable

    Initialization:
    - A: random Gaussian initialization
    - B: zero initialization
    """

    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # Frozen base weights stored as non-trainable Parameters so state_dict() to keep track of them in the model
        self.weight = nn.Parameter(
            original_linear.weight.data.clone(), requires_grad=False
        )
        if original_linear.bias is not None:
            self.bias = nn.Parameter(
                original_linear.bias.data.clone(), requires_grad=False
            )
        else:
            self.bias = None

        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.normal_(self.A)

        self.rank = rank
        self.alpha = alpha
        
        self.scaling = self.alpha / self.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight, self.bias)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scaling * lora_out


def inject_lora(model: nn.Module, rank: int, alpha: float = 1.0) -> nn.Module:
    """
    Replaces model's self-attention layers with custom LoRALinear layers, 
    replacing only the query and value projection matrices.
    """
    for name, module in model.named_modules():
        if name.endswith("attention.self"):
            module.query = LoRALinear(module.query, rank=rank, alpha=alpha)
            module.value = LoRALinear(module.value, rank=rank, alpha=alpha)
    return model


def freeze_base_weights(model: nn.Module) -> None:
    """
    Freezes pre-trained model weights so only A, B are trainable. Classifier head
    remains trainable
    """
    # Step 1: freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2: unfreeze LoRA matrices
    for name, param in model.named_parameters():
        if name.endswith(".A") or name.endswith(".B"):
            param.requires_grad = True

    # Step 3: unfreeze classifier head
    for param in model.classifier.parameters():
        param.requires_grad = True
