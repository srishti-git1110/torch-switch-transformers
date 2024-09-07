import torch

def update_gating_biases(
    gating_biases: torch.Tensor,
    error_sign: torch.Tensor,
    bias_update_rate_u: float = 0.001,
) -> torch.Tensor:
    return gating_biases + (error_sign * bias_update_rate_u)