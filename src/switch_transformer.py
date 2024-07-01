import torch
from torch import nn


class Router(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
    ):
        super(Router, self).__init__()
        self.inp_dim = inp_dim
        self.num_experts = num_experts
        self.layer = nn.Linear(inp_dim, num_experts)

    def forward(self, x: torch.Tensor):
        return nn.Softmax(self.layer(x), dim=-1)


class ExpertAllocation(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        alpha: float = 0.01,
    ):
        super(ExpertAllocation, self).__init__
        self.inp_dim = inp_dim
        self.num_experts = num_experts
        self.router = Router(inp_dim, num_experts)
        self.capacity_factor = capacity_factor
        self.alpha = alpha
        self.use_aux_loss = use_aux_loss

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        expert_capacity = (
            (x.shape[0] * x.shape[1]) / self.num_experts
        ) * self.capacity_factor

        expert_probs = self.router(x)
        top_prob, top_idx = expert_probs.topk(1, dim=-1)

        routed_experts = torch.zeros_like(expert_probs).scatter_(
            dim=-1,
            index=top_idx,
            src=torch.ones_like(top_prob),
        )

        aux_loss = 0
        if self.use_aux_loss:
            total_tokens = x.shape[0] * x.shape[1]
            f_i = torch.sum(routed_experts, dim=(0, 1)) * (1 / total_tokens)
            P_i = (torch.sum(expert_probs, dim=(0, 1))) * (1 / total_tokens)

            aux_loss = self.alpha * self.num_experts * torch.sum((f_i * P_i))

        flat_routed_experts = routed_experts.view(-1, self.num_experts)
        total_expert_allocation = torch.cumsum(flat_routed_experts, dim=0)
        expert_mask = (total_expert_allocation <= expert_capacity).float()
        revised_expert_allocation = expert_mask * flat_routed_experts
        routed_experts = revised_expert_allocation.view(
            x.shape[0], x.shape[1], self.num_experts
        )

        routed_expert_probs = expert_probs * routed_experts

        return routed_expert_probs, aux_loss


class SwitchLayer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        alpha: float = 0.01,
    ):
        super(SwitchLayer, self).__self__()
        self.inp_dim = inp_dim
        self.num_experts = num_experts
        self.expert_allocation = ExpertAllocation(
            inp_dim, num_experts, capacity_factor, use_aux_loss, alpha
        )
        self.experts = nn.ModuleList(
            [nn.Linear(inp_dim, inp_dim) for _ in range(num_experts)]
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        routed_expert_probs, aux_loss = self.expert_allocation(
            x
        )

        active_tokens = (routed_expert_probs.sum(dim=-1) > 0).view(-1)
        expert_probs, expert_indices = routed_expert_probs.topk(1, dim=-1)
        expert_probs, expert_indices = expert_probs.view(-1, 1), expert_indices.view(-1)
        active_experts = expert_indices[active_tokens]

        flat_x = x.view(-1, self.inp_dim)
        active_x = flat_x[active_tokens]
        active_out = torch.zeros_like(active_x)

        for i, expert in enumerate(self.num_experts):
            mask = active_experts == i
            if mask.any():
                expert_output = expert(active_x[mask])
                active_out[mask] = expert_output

        active_out *= expert_probs[active_tokens]
        out = torch.zeros_like(flat_x)
        out[active_tokens] = active_out
        out = out.view(x.shape)

        return out, aux_loss


class SwitchTransformerBlock(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        num_heads: int,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        alpha: float = 0.01,
        dropout: float = 0.1,
    ):
        super(SwitchTransformerBlock, self).__init__()
        self.inp_dim = inp_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.switch_layer = SwitchLayer(
            inp_dim, num_experts, capacity_factor, use_aux_loss, alpha
        )
        self.norm = nn.LayerNorm(inp_dim)
        self.attn_block = nn.MultiheadAttention(
            inp_dim, num_heads, dropout, batch_first=True
        )

    def forward(self, x: torch.Tensor)  -> (torch.Tensor, torch.Tensor):
        residual = x
        attn_output, _ = self.attn_block(x)
        x = attn_output + residual
        x = self.norm(x)
        normed_x = x

        x, aux_loss = self.switch_layer(x)
        x += normed_x
        x = self.norm(x)

        return x, aux_loss


class SwitchTransformer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        num_experts: int,
        num_heads: int,
        vocab_size: int,
        depth: int = 12,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = True,
        alpha: float = 0.01,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, inp_dim)
        self.layers = nn.ModuleList([])
        for _ in depth:
            self.layers.append(
                SwitchTransformerBlock(
                    inp_dim,
                    num_experts,
                    num_heads,
                    vocab_size,
                    capacity_factor,
                    use_aux_loss,
                    alpha,
                    dropout,
                )
            )
        self.output_layer = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.LayerNorm(inp_dim),
            nn.Linear(inp_dim, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.embedding(x)
        total_aux_loss = 0

        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss

        x = self.output_layer(x)

        return x, total_aux_loss
    