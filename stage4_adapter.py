"""Lightweight frozen-Chronos adapters for the Stage 4 experiments."""

from __future__ import annotations

import torch
from torch import nn
from einops import rearrange


class UniCATokenAdapter(nn.Module):
    """Single cross-attention layer with a zero-initialized residual scale."""

    def __init__(self, fusion_dim: int = 64, chronos_dim: int = 768, heads: int = 12,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.fusion_projection = nn.Linear(fusion_dim, chronos_dim)
        self.query_norm = nn.LayerNorm(chronos_dim)
        self.fusion_norm = nn.LayerNorm(chronos_dim)
        self.cross_attention = nn.MultiheadAttention(
            chronos_dim, heads, dropout=dropout, batch_first=True
        )
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, chronos_tokens: torch.Tensor, fusion_tokens: torch.Tensor) -> torch.Tensor:
        if chronos_tokens.ndim != 3 or fusion_tokens.ndim != 3:
            raise ValueError("chronos_tokens and fusion_tokens must be [batch, tokens, dim]")
        projected = self.fusion_norm(self.fusion_projection(fusion_tokens))
        update, _ = self.cross_attention(
            self.query_norm(chronos_tokens), projected, projected, need_weights=False
        )
        return chronos_tokens + self.alpha * update


class CoRACorrelationAdapter(nn.Module):
    """One-layer dynamic/global correlation adapter with zero residual gates."""

    def __init__(self, fusion_dim: int = 64, chronos_dim: int = 768,
                 heads: int = 12, global_hidden: int = 192,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.fusion_projection = nn.Linear(fusion_dim, chronos_dim)
        self.query_norm = nn.LayerNorm(chronos_dim)
        self.fusion_norm = nn.LayerNorm(chronos_dim)
        self.dynamic_attention = nn.MultiheadAttention(
            chronos_dim, heads, dropout=dropout, batch_first=True
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(chronos_dim, global_hidden), nn.GELU(),
            nn.Linear(global_hidden, chronos_dim),
        )
        self.alpha = nn.Parameter(torch.zeros(()))
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, chronos_tokens: torch.Tensor, fusion_tokens: torch.Tensor) -> torch.Tensor:
        projected = self.fusion_norm(self.fusion_projection(fusion_tokens))
        dynamic, _ = self.dynamic_attention(
            self.query_norm(chronos_tokens), projected, projected, need_weights=False
        )
        global_condition = self.global_mlp(projected.mean(dim=1)).unsqueeze(1)
        return chronos_tokens + self.alpha * dynamic + self.beta * global_condition


def adapter_forward(
    chronos: nn.Module,
    adapter: UniCATokenAdapter,
    context: torch.Tensor,
    fusion_tokens: torch.Tensor,
    prediction_length: int = 24,
) -> torch.Tensor:
    """Forecast with an adapter inserted after the frozen Chronos encoder.

    Returns all native quantiles with shape ``[B, Q, prediction_length]``.
    """
    patch_size = chronos.chronos_config.output_patch_size
    num_output_patches = (prediction_length + patch_size - 1) // patch_size
    batch_size = context.shape[0]
    group_ids = torch.arange(batch_size, device=context.device)
    encoded, loc_scale, _, _ = chronos.encode(
        context=context, group_ids=group_ids, num_output_patches=num_output_patches
    )
    hidden = adapter(encoded.last_hidden_state, fusion_tokens)
    forecast = hidden[:, -num_output_patches:]
    quantile_preds = chronos.output_patch_embedding(forecast)
    quantile_preds = rearrange(
        quantile_preds, "b n (q p) -> b q (n p)",
        q=chronos.num_quantiles, p=patch_size,
    )
    quantile_preds = rearrange(quantile_preds, "b q h -> b (q h)")
    quantile_preds = chronos.instance_norm.inverse(quantile_preds, loc_scale)
    quantile_preds = rearrange(
        quantile_preds, "b (q h) -> b q h", q=chronos.num_quantiles
    )
    return quantile_preds[..., :prediction_length]


def freeze_backbones(chronos: nn.Module, fusionsf: nn.Module) -> None:
    chronos.eval().requires_grad_(False)
    fusionsf.eval().requires_grad_(False)


def trainable_parameter_names(adapter: nn.Module) -> list[str]:
    return [name for name, value in adapter.named_parameters() if value.requires_grad]
