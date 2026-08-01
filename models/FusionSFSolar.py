"""Power-only FusionSF migration for the native Solar long-forecast pipeline."""

import torch
from torch import nn


class FeedForward(nn.Module):
    """FusionSF GEGLU feed-forward block."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.input = nn.Linear(dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        values, gates = self.input(x).chunk(2, dim=-1)
        return self.dropout(self.output(self.dropout(values * torch.nn.functional.gelu(gates))))


class Attention(nn.Module):
    """FusionSF multi-head self-attention."""

    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, inner * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, x):
        batch, length, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        reshape = lambda tensor: tensor.view(batch, length, self.heads, self.dim_head).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, -1)
        return self.to_out(output)


class Transformer(nn.Module):
    """FusionSF Transformer history encoder with learned time coordinates."""

    def __init__(self, dim, length, depth, heads, dim_head, dropout):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, length, dim))
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim), Attention(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim), FeedForward(dim, dim * 4, dropout),
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.pos_embedding
        for norm_attn, attention, norm_ff, feed_forward in self.layers:
            x = x + attention(norm_attn(x))
            x = x + feed_forward(norm_ff(x))
        return self.norm(x)


class CrossAttention(nn.Module):
    """FusionSF horizon-to-history attention (queries read history keys/values)."""

    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner = heads * dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_kv = nn.Linear(dim, inner * 2, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))

    def forward(self, history, horizon):
        batch, source_len, _ = history.shape
        target_len = horizon.shape[1]
        q = self.to_q(horizon).view(batch, target_len, self.heads, self.dim_head).transpose(1, 2)
        k, v = self.to_kv(history).chunk(2, dim=-1)
        k = k.view(batch, source_len, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(batch, source_len, self.heads, self.dim_head).transpose(1, 2)
        weights = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, target_len, -1)
        return self.to_out(output)


class CrossTransformer(nn.Module):
    """FusionSF residual cross-attention and GEGLU stack."""

    def __init__(self, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(dim), nn.LayerNorm(dim), CrossAttention(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim), FeedForward(dim, dim * 4, dropout),
            ]) for _ in range(depth)
        ])

    def forward(self, history, horizon):
        for history_norm, horizon_norm, attention, ff_norm, feed_forward in self.layers:
            horizon = horizon + attention(history_norm(history), horizon_norm(horizon))
            horizon = horizon + feed_forward(ff_norm(horizon))
        return horizon


class Model(nn.Module):
    """Solar-standard interface for ``fusionsf_solar_v1`` (Power-only track)."""

    model_family = "FusionSF"
    architecture_version = "fusionsf_solar_v1"
    experiment_track = "standard_power_only"

    def __init__(self, configs):
        super().__init__()
        self.seq_len = int(configs.seq_len)
        self.pred_len = int(configs.pred_len)
        self.c_out = int(configs.c_out)
        dim = int(configs.d_model)
        depth = int(configs.e_layers)
        heads = int(configs.n_heads)
        if dim % heads:
            raise ValueError("d_model must be divisible by n_heads")
        dim_head = dim // heads
        dropout = float(configs.dropout)
        if configs.embed == "timeF":
            frequency = str(configs.freq).lower()
            time_dim = 5 if ("min" in frequency or frequency in {"t"}) else 4
        else:
            time_dim = 4

        self.history_embedding = nn.Linear(1 + time_dim, dim)
        self.history_encoder = Transformer(dim, self.seq_len, depth, heads, dim_head, dropout)
        self.horizon_embedding = nn.Parameter(torch.randn(1, self.pred_len, dim))
        self.future_time_embedding = nn.Linear(time_dim, dim)
        self.cross_transformer = CrossTransformer(dim, depth, heads, dim_head, dropout)
        self.prediction_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.c_out))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        del x_dec, mask  # decoder values may contain labels; Power-only never reads them.
        if x_enc.shape[1] != self.seq_len:
            raise ValueError(f"expected seq_len={self.seq_len}, got {x_enc.shape[1]}")
        history_power = x_enc[..., -1:]
        history_tokens = self.history_encoder(
            self.history_embedding(torch.cat([history_power, x_mark_enc], dim=-1))
        )
        future_marks = x_mark_dec[:, -self.pred_len:, :]
        horizon = self.horizon_embedding.expand(x_enc.shape[0], -1, -1)
        horizon = horizon + self.future_time_embedding(future_marks)
        fused = self.cross_transformer(history_tokens, horizon)
        return self.prediction_head(fused)
