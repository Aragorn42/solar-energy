import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2
        )

        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout):
        super().__init__()

        layers = []
        for i in range(len(num_channels)):
            layers.append(
                TemporalBlock(
                    n_inputs=num_inputs if i == 0 else num_channels[i - 1],
                    n_outputs=num_channels[i],
                    kernel_size=kernel_size,
                    dilation=2 ** i,
                    dropout=dropout
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class Model(nn.Module):
    """
    PatchTST-compatible TCN
    """
    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.channel_independence = configs.channel_independence

        if self.channel_independence:
            self.enc_in = 1
            self.c_out = 1
        else:
            self.enc_in = configs.enc_in
            self.c_out = configs.c_out

        # TCN hidden dim
        d_model = configs.d_model
        num_layers = configs.e_layers

        self.tcn = TemporalConvNet(
            num_inputs=self.enc_in,
            num_channels=[d_model] * num_layers,
            kernel_size=configs.kernel_size if hasattr(configs, "kernel_size") else 3,
            dropout=configs.dropout
        )

        # PatchTST-style head: only last time step
        self.head = nn.Linear(d_model, self.pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, seq_len, enc_in]
        return: [B, pred_len, c_out]
        """

        B, L, C = x_enc.shape

        # ---------- Channel Independent ----------
        if self.channel_independence:
            outputs = []

            for c in range(C):
                x = x_enc[:, :, c:c+1]          # [B, L, 1]
                x = x.transpose(1, 2)           # [B, 1, L]

                feat = self.tcn(x)              # [B, d_model, L]
                last = feat[:, :, -1]           # [B, d_model]

                out = self.head(last)            # [B, pred_len]
                outputs.append(out.unsqueeze(-1))

            output = torch.cat(outputs, dim=-1)  # [B, pred_len, C]

        # ---------- Multivariate ----------
        else:
            x = x_enc.transpose(1, 2)             # [B, C, L]
            feat = self.tcn(x)                    # [B, d_model, L]
            last = feat[:, :, -1]                 # [B, d_model]

            out = self.head(last)                 # [B, pred_len]
            output = out.unsqueeze(-1)            # [B, pred_len, 1]
            if self.c_out > 1:
                output = output.repeat(1, 1, self.c_out)

        if self.output_attention:
            return output, None
        else:
            return output
