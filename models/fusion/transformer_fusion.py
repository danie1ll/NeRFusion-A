# --------------------------------------------------------------------------------------------------------- #
# ------------------------------------------- TransformerFusion ------------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

import math

import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerFusion(nn.Module):
    def __init__(self, cfg, ch_in=None):
        super(TransformerFusion, self).__init__()
        self.cfg = cfg
        self.model_dim = 512
        self.nhead = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1

        self.transformer = nn.Transformer(
            d_model=self.model_dim,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

        self.pos_encoder = PositionalEncoding(d_model=self.model_dim, dropout=self.dropout)

    def forward(self, coords, values_in, inputs, scale=2, outputs=None, save_mesh=False):
        
        src = self.pos_encoder(values_in)
        output = self.transformer(src, src)
        return output
