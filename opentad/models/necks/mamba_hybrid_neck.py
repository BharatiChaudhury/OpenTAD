import torch
import torch.nn as nn

from mamba_ssm import Mamba

from ..builder import NECKS


@NECKS.register_module()
class MambaHybridNeck(nn.Module):

    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_levels=6,
        num_heads=8,
    ):
        super().__init__()

        self.num_levels = num_levels

        self.mamba_blocks = nn.ModuleList([
            Mamba(
                d_model=in_channels,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            for _ in range(num_levels)
        ])

        #self.transformers = nn.ModuleList([
            self.lstm_blocks = nn.ModuleList([
            nn.LSTM(
                input_size=out_channels,
                hidden_size=out_channels // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True
            )
            for _ in range(num_levels)
            ])
        #])
            # nn.TransformerEncoder(
            #     nn.TransformerEncoderLayer(
            #         d_model=in_channels,
            #         nhead=num_heads,
            #         batch_first=True,
            #     ),
            #     num_layers=1,
            # )
        #     for _ in range(num_levels)
        # ])

        self.output_convs = nn.ModuleList([
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
            )
            for _ in range(num_levels)
        ])

    def forward(self, feats, masks):

        outputs = []

        for i in range(len(feats)):

            x = feats[i]

            # [B,C,T] -> [B,T,C]
            x = x.permute(0, 2, 1)

            # Mamba
            x = self.mamba_blocks[i](x)

            # Transformer
            #x = self.transformers[i](x)

            lstm_feat, _ = self.lstm_blocks[i](x)
            fused = feat_seq + 0.5 * (mamba_feat + lstm_feat)


            # back to [B,C,T]
            #x = x.permute(0, 2, 1)

            #x = self.output_convs[i](x)
            fused = fused.permute(0, 2, 1)

            outputs.append(x)

        return tuple(outputs), masks