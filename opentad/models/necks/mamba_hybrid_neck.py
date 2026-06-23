import torch
import torch.nn as nn

from mamba_ssm import Mamba

from ..builder import NECKS
from .etad_lstm import LSTMNeck


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
        self.etad_lstm = nn.ModuleList([
            LSTMNeck( in_channels=in_channels, out_channels=in_channels)
            for _ in range(num_levels)
            ])

        #self.transformers = nn.ModuleList([
            # self.lstm_blocks = nn.ModuleList([
            # nn.LSTM(
            #     input_size=out_channels,
            #     hidden_size=out_channels // 2,
            #     num_layers=1,
            #     batch_first=True,
            #     bidirectional=True
            # )
            # for _ in range(num_levels)
            # ])
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
            #print(type(masks))

            #if isinstance(masks, (list, tuple)):
            #    print("masks_len",len(masks))
            #else:
            #    print("masks_shape",masks.shape)

            feat_seq = feats[i]      # [B,C,T]

            # ----------------
            # Mamba branch
            # ----------------

            mamba_feat = feat_seq.permute(0,2,1)

            mamba_feat = self.mamba_blocks[i](mamba_feat)

            mamba_feat = mamba_feat.permute(0,2,1)

            # ----------------
            # ETAD branch
            # ----------------

            lstm_feat, _ = self.etad_lstm[i](
            feat_seq,
            masks[i]
            )

            # ----------------
            # Fusion
            # ----------------
            if i<3:
                lstm_feat, _ = self.etad_lstm[i](feat_seq, masks[i])
                fused = feat_seq + 0.5 * (mamba_feat + lstm_feat)
            else:
                fused = feat_seq + mamba_feat

            fused = self.output_convs[i](fused)
            #print("feature_shape", feat_seq.shape)
            #print("mamba_shape",mamba_feat.shape)
            #print("lstm_feature_shape",lstm_feat.shape)
            outputs.append(fused)
        
        return tuple(outputs), masks

        # for i in range(len(feats)):

        #     x = feats[i]
        #     feat_seq = feats[i]
        #     # [B,C,T] -> [B,T,C]
        #     x = x.permute(0, 2, 1)

        #     # Mamba
        #     mamba_feat = self.mamba_blocks[i](x)

        #     # Transformer
        #     #x = self.transformers[i](x)

        #     #lstm_feat, _ = self.lstm_blocks[i](x)
        #     lstm_feat, _ = self.etad_lstm[i](feat_seq,masks[i])
        #     fused = feat_seq + 0.5 * (mamba_feat + lstm_feat)


        #     # back to [B,C,T]
        #     #x = x.permute(0, 2, 1)

        #     #x = self.output_convs[i](x)
        #     fused = fused.permute(0, 2, 1)

        #     outputs.append(fused)

        # return tuple(outputs), masks
        
        