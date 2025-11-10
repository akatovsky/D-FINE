from basic.src.nn.backbone.hgnetv2 import HGNetv2
import torch

from basic.src.zoo.dfine.dfine_decoder import DFINETransformer
from basic.src.zoo.dfine.hybrid_encoder import HybridEncoder
from basic.src.zoo.dfine.dfine import DFINE

def main():
    n_hidden = 16
    width = 512
    height = 1024
    t_in = torch.randn(1, n_hidden, height, width)

    backbone = HGNetv2(
        stem_channels=[n_hidden, 16, 16],
        stage_config={
            # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
            "stage1": [16, 16, 64, 1, False, False, 3, 3],
            "stage2": [64, 32, 256, 1, True, False, 3, 3],
            "stage3": [256, 64, 512, 2, True, True, 5, 3],
            "stage4": [512, 128, 1024, 1, True, True, 5, 3],
        },
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        use_lab=True,
    )

    encoder = HybridEncoder(
        in_channels=[256, 512, 1024],  # Output channels from HGNetV2-B1 stages
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        expansion=0.5,
        depth_mult=0.34,  # Medium model depth multiplier
        act='silu',
        eval_spatial_size=None
    )

    # DFINETransformer decoder configuration for Medium model
    decoder = DFINETransformer(
        num_classes=1,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],  # Output from encoder
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[3, 6, 3],
        nhead=8,
        num_layers=3,
        dim_feedforward=1024,
        dropout=0.0,
        activation='relu',
        num_denoising=100,
        label_noise_ratio=0.0,
        box_noise_scale=1.0,
        learn_query_content=False,
        eval_spatial_size=None,
        eval_idx=-1,
        eps=1e-2,
        aux_loss=True,
        cross_attn_method='default',
        query_select_method='agnostic',
        reg_max=32,
        reg_scale=4.0,
        layer_scale=1
    )

    # Construct the complete D-FINE model
    model = DFINE(
        backbone=backbone,
        encoder=encoder,
        decoder=decoder
    )
    
    model.eval()
    with torch.no_grad():
        t_out = model(t_in)
        print(t_out)

if __name__ == "__main__":
    main()
