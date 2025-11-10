"""
Standalone script to construct a medium-sized D-FINE model.

This script creates a D-FINE-M (Medium) model with HGNetV2-B2 backbone,
configured according to the official D-FINE architecture.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.nn.backbone.hgnetv2 import HGNetv2
from src.zoo.dfine.hybrid_encoder import HybridEncoder
from src.zoo.dfine.dfine_decoder import DFINETransformer
from src.zoo.dfine.dfine import DFINE


def construct_dfine_medium(
    num_classes=80,
):
    """
    Construct a D-FINE Medium model.

    Args:
        num_classes (int): Number of object classes (default: 80 for COCO)
        pretrained_backbone (bool): Whether to load pretrained HGNetV2 weights
        local_model_dir (str): Directory for pretrained backbone weights

    Returns:
        DFINE: The constructed D-FINE Medium model
    """

    # HGNetV2-B2 backbone configuration (Medium model)
    backbone = HGNetv2(
        name='B2',
        return_idx=[1, 2, 3],
        freeze_at=-1,
        freeze_norm=False,
        use_lab=True,
        pretrained=False
    )

    # HybridEncoder configuration for Medium model
    encoder = HybridEncoder(
        in_channels=[384, 768, 1536],  # Output channels from HGNetV2-B2 stages
        feat_strides=[8, 16, 32],
        hidden_dim=256,
        nhead=8,
        dim_feedforward=1024,
        dropout=0.0,
        enc_act='gelu',
        use_encoder_idx=[2],
        num_encoder_layers=1,
        expansion=1.0,
        depth_mult=0.67,  # Medium model depth multiplier
        act='silu',
        eval_spatial_size=None
    )

    # DFINETransformer decoder configuration for Medium model
    decoder = DFINETransformer(
        num_classes=num_classes,
        hidden_dim=256,
        num_queries=300,
        feat_channels=[256, 256, 256],  # Output from encoder
        feat_strides=[8, 16, 32],
        num_levels=3,
        num_points=[3, 6, 3],
        nhead=8,
        num_layers=4,  # Medium model has 4 decoder layers
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

    return model


def main():
    """
    Main function to demonstrate model construction and basic usage.
    """
    print("Constructing D-FINE Medium model...")

    # Create model (without pretrained weights for quick initialization)
    model = construct_dfine_medium(
        num_classes=1,
        pretrained_backbone=False
    )

    # Set to evaluation mode
    model.eval()

    # Print model information
    print("\nModel constructed successfully!")
    print(f"Model architecture: D-FINE Medium (HGNetV2-B2)")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with dummy input
    print("\nTesting forward pass with dummy input...")
    dummy_input = torch.randn(1, 3, 640, 640)

    with torch.no_grad():
        try:
            output = model(dummy_input)
            print("Forward pass successful!")

            if isinstance(output, dict):
                print(f"\nOutput keys: {list(output.keys())}")
                if 'pred_logits' in output:
                    print(f"Prediction logits shape: {output['pred_logits'].shape}")
                if 'pred_boxes' in output:
                    print(f"Prediction boxes shape: {output['pred_boxes'].shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")

    print("\nModel ready for training or inference!")

    return model


if __name__ == "__main__":
    model = main()
