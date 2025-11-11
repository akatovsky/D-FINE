"""
Test script to understand D-FINE criterion losses.

Creates a criterion with dummy model outputs and targets,
then prints all individual loss components.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from basic.src.zoo.dfine.dfine import DFINE
from basic.src.zoo.dfine.dfine_decoder import DFINETransformer
from basic.src.zoo.dfine.hybrid_encoder import HybridEncoder

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from basic.src.nn.backbone.hgnetv2 import HGNetv2
from basic.src.zoo.dfine.dfine_criterion import DFINECriterion
from basic.src.zoo.dfine.matcher import HungarianMatcher



def create_model(n_hidden=16):
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
    return model



def create_dummy_targets(batch_size=2, num_objects_per_image=[3, 2], num_classes=1):
    """Create dummy ground truth targets."""
    targets = []

    for i in range(batch_size):
        num_objects = num_objects_per_image[i]

        # Random boxes in [cx, cy, w, h] format, normalized [0, 1]
        boxes = torch.rand(num_objects, 4)
        boxes[:, 2:] = boxes[:, 2:] * 0.3 + 0.05  # w, h in range [0.05, 0.35]

        # All objects are class 0 (single class)
        labels = torch.zeros(num_objects, dtype=torch.int64)

        # Random areas
        areas = boxes[:, 2] * boxes[:, 3] * 640 * 640  # Assuming 640x640 image

        targets.append({
            'boxes': boxes,
            'labels': labels,
            'area': areas,
            'iscrowd': torch.zeros(num_objects, dtype=torch.int64),
            'image_id': torch.tensor([i]),
        })

    return targets


def main():
    """Main test function."""
    print("="*80)
    print("D-FINE Criterion Loss Components Test")
    print("="*80)

    # Create model
    print("\nCreating model...")
    model = create_model(n_hidden=16)
    model.train()  # Set to training mode to get all auxiliary outputs
    print(f"  Model created with 16-channel input")
    print(f"  Model architecture: {type(model).__name__}")

    # Create matcher (hardcoded parameters from config)
    print("\nCreating matcher...")
    matcher = HungarianMatcher(
        weight_dict={
            'cost_class': 2,
            'cost_bbox': 5,
            'cost_giou': 2
        },
        use_focal_loss=True,  # From config: use_focal_loss: True
        alpha=0.25,
        gamma=2.0
    )
    print(f"  Matcher: HungarianMatcher")
    print(f"  Matcher costs: class=2, bbox=5, giou=2")

    # Create criterion (hardcoded parameters for single-class detection)
    print("\nCreating criterion...")
    num_classes = 1  # Single class detection
    criterion = DFINECriterion(
        matcher=matcher,
        weight_dict={
            'loss_vfl': 1,
            'loss_bbox': 5,
            'loss_giou': 2,
            'loss_fgl': 0.15,
            'loss_ddf': 1.5
        },
        losses=['vfl', 'boxes', 'local'],
        alpha=0.75,
        gamma=2.0,
        num_classes=num_classes,  # Single class
        reg_max=32,
        boxes_weight_format=None,
        share_matched_indices=False
    )
    criterion.train()  # Set to training mode

    print(f"  Criterion type: {type(criterion).__name__}")
    print(f"  Weight dict: {criterion.weight_dict}")
    print(f"  Losses: {criterion.losses}")
    print(f"  Alpha: {criterion.alpha}, Gamma: {criterion.gamma}")
    print(f"  Reg max: {criterion.reg_max}")
    print(f"  Num classes: {num_classes} (single-class detection)")
    print(f"  Query selection: agnostic")

    # Create dummy data
    batch_size = 2
    image_size = 640
    n_hidden = 16

    print(f"\nCreating dummy data...")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    print(f"  Input channels: {n_hidden}")

    # Create dummy input images
    dummy_images = torch.randn(batch_size, n_hidden, image_size, image_size)

    # Create dummy targets
    targets = create_dummy_targets(batch_size=batch_size, num_classes=num_classes)
    print(f"  Ground truth objects: {[len(t['boxes']) for t in targets]}")

    # Forward pass through model
    print(f"\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(dummy_images, targets=targets)

    # Compute losses
    print(f"\nComputing losses...")
    with torch.no_grad():
        loss_dict = criterion(outputs, targets, epoch=0, step=0, global_step=0)

    # Print all losses
    print("\n" + "="*80)
    print("LOSS COMPONENTS")
    print("="*80)

    # Group losses by type
    regular_losses = {}
    aux_losses = {}
    pre_losses = {}
    enc_losses = {}
    dn_losses = {}
    dn_pre_losses = {}

    for loss_name, loss_value in sorted(loss_dict.items()):
        if '_dn_pre' in loss_name:
            dn_pre_losses[loss_name] = loss_value
        elif '_dn_' in loss_name:
            dn_losses[loss_name] = loss_value
        elif '_aux_' in loss_name:
            aux_losses[loss_name] = loss_value
        elif '_pre' in loss_name:
            pre_losses[loss_name] = loss_value
        elif '_enc_' in loss_name:
            enc_losses[loss_name] = loss_value
        else:
            regular_losses[loss_name] = loss_value

    # Print by category
    def print_losses(title, losses_dict):
        if losses_dict:
            print(f"\n{title}:")
            print("-" * 60)
            for name, value in sorted(losses_dict.items()):
                weighted = value.item() if hasattr(value, 'item') else value
                print(f"  {name:30s}: {weighted:8.4f}")

    print_losses("FINAL LAYER (Regular Queries)", regular_losses)
    print_losses("AUXILIARY LAYERS (Regular Queries)", aux_losses)
    print_losses("PRE-DECODER HEAD (Regular Queries)", pre_losses)
    print_losses("ENCODER", enc_losses)
    print_losses("DENOISING QUERIES (All Layers)", dn_losses)
    print_losses("DENOISING PRE-DECODER", dn_pre_losses)

    # Compute total
    total_loss = sum(loss_dict.values())
    print("\n" + "="*80)
    print(f"TOTAL LOSS: {total_loss.item():.4f}")
    print("="*80)

    # Print summary statistics
    print(f"\nSUMMARY:")
    print(f"  Total loss components: {len(loss_dict)}")
    print(f"  Regular losses: {len(regular_losses)}")
    print(f"  Auxiliary losses: {len(aux_losses)}")
    print(f"  Pre-decoder losses: {len(pre_losses)}")
    print(f"  Encoder losses: {len(enc_losses)}")
    print(f"  Denoising losses: {len(dn_losses)}")
    print(f"  Denoising pre-decoder losses: {len(dn_pre_losses)}")

    print("\n" + "="*80)
    print("Test complete!")
    print("="*80)


if __name__ == "__main__":
    main()
