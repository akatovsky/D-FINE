"""
Simple inference script for D-FINE.

Loads a checkpoint and runs inference on images from a folder,
saving visualized results to /tmp/dfine/
"""

import os
import random
import sys
from pathlib import Path
import glob

import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import YAMLConfig


# COCO category names for visualization
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def load_model(checkpoint_path, config_path, device='cuda'):
    """Load model from checkpoint."""
    print(f"Loading config from {config_path}...")
    cfg = YAMLConfig(config_path)

    print(f"Building model...")
    # Build model
    model = cfg.model.to(device)

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model weights (handle EMA if present)
    if 'ema' in checkpoint and checkpoint['ema'] is not None:
        print("Using EMA weights")
        model_state = checkpoint['ema']['module']
    elif 'model' in checkpoint:
        model_state = checkpoint['model']
    else:
        model_state = checkpoint

    # Remove 'module.' prefix if present (DDP)
    model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
    model.load_state_dict(model_state)

    model.eval()
    print("Model loaded successfully!")
    return model, cfg


def load_images_from_folder(image_folder, num_images=None, extensions=('.png', '.jpg', '.jpeg')):
    """Load images from a folder."""
    image_folder = Path(image_folder)

    # Find all image files
    image_files = []
    for ext in extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
        image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))

    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    print(f"Found {len(image_files)} images in {image_folder}")

    # Randomly sample if num_images specified
    if num_images is not None and num_images < len(image_files):
        image_files = random.sample(image_files, num_images)

    images_data = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert('RGB')
            images_data.append({
                'image': img,
                'filename': img_path.name,
                'path': str(img_path)
            })
        except Exception as e:
            print(f"Warning: Failed to load {img_path}: {e}")

    return images_data


def preprocess_image(image, size=(640, 640)):
    """Preprocess image for model input."""
    # Resize
    image_resized = image.resize(size, Image.BILINEAR)

    # Convert to tensor and normalize
    img_tensor = F.to_tensor(image_resized)

    return img_tensor.unsqueeze(0)  # Add batch dimension


def postprocess_predictions(outputs, orig_size, conf_threshold=0.3):
    """Postprocess model outputs."""
    # Get predictions
    pred_logits = outputs['pred_logits'][0]  # [num_queries, num_classes]
    pred_boxes = outputs['pred_boxes'][0]    # [num_queries, 4]

    # Get scores and labels
    scores = pred_logits.sigmoid().max(dim=-1)
    conf_scores = scores.values
    labels = scores.indices

    # Filter by confidence
    keep = conf_scores > conf_threshold
    scores = conf_scores[keep]
    labels = labels[keep]
    boxes = pred_boxes[keep]

    # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] in pixels
    orig_h, orig_w = orig_size

    # Clone and extract values
    cx = boxes[:, 0].clone()
    cy = boxes[:, 1].clone()
    w = boxes[:, 2].clone()
    h = boxes[:, 3].clone()

    # Convert to corner format
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h

    # Stack back together
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # Clamp to image bounds
    boxes_xyxy[:, 0].clamp_(min=0, max=orig_w)
    boxes_xyxy[:, 1].clamp_(min=0, max=orig_h)
    boxes_xyxy[:, 2].clamp_(min=0, max=orig_w)
    boxes_xyxy[:, 3].clamp_(min=0, max=orig_h)

    return boxes_xyxy, labels, scores


def visualize_predictions(image, boxes, labels, scores, output_path):
    """Draw predictions on image and save."""
    draw = ImageDraw.Draw(image)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        font = ImageFont.load_default()

    # Color palette
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
              for _ in range(len(COCO_CLASSES))]

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.cpu().numpy()
        label_idx = label.item()
        score_val = score.item()

        # Get color for this class
        color = colors[label_idx % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label_text = f"{COCO_CLASSES[label_idx]}: {score_val:.2f}"
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        draw.rectangle([text_bbox[0]-2, text_bbox[1]-2, text_bbox[2]+2, text_bbox[3]+2], fill=color)
        draw.text((x1, y1), label_text, fill='white', font=font)

    # Save image
    image.save(output_path)
    print(f"Saved: {output_path}")


def main():
    """Main inference function."""
    import argparse

    parser = argparse.ArgumentParser(description='D-FINE Inference on Images from Folder')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, default='configs/dfine/dfine_hgnetv2_m_coco.yml',
                        help='Path to config file')
    parser.add_argument('--image-folder', type=str, required=True, help='Folder containing images')
    parser.add_argument('--num-images', type=int, default=None,
                        help='Number of random images (default: all)')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='/tmp/dfine', help='Output directory')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Load model
    model, cfg = load_model(args.checkpoint, args.config, args.device)

    # Load images from folder
    print(f"\nLoading images from {args.image_folder}...")
    images_data = load_images_from_folder(args.image_folder, args.num_images)

    # Run inference
    print(f"\nRunning inference...")
    device = torch.device(args.device)

    with torch.no_grad():
        for i, data in enumerate(images_data):
            image = data['image']
            filename = data['filename']
            orig_size = image.size[::-1]  # (height, width)

            print(f"[{i+1}/{len(images_data)}] Processing {filename}...")

            # Preprocess
            img_tensor = preprocess_image(image).to(device)

            # Inference
            outputs = model(img_tensor)

            # Postprocess
            boxes, labels, scores = postprocess_predictions(outputs, orig_size, args.conf_threshold)

            print(f"  Detected {len(boxes)} objects")

            # Visualize
            output_filename = f"result_{Path(filename).stem}.png"
            output_path = output_dir / output_filename
            visualize_predictions(image.copy(), boxes, labels, scores, output_path)

    print(f"\n{'='*60}")
    print(f"Inference complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
