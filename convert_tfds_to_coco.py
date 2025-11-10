"""
Convert TensorFlow Datasets COCO to standard COCO format.

This script downloads COCO 2017 via TensorFlow Datasets and converts it
to the standard COCO format expected by D-FINE.

Output structure:
~/data/COCO2017/
├── train2017/
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
├── val2017/
│   ├── 000000000001.jpg
│   └── ...
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
"""

import json
import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    import tensorflow_datasets as tfds
except ImportError:
    print("ERROR: tensorflow_datasets not installed")
    print("Install with: pip install tensorflow-datasets tensorflow")
    exit(1)


def convert_bbox_format(bbox, height, width):
    """
    Convert normalized bbox [ymin, xmin, ymax, xmax] to COCO format [x, y, w, h].

    TFDS format: [ymin, xmin, ymax, xmax] in normalized coordinates [0, 1]
    COCO format: [x, y, width, height] in absolute pixel coordinates
    """
    ymin, xmin, ymax, xmax = bbox

    # Convert to absolute coordinates
    x = xmin * width
    y = ymin * height
    w = (xmax - xmin) * width
    h = (ymax - ymin) * height

    return [float(x), float(y), float(w), float(h)]


def convert_split(split_name, output_dir):
    """
    Convert a single split (train/val) to COCO format.

    Args:
        split_name: 'train' or 'validation'
        output_dir: Path to output directory (~/data/COCO2017)
    """
    print(f"\n{'='*60}")
    print(f"Converting {split_name} split...")
    print(f"{'='*60}")

    # Load dataset
    print(f"Loading COCO 2017 {split_name} split from TensorFlow Datasets...")
    ds, info = tfds.load(
        "coco/2017",
        split=split_name,
        with_info=True,
        shuffle_files=False
    )

    # Get category names
    category_names = info.features['objects']['label'].names
    num_examples = info.splits[split_name].num_examples

    print(f"Dataset info:")
    print(f"  - Number of images: {num_examples}")
    print(f"  - Number of categories: {len(category_names)}")

    # Create output directories
    if split_name == 'train':
        img_dir = output_dir / "train2017"
        ann_file = output_dir / "annotations" / "instances_train2017.json"
    else:  # validation
        img_dir = output_dir / "val2017"
        ann_file = output_dir / "annotations" / "instances_val2017.json"

    img_dir.mkdir(parents=True, exist_ok=True)
    ann_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize COCO annotation structure
    coco_dict = {
        "info": {
            "description": "COCO 2017 Dataset (converted from TensorFlow Datasets)",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # COCO 2017 has 80 categories with non-contiguous IDs
    # Map from TFDS index (0-79) to original COCO category IDs
    coco_category_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
        80, 81, 82, 84, 85, 86, 87, 88, 89, 90
    ]

    # Add categories with correct COCO IDs
    for idx, name in enumerate(category_names):
        coco_dict['categories'].append({
            "id": coco_category_ids[idx],  # Use original COCO category IDs
            "name": name,
            "supercategory": "object"
        })

    # Process each example
    ann_id = 1  # Annotation ID counter

    print(f"\nProcessing {num_examples} images...")
    for example in tqdm(ds, total=num_examples, desc=f"Converting {split_name}"):
        # Get image data
        image_id = int(example['image/id'].numpy())
        image = example['image'].numpy()
        height, width = image.shape[:2]

        # Save image
        img_filename = f"{image_id:012d}.jpg"
        img_path = img_dir / img_filename

        # Convert numpy array to PIL Image and save
        pil_image = Image.fromarray(image)
        pil_image.save(img_path, quality=95)

        # Add image info
        coco_dict['images'].append({
            "id": image_id,
            "file_name": img_filename,
            "height": height,
            "width": width,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        })

        # Process annotations
        bboxes = example['objects']['bbox'].numpy()
        labels = example['objects']['label'].numpy()
        is_crowd = example['objects']['is_crowd'].numpy()
        areas = example['objects']['area'].numpy()

        for bbox, label, crowd, area in zip(bboxes, labels, is_crowd, areas):
            # Convert bbox format
            coco_bbox = convert_bbox_format(bbox, height, width)

            # Calculate area if not provided
            if area == 0:
                area = coco_bbox[2] * coco_bbox[3]
            else:
                area = float(area) * height * width  # Denormalize area

            # Add annotation
            coco_dict['annotations'].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": coco_category_ids[int(label)],  # Use original COCO category ID
                "bbox": coco_bbox,
                "area": float(area),
                "iscrowd": int(crowd),
                "segmentation": []
            })
            ann_id += 1

    # Save annotations JSON
    print(f"\nSaving annotations to {ann_file}...")
    with open(ann_file, 'w') as f:
        json.dump(coco_dict, f)

    print(f"\n{split_name.upper()} conversion complete!")
    print(f"  - Images saved to: {img_dir}")
    print(f"  - Annotations saved to: {ann_file}")
    print(f"  - Total images: {len(coco_dict['images'])}")
    print(f"  - Total annotations: {len(coco_dict['annotations'])}")

    return coco_dict


def main():
    """Main conversion function."""
    # Set output directory
    output_dir = Path.home() / "data" / "COCO2017"

    print("="*60)
    print("TensorFlow Datasets COCO → Standard COCO Format Converter")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")

    # Check if output directory exists and warn
    if output_dir.exists():
        response = input(f"\n{output_dir} already exists. Continue? (y/n): ")
        if response.lower() != 'y':
            print("Conversion cancelled.")
            return

    # Convert train split
    train_dict = convert_split('train', output_dir)

    # Convert validation split
    val_dict = convert_split('validation', output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDataset location: {output_dir}")
    print(f"\nTraining set:")
    print(f"  - Images: {len(train_dict['images'])}")
    print(f"  - Annotations: {len(train_dict['annotations'])}")
    print(f"  - Categories: {len(train_dict['categories'])}")
    print(f"\nValidation set:")
    print(f"  - Images: {len(val_dict['images'])}")
    print(f"  - Annotations: {len(val_dict['annotations'])}")
    print(f"  - Categories: {len(val_dict['categories'])}")

    print(f"\nTo use with D-FINE, update your config YAML:")
    print(f"  train_dataloader:")
    print(f"    dataset:")
    print(f"      img_folder: {output_dir}/train2017")
    print(f"      ann_file: {output_dir}/annotations/instances_train2017.json")
    print(f"  val_dataloader:")
    print(f"    dataset:")
    print(f"      img_folder: {output_dir}/val2017")
    print(f"      ann_file: {output_dir}/annotations/instances_val2017.json")


if __name__ == "__main__":
    main()
