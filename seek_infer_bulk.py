import argparse
import pandas as pd
import onnxruntime as ort
import json
import sys
from PIL import Image
import math
import os
import numpy as np
import torch

# ================== Configuration Constants ==================
NUM_SIZES = 8                   # Number of different square sizes to evaluate
OVERLAP = 0.333                  # Fractional overlap between squares (e.g., 0.2 for 20% overlap)
IMAGE_SIZE = 299                # Size to which each cropped image is resized
MIN_BOUNDING_BOX_RATIO = 0.33    # Minimum bounding box size as a ratio of max_size (e.g., 0.5 for 50%)
IMAGE_DIRECTORY = "data"        # Directory where images are stored (e.g., data/{photo_id}.jpg)
CROPPED_DIRECTORY = "data_cropped"  # Directory where cropped JSONs are saved (e.g., data_cropped/{photo_id}.json)
# =============================================================

def prepare_image(cropped_image, device):
    image_np = np.array(cropped_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)
    resized_tensor = torch.nn.functional.interpolate(image_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    resized_cpu = resized_tensor.cpu().numpy().transpose((0, 3, 2, 1))
    return resized_cpu  # Shape: (1, C, H, W)

def load_taxonomy(taxonomy_path):
    try:
        taxonomy_df = pd.read_csv(taxonomy_path, dtype={
            "parent_taxon_id": "Int64",
            "taxon_id": int,
            "rank_level": float,
            "leaf_class_id": "Int64",
            "iconic_class_id": "Int64",
            "spatial_class_id": "Int64",
            "name": pd.StringDtype()
        })
    except Exception as e:
        print(f"Error reading taxonomy CSV {taxonomy_path}: {e}", file=sys.stderr)
        sys.exit(1)

    leaf_df = taxonomy_df.dropna(subset=['leaf_class_id']).copy()
    leaf_df['leaf_class_id'] = leaf_df['leaf_class_id'].astype(int)
    leaf_df = leaf_df.sort_values('leaf_class_id').reset_index(drop=True)

    expected_start = 0
    actual_start = leaf_df['leaf_class_id'].min()
    if actual_start != expected_start:
        leaf_df['leaf_class_id'] = leaf_df['leaf_class_id'] - actual_start

    expected_ids = list(range(len(leaf_df)))
    actual_ids = leaf_df['leaf_class_id'].tolist()
    if actual_ids != expected_ids:
        sys.exit(1)

    return leaf_df

def sweep_bounding_squares(image, session, input_name, taxon_id, taxon_index, taxon_ids, device):
    width, height = image.size
    if taxon_id not in taxon_ids:
        shorter_side = min(width, height)
        square_size = int(shorter_side * 0.875)
        return (int((width - square_size) / 2), int((height - square_size) / 2), square_size, 0.0)

    crops = []
    square_size = min(width, height)
    center_x = (width - square_size) // 2
    center_y = (height - square_size) // 2
    crops.append((center_x, center_y, square_size))

    if width > height:
        crops.append((0, center_y, square_size))  # Left
        crops.append((width - square_size, center_y, square_size))  # Right
    else:
        crops.append((center_x, 0, square_size))  # Top
        crops.append((center_x, height - square_size, square_size))  # Bottom

    best_score = -math.inf
    best_crop = (0, 0, square_size, 0.0)

    for x, y, size in crops:
        crop = image.crop((x, y, x + size, y + size))
        processed_crop = prepare_image(crop, device)
        input_feed = {input_name: processed_crop}

        try:
            outputs = session.run(None, input_feed)
        except Exception as e:
            continue

        output_data = outputs[0][0]
        score = float(output_data[taxon_index])

        if score > best_score:
            best_score = score
            best_crop = (x, y, size, score)

    return best_crop

def process_image(row, taxon_ids, leaf_df, session, input_name, device):
    taxon_id = row['taxon_id']
    photo_id = row['photo_id']
    image_path = os.path.join(IMAGE_DIRECTORY, f"{photo_id}.jpg")
    output_json_path = os.path.join(CROPPED_DIRECTORY, f"{photo_id}.json")

    if os.path.exists(output_json_path):
        return

    if not os.path.isfile(image_path):
        return

    taxon_index = taxon_ids.index(taxon_id) if taxon_id in taxon_ids else -1

    try:
        image = Image.open(image_path)
    except Exception as e:
        return

    best_crop = sweep_bounding_squares(
        image=image,
        session=session,
        input_name=input_name,
        taxon_id=taxon_id,
        taxon_index=taxon_index,
        taxon_ids=taxon_ids,
        device=device
    )

    best_x, best_y, best_size, best_score = best_crop
    image_info = {
        "photo_id": photo_id,
        "taxon_id": taxon_id,
        "best_crop": {
            "x_offset": best_x,
            "y_offset": best_y,
            "size": best_size
        },
        "score": best_score
    }

    with open(output_json_path, 'w') as f:
        json.dump(image_info, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Run ONNX model on multiple JPEG images with bounding square search using GPU acceleration.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing taxon_id and photo_id pairs.')
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model file (.onnx).')
    parser.add_argument('--taxonomy', type=str, required=True, help='Path to the taxonomy CSV.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use (default: 0).')

    args = parser.parse_args()

    if not os.path.isfile(args.input_csv):
        sys.exit(1)
    if not os.path.isfile(args.model):
        sys.exit(1)
    if not os.path.isfile(args.taxonomy):
        sys.exit(1)

    leaf_df = load_taxonomy(args.taxonomy)
    taxon_ids = leaf_df['taxon_id'].tolist()

    # Set PyTorch device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    try:
        # Specify the GPU device ID for CUDAExecutionProvider
        provider_options = {"device_id": args.gpu_id}
        session = ort.InferenceSession(
            args.model,
            providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"],
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        input_name = session.get_inputs()[0].name
    except Exception as e:
        sys.exit(1)

    try:
        input_df = pd.read_csv(args.input_csv, dtype={"taxon_id": int, "photo_id": str})
    except Exception as e:
        sys.exit(1)

    if not {'taxon_id', 'photo_id'}.issubset(input_df.columns):
        sys.exit(1)

    if not os.path.exists(CROPPED_DIRECTORY):
        os.makedirs(CROPPED_DIRECTORY)

    for idx, row in input_df.iterrows():
        process_image(row, taxon_ids, leaf_df, session, input_name, device)

if __name__ == '__main__':
    main()

