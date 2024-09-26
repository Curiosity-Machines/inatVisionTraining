import argparse
import pandas as pd
import onnxruntime as ort
import sys
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm

# ================== Configuration Constants ==================
IMAGE_SIZE = 299                # Size to which each cropped image is resized
IMAGE_DIRECTORY = "data"        # Directory where images are stored (e.g., data/{photo_id}.jpg)
# =============================================================

def prepare_image(image, device):
    """
    Preprocess the image for the model:
    - Ensure RGB format
    - Normalize to [0, 1]
    - Resize to IMAGE_SIZE
    - Convert to tensor format suitable for the model
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate the center crop size (87.5% of the original size)
    width, height = image.size
    crop_size = int(min(width, height) * 0.875)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))

    # Resize the cropped image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    # Convert to numpy array and normalize
    image_np = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Convert to tensor and adjust dimensions
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)

    # Optionally, you can apply further normalization here if required by your model
    # For example:
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    # image_tensor = (image_tensor - mean) / std

    resized_cpu = image_tensor.cpu().numpy().transpose((0, 2, 3, 1))  # Shape: (1, C, H, W)
    return resized_cpu

def load_taxonomy(taxonomy_path):
    """
    Load taxonomy CSV and create mappings between taxon_id and model class indices.
    """
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

    # Filter to leaf classes
    leaf_df = taxonomy_df.dropna(subset=['leaf_class_id']).copy()
    leaf_df['leaf_class_id'] = leaf_df['leaf_class_id'].astype(int)
    leaf_df = leaf_df.sort_values('leaf_class_id').reset_index(drop=True)

    # Ensure leaf_class_id starts at 0 and is contiguous
    expected_start = 0
    actual_start = leaf_df['leaf_class_id'].min()
    if actual_start != expected_start:
        leaf_df['leaf_class_id'] = leaf_df['leaf_class_id'] - actual_start

    expected_ids = list(range(len(leaf_df)))
    actual_ids = leaf_df['leaf_class_id'].tolist()
    if actual_ids != expected_ids:
        print("Error: leaf_class_id does not start at 0 or has gaps.", file=sys.stderr)
        sys.exit(1)

    # Create mappings
    taxon_id_to_index = dict(zip(leaf_df['taxon_id'], leaf_df['leaf_class_id']))
    index_to_taxon_id = dict(zip(leaf_df['leaf_class_id'], leaf_df['taxon_id']))
    taxon_ids = leaf_df['taxon_id'].tolist()

    return leaf_df, taxon_id_to_index, index_to_taxon_id

def predict_taxon(image, session, input_name, index_to_taxon_id, device):
    """
    Run inference on the preprocessed image and return the top-1 predicted taxon_id.
    """
    processed_image = prepare_image(image, device)
    input_feed = {input_name: processed_image}

    try:
        outputs = session.run(None, input_feed)
    except Exception as e:
        print(f"Error during inference: {e}", file=sys.stderr)
        return None

    output_data = outputs[0][0]
    top1_index = np.argmax(output_data)
    predicted_taxon_id = index_to_taxon_id.get(top1_index, None)
    return predicted_taxon_id

def main():
    parser = argparse.ArgumentParser(description='Compute classification accuracy of an ONNX model on a dataset.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing taxon_id and photo_id pairs.')
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model file (.onnx).')
    parser.add_argument('--taxonomy', type=str, required=True, help='Path to the taxonomy CSV.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use (default: 0).')

    args = parser.parse_args()

    # Validate input files
    for file_path in [args.input_csv, args.model, args.taxonomy]:
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} does not exist.", file=sys.stderr)
            sys.exit(1)

    # Load taxonomy
    leaf_df, taxon_id_to_index, index_to_taxon_id = load_taxonomy(args.taxonomy)
    taxon_ids = leaf_df['taxon_id'].tolist()

    # Set PyTorch device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Initialize ONNX Runtime session
    try:
        provider_options = {"device_id": args.gpu_id}
        session = ort.InferenceSession(
            args.model,
            providers=[("CPUExecutionProvider", provider_options), "CPUExecutionProvider"],
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Error initializing ONNX Runtime session: {e}", file=sys.stderr)
        sys.exit(1)

    # Load input CSV
    try:
        input_df = pd.read_csv(args.input_csv, dtype={"taxon_id": int, "photo_id": str})
    except Exception as e:
        print(f"Error reading input CSV {args.input_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    if not {'taxon_id', 'photo_id'}.issubset(input_df.columns):
        print("Error: Input CSV must contain 'taxon_id' and 'photo_id' columns.", file=sys.stderr)
        sys.exit(1)

    total = len(input_df)
    correct = 0
    processed = 0

    # Iterate over each row with a progress bar
    for idx, row in tqdm(input_df.iterrows(), total=total, desc="Processing images"):
        actual_taxon_id = row['taxon_id']
        photo_id = row['photo_id']
        image_path = os.path.join(IMAGE_DIRECTORY, f"{photo_id}.jpg")

        if actual_taxon_id not in taxon_ids:
            continue

        if not os.path.isfile(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.", file=sys.stderr)
            continue

        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}. Skipping.", file=sys.stderr)
            continue

        predicted_taxon_id = predict_taxon(
            image=image,
            session=session,
            input_name=input_name,
            index_to_taxon_id=index_to_taxon_id,
            device=device
        )

        if predicted_taxon_id is None:
            print(f"Warning: Prediction failed for image {image_path}. Skipping.", file=sys.stderr)
            continue

        if predicted_taxon_id == actual_taxon_id:
            correct += 1

        if processed > 0 and processed % 100 == 0:
            accuracy = correct / processed * 100
            print(f"\nProcessed {processed} images out of {total}.")
            print(f"Correct Predictions: {correct}")
            print(f"Accuracy: {accuracy:.2f}%")

        processed += 1

    if processed == 0:
        print("No images were processed. Cannot compute accuracy.", file=sys.stderr)
        sys.exit(1)

    accuracy = correct / processed * 100
    print(f"\nProcessed {processed} images out of {total}.")
    print(f"Correct Predictions: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == '__main__':
    main()

