import argparse
import pandas as pd
import onnxruntime as ort
import sys
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm
import json  # Added for JSON handling

# ================== Configuration Constants ==================
IMAGE_SIZE = 300                # Size to which each cropped image is resized
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
    crop_size = int(min(width, height))
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))

    # Resize the cropped image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # Convert to numpy array and normalize
    image_np = np.array(image).astype(np.float32) # / 255.0  # Normalize to [0, 1]

    # Convert to tensor and adjust dimensions
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)

    # Optionally, you can apply further normalization here if required by your model
    # For example:
    # mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    # std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    # image_tensor = (image_tensor - mean) / std

    resized_cpu = image_tensor.cpu().numpy().transpose((0, 2, 3, 1))  # Shape: (1, C, H, W)
    return resized_cpu

def load_taxon_map_json(taxon_map_json_path):
    """
    Load taxon mapping from a JSON file and create mappings between taxon_id and model class indices.
    
    The JSON file should have the format:
    {
        "taxon_id_1": label_index_1,
        "taxon_id_2": label_index_2,
        ...
    }
    
    Label indices should start at 1.
    """
    try:
        with open(taxon_map_json_path, 'r') as f:
            taxon_map = json.load(f)
    except Exception as e:
        print(f"Error reading taxon mapping JSON {taxon_map_json_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Convert taxon_id from string to int and ensure label indices are integers starting at 1
    taxon_id_to_label_index = {}
    for taxon_id_str, label_index in taxon_map.items():
        try:
            taxon_id = int(taxon_id_str)
            label_idx = int(label_index) + 1
            if label_idx < 1:
                raise ValueError("Label indices should start at 1.")
            taxon_id_to_label_index[taxon_id] = label_idx
        except ValueError as ve:
            print(f"Invalid taxon_id or label_index in JSON: {taxon_id_str}: {label_index}. Error: {ve}", file=sys.stderr)
            sys.exit(1)
    
    # Check if label indices are contiguous and start at 1
    label_indices = sorted(taxon_id_to_label_index.values())
    expected_indices = list(range(1, len(label_indices) + 1))
    if label_indices != expected_indices:
        print("Error: Label indices in JSON do not start at 1 or are not contiguous.", file=sys.stderr)
        sys.exit(1)
    
    # Create mappings
    taxon_id_to_index = {taxon_id: label_index - 1 for taxon_id, label_index in taxon_id_to_label_index.items()}  # 0-based
    index_to_taxon_id = {label_index - 1: taxon_id for taxon_id, label_index in taxon_id_to_label_index.items()}  # 0-based

    taxon_ids = list(taxon_id_to_label_index.keys())
    
    return taxon_ids, taxon_id_to_index, index_to_taxon_id

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
    # Removed the --taxonomy argument
    parser.add_argument('--taxon_map_json', type=str, required=True, help='Path to the JSON file containing taxon_id to label index mapping.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use (default: 0).')

    args = parser.parse_args()

    # Validate input files
    for file_path in [args.input_csv, args.model, args.taxon_map_json]:
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} does not exist.", file=sys.stderr)
            sys.exit(1)

    # Load taxon mapping from JSON
    taxon_ids, taxon_id_to_index, index_to_taxon_id = load_taxon_map_json(args.taxon_map_json)

    # Set PyTorch device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Initialize ONNX Runtime session
    try:
        if device.type == 'cuda':
            providers = ['CUDAExecutionProvider']
            provider_options = [{"device_id": args.gpu_id}]
        elif device.type == 'mps':
            providers = ['CoreMLExecutionProvider']
            provider_options = {}
        else:
            providers = ['CPUExecutionProvider']
            provider_options = {}
        
        session = ort.InferenceSession(
            args.model,
            providers=['CPUExecutionProvider'],  # Added CoreMLExecutionProvidern
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

