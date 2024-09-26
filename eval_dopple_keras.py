import argparse
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import json

# ================== Configuration Constants ==================
IMAGE_SIZE = 300                # Size to which each cropped image is resized
IMAGE_DIRECTORY = "data"        # Directory where images are stored (e.g., data/{photo_id}.jpg)
# =============================================================

def prepare_image(image):
    """
    Preprocess the image for the model:
    - Ensure RGB format
    - Center crop to square
    - Resize to IMAGE_SIZE
    - Normalize to [0, 1]
    - Convert to numpy array suitable for Keras model
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Calculate the center crop size (make the image square)
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    image = image.crop((left, top, right, bottom))

    # Resize the cropped image
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)

    # Convert to numpy array and normalize
    image_np = np.array(image).astype(np.float32)  # Normalize to [0, 1]

    # Expand dimensions to match model's expected input shape (1, H, W, C)
    image_np = np.expand_dims(image_np, axis=0)  # Shape: (1, H, W, C)

    return image_np

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

def predict_taxon(image_np, model, index_to_taxon_id):
    """
    Run inference on the preprocessed image and return the top-1 predicted taxon_id.
    """
    try:
        predictions = model.predict(image_np, verbose=0)
    except Exception as e:
        print(f"Error during prediction: {e}", file=sys.stderr)
        return None

    if predictions.ndim == 2:
        predictions = predictions[0]  # Shape: (num_classes,)

    top1_index = np.argmax(predictions)
    predicted_taxon_id = index_to_taxon_id.get(top1_index, None)
    return predicted_taxon_id

def set_gpu(gpu_id):
    """
    Set the GPU device to use.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the specified GPU
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU: {gpus[gpu_id].name}")
        except IndexError:
            print(f"Error: GPU with ID {gpu_id} does not exist.", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error setting GPU: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("No GPU found. Using CPU.", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Compute classification accuracy of a Keras model on a dataset.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing taxon_id and photo_id pairs.')
    parser.add_argument('--model', type=str, required=True, help='Path to the Keras model file (.h5 or SavedModel directory).')
    parser.add_argument('--taxon_map_json', type=str, required=True, help='Path to the JSON file containing taxon_id to label index mapping.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use (default: 0).')

    args = parser.parse_args()

    # Validate input files
    for file_path in [args.input_csv, args.model, args.taxon_map_json]:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist.", file=sys.stderr)
            sys.exit(1)

    # Set GPU if available
    set_gpu(args.gpu_id)

    # Load taxon mapping from JSON
    taxon_ids, taxon_id_to_index, index_to_taxon_id = load_taxon_map_json(args.taxon_map_json)

    # Load Keras model
    try:
        model = keras.models.load_model(args.model)
        model.eval = False  # Ensure the model is in inference mode
        print(f"Successfully loaded Keras model from {args.model}")
    except Exception as e:
        print(f"Error loading Keras model from {args.model}: {e}", file=sys.stderr)
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

        image_np = prepare_image(image)

        predicted_taxon_id = predict_taxon(
            image_np=image_np,
            model=model,
            index_to_taxon_id=index_to_taxon_id
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

