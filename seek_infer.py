import argparse
import pandas as pd
import tensorflow as tf
import json
import sys
from PIL import Image
import math
import os
import numpy as np

# ================== Configuration Constants ==================
NUM_SIZES = 5                   # Number of different square sizes to evaluate
OVERLAP = 0.2                   # Fractional overlap between squares (e.g., 0.2 for 20% overlap)
IMAGE_SIZE = 299                # Size to which each cropped image is resized
MIN_BOUNDING_BOX_RATIO = 0.5    # Minimum bounding box size as a ratio of max_size (e.g., 0.5 for 50%)
# =============================================================

def prepare_image(cropped_image):
    """
    Prepares the cropped image for inference by resizing and normalizing.

    Args:
        cropped_image (PIL.Image.Image): Cropped PIL Image.

    Returns:
        np.ndarray: Preprocessed image array.
    """
    if cropped_image.mode != "RGB":
        cropped_image = cropped_image.convert("RGB")
    
    # Convert PIL Image to TensorFlow Tensor and normalize
    image = tf.image.convert_image_dtype(tf.convert_to_tensor(np.array(cropped_image)), tf.float32)
    
    # Resize to the model's expected input size
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE], method=tf.image.ResizeMethod.AREA)
    
    # Expand dims to add batch dimension
    image = tf.expand_dims(image, 0)
    return image.numpy()

def load_taxonomy(taxonomy_path):
    """
    Loads the taxonomy CSV and prepares a mapping from model output indices to taxon IDs.

    Args:
        taxonomy_path (str): Path to the taxonomy CSV file.

    Returns:
        pd.DataFrame: DataFrame containing taxonomy information sorted by leaf_class_id.
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

    # Filter rows where leaf_class_id is not null
    leaf_df = taxonomy_df.dropna(subset=['leaf_class_id']).copy()

    # Ensure leaf_class_id is integer
    leaf_df['leaf_class_id'] = leaf_df['leaf_class_id'].astype(int)

    # Sort by leaf_class_id to align with model output
    leaf_df = leaf_df.sort_values('leaf_class_id').reset_index(drop=True)

    # Verify that leaf_class_id starts at 0 or 1 and is contiguous
    expected_start = 0  # Change to 1 if your model starts indexing at 1
    actual_start = leaf_df['leaf_class_id'].min()
    if actual_start != expected_start:
        print(f"Warning: leaf_class_id starts at {actual_start}, expected {expected_start}. Adjusting indices.", file=sys.stderr)
        # Adjust leaf_class_id to start at 0
        leaf_df['leaf_class_id'] = leaf_df['leaf_class_id'] - actual_start

    # Check for contiguity
    expected_ids = list(range(len(leaf_df)))
    actual_ids = leaf_df['leaf_class_id'].tolist()
    if actual_ids != expected_ids:
        print("Error: leaf_class_id is not contiguous or does not start at the expected index.", file=sys.stderr)
        sys.exit(1)

    return leaf_df
def sweep_bounding_squares(image, sizes, overlap, model_interpreter, input_details, output_details, taxon_id, taxon_index):
    """
    Sweeps bounding squares across the image to find the best crop for the specified taxon.

    Args:
        image (PIL.Image.Image): Original PIL Image.
        sizes (List[int]): List of square sizes to evaluate.
        overlap (float): Fractional overlap between squares.
        model_interpreter (tf.lite.Interpreter): TensorFlow Lite interpreter.
        input_details (dict): Model input details.
        output_details (dict): Model output details.
        taxon_id (int): Taxon ID to score against.
        taxon_index (int): Index of the taxon in the model's output.

    Returns:
        Tuple[int, int, int, float]: (x_offset, y_offset, size, score)
    """
    width, height = image.size
    best_score = -math.inf
    best_crop = (0, 0, min(width, height), 0.0)  # Default crop

    for size in sizes:
        if size > width or size > height:
            continue  # Skip sizes larger than the image dimensions

        step_size = max(1, int(size * (1 - overlap)))

        # Calculate the number of steps in each direction
        x_steps = math.ceil((width - size) / step_size) + 1
        y_steps = math.ceil((height - size) / step_size) + 1

        for i in range(x_steps):
            for j in range(y_steps):
                x = i * step_size
                y = j * step_size

                # Ensure the crop is within image bounds
                if x + size > width:
                    x = width - size
                if y + size > height:
                    y = height - size

                crop = image.crop((x, y, x + size, y + size))
                processed_crop = prepare_image(crop)

                # Ensure the input data type matches the model's input data type
                if processed_crop.dtype != input_details[0]['dtype']:
                    processed_crop = processed_crop.astype(input_details[0]['dtype'])

                # Set the tensor to the first input of the model
                model_interpreter.set_tensor(input_details[0]['index'], processed_crop)

                # Run inference
                try:
                    model_interpreter.invoke()
                except Exception as e:
                    print(f"Error during model inference: {e}", file=sys.stderr)
                    continue  # Skip this crop

                # Get the output tensor
                try:
                    output_data = model_interpreter.get_tensor(output_details[0]['index'])[0]
                except Exception as e:
                    print(f"Error retrieving model output: {e}", file=sys.stderr)
                    continue  # Skip this crop

                # Get the score for the specified taxon
                if taxon_index >= len(output_data):
                    print(f"Error: Taxon index {taxon_index} out of bounds for model output.", file=sys.stderr)
                    sys.exit(1)
                score = float(output_data[taxon_index])

                if score > best_score:
                    best_score = score
                    best_crop = (x, y, size, score)

    return best_crop

def generate_square_sizes(max_size, num_sizes, min_ratio=0.5):
    """
    Generates a list of square sizes to evaluate, ensuring the minimum size
    is not below a specified ratio of the maximum size.

    Args:
        max_size (int): Maximum square size in pixels.
        num_sizes (int): Number of different sizes to generate.
        min_ratio (float): Minimum size ratio (e.g., 0.5 for 50%).

    Returns:
        List[int]: List of square sizes.
    """
    min_size = int(max_size * min_ratio)
    if num_sizes < 2:
        raise ValueError("num_sizes must be at least 2 to include max and min sizes.")

    step = (max_size - min_size) / (num_sizes - 1)
    sizes = [int(max_size - i * step) for i in range(num_sizes)]
    return sizes

def main():
    parser = argparse.ArgumentParser(description='Run TensorFlow Lite model on a JPEG image with bounding square search.')
    parser.add_argument('--image', type=str, required=True, help='Path to the JPEG image.')
    parser.add_argument('--model', type=str, required=True, help='Path to the TensorFlow Lite model.')
    parser.add_argument('--taxonomy', type=str, required=True, help='Path to the taxonomy CSV.')
    parser.add_argument('--taxon_id', type=int, required=True, help='Taxon ID to score against.')
    args = parser.parse_args()

    # Validate file paths
    if not os.path.isfile(args.image):
        print(f"Error: Image file {args.image} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: Model file {args.model} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.taxonomy):
        print(f"Error: Taxonomy CSV file {args.taxonomy} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load taxonomy
    leaf_df = load_taxonomy(args.taxonomy)
    taxon_ids = leaf_df['taxon_id'].tolist()

    # Verify the provided taxon_id exists in the taxonomy
    if args.taxon_id not in taxon_ids:
        print(f"Error: Taxon ID {args.taxon_id} not found in the taxonomy.", file=sys.stderr)
        sys.exit(1)

    # Get the index of the taxon_id in the sorted leaf_df
    taxon_index = taxon_ids.index(args.taxon_id)

    # Load TensorFlow Lite model
    try:
        interpreter = tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Assume single input and single output
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
    except Exception as e:
        print(f"Error loading TensorFlow Lite model {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    # Open image using PIL
    try:
        image = Image.open(args.image)
    except Exception as e:
        print(f"Error opening image file {args.image}: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine maximum square size based on image dimensions
    max_square_size = min(image.size)

    # Generate square sizes to evaluate with minimum bounding box ratio
    sizes = generate_square_sizes(
        max_size=max_square_size, 
        num_sizes=NUM_SIZES, 
        min_ratio=MIN_BOUNDING_BOX_RATIO
    )

    # Sweep bounding squares to find the best crop
    best_x, best_y, best_size, best_score = sweep_bounding_squares(
        image=image,
        sizes=sizes,
        overlap=OVERLAP,
        model_interpreter=interpreter,
        input_details=input_details,
        output_details=output_details,
        taxon_id=args.taxon_id,
        taxon_index=taxon_index
    )

    # ... [output formatting code] ...
    result = {
        "taxon_id": args.taxon_id,
        "best_crop": {
            "x_offset": best_x,
            "y_offset": best_y,
            "size": best_size
        },
        "score": best_score
    }

    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
