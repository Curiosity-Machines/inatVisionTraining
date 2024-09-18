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

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using CUDA GPU for image resizing.", file=sys.stderr)
elif torch.backends.mps.is_available():
    device = torch.device('mps')  # For MacOS Metal Performance Shaders
    print("Using MPS (Metal) GPU for image resizing.", file=sys.stderr)
else:
    device = torch.device('cpu')
    print("Warning: No GPU detected. Image resizing will be performed on CPU.", file=sys.stderr)
# ================== Configuration Constants ==================
NUM_SIZES = 8                   # Number of different square sizes to evaluate
OVERLAP = 0.333                  # Fractional overlap between squares (e.g., 0.2 for 20% overlap)
IMAGE_SIZE = 299                # Size to which each cropped image is resized
MIN_BOUNDING_BOX_RATIO = 0.33    # Minimum bounding box size as a ratio of max_size (e.g., 0.5 for 50%)
IMAGE_DIRECTORY = "data"        # Directory where images are stored (e.g., data/{photo_id}.jpg)
# =============================================================


def prepare_image(cropped_image):
    """
    Prepares the cropped image for inference by resizing and normalizing.

    Args:
        cropped_image (PIL.Image.Image): Cropped PIL Image.

    Returns:
        np.ndarray: Preprocessed image array in CHW format.
    """
    #if cropped_image.mode != "RGB":
    #    cropped_image = cropped_image.convert("RGB")
    
    # Resize to the model's expected input size using a high-quality resampling filter
    #resized_image = cropped_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    
    # Convert PIL Image to NumPy array and normalize to [0, 1]
    #image_array = np.array(resized_image).astype(np.float32) / 255.0
    
    # Transpose to CHW format (Channels, Height, Width) if required by the model
    #image_array = np.transpose(image_array, (2, 0, 1))
    
    # Add batch dimension
    #image_array = np.expand_dims(image_array, axis=0)
    image_np = np.array(cropped_image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Convert to PyTorch tensor and rearrange dimensions to CHW
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # Shape: (1, C, H, W)
    
    # Resize using PyTorch on GPU
    resized_tensor = torch.nn.functional.interpolate(image_tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
    
    # Move back to CPU and convert to NumPy
    resized_cpu = resized_tensor.cpu().numpy().transpose((0, 3, 2, 1))
    
    return resized_cpu  # Shape: (1, C, H, W)


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

def sweep_bounding_squares(image, session, input_name, taxon_id, taxon_index, taxon_ids):
    """
    Evaluates three possible crops (center, fully left/right or fully top/bottom depending on the aspect ratio).
    If taxon_id is not in the taxonomy, skips inference and returns a 87.5% square of the shorter side and a score of 0.0.

    Args:
        image (PIL.Image.Image): Original PIL Image.
        session (onnxruntime.InferenceSession): ONNX Runtime session.
        input_name (str): Name of the input node in the ONNX model.
        taxon_id (int): Taxon ID to score against.
        taxon_index (int): Index of the taxon in the model's output.
        taxon_ids (List[int]): List of valid taxon_ids from the taxonomy.

    Returns:
        Tuple[int, int, int, float]: (x_offset, y_offset, size, score)
    """
    width, height = image.size

    # If taxon_id is not in the taxonomy, skip inference
    if taxon_id not in taxon_ids:
        # Return a square that is 87.5% of the shorter dimension
        shorter_side = min(width, height)
        square_size = int(shorter_side * 0.875)
        return (int((width - square_size) / 2), int((height - square_size) / 2), square_size, 0.0)

    # Possible crops: center, fully left/right or top/bottom depending on aspect ratio
    crops = []

    # Central crop
    square_size = min(width, height)
    center_x = (width - square_size) // 2
    center_y = (height - square_size) // 2
    crops.append((center_x, center_y, square_size))

    # Landscape: left and right crops
    if width > height:
        crops.append((0, center_y, square_size))  # Left
        crops.append((width - square_size, center_y, square_size))  # Right
    # Portrait: top and bottom crops
    else:
        crops.append((center_x, 0, square_size))  # Top
        crops.append((center_x, height - square_size, square_size))  # Bottom

    best_score = -math.inf
    best_crop = (0, 0, square_size, 0.0)  # Default crop

    for x, y, size in crops:
        crop = image.crop((x, y, x + size, y + size))
        processed_crop = prepare_image(crop)  # Shape: (1, C, H, W)

        # Create the input dictionary
        input_feed = {input_name: processed_crop}

        # Run inference
        try:
            outputs = session.run(None, input_feed)
        except Exception as e:
            print(f"Error during model inference for taxon_id {taxon_id} and photo_id at position ({x}, {y}, {size}): {e}", file=sys.stderr)
            continue

        # Assuming the model outputs logits or probabilities
        output_data = outputs[0][0]  # Assuming output shape is (1, num_classes)

        # Get the score for the specified taxon
        score = float(output_data[taxon_index])

        if score > best_score:
            best_score = score
            best_crop = (x, y, size, score)

    return best_crop

def main():
    parser = argparse.ArgumentParser(description='Run ONNX model on multiple JPEG images with bounding square search using GPU acceleration on MacOS.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing taxon_id and photo_id pairs.')
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX model file (.onnx).')
    parser.add_argument('--taxonomy', type=str, required=True, help='Path to the taxonomy CSV.')
    parser.add_argument('--output_json', type=str, required=False, help='Path to save the output JSON. If not provided, prints to stdout.')
    args = parser.parse_args()

    # Validate file paths
    if not os.path.isfile(args.input_csv):
        print(f"Error: Input CSV file {args.input_csv} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"Error: ONNX model file {args.model} does not exist.", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(args.taxonomy):
        print(f"Error: Taxonomy CSV file {args.taxonomy} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load taxonomy
    leaf_df = load_taxonomy(args.taxonomy)
    taxon_ids = leaf_df['taxon_id'].tolist()

    # Load ONNX model with Metal execution provider
    try:
        # Set providers to use Metal for GPU acceleration
        print(ort.get_available_providers())
        session = ort.InferenceSession(args.model, providers=["CoreMLExecutionProvider", "CUDAExecutionProvider"], graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Error loading ONNX model from {args.model}: {e}", file=sys.stderr)
        sys.exit(1)

    # Read the input CSV
    try:
        input_df = pd.read_csv(args.input_csv, dtype={
            "taxon_id": int,
            "photo_id": str
        })
    except Exception as e:
        print(f"Error reading input CSV {args.input_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate required columns
    if not {'taxon_id', 'photo_id'}.issubset(input_df.columns):
        print(f"Error: Input CSV must contain 'taxon_id' and 'photo_id' columns.", file=sys.stderr)
        sys.exit(1)

    # Initialize the results list
    image_info_list = []

    # Process each row in the input CSV
    for idx, row in input_df.iterrows():
        taxon_id = row['taxon_id']
        photo_id = row['photo_id']

        # Locate the image file
        image_path = os.path.join(IMAGE_DIRECTORY, f"{photo_id}.jpg")
        if not os.path.isfile(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.", file=sys.stderr)
            continue

        # Verify the provided taxon_id exists in the taxonomy
        if taxon_id not in taxon_ids:
            print(f"Error: Taxon ID {taxon_id} not found in the taxonomy. Skipping photo_id {photo_id}.", file=sys.stderr)
            continue

        # Get the index of the taxon_id in the sorted leaf_df
        taxon_index = taxon_ids.index(taxon_id)

        # Open image using PIL
        try:
            image = Image.open(image_path)
        except Exception as e:
            print(f"Error opening image file {image_path}: {e}. Skipping.", file=sys.stderr)
            continue

        # Determine maximum square size based on image dimensions
        max_square_size = min(image.size)

        # Generate square sizes to evaluate with minimum bounding box ratio
        sizes = generate_square_sizes(
            max_size=max_square_size, 
            num_sizes=NUM_SIZES, 
            min_ratio=MIN_BOUNDING_BOX_RATIO
        )
        print(f"Processing photo_id {photo_id} with taxon_id {taxon_id}.", file=sys.stderr)

        # Sweep bounding squares to find the best crop
        best_crop = sweep_bounding_squares(
            image=image,
            session=session,
            input_name=input_name,
            taxon_id=taxon_id,
            taxon_index=taxon_index,
            taxon_ids=taxon_ids
        )

        best_x, best_y, best_size, best_score = best_crop

        # Append the result to the image_info_list
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
        image_info_list.append(image_info)

    # Prepare the final JSON structure
    final_result = {
        "image_info": image_info_list
    }

    # Output the JSON to a file or stdout
    if args.output_json:
        try:
            with open(args.output_json, 'w') as f:
                json.dump(final_result, f, indent=2)
            print(f"Output successfully written to {args.output_json}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing to output JSON file {args.output_json}: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Print JSON to stdout
        print(json.dumps(final_result, indent=2))

if __name__ == '__main__':
    main()

