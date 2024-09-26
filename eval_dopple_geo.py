import argparse
import pandas as pd
import onnxruntime as ort
import sys
from PIL import Image
import os
import numpy as np
import torch
from tqdm import tqdm
import json
import math

# ================== Configuration Constants ==================
IMAGE_SIZE = 300                # Size to which each cropped image is resized
IMAGE_DIRECTORY = "data"        # Directory where images are stored (e.g., data/{photo_id}.jpg)
# =============================================================

# ================== Geographic Model Imports ==================
# Assuming you have access to the necessary modules from your geographic model script
# such as models, utils, CoordEncoder, etc.
# If they are in separate files, ensure they are accessible in the PYTHONPATH
from models import get_geo_model  # Modify based on your actual module structure
from utils import CoordEncoder  # Modify based on your actual module structure
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
    image_index_to_taxon_id = {label_index - 1: taxon_id for taxon_id, label_index in taxon_id_to_label_index.items()}  # 0-based

    taxon_ids = list(taxon_id_to_label_index.keys())
    
    return taxon_ids, taxon_id_to_index, image_index_to_taxon_id

def predict_taxon_image_classifier(image, session, input_name, image_index_to_taxon_id, device):
    """
    Run inference on the preprocessed image using the image classifier and return the probabilities.
    """
    processed_image = prepare_image(image, device)
    input_feed = {input_name: processed_image}

    try:
        outputs = session.run(None, input_feed)
    except Exception as e:
        print(f"Error during image classifier inference: {e}", file=sys.stderr)
        return None, None

    # Assuming the output is a single output tensor with shape (1, num_classes)
    output_data = outputs[0][0]  # Shape: (num_classes,)
    probabilities = torch.softmax(torch.from_numpy(output_data), dim=0).numpy()  # Softmax to get probabilities
    top1_index = np.argmax(probabilities)
    predicted_taxon_id = image_index_to_taxon_id.get(top1_index, None)
    return predicted_taxon_id, probabilities

def load_geographic_model(geo_model_path, env_data_path, device):
    """
    Load the geographic model and environmental data.
    """
    # Load the geographic model
    geo_train_params = torch.load(geo_model_path, map_location=device)
    geo_model = get_geo_model(geo_train_params['params'])
    geo_model.load_state_dict(geo_train_params['state_dict'], strict=True)
    geo_model = geo_model.to(device)
    geo_model.eval()

    # Load environmental data
    env_feats = np.load(env_data_path).astype(np.float32)
    env_feats = torch.from_numpy(env_feats).to(device)

    # Initialize CoordEncoder
    coord_encoder = CoordEncoder('sin_cos_env', raster=env_feats)

    return geo_model, coord_encoder, geo_train_params

def predict_taxon_geographic(lat, lon, geo_model, geo_params, coord_encoder, device):
    """
    Predict taxon probabilities using the geographic model based on latitude and longitude.
    """
    # Prepare input coordinates
    loc = np.array([lon, lat]).astype(np.float32)  # Shape: (2,)
    loc_tensor = torch.from_numpy(loc).unsqueeze(0).to(device)  # Shape: (1, 2)

    # Encode coordinates
    loc_feat = coord_encoder.encode(loc_tensor)

    # Run geographic model inference
    with torch.no_grad():
        geo_pred = geo_model(loc_feat, return_feats=True)  # Assuming return_feats=True returns embeddings
        # Depending on your geo_model's architecture, adjust the following accordingly
        # For example, if it outputs probabilities directly:
        # geo_probs = torch.sigmoid(geo_pred).cpu().numpy()[0]
        # Here, we'll assume it outputs logits or raw scores
        geo_logits = geo_model.class_emb(geo_pred).cpu().numpy()[0]  # Shape: (num_classes,)
        geo_probs = torch.softmax(torch.from_numpy(geo_logits), dim=0).numpy()  # Shape: (num_classes,)

    return geo_probs

def combine_scores(vision_probs, geo_probs):
    """
    Combine vision and geographic probabilities by multiplying them.
    """
    combined = vision_probs * geo_probs
    combined /= combined.sum()  # Normalize to sum to 1
    return combined

def combine_scores_top_n(vision_probs, geo_probs, top_n=3, image_index_to_taxon_id=None, taxon_id_to_geo_index=None, actual_taxon_id=None):
    """
    Combine vision and geographic probabilities by:
    1. Selecting the top N taxa from vision_probs.
    2. Re-ranking these top N based on geo_probs.
    3. Selecting the top 1 taxon as the final prediction.

    Returns:
        combined_predicted_taxon_id: The final predicted taxon_id.
    """
    # Get top N indices from vision_probs
    top_n_vision_indexes = np.argsort(vision_probs)[::-1][:top_n]
    top_n_taxon_ids = [image_index_to_taxon_id.get(idx, None) for idx in top_n_vision_indexes]
    top_n_vision_probs = vision_probs[top_n_vision_indexes]

    # Now create a list of pairs (taxon_id, vison_prob * geo_prob) for the top N taxa
    top_n_combined_probs = []

    wild_count = 0
    wild_geo_probs = []
    num_geo_prob = 0
    min_geo_prob = 1
    sum_geo_prob = 0

    for i, taxon_id in enumerate(top_n_taxon_ids):
        if taxon_id_to_geo_index.get(taxon_id) is not None:
            geo_prob = geo_probs[taxon_id_to_geo_index.get(taxon_id)]
            
            if geo_prob > 1e-8:
                vision_prob = top_n_vision_probs[i]

                sqrt_geo_prob = math.sqrt(math.sqrt(math.sqrt(geo_prob))) # Precision issues

                wild_geo_probs.append((taxon_id, vision_prob * sqrt_geo_prob))
            else:
                break
        else:
            break

    # if the top matches are wild, then use the geography model
    if len(wild_geo_probs) > 0:
        # Sort by combined probability
        wild_geo_probs = sorted(wild_geo_probs, key=lambda x: x[1], reverse=True)
        top_n_combined_probs = wild_geo_probs[:top_n]
        reordered_taxa_id = top_n_combined_probs[0][0]
        previous_taxa_id = top_n_taxon_ids[0]

        if reordered_taxa_id != previous_taxa_id:
            if reordered_taxa_id == actual_taxon_id or previous_taxa_id == actual_taxon_id:
                if reordered_taxa_id == actual_taxon_id:
                    print(f"Correctly reordered taxa: {actual_taxon_id}")
                else:
                    print(f"Incorrectly reordered taxa: {actual_taxon_id}")

        return reordered_taxa_id
        
    else:
        return top_n_taxon_ids[0]

def main():
    parser = argparse.ArgumentParser(description='Compute classification accuracy of an ONNX model on a dataset with geographic integration.')
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file containing taxon_id, photo_id, lat, and lon columns.')
    parser.add_argument('--model', type=str, required=True, help='Path to the ONNX image classifier model file (.onnx).')
    parser.add_argument('--taxon_map_json', type=str, required=True, help='Path to the JSON file containing taxon_id to label index mapping.')
    parser.add_argument('--geo_model', type=str, required=True, help='Path to the geographic model file (.pt).')
    parser.add_argument('--env_data', type=str, required=True, help='Path to the environmental data file (bioclim_elevation_scaled.npy).')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID to use (default: 0).')
    parser.add_argument('--top_n', type=int, default=3, help='Number of top vision predictions to consider for re-ranking (default: 3).')

    args = parser.parse_args()

    # Validate input files
    for file_path in [args.input_csv, args.model, args.taxon_map_json, args.geo_model, args.env_data]:
        if not os.path.isfile(file_path):
            print(f"Error: File {file_path} does not exist.", file=sys.stderr)
            sys.exit(1)

    # Load taxon mapping from JSON
    taxon_ids, taxon_id_to_index, image_index_to_taxon_id = load_taxon_map_json(args.taxon_map_json)

    # Set PyTorch device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Initialize ONNX Runtime session for image classifier
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
            providers=providers,
            provider_options=provider_options,
            graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        )
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Error initializing ONNX Runtime session: {e}", file=sys.stderr)
        sys.exit(1)

    # Load the geographic model and environmental data
    try:
        geo_model, coord_encoder, geo_params = load_geographic_model(args.geo_model, args.env_data, device)
    except Exception as e:
        print(f"Error loading geographic model: {e}", file=sys.stderr)
        sys.exit(1)

    # Load input CSV
    try:
        input_df = pd.read_csv(args.input_csv, dtype={"taxon_id": int, "photo_id": str, "lat": float, "lon": float})
    except Exception as e:
        print(f"Error reading input CSV {args.input_csv}: {e}", file=sys.stderr)
        sys.exit(1)

    required_columns = {'taxon_id', 'photo_id', 'lat', 'lon'}
    if not required_columns.issubset(input_df.columns):
        print(f"Error: Input CSV must contain columns: {required_columns}", file=sys.stderr)
        sys.exit(1)

    total = len(input_df)
    correct_vision = 0
    correct_geo = 0
    correct_combined = 0
    processed = 0

    # Initialize lists to store probabilities if needed
    combined_probabilities = []

    # Iterate over each row with a progress bar
    for idx, row in tqdm(input_df.iterrows(), total=total, desc="Processing images"):
        actual_taxon_id = row['taxon_id']
        photo_id = row['photo_id']
        lat = row['lat']
        lon = row['lon']
        image_path = os.path.join(IMAGE_DIRECTORY, f"{photo_id}.jpg")

        if not os.path.isfile(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping.", file=sys.stderr)
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not open image {image_path}: {e}. Skipping.", file=sys.stderr)
            continue

        # Image Classifier Prediction
        predicted_taxon_id_vision, vision_probs = predict_taxon_image_classifier(
            image=image,
            session=session,
            input_name=input_name,
            image_index_to_taxon_id=image_index_to_taxon_id,
            device=device
        )

        if predicted_taxon_id_vision is None:
            print(f"Warning: Image classifier prediction failed for image {image_path}. Skipping.", file=sys.stderr)
            continue

        # Increment vision accuracy
        if predicted_taxon_id_vision == actual_taxon_id:
            correct_vision += 1

        # Geographic Model Prediction
        geo_probs = predict_taxon_geographic(
            lat=lat,
            lon=lon,
            geo_model=geo_model,
            geo_params=geo_params,
            coord_encoder=coord_encoder,
            device=device
        )

        if geo_probs is None:
            print(f"Warning: Geographic model prediction failed for image {image_path}. Skipping geographic scoring.", file=sys.stderr)
            # Optionally, you can decide to continue without geo_scores
            continue

        geo_index_to_taxons = geo_params['params']['class_to_taxa'] # this is a list
        geo_index_to_taxon_id = {geo_index: geo_index_to_taxons[geo_index] for geo_index in range(len(geo_index_to_taxons))}

        taxon_id_to_geo_index = {taxon_id: geo_index for geo_index, taxon_id in geo_index_to_taxon_id.items()}

        # Combine Vision and Geographic Scores using Top N Re-ranking
        combined_predicted_taxon_id = combine_scores_top_n(
            vision_probs=vision_probs,
            geo_probs=geo_probs,
            top_n=args.top_n,
            image_index_to_taxon_id=image_index_to_taxon_id,
            taxon_id_to_geo_index=taxon_id_to_geo_index,
            actual_taxon_id=actual_taxon_id
        )

        if combined_predicted_taxon_id is None:
            print(f"Warning: Combined prediction failed for image {image_path}. Skipping.", file=sys.stderr)
            continue

        # Increment combined accuracy
        if combined_predicted_taxon_id == actual_taxon_id:
            correct_combined += 1

        # Increment geographic-only accuracy based on geo_probs
        geo_top1_index = np.argmax(geo_probs)
        geo_predicted_taxon_id = image_index_to_taxon_id.get(geo_top1_index, None)
        if geo_predicted_taxon_id == actual_taxon_id:
            correct_geo += 1

        # Optional: Print progress every 100 images
        if processed > 0 and processed % 100 == 0:
            accuracy_vision = correct_vision / processed * 100
            accuracy_geo = correct_geo / processed * 100
            accuracy_combined = correct_combined / processed * 100
            print(f"\nProcessed {processed} images out of {total}.")
            print(f"Correct Predictions (Vision Only): {correct_vision}")
            print(f"Accuracy (Vision Only): {accuracy_vision:.2f}%")
            print(f"Correct Predictions (Geographic Only): {correct_geo}")
            print(f"Accuracy (Geographic Only): {accuracy_geo:.2f}%")
            print(f"Correct Predictions (Combined): {correct_combined}")
            print(f"Accuracy (Combined): {accuracy_combined:.2f}%")

        processed += 1

    if processed == 0:
        print("No images were processed. Cannot compute accuracy.", file=sys.stderr)
        sys.exit(1)

    # Final Accuracy Calculations
    accuracy_vision = correct_vision / processed * 100
    accuracy_geo = correct_geo / processed * 100
    accuracy_combined = correct_combined / processed * 100

    print(f"\nProcessed {processed} images out of {total}.")
    print(f"Correct Predictions (Vision Only): {correct_vision}")
    print(f"Accuracy (Vision Only): {accuracy_vision:.2f}%")
    print(f"Correct Predictions (Geographic Only): {correct_geo}")
    print(f"Accuracy (Geographic Only): {accuracy_geo:.2f}%")
    print(f"Correct Predictions (Combined): {correct_combined}")
    print(f"Accuracy (Combined): {accuracy_combined:.2f}%")

    # Optionally, save combined predictions to a CSV
    # Adding a new column to the input_df with the combined taxon_id
    # This approach avoids re-predicting and ensures consistency
    # We'll append the combined_predicted_taxon_id during the loop
    # To do this, let's collect all combined predictions

    # Ensure that combined_probabilities list matches the processed rows
    # Since some images were skipped, align the indices accordingly
    # We'll add a new column 'predicted_taxon_id_combined' with NaNs initially
    input_df = input_df.reset_index(drop=True)
    input_df['predicted_taxon_id_combined'] = np.nan

    # Re-initialize counters for combined predictions
    processed_combined = 0

if __name__ == '__main__':
    main()

