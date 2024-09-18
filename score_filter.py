#!/usr/bin/env python3

import sys
import os
import json
import csv
import argparse
from collections import defaultdict

# Constants
SCORE_THRESHOLD = 0.2  # Score threshold to filter out low-quality images

def parse_arguments():
    parser = argparse.ArgumentParser(description="Filter images based on JSON score and ensure minimum N images per taxon.")
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file with taxon_id and photo_id fields.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where <photo_id>.json files are stored.')
    parser.add_argument('--min_images_per_taxon', type=int, default=25, help='Minimum number of images to include per taxon, even if scores are low.')
    return parser.parse_args()

def load_json(json_path):
    """Load JSON data from the given file path."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {json_path}: {e}")
        return None

def process_csv(input_csv, data_dir, min_images_per_taxon):
    """Process the input CSV file, grouping images by taxon and filtering based on score."""
    taxon_groups = defaultdict(list)  # Dictionary to group photos by taxon_id
    
    # Read the input CSV and group photo_ids by taxon_id
    with open(input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            taxon_id = row.get('taxon_id')
            photo_id = row.get('photo_id')
            
            # Check if JSON file exists for this photo_id
            json_path = os.path.join(data_dir, f"{photo_id}.json")
            if not os.path.isfile(json_path):
                continue  # Skip if JSON file doesn't exist
            
            # Load JSON and check the score
            json_data = load_json(json_path)
            if json_data is None:
                continue  # Skip if there was an error loading the JSON
            
            score = json_data.get('score', 0)
            taxon_groups[taxon_id].append({
                'photo_id': photo_id,
                'score': score
            })
    
    return taxon_groups

def select_top_images(taxon_groups, min_images_per_taxon):
    """Select the top images for each taxon based on score, ensuring at least min_images_per_taxon are selected."""
    selected_images = []

    for taxon_id, photos in taxon_groups.items():
        # Sort photos by score in descending order
        photos_sorted = sorted(photos, key=lambda x: x['score'], reverse=True)

        # Filter out photos with score > SCORE_THRESHOLD
        valid_photos = [photo for photo in photos_sorted if photo['score'] > SCORE_THRESHOLD]

        # If we don't have enough valid photos, include more until we reach min_images_per_taxon
        if len(valid_photos) < min_images_per_taxon:
            # Add remaining photos (with the highest scores) to make up the difference
            valid_photos.extend(photos_sorted[len(valid_photos):min_images_per_taxon])
        
        # Add the selected valid photos for this taxon
        selected_images.extend([{'taxon_id': taxon_id, 'photo_id': photo['photo_id']} for photo in valid_photos])
    
    return selected_images

def main():
    args = parse_arguments()

    # Process the input CSV and group photo_ids by taxon
    taxon_groups = process_csv(args.input_csv, args.data_dir, args.min_images_per_taxon)

    # Select the top N images per taxon, ensuring at least min_images_per_taxon are included
    selected_images = select_top_images(taxon_groups, args.min_images_per_taxon)

    # Write the selected taxon_id and photo_id pairs to stdout as CSV
    writer = csv.DictWriter(sys.stdout, fieldnames=["taxon_id", "photo_id"])
    writer.writeheader()
    writer.writerows(selected_images)

if __name__ == "__main__":
    main()

