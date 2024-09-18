#!/usr/bin/env python3

import os
import csv
import argparse
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

def parse_arguments():
    parser = argparse.ArgumentParser(description="Resize images so that the shortest axis is 512px.")
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file with taxon_id and photo_id fields.')
    parser.add_argument('--source_dir', type=str, required=True, help='Directory where the original <photo_id>.jpg files are located.')
    parser.add_argument('--dest_dir', type=str, required=True, help='Directory where the resized <photo_id>.jpg files will be saved.')
    parser.add_argument('--num_threads', type=int, default=4, help='Number of threads to use for parallel processing.')
    return parser.parse_args()

def resize_image(photo_id, source_dir, dest_dir):
    """
    Resize an image so that its shortest axis is 512px while maintaining aspect ratio.
    Saves the resulting image in the destination directory.
    """
    source_image_path = os.path.join(source_dir, f"{photo_id}.jpg")
    dest_image_path = os.path.join(dest_dir, f"{photo_id}.jpg")

    # If the destination image already exists, skip processing
    if os.path.isfile(dest_image_path):
        return

    # Check if the source image exists
    if not os.path.isfile(source_image_path):
        return

    try:
        with Image.open(source_image_path) as img:
            original_width, original_height = img.size

            # Determine the scaling factor based on the shortest axis
            if original_width < original_height:
                new_width = 512
                new_height = int((512 / original_width) * original_height)
            else:
                new_height = 512
                new_width = int((512 / original_height) * original_width)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Save the resized image to the destination directory
            resized_img.save(dest_image_path)

    except Exception as e:
        print(f"Error processing image {source_image_path}: {e}")

def process_row(row, source_dir, dest_dir):
    """Process a single row from the CSV file by resizing the image."""
    photo_id = row.get('photo_id')
    if not photo_id:
        print("No photo_id found in row. Skipping.")
        return

    resize_image(photo_id, source_dir, dest_dir)

def process_csv(input_csv, source_dir, dest_dir, num_threads):
    """Process the input CSV file and resize images using multiple threads."""
    with open(input_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)  # Read all rows into memory to parallelize processing

        # Use a ThreadPoolExecutor to process rows in parallel
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_row, row, source_dir, dest_dir) for row in rows]

            # Wait for all threads to complete
            for future in as_completed(futures):
                try:
                    future.result()  # Will raise any exceptions that occurred in threads
                except Exception as e:
                    print(f"Error in thread: {e}")

def main():
    args = parse_arguments()
    
    # Ensure destination directory exists
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    # Process the input CSV file
    process_csv(args.input_csv, args.source_dir, args.dest_dir, args.num_threads)

if __name__ == "__main__":
    main()

