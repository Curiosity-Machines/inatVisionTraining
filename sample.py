import os
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from datasets import inat_dataset
import augment

# Try to import PIL, but provide a fallback if it's not available
try:
    from PIL import Image
    has_pil = True
except ImportError:
    has_pil = False
    print("Warning: PIL not found. Will save images using tensorflow instead.")

def save_image(img, filepath):
    if has_pil:
        img_array = tf.keras.preprocessing.image.array_to_img(img[0])
        img_array.save(filepath, format='PNG')
    else:
        # Fallback method using tensorflow
        tf.io.write_file(filepath, tf.io.encode_png(tf.cast(img[0] * 255, tf.uint8)))

def verify_dataset(dataset_path, sample_size=1000, augment_magnitude=0.5):
    print(f"Verifying dataset: {dataset_path}")

    # Create samples directory if it doesn't exist
    samples_dir = "samples"
    os.makedirs(samples_dir, exist_ok=True)

    # Read the CSV file to get photo_ids
    df = pd.read_csv(dataset_path)
    photo_ids = df['photo_id'].tolist()

    # Load the dataset using the provided code
    ds, num_examples, _ = inat_dataset.make_dataset(
        dataset_path,
        label_column_name="taxon_id",
        image_size=(300, 300),  # You can adjust this size as needed
        batch_size=1,  # Process one image at a time for verification
        shuffle_buffer_size=10000,
        repeat_forever=False,
        augment_magnitude=augment_magnitude
    )

    print(f"Total examples: {num_examples}")

    # Calculate sampling rate
    sampling_rate = min(1.0, sample_size / num_examples)

    nan_count = 0
    sample_count = 0
    error_count = 0

    for i, (img, label) in enumerate(ds):
        photo_id = photo_ids[i]

        # Check for errors (including NaNs)
        if tf.math.reduce_any(tf.math.is_nan(img)) or tf.math.reduce_any(tf.math.is_nan(label)):
            nan_count += 1
            error_count += 1
            print(f"Error: NaN found in example {i}, photo_id: {photo_id}")
        elif tf.math.reduce_any(tf.math.is_inf(img)) or tf.math.reduce_any(tf.math.is_inf(label)):
            error_count += 1
            print(f"Error: Inf found in example {i}, photo_id: {photo_id}")

        # Randomly decide whether to save this image
        if random.random() < sampling_rate and sample_count < sample_size:
            save_image(img, os.path.join(samples_dir, f"sample_{photo_id}.png"))
            sample_count += 1

        if i % 1000 == 0:
            print(f"Processed {i} images")

    print("\nVerification Results:")
    print(f"Total images processed: {i+1}")
    print(f"Images with errors: {error_count}")
    print(f"Images with NaN values: {nan_count}")
    print(f"Samples saved: {sample_count}")

if __name__ == "__main__":
    # Verify training dataset
    verify_dataset("training_set.trim.csv")
    
    # Verify validation dataset
    verify_dataset("validation_set.csv")
