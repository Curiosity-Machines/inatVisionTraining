# inat_dataset.py

import pandas as pd
import tensorflow as tf
from functools import partial
import json
from tensorflow.keras import backend as K

AUTOTUNE = tf.data.AUTOTUNE

def _decode_img(img):
    # Decode JPEG, convert to float32 in [0,1]
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def _process(photo_id, label, num_classes):
    # Load and preprocess image
    file_path = tf.strings.format("data/{}.jpg", photo_id)
    img = tf.io.read_file(file_path)
    img = _decode_img(img)
    label = tf.one_hot(label, num_classes)
    return img, label

def _load_dataframe(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)
    return df

def _prepare_dataset(
    ds,
    image_size=(299, 299),
    batch_size=32,
    repeat_forever=True,
    augment=False,
):
    if repeat_forever:
        ds = ds.repeat()

    target_size = tf.constant([image_size[0], image_size[1]], dtype=tf.int32)

    def preprocess_image(image):
        # Determine the shape of the image
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        # Calculate the size of the square crop
        crop_size = tf.minimum(height, width)
        
        # Inset the crop size by 12.5%
        inset_factor = 0.875  # 1 - 0.125
        inset_crop_size = tf.cast(tf.cast(crop_size, tf.float32) * inset_factor, tf.int32)

        # Calculate offsets for center crop
        height_offset = (height - inset_crop_size) // 2
        width_offset = (width - inset_crop_size) // 2

        # Perform center crop
        cropped_image = tf.image.crop_to_bounding_box(
            image, 
            height_offset, 
            width_offset, 
            inset_crop_size, 
            inset_crop_size
        )

        # Resize the image to the target size
        resized_image = tf.image.resize(cropped_image, target_size)

        return resized_image

    ds = ds.map(lambda x, y: (preprocess_image(x), y), num_parallel_calls=AUTOTUNE)

    # Apply data augmentation if specified
    if augment:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
            # Add more augmentation layers if needed
        ])
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size)
    
    # Prefetch to optimize pipeline
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

def make_dataset(
    path,
    label_column_name,
    image_size=(299, 299),
    batch_size=32,
    shuffle_buffer_size=10_000,
    repeat_forever=True,
    augment=False,
    label_to_index=None,  # New parameter
):
    df = _load_dataframe(path)
    num_examples = len(df)
    
    if label_to_index is None:
        # **Training Set**: Create label mapping
        labels = sorted(df[label_column_name].unique())
        label_to_index = {int(label): index for index, label in enumerate(labels)}
        num_classes = len(labels)

        # Write it to a file
        with open(path.replace(".csv", "_label_mapping.json"), "w") as f:
            json.dump(label_to_index, f)
    else:
        # **Validation Set**: Use existing label mapping
        labels = None  # Not needed for validation
        num_classes = len(label_to_index)
        # Verify that all labels in validation set are present in training set
        unique_val_labels = set(df[label_column_name].unique())
        missing_labels = unique_val_labels - set(label_to_index.keys())
        if missing_labels:
            raise ValueError(f"Validation set contains labels not in training set: {missing_labels}")

    # Map labels to indices
    df['label_index'] = df[label_column_name].map(label_to_index)
    
    ds = tf.data.Dataset.from_tensor_slices((df["photo_id"], df["label_index"]))
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)

    process_partial = partial(_process, num_classes=num_classes)
    ds = ds.map(process_partial, num_parallel_calls=AUTOTUNE)

    ds = _prepare_dataset(
        ds,
        image_size=image_size,
        batch_size=batch_size,
        repeat_forever=repeat_forever,
        augment=augment,
    )

    if labels is not None:
        return (ds, num_examples, labels, label_to_index)
    else:
        return (ds, num_examples)

