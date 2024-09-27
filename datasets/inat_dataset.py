import pandas as pd
import tensorflow as tf
from functools import partial
from tensorflow.keras import backend as K
import json
import augment
import os

AUTOTUNE = tf.data.AUTOTUNE

def _decode_img(img, file_path):
    img = tf.image.decode_jpeg(img, channels=3)
     #img = tf.image.convert_image_dtype(img, tf.float32)
    
    return img

def _process(photo_id, label, num_classes):
    # Load and preprocess image
    file_path = tf.strings.format("data/{}.jpg", photo_id)
    img = tf.io.read_file(file_path)
    img = _decode_img(img, file_path)
    label = tf.one_hot(label, num_classes)
    return img, label

def _flip(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    # right left only
    x = tf.image.random_flip_left_right(x)
    return x, y


def _color(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x, y

def crop_to_square(image, label):
    shape = tf.shape(image)
    min_dim = tf.minimum(shape[0], shape[1])
    return tf.image.resize_with_crop_or_pad(image, min_dim, min_dim), label

def random_crop_to_square(image, label):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        area_range=(0.66, 1.0),
        aspect_ratio_range=(1.0, 1.0),
        max_attempts=100,
        min_object_covered=0.1,
    )
    image = tf.slice(image, begin, size)

    return image, label

def _random_crop(x: tf.Tensor, y: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(x),
        bounding_boxes=bbox,
        #area_range=(0.08, 1.0),
        area_range=(0.5, 1.0),
        aspect_ratio_range=(0.75, 1.33),
        max_attempts=100,
        min_object_covered=0.1,
    )
    x = tf.slice(x, begin, size)

    return x, y


def _load_dataframe(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)
    return df

def _prepare_dataset(
    ds,
    image_size=None,
    batch_size=32,
    repeat_forever=True,
    augment_magnitude=0.0,
    label_to_index=None,
):
    if repeat_forever:
        ds = ds.repeat()

    if augment_magnitude > 0.0:
        aa = augment.RandAugment(magnitude=augment_magnitude, num_layers=3, exclude_ops=['Invert', 'Posterize', 'Solarize', 'SolarizeAdd'])
            
        ds = ds.map(lambda x, y: random_crop_to_square(x, y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (aa.distort(x), y), num_parallel_calls=AUTOTUNE)

    else:
        ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y), num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda x, y: (tf.image.random_flip_left_right(x), y), num_parallel_calls=AUTOTUNE)

    # Batch the dataset
    ds = ds.batch(batch_size)
    
    # Prefetch to optimize pipeline
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    
    return ds

def make_dataset(
    path,
    label_column_name,
    image_size=None,
    batch_size=32,
    shuffle_buffer_size=10_000,
    repeat_forever=True,
    augment_magnitude=0.0,
    label_to_index=None
):
    df = _load_dataframe(path)
    num_examples = len(df)
    
    if label_to_index is None:
        # If the json file exists, open it and start from those labels
        if os.path.exists(path.replace(".csv", "_label_mapping.json")):
            with open(path.replace(".csv", "_label_mapping.json"), "r") as f:
                label_to_index = json.load(f)
                label_to_index = {int(label): value for label, value in label_to_index.items()}
        else:
            label_to_index = {}

        # Append new labels to the existing label_to_index
        existing_labels = set(label_to_index.keys())
        dataset_labels = set(df[label_column_name].unique())
        new_labels = dataset_labels - existing_labels

        if new_labels:
            max_index = max(label_to_index.values())
            new_label_to_index = {int(label): idx + max_index for idx, label in enumerate(new_labels)}
            # Merge the existing and new label_to_index
            updated_label_to_index = {**label_to_index, **new_label_to_index}
            label_to_index = updated_label_to_index
            
        # No new labels to append
        num_classes = len(label_to_index)

        # Write the updated label_to_index to JSON
        with open(path.replace(".csv", "_label_mapping.json"), "w") as f:
            dump_label_to_index = {str(label): value for label, value in label_to_index.items()}
            json.dump(dump_label_to_index, f)
    else:
        # **Validation Set**: Use existing label mapping without appending
        num_classes = len(label_to_index)
        # Verify that all labels in validation set are present in training set
        unique_val_labels = set(df[label_column_name].unique())
        missing_labels = unique_val_labels - set(label_to_index.keys())
        if missing_labels:
            raise ValueError(f"Dataset contains labels not in the provided label_to_index: {len(missing_labels)} labels")
    
    df['label_index'] = df[label_column_name].map(label_to_index)
    
    # Check for any unmapped labels after mapping
    if df['label_index'].isnull().any():
        unmapped = df[df['label_index'].isnull()][label_column_name].unique()
        raise ValueError(f"Found labels in the dataset that are not in label_to_index: {unmapped}")
    
    df['label_index'] = df['label_index'].astype(int)
    
    ds = tf.data.Dataset.from_tensor_slices((df["photo_id"], df["label_index"]))
    
    ds = ds.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    
    process_partial = partial(_process, num_classes=num_classes)
    ds = ds.map(process_partial, num_parallel_calls=AUTOTUNE)
    
    ds = _prepare_dataset(
        ds,
        image_size=image_size,
        batch_size=batch_size,
        repeat_forever=repeat_forever,
        augment_magnitude=augment_magnitude,
        label_to_index=label_to_index,  # Pass label_to_index if needed
    )
    
    return (ds, num_examples, label_to_index)
