import pandas as pd
import tensorflow as tf
from functools import partial
from tensorflow.keras import backend as K
import json
import augment

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

    #if augment:
    #    # crop 100% of the time
    #    ds = ds.map(lambda x, y: _random_crop(x, y), num_parallel_calls=AUTOTUNE)
    #else:
    #    # central crop
    #    ds = ds.map(lambda x, y: (tf.image.central_crop(x, 0.875), y), num_parallel_calls=AUTOTUNE)

    #ds = ds.map(lambda x, y: (tf.image.resize(x, image_size), y), num_parallel_calls=AUTOTUNE)

    #if augment:
    #    # flip 50% of the time
    #    # the function already flips 50% of the time, so we call it 100% of the time
    #    ds = ds.map(lambda x, y: _flip(x, y), num_parallel_calls=AUTOTUNE)
    #    # do color 30% of the time
    #    ds = ds.map(
    #        lambda x, y: tf.cond(
    #            tf.random.uniform([], 0, 1) > 0.7, lambda: _color(x, y), lambda: (x, y)
    #        ),
    #        num_parallel_calls=AUTOTUNE,
    #    )
    #    # make sure the color transforms haven't move any of the pixels outside of [0,1]
    #    ds = ds.map(
    #        lambda x, y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=AUTOTUNE
    #    )

    ## Convert to uint8 0-255
    #ds = ds.map(lambda x, y: (tf.cast(x * 255, tf.uint8), y), num_parallel_calls=AUTOTUNE)

    if augment_magnitude > 0.0:
        aa = augment.RandAugment(magnitude=augment_magnitude)
            
        ds = ds.map(lambda x, y: crop_to_square(x, y), num_parallel_calls=AUTOTUNE)
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
    num_classes = len(df[label_column_name].unique())

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
        num_classes = len(label_to_index)
        # Verify that all labels in validation set are present in training set
        unique_val_labels = set(df[label_column_name].unique())
        missing_labels = unique_val_labels - set(label_to_index.keys())
        if missing_labels:
            raise ValueError(f"set contains labels not in training set: {missing_labels}")

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
        augment_magnitude=augment_magnitude,
    )

    return (ds, num_examples, label_to_index)

