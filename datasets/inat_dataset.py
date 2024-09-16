import pandas as pd
import tensorflow as tf
from functools import partial
from tensorflow.keras import backend as K
import json

AUTOTUNE = tf.data.AUTOTUNE

def _decode_img(img, file_path):
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Assert that the image has three channels
    tf.debugging.assert_equal(
        tf.shape(img)[-1],
        3,
        message="Image does not have 3 channels (RGB)."
    )
    
    # Compute min and max pixel values
    min_value = tf.reduce_min(img)
    max_value = tf.reduce_max(img)
    
    # Assert that pixel values are within [0, 255]
    tf.debugging.assert_greater_equal(
        min_value, tf.constant(0, dtype=img.dtype), message="Image has pixel values below 0."
    )
    tf.debugging.assert_less_equal(
        max_value, tf.constant(255, dtype=img.dtype), message="Image has pixel values above 255."
    )
    
    # Assert that the image is not entirely zero
    total_sum = tf.reduce_sum(tf.cast(img, tf.int32))
    tf.debugging.assert_greater(
        total_sum, 0, message="Image is entirely zeros."
    )
    
    return img


def _process(photo_id, label, num_classes):
    # Load and preprocess image
    file_path = tf.strings.format("data/{}.jpg", photo_id)
    img = tf.io.read_file(file_path)
    img = _decode_img(img, file_path)
    label = tf.one_hot(label, num_classes)
    return img, label

def _load_dataframe(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42)
    return df

def preprocess_image(image, target_size):
    # Determine the shape of the image
    #shape = tf.shape(image)
    #height, width = shape[0], shape[1]

    ## Calculate the size of the square crop
    #crop_size = tf.minimum(height, width)
    #
    ## Inset the crop size by 12.5%
    #inset_factor = 0.875  # 1 - 0.125
    #inset_crop_size = tf.cast(tf.cast(crop_size, tf.float32) * inset_factor, tf.int32)

    ## Calculate offsets for center crop
    #height_offset = (height - inset_crop_size) // 2
    #width_offset = (width - inset_crop_size) // 2

    ## Perform center crop
    #cropped_image = tf.image.crop_to_bounding_box(
    #    image, 
    #    height_offset, 
    #    width_offset, 
    #    inset_crop_size, 
    #    inset_crop_size
    #)

    ## Resize the image to the target size
    #resized_image = tf.image.resize(cropped_image, target_size)
    #print("Shape:", resized_image.shape)
    #print("Data type:", resized_image.dtype)

    ## Calculate and print min and max values
    #min_value = tf.reduce_min(resized_image)
    #max_value = tf.reduce_max(resized_image)
    #print("Min value:", min_value)
    #print("Max value:", max_value)
    resized_image = tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])

    return resized_image

def _prepare_dataset(
    ds,
    image_size=(299, 299),
    batch_size=32,
    repeat_forever=True,
    augment=False,
    label_to_index=None,
):
    if repeat_forever:
        ds = ds.repeat()

    target_size = tf.constant([image_size[0], image_size[1]], dtype=tf.int32)
    
    ds = ds.map(lambda x, y: (preprocess_image(x, target_size), y), num_parallel_calls=AUTOTUNE)

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
        labels = None  # Not needed for validation
        num_classes = len(label_to_index)
        # Verify that all labels in validation set are present in training set
        unique_val_labels = set(df[label_column_name].unique())
        missing_labels = unique_val_labels - set(label_to_index.keys())
        if missing_labels:
            raise ValueError(f"Validation set contains labels not in training set: {missing_labels}")

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

