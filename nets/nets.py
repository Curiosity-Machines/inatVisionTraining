import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomCrop, RandomRotation, Resizing, RandomBrightness, BatchNormalization, GlobalAveragePooling2D, Activation, Dropout, Dense
from dynamic_dropout import DynamicDropout

def make_neural_network(
    base_arch_name,
    weights,
    image_size,
    n_classes,
    input_dtype,
    train_full_network,
    ckpt,
    factorize=False,
    fact_rank=None,
    dropout_rate=0.5,
    l2_reg=1e-5,
    augment=True
):
    input_size = image_size[0]  # Assuming square images
    # Define the target size for the base architecture
    # Removed Resizing layer, as images are already resized in the data pipeline

    base_arch = None
    if base_arch_name == "xception":
        base_arch = keras.applications.Xception
    elif base_arch_name == "efficientnetb1":
        base_arch = keras.applications.EfficientNetV2B1
    elif base_arch_name == "efficientnetb3":
        base_arch = keras.applications.EfficientNetV2B3
    elif base_arch_name == "efficientnetb4":
        base_arch = keras.applications.EfficientNetV2B4
    if not base_arch:
        print("Unsupported base architecture.")
        return None

    inputs = keras.layers.Input(shape=(input_size, input_size, 3), dtype=input_dtype)

    data_augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.25),
        layers.RandomRotation(0.3),
        layers.RandomBrightness(0.2)
    ], name="data_augmentation")

    x = data_augmentation(inputs) if augment else inputs

    # Removed the Resizing layer
    # x = layers.Resizing(target_size_with_channels[0], target_size_with_channels[1])(x)

    base_model = base_arch(
        input_shape=(input_size, input_size, 3), weights=weights, include_top=False
    )
    base_model.trainable = train_full_network
    if ckpt is not None:
        base_model.load_weights(ckpt)

    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)

    if factorize and fact_rank is not None:
        x = tf.reshape(x, (-1, 1, 1, x.shape[-1]))
        svd_u = layers.Conv2D(fact_rank, (1, 1))(x)
        svd_u = layers.BatchNormalization()(svd_u)
        svd_u = layers.Activation('relu')(svd_u)
        svd_u = DynamicDropout(initial_rate=dropout_rate)(svd_u)
        logits = layers.Conv2D(n_classes, (1, 1))(svd_u)
        logits = layers.Reshape([n_classes])(logits)
    else:
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = DynamicDropout(initial_rate=dropout_rate)(x)
        logits = layers.Dense(n_classes, name="dense_logits")(x)

    output = layers.Activation("softmax", dtype="float32", name="predictions")(logits)
    model = keras.Model(inputs=inputs, outputs=output)
    return model

