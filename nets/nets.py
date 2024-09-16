import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomCrop, RandomRotation, Resizing, RandomBrightness, BatchNormalization, GlobalAveragePooling2D, Activation, Dropout, Dense

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
    input_size = 768
    # Define the target size for the base architecture
    target_size_with_channels = image_size + [3]  # e.g., [299, 299, 3]

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
        RandomFlip("horizontal"),
        RandomZoom(0.25),
        RandomRotation(0.3),
        RandomBrightness(0.2)
        # RandomCrop can be added if needed, ensure the output size matches image_size
        # For slight cropping, we can use Resizing with cropping
    ], name="data_augmentation")

    x = None

    if augment:
        x = data_augmentation(inputs)
    else:
        x = inputs

    x = Resizing(target_size_with_channels[0], target_size_with_channels[1])(x)

    base_model = base_arch(
        input_shape=target_size_with_channels, weights=weights, include_top=False
    )
    base_model.trainable = train_full_network
    if ckpt is not None:
        base_model.load_weights(ckpt)

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)

    if factorize and fact_rank is not None:
        x = tf.reshape(x, (-1, 1, 1, x.shape[-1]))
        svd_u = Conv2D(fact_rank, (1, 1))(x)
        svd_u = BatchNormalization()(svd_u)
        svd_u = Activation('relu')(svd_u)
        svd_u = Dropout(dropout_rate)(svd_u)
        logits = Conv2D(n_classes, (1, 1))(svd_u)
        logits = Reshape([n_classes])(logits)
    else:
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)
        logits = Dense(n_classes, name="dense_logits")(x)

    output = keras.layers.Activation("softmax", dtype="float32", name="predictions")(logits)
    model = keras.Model(inputs=inputs, outputs=output)
    return model
