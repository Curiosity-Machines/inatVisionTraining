import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.layers import RandomFlip, RandomZoom, RandomCrop, RandomRotation, Resizing, RandomBrightness, BatchNormalization, GlobalAveragePooling2D, Activation, Dropout, Dense
from efficientnetv2 import EfficientNetV2_S

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
    dropout=0.0
):
    input_size = image_size[0]
    # Define the target size for the base architecture
    # Removed Resizing layer, as images are already resized in the data pipeline

    #inputs = keras.layers.Input(shape=(input_size, input_size, 3), dtype=input_dtype)
    print(f"Building efficientnet S with shape {input_size}, num classes {n_classes}, dropout {dropout}")

    return EfficientNetV2_S(input_shape=(input_size, input_size, 3), pretrained=weights, num_classes=n_classes, dropout_rate=dropout, include_top=True, final_dropout_rate=dropout, weights=weights, pooling="avg", classes=n_classes)
    #base_model.trainable = train_full_network

    #if ckpt is not None:
    #    base_model.load_weights(ckpt)

    #if factorize and fact_rank is not None:
    #    x = keras.layers.GlobalAveragePooling2D(keepdims=True)(base_model.output)
    #    x = BatchNormalization()(x)
    #    svd_u = layers.Conv2D(fact_rank, (1, 1))(x)
    #    svd_u = Dropout(dropout)(svd_u)
    #    logits = layers.Conv2D(n_classes, (1, 1))(svd_u)
    #    logits = layers.Reshape([n_classes])(logits)
    #else:
    #    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    #    x = BatchNormalization()(x)
    #    x = Dropout(dropout)(x)
    #    logits = layers.Dense(n_classes, name="dense_logits")(x)

    #output = layers.Activation("softmax", dtype="float32", name="predictions")(logits)
    #model = keras.Model(inputs=inputs, outputs=output)
    #return model

