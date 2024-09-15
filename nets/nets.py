import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

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
    dropout_rate=0.5,  # Dropout rate parameter
    l2_reg=1e-4        # L2 regularization strength
):
    image_size_with_channels = image_size + [3]
    base_arch = None
    if base_arch_name == "xception":
        base_arch = keras.applications.Xception
    elif base_arch_name == "efficientnetb1":
        base_arch = keras.applications.EfficientNetB1
    if not base_arch:
        print("Unsupported base architecture.")
        return None

    inputs = keras.layers.Input(shape=image_size_with_channels, dtype=input_dtype)
    base_model = base_arch(
        input_shape=image_size_with_channels, weights=weights, include_top=False
    )
    base_model.trainable = train_full_network
    if ckpt is not None:
        base_model.load_weights(ckpt)

    x = base_model(inputs)
    
    # Add dropout after the base model
    x = keras.layers.Dropout(dropout_rate, name="base_dropout")(x)

    if factorize and fact_rank is not None:
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        
        # Add Conv2D with L2 regularization
        svd_u = keras.layers.Conv2D(
            fact_rank, 
            kernel_size=(1, 1),
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name="svd_conv"
        )(tf.expand_dims(x, axis=1))
        
        # Add dropout before the final Conv2D layer
        svd_u = keras.layers.Dropout(dropout_rate, name="svd_dropout")(svd_u)
        
        # Add final Conv2D with L2 regularization
        logits = keras.layers.Conv2D(
            n_classes, 
            kernel_size=(1, 1),
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg),
            name="logits_conv"
        )(svd_u)
        
        logits = keras.layers.Reshape([n_classes], name="logits_reshape")(logits)
    else:
        x = keras.layers.GlobalAveragePooling2D(name="gap")(x)
        
        # Add dropout before the Dense layer
        x = keras.layers.Dropout(dropout_rate, name="dense_dropout")(x)
        
        # Add Dense layer with L2 regularization
        logits = keras.layers.Dense(
            n_classes, 
            activation=None,
            kernel_regularizer=regularizers.l2(l2_reg),
            name="dense_logits"
        )(x)

    # Softmax activation for the output
    output = keras.layers.Activation(
        "softmax", 
        dtype="float32", 
        name="predictions"
    )(logits)
    
    model = keras.Model(inputs=inputs, outputs=output, name="custom_model_with_l2")
    return model

