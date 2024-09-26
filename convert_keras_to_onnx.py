import tensorflow as tf
import tf2onnx

# Path to your saved Keras model
keras_model_path = 'trained_10k/checkpoint-300x300-e4-755valacc.keras'

# Load the Keras model
model = tf.keras.models.load_model(keras_model_path)

input_shape = model.inputs[0].shape
input_dtype = tf.float32  # Change if your model uses a different dtype

# Create TensorSpec for the input
spec = (tf.TensorSpec(input_shape, input_dtype, name="input"),)

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=18,  # You can specify the ONNX opset version; 13 is commonly used
    output_path="dopple_species.onnx"
)
