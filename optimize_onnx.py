import onnx
import onnxoptimizer

# Load the ONNX model
onnx_model = onnx.load("dopple_species_big.onnx")

# Specify optimization passes
passes = ["eliminate_identity", "fuse_bn_into_conv", "fuse_add_bias_into_conv"]

# Apply optimizations
optimized_model = onnxoptimizer.optimize(onnx_model)

# Save the optimized model
optimized_onnx_model_path = 'dopple_species_big_optimized.onnx'
onnx.save(optimized_model, optimized_onnx_model_path)

print(f"Optimized ONNX model saved to {optimized_onnx_model_path}")

