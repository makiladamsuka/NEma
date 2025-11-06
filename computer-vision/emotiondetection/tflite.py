import tensorflow as tf
import os

# Define file names
H5_MODEL_PATH = 'media.h5'
TFLITE_MODEL_PATH = 'media.tflite'

print(f"--- Starting TFLite Conversion for {H5_MODEL_PATH} ---")

try:
    # 1. Load the Keras Model
    # Using compile=False is safer, especially if you have custom losses/optimizers that TFLite doesn't care about.
    model = tf.keras.models.load_model(H5_MODEL_PATH, compile=False)
    print("✅ H5 Model loaded successfully.")

    # --- Robust Converter Initialization using Concrete Functions ---
    # This method forces TensorFlow to trace the *entire* computation graph explicitly,
    # which often resolves "missing attribute" and unsupported operation errors.
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tf.function(lambda x: model(x)).get_concrete_function(
             # This line is crucial: Define the EXACT expected input shape and type (e.g., float32)
             # We assume your model has at least one input.
             tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input_tensor")
        )]
    )
    print("✅ TFLite Converter initialized using Concrete Function trace.")


    # --- Conversion Optimizations for Raspberry Pi ---
    # 1. Optimization Level (crucial for size and speed)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # OPTIONAL: If you need it even smaller and faster (but potentially slightly less accurate)
    # This is highly recommended for Raspberry Pi or similar edge devices.
    # converter.target_spec.supported_types = [tf.float16]
    
    # 2. Allow TensorFlow Operations (Essential for custom layers or complex logic)
    # This is the flag that allows the TFLite runtime to call back into TensorFlow for operations 
    # it doesn't natively support (the 'SELECT_TF_OPS').
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    print("✅ Conversion target set to allow built-in and selected TF operations.")

    # --- Convert and Save ---
    tflite_model = converter.convert()

    # Save the TFLite model
    with open(TFLITE_MODEL_PATH, 'wb') as f:
      f.write(tflite_model)

    print(f"\n🎉 SUCCESS! Model successfully converted and saved to {TFLITE_MODEL_PATH}")

except Exception as e:
    print("\n--- CRITICAL CONVERSION ERROR ---")
    print(f"The conversion failed. Details: {e}")
    print("\nTroubleshooting Steps:")
    print("1. Inspect the Layers: The error message might mention a specific layer (e.g., 'SparseTensor', 'CustomLayer').")
    print("2. Try Float16 Quantization: Uncomment the `converter.target_spec.supported_types = [tf.float16]` line for a smaller model.")
    print("3. Check Dependencies: Ensure your Python environment's TensorFlow version matches the one used to create `media.h5`.")
    print("---------------------------------")