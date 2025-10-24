import tensorflowjs as tfjs
import os
import time

print("--- Starting Model Conversion to TensorFlow.js ---")

# --- Define File Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ai_model directory
extension_dir = os.path.join(os.path.dirname(base_dir), 'extension') # Go up one and into 'extension'

input_model_path = os.path.join(base_dir, 'autocorrect_model.h5')
output_model_dir = os.path.join(extension_dir, 'model') # Output directly to extension/model

print(f"Input Keras model (.h5): {input_model_path}")
print(f"Output TF.js model directory: {output_model_dir}")

# Ensure the output directory exists
os.makedirs(output_model_dir, exist_ok=True)
print(f"Ensured output directory exists: {output_model_dir}")

# --- Perform Conversion ---
# This uses the command-line tool functionality via the Python API
# `input_format='keras'` tells it we are converting from a .h5 file
# `output_format='tfjs_layers_model'` creates the model.json + weights format
start_time = time.time()
print("\nStarting conversion process...")

try:
    tfjs.converters.save_keras_model(input_model_path, output_model_dir)
    end_time = time.time()
    print("\n--- Conversion Successful ---")
    print(f"Converted model saved to: {output_model_dir}")
    print(f"Conversion took {end_time - start_time:.2f} seconds.")
    print("\nYou should now find 'model.json' and '*.bin' files in that directory.")

except Exception as e:
    print(f"\n--- Conversion Failed ---")
    print(f"An error occurred: {e}")
    print("Please ensure the input model file exists and TensorFlow.js is installed correctly.")

print(f"\n--- Conversion Script Finished ---")
