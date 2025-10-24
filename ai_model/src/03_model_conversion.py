import tensorflowjs as tfjs
import os
import time
import sys
from tensorflow import keras # *** ADDED: Import Keras ***

print("--- Starting Model Conversion to TensorFlow.js ---")
print(f"Using TensorFlow.js version: {tfjs.__version__}") # Verify version

# --- Define File Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
project_dir = os.path.dirname(base_dir)
extension_dir = os.path.join(project_dir, 'extension')

input_model_path = os.path.join(base_dir, 'autocorrect_model.h5')
output_model_dir = os.path.join(extension_dir, 'model')

print(f"Input Keras model (.h5): {input_model_path}")
print(f"Output TF.js model directory: {output_model_dir}")

# Ensure the output directory exists
try:
    os.makedirs(output_model_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_model_dir}")
except OSError as e:
    print(f"Error creating directory {output_model_dir}: {e}")
    sys.exit(1)

# --- Load the Keras Model *** (NEW STEP) *** ---
print("\nLoading Keras model from .h5 file...")
try:
    if not os.path.exists(input_model_path):
        print(f"ERROR: Input model file not found at {input_model_path}")
        sys.exit(1)
    # Load the model object
    model_to_convert = keras.models.load_model(input_model_path)
    print("Keras model loaded successfully.")
    model_to_convert.summary() # Optional: Show summary to confirm load
except Exception as e:
    print(f"Error loading Keras model: {e}")
    sys.exit(1)

# --- Perform Conversion ---
start_time = time.time()
print("\nStarting conversion process...")

try:
    # *** CHANGED: Pass the loaded model object, not the path ***
    tfjs.converters.save_keras_model(model_to_convert, output_model_dir)
    end_time = time.time()
    print("\n--- Conversion Successful ---")
    print(f"Converted model saved to: {output_model_dir}")
    print(f"Conversion took {end_time - start_time:.2f} seconds.")
    print("\nYou should now find 'model.json' and '*.bin' files in that directory.")

except Exception as e:
    print(f"\n--- Conversion Failed ---")
    print(f"An error occurred: {e}")
    print("Please ensure TensorFlow.js is installed correctly and the input model is valid.")
    sys.exit(1)

print(f"\n--- Conversion Script Finished ---")
