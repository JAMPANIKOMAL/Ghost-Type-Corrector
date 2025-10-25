"""
Model Conversion Script - Modern Approach
Converts Keras .h5 model to TensorFlow.js format using Keras 3 native export.
"""

import os
import subprocess
import sys

print("--- Starting Model Conversion to TensorFlow.js ---")
print(f"Python version: {sys.version}")

# --- Define File Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai_model directory
extension_dir = os.path.join(os.path.dirname(base_dir), 'extension')  # Go up one and into 'extension'

input_model_path = os.path.join(base_dir, 'autocorrect_model.h5')
output_model_dir = os.path.join(extension_dir, 'model')  # Output directly to extension/model

print(f"\nInput Keras model (.h5): {input_model_path}")
print(f"Output TF.js model directory: {output_model_dir}")

# Check if input model exists
if not os.path.exists(input_model_path):
    print(f"\nERROR: Model file not found at {input_model_path}")
    print("Please train the model first by running 02_model_training.py")
    sys.exit(1)

# Ensure the output directory exists
os.makedirs(output_model_dir, exist_ok=True)
print(f"Output directory ready: {output_model_dir}")

# --- Method: Use tensorflowjs_converter command-line tool ---
print("\n" + "="*60)
print("Using tensorflowjs_converter command-line tool")
print("="*60)

try:
    # The tensorflowjs_converter is installed as a command-line tool
    # when you install tensorflowjs package
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format", "keras",
        "--output_format", "tfjs_layers_model",
        input_model_path,
        output_model_dir
    ]
    
    print(f"\nRunning command:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n--- Conversion Successful ---")
        print(result.stdout)
        print(f"\nConverted model saved to: {output_model_dir}")
        print("\nGenerated files:")
        for file in os.listdir(output_model_dir):
            file_path = os.path.join(output_model_dir, file)
            size = os.path.getsize(file_path)
            print(f"  - {file} ({size:,} bytes)")
    else:
        print("\n--- Conversion Failed ---")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
        
except FileNotFoundError:
    print("\nERROR: tensorflowjs converter not found.")
    print("Please install tensorflowjs: pip install tensorflowjs")
    sys.exit(1)
except Exception as e:
    print(f"\nERROR: An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print(f"\n--- Conversion Script Finished ---")
