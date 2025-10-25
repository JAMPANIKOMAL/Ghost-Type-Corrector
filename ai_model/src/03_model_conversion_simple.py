"""
Simple Model Conversion Script
Converts Keras .h5 model to TensorFlow SavedModel, then to TensorFlow.js
This bypasses the complex tensorflowjs converter dependencies.
"""

import os
import sys

print("--- Starting Model Conversion to TensorFlow.js ---")
print(f"Python version: {sys.version}\n")

# --- Step 1: Import TensorFlow and load the model ---
try:
    import tensorflow as tf
    import keras
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}\n")
except ImportError as e:
    print(f"ERROR: Could not import TensorFlow/Keras: {e}")
    sys.exit(1)

# --- Define File Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ai_model directory
extension_dir = os.path.join(os.path.dirname(base_dir), 'extension')  # extension directory

input_model_path = os.path.join(base_dir, 'autocorrect_model.h5')
saved_model_dir = os.path.join(base_dir, 'saved_model_temp')  # Temporary SavedModel directory
output_model_dir = os.path.join(extension_dir, 'model')  # Final TF.js model directory

print(f"Input Keras model (.h5): {input_model_path}")
print(f"Temporary SavedModel dir: {saved_model_dir}")
print(f"Output TF.js model directory: {output_model_dir}\n")

# Check if input model exists
if not os.path.exists(input_model_path):
    print(f"ERROR: Model file not found at {input_model_path}")
    print("Please train the model first by running 02_model_training.py")
    sys.exit(1)

# --- Step 2: Load the Keras model ---
print("="*60)
print("Step 1: Loading Keras model...")
print("="*60)

try:
    # Try loading with tf.keras which has better Keras 2 compatibility
    import tensorflow as tf
    model = tf.keras.models.load_model(input_model_path, compile=False)
    print("✓ Model loaded successfully\n")
    model.summary()
except Exception as e:
    print(f"\nERROR: Failed to load model: {e}")
    print("\nTrying alternative loading method with tf_keras...")
    try:
        # Try using tf_keras (Keras 2 compatibility)
        import tf_keras
        model = tf_keras.models.load_model(input_model_path)
        print("✓ Model loaded successfully using tf_keras\n")
        model.summary()
    except Exception as e2:
        print(f"\nERROR: Both loading methods failed: {e2}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# --- Step 3: Save as TensorFlow SavedModel format ---
print("\n" + "="*60)
print("Step 2: Converting to TensorFlow SavedModel...")
print("="*60)

try:
    # Create output directory
    os.makedirs(saved_model_dir, exist_ok=True)
    
    # Save in SavedModel format
    tf.saved_model.save(model, saved_model_dir)
    print(f"✓ SavedModel created at: {saved_model_dir}\n")
except Exception as e:
    print(f"\nERROR: Failed to save as SavedModel: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# --- Step 4: Convert SavedModel to TensorFlow.js ---
print("="*60)
print("Step 3: Converting SavedModel to TensorFlow.js...")
print("="*60)

try:
    # Use subprocess to call tensorflowjs_converter
    import subprocess
    
    # Create output directory
    os.makedirs(output_model_dir, exist_ok=True)
    
    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format", "tf_saved_model",
        "--output_format", "tfjs_graph_model",
        saved_model_dir,
        output_model_dir
    ]
    
    print(f"Running command:")
    print(" ".join(cmd))
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Conversion to TensorFlow.js successful!\n")
        if result.stdout:
            print("Output:", result.stdout)
    else:
        print("✗ Conversion failed")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        print("\nAttempting alternative conversion method...")
        raise Exception("Converter failed")
        
except Exception as e:
    print(f"\nWARNING: tensorflowjs converter failed: {e}")
    print("This is likely due to dependency conflicts with Python 3.13")
    print("\n" + "="*60)
    print("ALTERNATIVE SOLUTION")
    print("="*60)
    print("\nThe model has been successfully converted to SavedModel format.")
    print(f"Location: {saved_model_dir}")
    print("\nTo complete the conversion to TensorFlow.js, you have two options:")
    print("\n1. Use the tensorflowjs_converter command-line tool manually:")
    print(f"   tensorflowjs_converter --input_format=tf_saved_model \\")
    print(f"     {saved_model_dir} \\")
    print(f"     {output_model_dir}")
    print("\n2. Install Node.js and use the tfjs-converter npm package:")
    print("   npm install -g @tensorflow/tfjs-converter")
    print(f"   tensorflowjs_converter --input_format=tf_saved_model \\")
    print(f"     {saved_model_dir} \\")
    print(f"     {output_model_dir}")
    print("\nRecommendation: Option 2 (Node.js) is more reliable with Python 3.13")
    sys.exit(1)

# --- Step 5: Verify output ---
print("="*60)
print("Step 4: Verifying output...")
print("="*60)

try:
    files = os.listdir(output_model_dir)
    print(f"\nGenerated files in {output_model_dir}:")
    for file in files:
        file_path = os.path.join(output_model_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({size:,} bytes)")
    
    if 'model.json' in files:
        print("\n✓ Conversion complete! model.json found.")
    else:
        print("\n⚠ Warning: model.json not found. Conversion may be incomplete.")
        
except Exception as e:
    print(f"\nERROR: Could not verify output: {e}")

print("\n" + "="*60)
print("Conversion process finished!")
print("="*60)
