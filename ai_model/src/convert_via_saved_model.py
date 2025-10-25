#!/usr/bin/env python3
"""
Alternative Model Conversion Script
====================================
Converts Keras .h5 model to TensorFlow.js via SavedModel format.

This approach avoids the tensorflow-decision-forests dependency issue
by converting in two steps:
  1. Keras .h5 → TensorFlow SavedModel
  2. SavedModel → TensorFlow.js (using command-line tool)
"""

import sys
import subprocess
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

def main():
    print("=" * 70)
    print("ALTERNATIVE MODEL CONVERSION (via SavedModel)")
    print("=" * 70)
    print()
    
    # Define paths
    script_dir = Path(__file__).parent
    ai_model_dir = script_dir.parent
    project_root = ai_model_dir.parent
    
    h5_model_path = ai_model_dir / 'autocorrect_model.h5'
    saved_model_dir = ai_model_dir / 'saved_model_temp'
    tfjs_output_dir = project_root / 'extension' / 'model'
    
    # Check if .h5 model exists
    if not h5_model_path.exists():
        print(f"ERROR: Model not found at {h5_model_path}")
        print("Please run 02_model_training.py first.")
        return 1
    
    print(f"Input:  {h5_model_path}")
    print(f"Temp:   {saved_model_dir}")
    print(f"Output: {tfjs_output_dir}")
    print()
    
    # STEP 1: Load Keras model and save as SavedModel
    print("STEP 1: Converting .h5 to SavedModel format")
    print("-" * 70)
    
    try:
        print("Loading Keras model...")
        model = keras.models.load_model(str(h5_model_path))
        print(f"✓ Model loaded successfully")
        print()
        
        print("Saving as TensorFlow SavedModel...")
        # Remove old saved_model if it exists
        if saved_model_dir.exists():
            import shutil
            shutil.rmtree(saved_model_dir)
        
        model.save(str(saved_model_dir), save_format='tf')
        print(f"✓ SavedModel created at {saved_model_dir}")
        print()
        
    except Exception as e:
        print(f"ERROR: Failed to convert to SavedModel")
        print(f"Error: {e}")
        return 1
    
    # STEP 2: Convert SavedModel to TensorFlow.js
    print("STEP 2: Converting SavedModel to TensorFlow.js")
    print("-" * 70)
    
    # Create output directory
    tfjs_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build conversion command
    cmd = [
        "tensorflowjs_converter",
        "--input_format", "tf_saved_model",
        "--output_format", "tfjs_graph_model",
        "--signature_name", "serving_default",
        str(saved_model_dir),
        str(tfjs_output_dir)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout:
            print(result.stdout)
        
        print()
        print("✓ Conversion successful!")
        
    except subprocess.CalledProcessError as e:
        print("ERROR: tensorflowjs_converter failed")
        print()
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        print()
        print("Alternative: Try installing the missing dependency:")
        print("  pip install tensorflow-decision-forests")
        print()
        print("Or use an older version of tensorflowjs:")
        print("  pip uninstall tensorflowjs -y")
        print("  pip install tensorflowjs==3.18.0")
        return 1
    
    except FileNotFoundError:
        print("ERROR: tensorflowjs_converter command not found!")
        print()
        print("The command should be in your PATH after installing tensorflowjs.")
        print("Check your installation:")
        print("  pip show tensorflowjs")
        print("  where tensorflowjs_converter")
        return 1
    
    # STEP 3: Verify output
    print()
    print("STEP 3: Verifying output files")
    print("-" * 70)
    
    model_json = tfjs_output_dir / 'model.json'
    if not model_json.exists():
        print("ERROR: model.json not found in output directory!")
        return 1
    
    print("✓ model.json created")
    
    # List all output files
    output_files = sorted(tfjs_output_dir.glob('*'))
    total_size = 0
    print()
    print("Output files:")
    for file_path in output_files:
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            print(f"  - {file_path.name:40s} {size:10,} bytes")
    
    print()
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print()
    
    # Cleanup
    print("Cleaning up temporary files...")
    import shutil
    shutil.rmtree(saved_model_dir)
    print("✓ Temporary SavedModel deleted")
    
    # Success!
    print()
    print("=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print()
    print(f"✓ TensorFlow.js model saved to: {tfjs_output_dir}")
    print()
    print("Next steps:")
    print("1. Load the model in your browser extension:")
    print("   const model = await tf.loadGraphModel('model/model.json');")
    print()
    print("2. Note: This is a GraphModel (not LayersModel)")
    print("   Use model.predict() for inference")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
