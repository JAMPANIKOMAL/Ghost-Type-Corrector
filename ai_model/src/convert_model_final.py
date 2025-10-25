#!/usr/bin/env python3
"""
Final Model Conversion Script
Run this AFTER patching tensorflowjs with direct_patch.py
"""

import sys
import os
import shutil
from pathlib import Path

print("=" * 70)
print("TENSORFLOW.JS MODEL CONVERSION")
print("=" * 70)
print()

# Now safe to import
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# Define paths
script_dir = Path(__file__).parent
ai_model_dir = script_dir.parent
project_root = ai_model_dir.parent

model_path = ai_model_dir / 'autocorrect_model.h5'
savedmodel_dir = ai_model_dir / 'saved_model_temp'
output_dir = project_root / 'extension' / 'model'

print(f"Input model:       {model_path}")
print(f"Temp SavedModel:   {savedmodel_dir}")
print(f"Output directory:  {output_dir}")
print()

# Check model exists
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

# Step 1: Load Keras model
print("Step 1: Loading Keras model...")
try:
    model = keras.models.load_model(str(model_path))
    print(f"✓ Model loaded: {model.name}")
    model.summary()
    print()
except Exception as e:
    print(f"✗ Failed to load model: {e}")
    sys.exit(1)

# Step 2: Save as SavedModel
print("Step 2: Converting to SavedModel format...")
try:
    # Remove existing directory with better error handling
    if savedmodel_dir.exists():
        print(f"  Removing existing directory...")
        try:
            shutil.rmtree(savedmodel_dir)
        except PermissionError:
            # Try harder on Windows
            import time
            time.sleep(0.5)
            shutil.rmtree(savedmodel_dir, ignore_errors=True)
            if savedmodel_dir.exists():
                print("  ⚠ Could not fully remove old directory, trying to continue...")
    
    model.save(str(savedmodel_dir), save_format='tf')
    print(f"✓ SavedModel created at {savedmodel_dir}")
    print()
except Exception as e:
    print(f"✗ Failed to create SavedModel: {e}")
    print("\nTry manually deleting this folder and run again:")
    print(f"  {savedmodel_dir}")
    sys.exit(1)

# Step 3: Convert to TensorFlow.js
print("Step 3: Converting SavedModel to TensorFlow.js...")
try:
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert using tensorflowjs
    tfjs.converters.convert_tf_saved_model(
        str(savedmodel_dir),
        str(output_dir),
        signature_def='serving_default',
        saved_model_tags='serve'
    )
    
    print(f"✓ TensorFlow.js model created at {output_dir}")
    print()
    
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup temp directory
    if savedmodel_dir.exists():
        shutil.rmtree(savedmodel_dir)
        print("✓ Cleaned up temporary files")
        print()

# Verify output
print("Step 4: Verifying output...")
model_json = output_dir / 'model.json'

if not model_json.exists():
    print("✗ model.json not found!")
    sys.exit(1)

print(f"✓ model.json exists")

# List all files
output_files = sorted(output_dir.glob('*'))
total_size = 0

print("\nGenerated files:")
for file_path in output_files:
    if file_path.is_file():
        size = file_path.stat().st_size
        total_size += size
        print(f"  - {file_path.name} ({size:,} bytes)")

print(f"\nTotal size: {total_size / (1024*1024):.2f} MB")
print()

# Success!
print("=" * 70)
print("✓ CONVERSION SUCCESSFUL!")
print("=" * 70)
print()
print("Model is ready to use in the Chrome extension.")
print()
print("Next steps:")
print("1. Update extension/js/sandbox_logic.js to use tf.loadGraphModel()")
print("2. Reload the extension in Chrome (chrome://extensions/)")
print("3. Test the autocorrection on any webpage")
print()
