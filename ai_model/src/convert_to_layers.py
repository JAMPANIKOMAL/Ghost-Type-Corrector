#!/usr/bin/env python3
"""
Direct HDF5 to TensorFlow.js Layers Model Conversion
Uses the official tensorflowjs converter directly on HDF5 file.
"""

import sys
import os
import shutil
from pathlib import Path

print("=" * 70)
print("DIRECT HDF5 TO TFJS LAYERS MODEL CONVERSION")
print("=" * 70)
print()

# Import after patching
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

# Define paths
script_dir = Path(__file__).parent
ai_model_dir = script_dir.parent
project_root = ai_model_dir.parent

model_path = ai_model_dir / 'autocorrect_model.h5'
output_dir = project_root / 'extension' / 'model'

print(f"Input HDF5:    {model_path}")
print(f"Output TFJS:   {output_dir}")
print()

# Check model exists
if not model_path.exists():
    print(f"ERROR: Model not found at {model_path}")
    sys.exit(1)

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

print("Converting HDF5 to TensorFlow.js Layers Model...")
print()

try:
    # Load the Keras model first
    print("Loading Keras model...")
    model = keras.models.load_model(str(model_path))
    print(f"✓ Model loaded: {model.name}")
    print()
    
    # Use the Python API to convert
    print("Converting to TensorFlow.js format...")
    
    # Use tfjs.converters.save_keras_model for layers format
    tfjs.converters.save_keras_model(model, str(output_dir))
    
    print("✓ Conversion successful!")
    print()
    
except Exception as e:
    print(f"✗ Conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Verify output
model_json = output_dir / 'model.json'

if not model_json.exists():
    print("✗ model.json not found!")
    sys.exit(1)

print("✓ model.json created")

# List files
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

print("=" * 70)
print("✓ CONVERSION COMPLETE!")
print("=" * 70)
print()
print("Model converted to LAYERS format (not graph).")
print("This format supports the .predict() API needed for Seq2Seq.")
print()
