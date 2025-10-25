#!/usr/bin/env python3
"""
TensorFlow.js Model Conversion with NumPy Patch
================================================
Patches NumPy compatibility issues and converts the model properly.
"""

import sys
import numpy as np
from pathlib import Path

# Patch numpy compatibility issues BEFORE importing tensorflowjs
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

print("NumPy patches applied")

# Now we can safely import tensorflowjs
import tensorflowjs as tfjs
from tensorflowjs.converters import converter

def convert_model():
    """
    Convert Keras model to TensorFlow.js format using the Python API.
    """
    print("=" * 70)
    print("TENSORFLOW.JS MODEL CONVERSION (with NumPy Patch)")
    print("=" * 70)
    print()
    
    # Define paths
    script_dir = Path(__file__).parent
    ai_model_dir = script_dir.parent
    project_root = ai_model_dir.parent
    
    model_path = ai_model_dir / 'autocorrect_model.h5'
    output_dir = project_root / 'extension' / 'model'
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    print(f"Input:  {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready")
    print()
    
    # Convert using Python API (which respects our numpy patches)
    print("Converting model...")
    print("-" * 70)
    
    try:
        converter.convert(
            input_path=str(model_path),
            input_format='keras',
            output_path=str(output_dir),
            output_format='tfjs_layers_model'
        )
        
        print()
        print("✓ Conversion successful!")
        
    except Exception as e:
        print(f"ERROR: Conversion failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify output
    print()
    print("Verifying output files...")
    print("-" * 70)
    
    model_json = output_dir / 'model.json'
    if not model_json.exists():
        print("ERROR: model.json not created!")
        return 1
    
    print("✓ model.json found")
    
    # List all files
    output_files = sorted(output_dir.glob('*'))
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
    
    # Success!
    print("=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print()
    print(f"✓ Model ready at: {output_dir}")
    print()
    print("Next steps:")
    print("1. Close Chrome completely (to unlock model files)")
    print("2. Reload the extension in chrome://extensions/")
    print("3. Test on any webpage with text inputs")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(convert_model())
