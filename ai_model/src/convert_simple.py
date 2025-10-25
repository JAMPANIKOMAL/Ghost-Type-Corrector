#!/usr/bin/env python3
"""
Simplified Direct Conversion - No Intermediate SavedModel
==========================================================
"""

import sys
from pathlib import Path

# Patch numpy compatibility issues
import numpy
if not hasattr(numpy, 'object'):
    numpy.object = object
if not hasattr(numpy, 'bool'):
    numpy.bool = bool

import tensorflowjs as tfjs

def main():
    print("=" * 70)
    print("DIRECT KERAS TO TFJS CONVERSION")
    print("=" * 70)
    print()
    
    # Paths
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
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert using tensorflowjs API directly
    print("Converting Keras model to TensorFlow.js...")
    print()
    
    try:
        tfjs.converters.convert_tf_keras_model(
            str(model_path),
            str(output_dir)
        )
        
        print()
        print("✓ Conversion successful!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Verify
    print()
    print("Verifying output...")
    print("-" * 70)
    
    model_json = output_dir / 'model.json'
    if not model_json.exists():
        print("ERROR: model.json not created!")
        return 1
    
    print("✓ model.json found")
    
    # List files
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
    
    # Success!
    print()
    print("=" * 70)
    print("SUCCESS! MODEL READY FOR BROWSER")
    print("=" * 70)
    print()
    print(f"✓ TensorFlow.js model: {output_dir}")
    print()
    print("Load in extension:")
    print("  const model = await tf.loadLayersModel('model/model.json');")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
