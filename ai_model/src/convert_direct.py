#!/usr/bin/env python3
"""
Direct Python-based Model Conversion
=====================================
Converts Keras model to TensorFlow.js using Python API directly,
avoiding command-line tool issues.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Patch numpy before importing tensorflowjs
import numpy
if not hasattr(numpy, 'object'):
    numpy.object = object
if not hasattr(numpy, 'bool'):
    numpy.bool = bool
if not hasattr(numpy, 'int'):
    numpy.int = int
if not hasattr(numpy, 'float'):
    numpy.float = float

import tensorflow as tf
from tensorflow import keras

def convert_keras_to_tfjs_python(model_path, output_dir):
    """
    Convert Keras model using direct Python API.
    
    This avoids the command-line tool and its dependency issues.
    """
    print("Loading Keras model...")
    model = keras.models.load_model(str(model_path))
    print(f"✓ Model loaded: {model_path.name}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as TensorFlow SavedModel first
    print("Converting to SavedModel format...")
    saved_model_dir = model_path.parent / 'saved_model_temp'
    
    # Clean up old saved_model if it exists
    if saved_model_dir.exists():
        import shutil
        shutil.rmtree(saved_model_dir)
    
    model.save(str(saved_model_dir), save_format='tf')
    print(f"✓ SavedModel created")
    print()
    
    # Now use tensorflowjs converter programmatically
    print("Converting SavedModel to TensorFlow.js...")
    
    try:
        import tensorflowjs.converters.converter as converter
        
        # Call the convert function directly
        converter.convert(
            input_path=str(saved_model_dir),
            input_format='tf_saved_model',
            output_path=str(output_dir),
            signature_name='serving_default',
            saved_model_tags='serve'
        )
        
        print("✓ Conversion successful!")
        return True
        
    except Exception as e:
        print(f"ERROR: Python API conversion failed: {e}")
        print()
        print("Falling back to manual weight extraction...")
        
        # Fallback: Extract weights manually
        return convert_manual(model, output_dir)
    
    finally:
        # Clean up temporary SavedModel
        if saved_model_dir.exists():
            import shutil
            shutil.rmtree(saved_model_dir)
            print("✓ Cleaned up temporary files")


def convert_manual(model, output_dir):
    """
    Manual conversion by extracting weights and creating model.json.
    """
    print("Extracting model architecture and weights...")
    
    # Get model configuration
    config = model.get_config()
    
    # Extract weights
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights.extend(layer_weights)
    
    # Save weights as binary files
    weight_data = []
    weight_specs = []
    
    for i, weight in enumerate(weights):
        # Save weight to file
        weight_file = output_dir / f'group1-shard{i+1}of{len(weights)}.bin'
        weight.tofile(str(weight_file))
        
        weight_spec = {
            'name': f'weight_{i}',
            'shape': list(weight.shape),
            'dtype': str(weight.dtype)
        }
        weight_specs.append(weight_spec)
    
    # Create model.json
    model_json = {
        'format': 'layers-model',
        'generatedBy': 'Ghost Type Corrector Manual Converter',
        'convertedBy': 'Python 3.10',
        'modelTopology': config,
        'weightsManifest': [{
            'paths': [f'group1-shard{i+1}of{len(weights)}.bin' for i in range(len(weights))],
            'weights': weight_specs
        }]
    }
    
    # Save model.json
    model_json_path = output_dir / 'model.json'
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"✓ Manual conversion complete")
    print(f"  - Created model.json")
    print(f"  - Created {len(weights)} weight files")
    
    return True


def main():
    print("=" * 70)
    print("PYTHON-BASED MODEL CONVERSION")
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
    
    # Convert
    try:
        success = convert_keras_to_tfjs_python(model_path, output_dir)
        
        if not success:
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
        print("1. Load in browser extension:")
        print("   const model = await tf.loadLayersModel('model/model.json');")
        print()
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
