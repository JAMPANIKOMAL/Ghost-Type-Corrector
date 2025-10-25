#!/usr/bin/env python3
"""
Fixed Model Conversion - Properly Formats for TensorFlow.js
============================================================
Converts the model format from Python naming to JavaScript naming conventions.
"""

import sys
import json
from pathlib import Path
import numpy as np

# Patch numpy before importing TensorFlow
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

import tensorflow as tf
from tensorflow import keras


def convert_keys_to_camel_case(obj):
    """
    Recursively convert snake_case keys to camelCase for TensorFlow.js compatibility.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            # Convert snake_case to camelCase
            new_key = key
            if '_' in key:
                parts = key.split('_')
                new_key = parts[0] + ''.join(word.capitalize() for word in parts[1:])
            new_obj[new_key] = convert_keys_to_camel_case(value)
        return new_obj
    elif isinstance(obj, list):
        return [convert_keys_to_camel_case(item) for item in obj]
    else:
        return obj


def convert_model():
    """
    Convert Keras model to TensorFlow.js with proper naming conventions.
    """
    print("=" * 70)
    print("FIXED TENSORFLOW.JS MODEL CONVERSION")
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
    
    # Load model
    print("Loading Keras model...")
    model = keras.models.load_model(str(model_path))
    print(f"✓ Model loaded: {model.name}")
    print()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model configuration and convert to camelCase
    print("Extracting model architecture...")
    config = model.get_config()
    config_camel = convert_keys_to_camel_case(config)
    print("✓ Model architecture extracted")
    print()
    
    # Extract weights
    print("Extracting model weights...")
    weights = []
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if layer_weights:
            weights.extend(layer_weights)
    
    print(f"✓ Extracted {len(weights)} weight tensors")
    print()
    
    # Save weights as binary files with proper layer names
    print("Saving weight files...")
    weight_specs = []
    weight_index = 0
    
    for layer in model.layers:
        layer_weights = layer.get_weights()
        if not layer_weights:
            continue
            
        layer_name = layer.name
        layer_class = layer.__class__.__name__
        
        print(f"  Processing layer: {layer_name} ({layer_class}) - {len(layer_weights)} weights")
        
        # Map weight types based on layer class and actual weight shapes
        for i, weight in enumerate(layer_weights):
            # Save weight to file
            weight_file = output_dir / f'group1-shard{weight_index+1}of{len(weights)}.bin'
            weight.tofile(str(weight_file))
            
            # Determine weight name based on layer type and weight index
            if layer_class == 'Embedding':
                weight_name = f'{layer_name}/embeddings:0'
            elif layer_class == 'LSTM':
                # LSTM has 3 weights: kernel, recurrent_kernel, bias
                if i == 0:
                    weight_name = f'{layer_name}/kernel:0'
                elif i == 1:
                    weight_name = f'{layer_name}/recurrent_kernel:0'
                else:
                    weight_name = f'{layer_name}/bias:0'
            elif layer_class == 'Dense' or layer_class == 'TimeDistributed':
                # Dense has 2 weights: kernel, bias
                if i == 0:
                    weight_name = f'{layer_name}/kernel:0'
                else:
                    weight_name = f'{layer_name}/bias:0'
            else:
                # Fallback for unknown layer types
                weight_name = f'{layer_name}/weight_{i}:0'
            
            weight_spec = {
                'name': weight_name,
                'shape': list(weight.shape),
                'dtype': 'float32'
            }
            weight_specs.append(weight_spec)
            weight_index += 1
    
    print(f"✓ Saved {len(weights)} weight files")
    print()
    
    # Create model.json with proper format
    # TensorFlow.js expects the model topology to be wrapped in className and config
    print("Creating model.json...")
    model_json = {
        'format': 'layers-model',
        'generatedBy': 'TensorFlow.js Converter v3.18.0',
        'convertedBy': 'Ghost Type Corrector (Fixed)',
        'modelTopology': {
            'className': 'Functional',
            'config': config_camel
        },
        'weightsManifest': [{
            'paths': [f'group1-shard{i+1}of{len(weights)}.bin' for i in range(len(weights))],
            'weights': weight_specs
        }]
    }
    
    # Save model.json
    model_json_path = output_dir / 'model.json'
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print("✓ model.json created")
    print()
    
    # Verify output
    print("Verifying output files...")
    print("-" * 70)
    
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
    print("1. Reload the extension in chrome://extensions/")
    print("2. Refresh any webpage to test")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(convert_model())
