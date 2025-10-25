#!/usr/bin/env python3
"""
Official TensorFlow.js Conversion via SavedModel
=================================================
Uses SavedModel as intermediate format to avoid HDF5 conversion issues.
"""

import sys
import subprocess
import shutil
from pathlib import Path

# Patch numpy BEFORE any imports
import numpy as np
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

def convert_via_savedmodel():
    """
    Convert model using SavedModel intermediate format.
    This works better with tensorflowjs_converter.
    """
    print("=" * 70)
    print("TENSORFLOW.JS CONVERSION VIA SAVEDMODEL")
    print("=" * 70)
    print()
    
    # Define paths
    script_dir = Path(__file__).parent
    ai_model_dir = script_dir.parent
    project_root = ai_model_dir.parent
    
    model_path = ai_model_dir / 'autocorrect_model.h5'
    savedmodel_dir = ai_model_dir / 'saved_model_temp'
    output_dir = project_root / 'extension' / 'model'
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return 1
    
    print(f"Input HDF5:    {model_path}")
    print(f"Temp SavedModel: {savedmodel_dir}")
    print(f"Output TF.js:  {output_dir}")
    print()
    
    # Step 1: Load Keras model
    print("Step 1: Loading Keras model...")
    model = keras.models.load_model(str(model_path))
    print(f"✓ Model loaded: {model.name}")
    model.summary()
    print()
    
    # Step 2: Save as SavedModel
    print("Step 2: Converting to SavedModel format...")
    if savedmodel_dir.exists():
        shutil.rmtree(savedmodel_dir)
    
    # SavedModel format is more stable for conversion
    model.save(str(savedmodel_dir), save_format='tf')
    print(f"✓ SavedModel created at {savedmodel_dir}")
    print()
    
    # Step 3: Use official tensorflowjs_converter
    print("Step 3: Converting SavedModel to TensorFlow.js...")
    print("-" * 70)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        'tensorflowjs_converter',
        '--input_format', 'tf_saved_model',
        '--output_format', 'tfjs_graph_model',  # Use graph model for better compatibility
        '--signature_name', 'serving_default',
        '--saved_model_tags', 'serve',
        str(savedmodel_dir),
        str(output_dir)
    ]
    
    print("Command:", ' '.join(cmd))
    print()
    
    try:
        # Import and patch tensorflowjs modules
        import tensorflowjs.converters.converter as converter
        
        print("Using Python API for conversion...")
        converter.convert(
            [str(savedmodel_dir)],
            output_dir=str(output_dir),
            input_format='tf_saved_model',
            output_format='tfjs_graph_model',
            signature_name='serving_default',
            saved_model_tags='serve'
        )
        
        print()
        print("✓ Conversion successful!")
        
    except Exception as e:
        print(f"Python API failed: {e}")
        print("Trying command line...")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("Warnings:", result.stderr)
            print()
            print("✓ Conversion successful!")
            
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Command line conversion failed!")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return 1
        except FileNotFoundError:
            print("ERROR: tensorflowjs_converter not found!")
            print("Install with: pip install tensorflowjs==3.18.0")
            return 1
    
    # Step 4: Cleanup
    print()
    print("Step 4: Cleaning up temporary files...")
    if savedmodel_dir.exists():
        shutil.rmtree(savedmodel_dir)
    print("✓ Temporary files removed")
    print()
    
    # Verify output
    print("Verifying output files...")
    print("-" * 70)
    
    model_json = output_dir / 'model.json'
    if not model_json.exists():
        print("ERROR: model.json not created!")
        return 1
    
    print("✓ model.json found")
    
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
    
    print("=" * 70)
    print("CONVERSION COMPLETE!")
    print("=" * 70)
    print()
    print(f"✓ Model ready at: {output_dir}")
    print()
    print("NOTE: Model was converted to tfjs_graph_model format.")
    print("Load in browser with: tf.loadGraphModel('model/model.json')")
    print()
    print("Next steps:")
    print("1. Update sandbox_logic.js to use tf.loadGraphModel() only")
    print("2. Reload extension in chrome://extensions/")
    print("3. Test on any webpage")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(convert_via_savedmodel())
