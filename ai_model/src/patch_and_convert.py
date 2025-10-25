#!/usr/bin/env python3
"""
Patch TensorFlowJS Library and Convert Model
=============================================
Patches the tensorflowjs library's numpy compatibility issues,
then converts the model.
"""

import sys
import os
import shutil
from pathlib import Path

def patch_tensorflowjs():
    """Patch the tensorflowjs library to fix numpy compatibility."""
    print("=" * 70)
    print("PATCHING TENSORFLOWJS LIBRARY")
    print("=" * 70)
    print()
    
    # Find tensorflowjs installation
    try:
        import tensorflowjs
        tfjs_path = Path(tensorflowjs.__file__).parent
        print(f"TensorFlowJS location: {tfjs_path}")
    except ImportError:
        print("ERROR: tensorflowjs not installed!")
        print("Install with: pip install tensorflowjs==3.18.0")
        return False
    
    # Patch read_weights.py
    read_weights_file = tfjs_path / 'read_weights.py'
    if read_weights_file.exists():
        print(f"\nPatching: {read_weights_file}")
        
        with open(read_weights_file, 'r') as f:
            content = f.read()
        
        # Patch the problematic line
        original_line = "np.uint8, np.uint16, np.object, np.bool]"
        patched_line = "np.uint8, np.uint16, object, bool]"
        
        if original_line in content:
            content = content.replace(original_line, patched_line)
            
            # Backup original
            backup_file = read_weights_file.with_suffix('.py.backup')
            if not backup_file.exists():
                shutil.copy(read_weights_file, backup_file)
                print(f"  ✓ Backed up to: {backup_file}")
            
            # Write patched version
            with open(read_weights_file, 'w') as f:
                f.write(content)
            print(f"  ✓ Patched successfully")
        else:
            print(f"  ℹ Already patched or different version")
    else:
        print(f"  ✗ File not found: {read_weights_file}")
        return False
    
    print()
    print("✓ TensorFlowJS library patched successfully")
    print()
    return True


def convert_model():
    """Convert the model after patching."""
    print("=" * 70)
    print("CONVERTING MODEL TO TENSORFLOW.JS")
    print("=" * 70)
    print()
    
    # Now import after patching
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    
    # Define paths
    script_dir = Path(__file__).parent
    ai_model_dir = script_dir.parent
    project_root = ai_model_dir.parent
    
    model_path = ai_model_dir / 'autocorrect_model.h5'
    savedmodel_dir = ai_model_dir / 'saved_model_temp'
    output_dir = project_root / 'extension' / 'model'
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return False
    
    print(f"Input:  {model_path}")
    print(f"Temp:   {savedmodel_dir}")
    print(f"Output: {output_dir}")
    print()
    
    # Load and convert
    print("Loading Keras model...")
    model = keras.models.load_model(str(model_path))
    print(f"✓ Loaded: {model.name}")
    print()
    
    print("Saving as SavedModel...")
    if savedmodel_dir.exists():
        shutil.rmtree(savedmodel_dir)
    model.save(str(savedmodel_dir), save_format='tf')
    print("✓ SavedModel created")
    print()
    
    print("Converting to TensorFlow.js...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import tensorflowjs as tfjs
        
        tfjs.converters.convert_tf_saved_model(
            str(savedmodel_dir),
            str(output_dir),
            signature_def='serving_default',
            saved_model_tags='serve'
        )
        print("✓ Conversion successful!")
        
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if savedmodel_dir.exists():
            shutil.rmtree(savedmodel_dir)
    
    # Verify
    print()
    print("Verifying output...")
    model_json = output_dir / 'model.json'
    if not model_json.exists():
        print("✗ model.json not found!")
        return False
    
    print("✓ model.json created")
    
    output_files = sorted(output_dir.glob('*'))
    total_size = 0
    for file_path in output_files:
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
    
    print(f"✓ Total size: {total_size / (1024*1024):.2f} MB")
    print()
    
    print("=" * 70)
    print("SUCCESS!")
    print("=" * 70)
    print()
    print(f"Model ready at: {output_dir}")
    print()
    print("Next steps:")
    print("1. Update sandbox_logic.js to use tf.loadGraphModel()")
    print("2. Reload extension in Chrome")
    print()
    
    return True


def main():
    # First patch the library
    if not patch_tensorflowjs():
        print("\nFailed to patch tensorflowjs library")
        return 1
    
    # Then convert the model
    if not convert_model():
        print("\nFailed to convert model")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
