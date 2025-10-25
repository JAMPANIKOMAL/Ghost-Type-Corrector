#!/usr/bin/env python3
"""
Fixed TensorFlow.js Model Conversion
=====================================
Properly converts Keras Functional API model to TensorFlow.js format.
Uses the official tensorflowjs converter which handles naming conventions correctly.
"""

import sys
import subprocess
from pathlib import Path
import shutil

def convert_using_cli():
    """
    Use the tensorflowjs_converter CLI tool which properly handles
    the Functional API model format and naming conventions.
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
    
    # Ensure output directory exists (don't delete - Chrome might have it open)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("✓ Output directory ready (files will be overwritten)")
    print()
    
    # Run tensorflowjs_converter CLI
    print("Running tensorflowjs_converter...")
    print("-" * 70)
    
    cmd = [
        'tensorflowjs_converter',
        '--input_format', 'keras',
        '--output_format', 'tfjs_layers_model',
        str(model_path),
        str(output_dir)
    ]
    
    print("Command:", ' '.join(cmd))
    print()
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print(result.stdout)
        if result.stderr:
            print("Warnings:")
            print(result.stderr)
        
        print()
        print("✓ Conversion successful!")
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Conversion failed!")
        print(f"Return code: {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 1
    except FileNotFoundError:
        print("ERROR: tensorflowjs_converter command not found!")
        print()
        print("Please install tensorflowjs:")
        print("  pip install tensorflowjs==3.18.0")
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
    print("The model should now load properly in the browser extension.")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(convert_using_cli())
