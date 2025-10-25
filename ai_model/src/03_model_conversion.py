#!/usr/bin/env python3
"""
Ghost Type Corrector - Model Conversion to TensorFlow.js
=========================================================
Converts trained Keras model to TensorFlow.js format for browser deployment.

This script:
1. Loads the trained Keras model
2. Converts it to TensorFlow.js format
3. Saves model.json and weight files for use in the browser extension

Author: Ghost Type Corrector Team
License: MIT
"""

import subprocess
import sys
import os
from pathlib import Path

# =============================================================================
# CONVERSION CONFIGURATION
# =============================================================================

# Output format options:
# - tfjs_layers_model: For use with tf.loadLayersModel() (recommended)
# - tfjs_graph_model: For use with tf.loadGraphModel() (smaller, faster)
OUTPUT_FORMAT = 'tfjs_layers_model'


# =============================================================================
# CONVERSION PIPELINE
# =============================================================================

def check_tensorflowjs_installed():
    """Check if tensorflowjs_converter command is available."""
    try:
        # Try running the converter with --help to verify it's installed
        result = subprocess.run(
            [sys.executable, "-m", "tensorflowjs.converters.converter", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5
        )
        
        if result.returncode == 0 or "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower():
            print(f"✓ TensorFlowJS converter found")
            return True
    except Exception as e:
        pass
    
    print("ERROR: tensorflowjs converter not accessible!")
    print()
    print("Please install it:")
    print("  conda activate ghost-corrector-gpu")
    print("  pip install tensorflowjs==4.4.0 --no-deps")
    print("  pip install packaging importlib-resources tensorflow-hub")
    return False


def convert_model(input_model_path: Path, output_dir: Path):
    """
    Convert Keras model to TensorFlow.js format.
    
    Args:
        input_model_path: Path to trained .h5 model
        output_dir: Directory to save converted model
    """
    print(f"Converting model:")
    print(f"  Input:  {input_model_path}")
    print(f"  Output: {output_dir}")
    print(f"  Format: {OUTPUT_FORMAT}")
    print()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build conversion command
    cmd = [
        sys.executable,
        "-m", "tensorflowjs.converters.converter",
        "--input_format", "keras",
        "--output_format", OUTPUT_FORMAT,
        str(input_model_path),
        str(output_dir)
    ]
    
    print("Running conversion...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run conversion
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Show output
        if result.stdout:
            print(result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("ERROR: Conversion failed!")
        print()
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        print()
        print("This is likely due to missing dependencies.")
        print("TensorFlowJS requires some packages we didn't install to avoid conflicts.")
        print()
        print("To fix, you can try converting manually using the command line:")
        print()
        print(f"  tensorflowjs_converter --input_format keras --output_format {OUTPUT_FORMAT} \\")
        print(f"    \"{input_model_path}\" \\")
        print(f"    \"{output_dir}\"")
        print()
        return False
    except FileNotFoundError as e:
        print("ERROR: tensorflowjs_converter not found!")
        print()
        print("Install tensorflowjs:")
        print("  pip install tensorflowjs==4.4.0 --no-deps")
        print("  pip install packaging importlib-resources tensorflow-hub")
        print()
        return False


def verify_conversion(output_dir: Path):
    """
    Verify that conversion produced expected output files.
    
    Args:
        output_dir: Directory containing converted model
        
    Returns:
        True if verification passed, False otherwise
    """
    print()
    print("Verifying conversion...")
    print("-" * 70)
    
    # Check for model.json
    model_json = output_dir / 'model.json'
    if not model_json.exists():
        print("✗ ERROR: model.json not found!")
        return False
    
    print(f"✓ model.json found ({model_json.stat().st_size:,} bytes)")
    
    # List all generated files
    all_files = sorted(output_dir.iterdir())
    print()
    print("Generated files:")
    total_size = 0
    for file_path in all_files:
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            print(f"  - {file_path.name:40s} {size:10,} bytes")
    
    print()
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    
    return True


# =============================================================================
# MAIN CONVERSION PIPELINE
# =============================================================================

def main():
    """Main conversion pipeline."""
    
    print("=" * 70)
    print("GHOST TYPE CORRECTOR - MODEL CONVERSION")
    print("=" * 70)
    print()
    
    # Define paths
    script_dir = Path(__file__).parent
    ai_model_dir = script_dir.parent  # ai_model directory
    project_root = ai_model_dir.parent  # Ghost Type Corrector directory
    
    model_path = ai_model_dir / 'autocorrect_model.h5'
    output_dir = project_root / 'extension' / 'model'
    
    # Check if model exists
    if not model_path.exists():
        print("ERROR: Trained model not found!")
        print(f"Expected location: {model_path}")
        print()
        print("Please run 02_model_training.py first to train the model.")
        return 1
    
    print(f"Model file: {model_path}")
    print(f"Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Convert model (skip dependency check, just try to run)
    print("STEP 1: Converting Model")
    print("-" * 70)
    success = convert_model(model_path, output_dir)
    
    if not success:
        print()
        print("=" * 70)
        print("CONVERSION FAILED")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("1. Make sure you're using the correct conda environment:")
        print("   conda activate ghost-corrector-gpu")
        print()
        print("2. Verify tensorflowjs is installed:")
        print("   pip install tensorflowjs==4.4.0")
        print()
        print("3. If using Python 3.11+, try Python 3.10:")
        print("   conda create -n ghost-corrector-gpu python=3.10")
        print()
        return 1
    
    # Verify conversion
    print()
    print("STEP 2: Verifying Output")
    print("-" * 70)
    if not verify_conversion(output_dir):
        return 1
    
    # Success!
    print()
    print("=" * 70)
    print("CONVERSION SUCCESSFUL")
    print("=" * 70)
    print()
    print(f"✓ TensorFlow.js model saved to: {output_dir}")
    print()
    print("Next steps:")
    print("1. The model is ready to use in your browser extension")
    print("2. Update your extension's content.js to load the model:")
    print(f"   const model = await tf.loadLayersModel('model/model.json');")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
