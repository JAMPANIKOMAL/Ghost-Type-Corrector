"""
Final Model Conversion Script
Direct conversion without problematic dependencies
"""

import os
import sys
import subprocess

print("=" * 70)
print("TENSORFLOWJS MODEL CONVERSION TOOL")
print("=" * 70)
print()

# --- Define File Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
extension_dir = os.path.join(os.path.dirname(base_dir), 'extension')

input_model_path = os.path.join(base_dir, 'autocorrect_model.h5')
output_model_dir = os.path.join(extension_dir, 'model')

print(f"Input model:  {input_model_path}")
print(f"Output dir:   {output_model_dir}")
print()

# Check if input exists
if not os.path.exists(input_model_path):
    print("ERROR: Model file not found!")
    print(f"Expected location: {input_model_path}")
    print()
    print("Please train the model first by running:")
    print("  python 02_model_training.py")
    sys.exit(1)

# Create output directory
os.makedirs(output_model_dir, exist_ok=True)

# Try using tensorflowjs_wizard package which has fewer dependencies
print("Attempting conversion...")
print()

# Method: Install and use tensorflowjs-wizard (lighter weight)
try:
    print("Step 1: Installing tensorflowjs-wizard (lightweight converter)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "tensorflowjs-wizard", "--quiet"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Installed successfully")
    else:
        print("✗ Installation failed, trying alternative method...")
        raise Exception("Installation failed")
    
    print()
    print("Step 2: Converting model...")
    
    # Use tensorflowjs-wizard
    result = subprocess.run(
        [
            sys.executable, "-m", "tensorflowjs_wizard",
            "--input_path", input_model_path,
            "--output_path", output_model_dir
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Conversion successful!")
        print()
        print(result.stdout)
    else:
        raise Exception("Conversion failed")
        
except Exception as e:
    print(f"\nWizard method failed: {e}")
    print()
    print("=" * 70)
    print("ALTERNATIVE APPROACH REQUIRED")
    print("=" * 70)
    print()
    print("Due to Python 3.13 compatibility issues, please use one of these methods:")
    print()
    print("METHOD 1: Use online converter")
    print("-" * 70)
    print("1. Visit: https://www.npmjs.com/package/@tensorflow/tfjs-converter")
    print("2. Follow the documentation for Keras model conversion")
    print()
    print("METHOD 2: Use Python 3.10")
    print("-" * 70)
    print("1. Install Python 3.10")
    print("2. Create a new virtual environment")
    print("3. Install: pip install tensorflowjs==4.4.0 tensorflow==2.13")
    print("4. Run: tensorflowjs_converter --input_format keras \\")
    print(f"        {input_model_path} \\")
    print(f"        {output_model_dir}")
    print()
    print("METHOD 3: Retrain with current environment")
    print("-" * 70)
    print("Retrain the model so it's compatible with current Keras/TF versions:")
    print("  python 02_model_training.py")
    print()
    sys.exit(1)

# Verify output
print()
print("=" * 70)
print("VERIFICATION")
print("=" * 70)
print()

try:
    files = os.listdir(output_model_dir)
    if files:
        print(f"Files created in {output_model_dir}:")
        for file in files:
            file_path = os.path.join(output_model_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  ✓ {file} ({size:,} bytes)")
        
        if 'model.json' in files:
            print()
            print("✓ SUCCESS: model.json found!")
            print("✓ The model is ready to use in your browser extension")
        else:
            print()
            print("⚠ WARNING: model.json not found")
            print("  Conversion may be incomplete")
    else:
        print("✗ ERROR: No files were created")
        sys.exit(1)
        
except Exception as e:
    print(f"ERROR during verification: {e}")
    sys.exit(1)

print()
print("=" * 70)
print("CONVERSION COMPLETE")
print("=" * 70)
