#!/usr/bin/env python3
"""
Direct Patch Script - Patches tensorflowjs without importing it first
"""

import os
import sys
import shutil
from pathlib import Path

def find_and_patch():
    """Find and patch tensorflowjs files directly."""
    
    # Get the conda environment path
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        print("ERROR: Not in a conda environment!")
        print("Activate your environment first: conda activate ghost-corrector-gpu")
        return False
    
    print("=" * 70)
    print("PATCHING TENSORFLOWJS LIBRARY")
    print("=" * 70)
    print(f"\nConda environment: {conda_prefix}")
    
    # Path to tensorflowjs
    tfjs_path = Path(conda_prefix) / 'lib' / 'site-packages' / 'tensorflowjs'
    
    if not tfjs_path.exists():
        print(f"\nERROR: tensorflowjs not found at {tfjs_path}")
        return False
    
    print(f"TensorFlowJS path: {tfjs_path}")
    
    # Files to patch
    files_to_patch = {
        'read_weights.py': [
            ('np.uint8, np.uint16, np.object, np.bool]', 
             'np.uint8, np.uint16, object, bool]')
        ],
        'write_weights.py': [
            ('np.uint8, np.uint16, np.bool, np.object]', 
             'np.uint8, np.uint16, bool, object]'),
            ('if data.dtype == np.object:', 
             'if data.dtype == object:'),
            ('data.dtype == np.bool', 
             'data.dtype == bool')
        ]
    }
    
    patched_count = 0
    
    for filename, replacements in files_to_patch.items():
        file_path = tfjs_path / filename
        
        if not file_path.exists():
            print(f"\n⚠ File not found: {filename}")
            continue
        
        print(f"\n📝 Processing: {filename}")
        
        # Read file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"  ✗ Error reading file: {e}")
            continue
        
        # Check if already patched
        original_content = content
        
        # Apply replacements
        for old_text, new_text in replacements:
            if old_text in content:
                print(f"  Found: {old_text}")
                content = content.replace(old_text, new_text)
                print(f"  Replaced with: {new_text}")
            else:
                print(f"  ℹ Pattern not found or already patched")
        
        # Only write if changed
        if content != original_content:
            # Backup original
            backup_path = file_path.with_suffix('.py.backup')
            if not backup_path.exists():
                shutil.copy(file_path, backup_path)
                print(f"  ✓ Backup created: {backup_path.name}")
            
            # Write patched version
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  ✓ Patched successfully")
                patched_count += 1
            except Exception as e:
                print(f"  ✗ Error writing file: {e}")
                # Restore backup
                if backup_path.exists():
                    shutil.copy(backup_path, file_path)
                return False
        else:
            print(f"  ℹ No changes needed")
    
    print("\n" + "=" * 70)
    if patched_count > 0:
        print(f"✓ SUCCESSFULLY PATCHED {patched_count} FILE(S)")
    else:
        print("ℹ NO PATCHING NEEDED (already patched or different version)")
    print("=" * 70)
    print()
    
    return True


if __name__ == "__main__":
    if find_and_patch():
        print("Now run the conversion script:")
        print("  python convert_model_final.py")
        sys.exit(0)
    else:
        print("\nPatching failed!")
        sys.exit(1)
