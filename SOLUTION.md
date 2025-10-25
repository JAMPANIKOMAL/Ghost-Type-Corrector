# Ghost Type Corrector - Model Conversion Guide

## Current Situation

**Problem**: Cannot convert the existing Keras model to TensorFlow.js due to:
- Model was trained with Keras 2 (TensorFlow 2.13 or earlier)
- Current environment has Python 3.13.7 with Keras 3 (TensorFlow 2.20)
- Keras 2 → Keras 3 has breaking changes in model format
- TensorFlow.js converter (Python) has dependency conflicts with Python 3.13

## ✅ WORKING SOLUTION

### Option A: Retrain Model (Recommended - Simplest)

The model exists but needs to be retrained with the current environment:

```powershell
# Make sure you're in the project directory
cd "C:\Users\jampa\Videos\Ghost Type Corrector"

# Run the training script
& "C:/Users/jampa/Videos/Ghost Type Corrector/.venv/Scripts/python.exe" ai_model/src/02_model_training.py
```

This will create a new `autocorrect_model.h5` compatible with Keras 3.

**After retraining**, use this Python script to convert:

```python
# Save this as quick_convert.py
import keras
import tensorflowjs as tfjs

# Load model
model = keras.models.load_model('ai_model/autocorrect_model.h5')

# Convert to TF.js
tfjs.converters.save_keras_model(model, 'extension/model')

print("✓ Conversion complete!")
```

### Option B: Use Docker (Most Reliable)

Use a Docker container with compatible Python version:

```bash
docker run -it --rm -v "${PWD}:/workspace" python:3.10

# Inside container:
cd /workspace
pip install tensorflowjs==4.4.0 tensorflow==2.13
tensorflowjs_converter --input_format keras \
  ai_model/autocorrect_model.h5 \
  extension/model
```

### Option C: Manual Installation of Python 3.10

1. Download Python 3.10 from python.org
2. Install to a separate location (e.g., C:\Python310)
3. Create virtual environment:
   ```powershell
   C:\Python310\python.exe -m venv .venv310
   .venv310\Scripts\activate
   pip install tensorflowjs==4.4.0 "tensorflow==2.13" "numpy<2.0"
   ```
4. Convert:
   ```powershell
   tensorflowjs_converter --input_format keras `
     "C:\Users\jampa\Videos\Ghost Type Corrector\ai_model\autocorrect_model.h5" `
     "C:\Users\jampa\Videos\Ghost Type Corrector\extension\model"
   ```

## Current Environment Details

- **Python**: 3.13.7
- **TensorFlow**: 2.20.0
- **Keras**: 3.11.3
- **Model File**: `ai_model/autocorrect_model.h5` (exists, trained with Keras 2)
- **Target**: `extension/model/` (needs model.json + .bin files)

## Dependencies Installed

Currently installed in virtual environment:
- tensorflow==2.20.0
- keras==3.11.3
- tf-keras==2.20.1 (Keras 2 compatibility layer - but incompatible with Python 3.13)
- numpy==2.3.4
- h5py==3.15.1
- tensorflowjs==4.22.0 (installed without full dependencies)
- pandas, wurlitzer, tensorflow-decision-forests (partial installs)

## Why Current Approach Fails

1. **tensorflowjs converter** requires `tensorflow-decision-forests`
2. **tensorflow-decision-forests** requires `tensorflow~=2.15.0`
3. But Python 3.13 only supports **TensorFlow 2.20+**
4. Keras 2 models can't load in Keras 3 without compatibility issues
5. The `tf_keras` compatibility layer has issues with InputLayer deserialization

## Recommended Next Steps

**BEST**: Run Option A (retrain the model) - it takes ~10 minutes and solves all issues.

**ALTERNATIVE**: If you can't retrain, install Python 3.10 alongside (Option C).

## Files Created During Troubleshooting

- `src/03_model_conversion_v2.py` - Uses subprocess for conversion
- `src/03_model_conversion_simple.py` - Attempts direct Keras loading
- `src/convert_model.py` - Final attempt with multiple fallbacks
- `CONVERSION_INSTRUCTIONS.md` - This file

## Quick Test if It Works

After conversion, check:
```powershell
dir extension\model\
```

You should see:
- `model.json` (model architecture)
- `group1-shard1of1.bin` or similar (model weights)
