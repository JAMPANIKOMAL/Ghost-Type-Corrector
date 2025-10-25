# Model Conversion Instructions

## Problem
The model was trained with Keras 2/TensorFlow 2.x but the current Python 3.13 environment uses Keras 3/TensorFlow 2.20, which has compatibility issues.

## Solutions

### Option 1: Use Node.js tensorflowjs converter (RECOMMENDED)

The Node.js version of the tensorflowjs converter is more reliable and doesn't have Python version compatibility issues.

#### Steps:

1. **Install Node.js** (if not already installed):
   - Download from: https://nodejs.org/
   - Install the LTS version

2. **Install the TensorFlow.js converter**:
   ```bash
   npm install -g @tensorflow/tfjs-converter
   ```

3. **Convert the model**:
   ```bash
   tensorflowjs_converter `
     --input_format=keras `
     "C:\Users\jampa\Videos\Ghost Type Corrector\ai_model\autocorrect_model.h5" `
     "C:\Users\jampa\Videos\Ghost Type Corrector\extension\model"
   ```

### Option 2: Retrain the model

Retrain the model using the current environment (TensorFlow 2.20 / Keras 3):

```powershell
cd "C:\Users\jampa\Videos\Ghost Type Corrector\ai_model\src"
& "C:/Users/jampa/Videos/Ghost Type Corrector/.venv/Scripts/python.exe" 02_model_training.py
```

After retraining, the model will be compatible with the current Keras version.

### Option 3: Use Python 3.9-3.11

Create a new virtual environment with Python 3.9, 3.10, or 3.11 (which have better TensorFlow.js compatibility):

1. Install Python 3.10 from python.org
2. Create a new virtual environment
3. Install the required packages
4. Run the conversion script

## Current Status

- Model file exists: `ai_model/autocorrect_model.h5`
- Model was trained with TensorFlow/Keras 2.x
- Current environment: Python 3.13.7, TensorFlow 2.20.0, Keras 3.11.3
- Compatibility issue: Keras 2 → Keras 3 model format changed

## Recommendation

**Use Option 1 (Node.js converter)** - it's the fastest and most reliable solution.
