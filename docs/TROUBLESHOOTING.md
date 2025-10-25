# Troubleshooting Guide

Common issues and solutions for Ghost Type Corrector setup and training.

## Environment Issues

### GPU Not Detected

**Symptoms:**
```
GPU: []
```

**Solutions:**

1. Check NVIDIA driver:
```powershell
nvidia-smi
```

2. Reinstall CUDA toolkit:
```powershell
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --force-reinstall
```

3. Verify TensorFlow build:
```powershell
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

Expected output: `True`

4. Check GPU visibility:
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### CUDA Version Mismatch

**Error:**
```
Could not load dynamic library 'cudart64_110.dll'
```

**Solution:**
TensorFlow 2.10 requires CUDA 11.2 specifically:
```powershell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

### Conda Environment Creation Fails

**Error:**
```
PackagesNotFoundError: The following packages are not available
```

**Solution:**
Ensure conda is updated:
```powershell
conda update conda -y
conda env create -f environment-gpu.yml
```

If still failing, create manually:
```powershell
conda create -n ghost-corrector-gpu python=3.10 -y
conda activate ghost-corrector-gpu
pip install tensorflow==2.10.1 numpy==1.24.3 tqdm
```

## Training Issues

### Out of Memory Error

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

1. Reduce batch size in `02_model_training.py`:
```python
BATCH_SIZE = 32  # From 64
```

2. Reduce model size:
```python
LATENT_DIM = 128  # From 256
```

3. Limit dataset size:
```python
NUM_SAMPLES = 50000  # From 100000
```

4. Enable memory growth:
```python
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Training Accuracy Not Improving

**Symptoms:**
- Accuracy stuck below 50%
- Loss not decreasing

**Solutions:**

1. Check data quality:
```powershell
# Verify data files exist and contain data
cd ai_model\data
Get-Content train_clean.txt | Select-Object -First 10
Get-Content train_noisy.txt | Select-Object -First 10
```

2. Increase training duration:
```python
EPOCHS = 20  # From 10
```

3. Increase model capacity:
```python
LATENT_DIM = 512  # From 256
```

4. Use more training data:
```python
NUM_SAMPLES = None  # Use all available data
```

### NaN Loss During Training

**Symptoms:**
```
loss: nan
accuracy: 0.0000
```

**Solutions:**

1. Reduce learning rate (edit `02_model_training.py`):
```python
optimizer = Adam(learning_rate=0.0001)  # Add custom optimizer
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
```

2. Check for data corruption:
```powershell
python 01_data_preprocessing.py  # Regenerate data
```

3. Add gradient clipping:
```python
optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
```

## Conversion Issues

### TensorFlowJS Import Error

**Error:**
```
ImportError: cannot import name 'load_keras_model'
```

**Solution:**
Use TensorFlowJS 3.18.0 specifically:
```powershell
pip uninstall tensorflowjs -y
pip install tensorflowjs==3.18.0
```

### tensorflow-decision-forests Conflict

**Error:**
```
ERROR: Could not find a version that satisfies tensorflow-decision-forests
```

**Solution:**
Install TensorFlowJS separately (not in environment YAML):
```powershell
conda activate ghost-corrector-gpu
pip install tensorflowjs==3.18.0
```

### NumPy Compatibility Warnings

**Warning:**
```
AttributeError: module 'numpy' has no attribute 'object'
```

**Solution:**
The `convert_direct.py` script handles this automatically with patches. Use it instead of `03_model_conversion.py`:
```powershell
python convert_direct.py
```

### Conversion Produces No Output

**Symptoms:**
- No error messages
- `extension/model/` directory empty

**Solutions:**

1. Check model file exists:
```powershell
cd ai_model
dir autocorrect_model.h5
```

2. Run conversion with verbose output:
```powershell
python convert_direct.py
```

3. Check permissions on output directory:
```powershell
cd ..\extension
mkdir model -Force
cd ..\ai_model\src
python convert_direct.py
```

### SavedModel Permission Error

**Error:**
```
PermissionError: [WinError 32] The process cannot access the file
```

**Solution:**
Remove temporary directories and use direct conversion:
```powershell
cd ai_model
rmdir /s /q saved_model_temp
cd src
python convert_direct.py
```

## Data Issues

### Corpus File Not Found

**Error:**
```
FileNotFoundError: ai_model/data/corpus.txt
```

**Solution:**
Place your corpus file in the correct location:
```powershell
# Create data directory
mkdir ai_model\data -Force

# Add your corpus.txt file
# Download from: https://wortschatz.uni-leipzig.de/en/download
```

### Empty Training Files

**Symptoms:**
- `train_clean.txt` or `train_noisy.txt` is 0 KB
- Training fails immediately

**Solutions:**

1. Check corpus.txt has content:
```powershell
cd ai_model\data
Get-Content corpus.txt | Measure-Object -Line
```

2. Rerun preprocessing:
```powershell
cd ..\src
python 01_data_preprocessing.py
```

3. Verify output:
```powershell
cd ..\data
Get-Content train_clean.txt | Select-Object -First 5
```

### Encoding Errors

**Error:**
```
UnicodeDecodeError: 'utf-8' codec can't decode byte
```

**Solution:**
Ensure corpus.txt is UTF-8 encoded. Convert in PowerShell:
```powershell
Get-Content corpus.txt | Out-File corpus_utf8.txt -Encoding UTF8
```

## Performance Issues

### Training Too Slow

**Current:** 5+ minutes per epoch on GPU

**Solutions:**

1. Increase batch size:
```python
BATCH_SIZE = 128  # From 64
```

2. Reduce dataset size:
```python
NUM_SAMPLES = 50000  # From 100000
```

3. Reduce sequence length:
```python
MAX_SENTENCE_LENGTH = 50  # From 100
```

4. Verify GPU usage:
```powershell
# In another terminal while training
nvidia-smi
```

GPU utilization should be >80%.

### Conversion Too Slow

**Current:** >2 minutes

**Solution:**
The `convert_direct.py` script is optimized. If still slow:
```powershell
# Ensure no antivirus scanning
python convert_direct.py
```

## Validation Issues

### Model Files Not Loading in Browser

**Symptoms:**
- `model.json` loads but weights fail
- Console errors in browser

**Solutions:**

1. Verify all weight files exist:
```powershell
cd extension\model
dir *.bin
```

Should show 10 files: `group1-shard1of10.bin` through `group1-shard10of10.bin`

2. Check model.json format:
```powershell
Get-Content model.json | Select-Object -First 20
```

Should contain `"modelTopology"` and `"weightsManifest"`.

3. Test loading with TensorFlow.js:
```javascript
const model = await tf.loadLayersModel('file://extension/model/model.json');
console.log(model.summary());
```

### Tokenizer Config Missing

**Error:**
```
FileNotFoundError: tokenizer_config.json
```

**Solution:**
Regenerate during training:
```powershell
cd ai_model\src
python 02_model_training.py
```

The tokenizer is automatically saved after training completes.

## System Issues

### Disk Space Full

**Error:**
```
OSError: [Errno 28] No space left on device
```

**Solution:**
Free up space. Model files require:
- Corpus: 100 MB - 1 GB
- Training data: 200-500 MB
- Trained model: 10 MB
- TensorFlow.js model: 3-5 MB
- Conda environment: 2-3 GB
- Training checkpoints: 50-100 MB

Total: ~4-5 GB

### Python Version Issues

**Error:**
```
Python 3.13 is not compatible with TensorFlow 2.10
```

**Solution:**
Use Python 3.10:
```powershell
conda create -n ghost-corrector-gpu python=3.10 -y
conda activate ghost-corrector-gpu
```

### DLL Load Failed (Windows)

**Error:**
```
ImportError: DLL load failed while importing _pywrap_tensorflow_internal
```

**Solution:**
Install Visual C++ Redistributable:
- Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Install and restart

## Getting Help

If issues persist:

1. Check all code files for errors:
```powershell
cd ai_model\src
python -m py_compile 01_data_preprocessing.py
python -m py_compile 02_model_training.py
python -m py_compile convert_direct.py
```

2. Verify environment:
```powershell
conda activate ghost-corrector-gpu
conda list
```

3. Create a clean environment:
```powershell
conda deactivate
conda env remove -n ghost-corrector-gpu
conda env create -f environment-gpu.yml
```

4. Review documentation:
- [INSTALLATION.md](INSTALLATION.md) - Setup instructions
- [CONFIGURATION.md](CONFIGURATION.md) - Parameter tuning
- [QUICKSTART.md](QUICKSTART.md) - Fast commands

---

**Last Updated:** October 25, 2025
