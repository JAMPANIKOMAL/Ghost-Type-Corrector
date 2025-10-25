# Ghost Type Corrector - Quick Reference

## Environment Commands

### GPU Environment

```powershell
# Create
conda env create -f environment-gpu.yml

# Activate
conda activate ghost-corrector-gpu

# Install CUDA support
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Deactivate
conda deactivate

# Remove
conda env remove -n ghost-corrector-gpu
```

### CPU Environment

```powershell
# Create
conda env create -f environment-cpu.yml

# Activate
conda activate ghost-corrector-cpu

# Verify
python -c "import tensorflow as tf; print(tf.__version__)"

# Deactivate
conda deactivate

# Remove
conda env remove -n ghost-corrector-cpu
```

---

## Training Pipeline

### Full Workflow

```powershell
# 1. Activate environment
conda activate ghost-corrector-gpu

# 2. Navigate to source
cd ai_model/src

# 3. Run preprocessing
python 01_data_preprocessing.py

# 4. Train model
python 02_model_training.py

# 5. Convert to TensorFlow.js
python 03_model_conversion.py
```

### Individual Steps

```powershell
# Preprocessing only
python ai_model/src/01_data_preprocessing.py

# Training only
python ai_model/src/02_model_training.py

# Conversion only
python ai_model/src/03_model_conversion.py
```

---

## Configuration Quick Edit

### Data Preprocessing

**File:** `ai_model/src/01_data_preprocessing.py`

```python
NOISE_LEVEL = 0.15              # Typo probability (0.0-1.0)
MIN_SENTENCE_LENGTH = 3         # Minimum words
RANDOM_SEED = 42                # Reproducibility
```

### Model Training

**File:** `ai_model/src/02_model_training.py`

```python
NUM_SAMPLES = 100000            # Training samples
MAX_SENTENCE_LENGTH = 100       # Max characters
EMBEDDING_DIM = 128             # Embedding size
LATENT_DIM = 256                # LSTM hidden size
EPOCHS = 10                     # Training iterations
BATCH_SIZE = 64                 # Batch size
VALIDATION_SPLIT = 0.2          # Validation fraction
```

### Recommended Settings

| Use Case                | BATCH_SIZE | EPOCHS | LATENT_DIM |
|-------------------------|------------|--------|------------|
| **Quick Test (CPU)**    | 32         | 5      | 128        |
| **Standard (GPU)**      | 128        | 10     | 256        |
| **High Accuracy (GPU)** | 256        | 20     | 512        |
| **Memory Limited**      | 16         | 10     | 128        |

---

## Common Issues

### GPU Not Detected

```powershell
# Check driver
nvidia-smi

# Reinstall CUDA
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --force-reinstall
```

### Out of Memory

**Edit** `ai_model/src/02_model_training.py`:

```python
BATCH_SIZE = 32        # Reduce from 64
LATENT_DIM = 128       # Reduce from 256
NUM_SAMPLES = 50000    # Reduce from 100000
```

### Conversion Fails

```powershell
# Reinstall tensorflowjs
conda activate ghost-corrector-gpu
pip install tensorflowjs==4.4.0 protobuf<3.20 --force-reinstall
```

### Model Won't Load

```powershell
# Retrain with correct environment
conda activate ghost-corrector-gpu
cd ai_model/src
python 02_model_training.py
```

---

## File Locations

### Input Files

```
ai_model/data/corpus.txt          # Your training data (required)
```

### Generated Files

```
ai_model/data/train_clean.txt                 # Preprocessing output
ai_model/data/train_noisy.txt                 # Preprocessing output
ai_model/data/tokenizer_config.json           # Training output
ai_model/autocorrect_model.h5                 # Training output
extension/model/model.json                    # Conversion output
extension/model/*.bin                         # Conversion output
```

---

## Verification Commands

### Check Installations

```powershell
# TensorFlow version
python -c "import tensorflow as tf; print(tf.__version__)"

# GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# TensorFlowJS version
python -c "import tensorflowjs; print(tensorflowjs.__version__)"

# CUDA build
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

### Check Generated Files

```powershell
# Training data
dir ai_model\data\*.txt

# Trained model
dir ai_model\*.h5

# Converted model
dir extension\model\*.*
```

---

## Performance Tips

### GPU Training

1. **Maximize batch size** without OOM errors
2. **Monitor GPU usage:** `nvidia-smi`
3. **Close other GPU applications** during training
4. **Use mixed precision** (advanced):
   ```python
   from tensorflow.keras import mixed_precision
   mixed_precision.set_global_policy('mixed_float16')
   ```

### CPU Training

1. **Reduce batch size** to 32 or 16
2. **Reduce model size** (LATENT_DIM = 128)
3. **Use fewer samples** for testing
4. **Close memory-intensive applications**

---

## Data Requirements

### Corpus Format

```
the quick brown fox jumps over the lazy dog
machine learning is transforming technology
natural language processing enables better communication
```

**Rules:**
- One sentence per line
- UTF-8 encoding
- Minimum 10,000 sentences (100,000+ recommended)
- Clean text (no special formatting)

### Recommended Sources

1. **Leipzig Corpora Collection** (public domain)
2. **Wikipedia dumps** (Creative Commons)
3. **Project Gutenberg** (public domain books)
4. **Custom domain-specific text** (emails, documents)

---

## Next Steps

1. **Install environment:** See [SETUP.md](SETUP.md)
2. **Prepare corpus:** Place in `ai_model/data/corpus.txt`
3. **Run pipeline:** Execute all three scripts
4. **Test model:** Check `extension/model/` for outputs
5. **Integrate:** Use TensorFlow.js model in browser extension

---

**Quick Start:** See [SETUP.md](SETUP.md) for detailed instructions  
**Visual Guide:** See [WORKFLOW.md](WORKFLOW.md) for process diagrams  
**Version:** 1.0.0
