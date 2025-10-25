# Quick Reference - Ghost Type Corrector

## 🔧 Environment Setup

### Create Environment (Choose One)

**GPU (NVIDIA with CUDA):**
```powershell
conda env create -f environment-gpu.yml
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

**CPU (Universal):**
```powershell
conda env create -f environment-cpu.yml
conda activate ghost-corrector-cpu
```

### Verify Setup

**Check TensorFlow:**
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Check GPU (if using GPU env):**
```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## 📝 Training Workflow

```powershell
# Activate environment
conda activate ghost-corrector-gpu  # or ghost-corrector-cpu

# Navigate to ai_model directory
cd ai_model

# 1. Preprocess data (~1 minute)
python 01_data_preprocessing.py

# 2. Train model (~5-50 minutes)
python 02_model_training.py

# 3. Convert to TensorFlow.js (~30 seconds)
python 03_model_conversion.py
```

---

## ⚙️ Key Configuration Parameters

### Data Preprocessing (`01_data_preprocessing.py`)
```python
NOISE_LEVEL = 0.15              # Typo frequency (15%)
MIN_SENTENCE_LENGTH = 3         # Min words per sentence
RANDOM_SEED = 42                # For reproducibility
```

### Model Training (`02_model_training.py`)
```python
NUM_SAMPLES = 100000            # Training samples (None = all)
EMBEDDING_DIM = 128             # Character embedding size
LATENT_DIM = 256                # LSTM units
EPOCHS = 10                     # Training iterations
BATCH_SIZE = 64                 # Batch size (↑ for GPU, ↓ if OOM)
```

---

## 📊 Expected Performance

### Training Time (100K samples, 10 epochs)
| Hardware | Time |
|----------|------|
| RTX 3080 | ~5 min |
| GTX 1060 | ~10 min |
| CPU (8-core) | ~50 min |

### Model Size
- Keras model (.h5): ~5-10 MB
- TensorFlow.js: ~3-8 MB (sharded)

---

## 🔍 Troubleshooting

### GPU Not Detected
```powershell
# Check drivers
nvidia-smi

# Reinstall CUDA
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Out of Memory
Reduce batch size:
```python
BATCH_SIZE = 32  # or 16
```

### Conversion Fails
```powershell
pip install tensorflowjs==4.4.0 protobuf<3.20
```

---

## 📁 Generated Files

After running all scripts:
```
ai_model/
├── data/
│   ├── train_clean.txt          ← Step 1
│   ├── train_noisy.txt          ← Step 1
│   └── tokenizer_config.json    ← Step 2
│
├── autocorrect_model.h5         ← Step 2
│
extension/model/
├── model.json                   ← Step 3
└── group*.bin                   ← Step 3
```

---

## 🎯 Common Commands

```powershell
# List conda environments
conda env list

# Activate environment
conda activate ghost-corrector-gpu

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n ghost-corrector-gpu

# Update packages
conda update --all

# Check installed packages
conda list
```

---

## 💡 Tips

- **Use GPU** whenever possible (10x faster)
- **Start small** (100K samples) for testing
- **Increase EPOCHS** to 20-50 for better accuracy
- **Monitor training** - stop if loss stops decreasing
- **Save checkpoints** for long training runs

---

## 📖 Full Documentation

See **[SETUP.md](SETUP.md)** for complete guide.
