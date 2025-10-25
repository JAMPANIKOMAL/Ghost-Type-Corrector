# Ghost Type Corrector - Setup and Training Guide

## Overview

This guide walks you through setting up your environment and training the AI autocorrection model.

---

## 📋 Prerequisites

- **Anaconda** or **Miniconda** installed
- For GPU training: **NVIDIA GPU** with CUDA support + latest drivers
- At least **8GB RAM** (16GB+ recommended for GPU training)
- **10GB free disk space**

---

## 🚀 Quick Start

### Step 1: Choose Your Environment

**Option A: GPU Training (Recommended for Speed)**
```powershell
# Create environment
conda env create -f environment-gpu.yml

# Activate
conda activate ghost-corrector-gpu

# Install CUDA support (CRITICAL!)
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

# Verify GPU
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

**Option B: CPU Training (Universal)**
```powershell
# Create environment
conda env create -f environment-cpu.yml

# Activate
conda activate ghost-corrector-cpu

# Verify installation
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
```

---

### Step 2: Prepare Your Data

Place your corpus file in the data directory:
```
Ghost Type Corrector/
└── ai_model/
    └── data/
        └── corpus.txt  ← Your training text here
```

The corpus should be a text file with one sentence per line.

---

### Step 3: Run the Training Pipeline

Execute the scripts in order:

**A. Preprocess Data**
```powershell
cd ai_model
python 01_data_preprocessing.py
```
This creates:
- `data/train_clean.txt` - Clean sentences
- `data/train_noisy.txt` - Same sentences with synthetic typos

**B. Train Model**
```powershell
python 02_model_training.py
```
This creates:
- `autocorrect_model.h5` - Trained Keras model
- `data/tokenizer_config.json` - Character vocabulary

**C. Convert to TensorFlow.js**
```powershell
python 03_model_conversion.py
```
This creates:
- `extension/model/model.json` - Model architecture
- `extension/model/group*.bin` - Model weights

---

## ⚙️ Configuration

Edit the configuration sections in each script:

### 01_data_preprocessing.py
```python
NOISE_LEVEL = 0.15           # 15% of words get typos
MIN_SENTENCE_LENGTH = 3      # Minimum words per sentence
RANDOM_SEED = 42             # For reproducibility
```

### 02_model_training.py
```python
NUM_SAMPLES = 100000         # Training samples (None = all)
EMBEDDING_DIM = 128          # Character embedding size
LATENT_DIM = 256             # LSTM hidden units
EPOCHS = 10                  # Training epochs
BATCH_SIZE = 64              # Batch size (increase for GPU)
```

---

## 📊 Expected Training Time

With 100,000 samples:

| Setup | Time per Epoch | Total (10 epochs) |
|-------|----------------|-------------------|
| **NVIDIA RTX 3080** | ~30s | ~5 minutes |
| **NVIDIA GTX 1060** | ~60s | ~10 minutes |
| **CPU (8 cores)** | ~5min | ~50 minutes |

---

## 🔧 Troubleshooting

### GPU Not Detected
```powershell
# Check CUDA installation
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If empty:
1. Verify NVIDIA drivers are installed: `nvidia-smi`
2. Reinstall CUDA support:
   ```powershell
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

### Out of Memory (OOM) Error

Reduce batch size in `02_model_training.py`:
```python
BATCH_SIZE = 32  # or 16 for very limited GPU memory
```

### Conversion Failed

Make sure you're in the correct environment:
```powershell
conda activate ghost-corrector-gpu
pip install tensorflowjs==4.4.0 protobuf<3.20
```

---

## 📁 Project Structure

```
Ghost Type Corrector/
├── environment-gpu.yml           # GPU environment config
├── environment-cpu.yml           # CPU environment config
├── SETUP.md                      # This file
├── README.md                     # Project overview
│
├── ai_model/
│   ├── 01_data_preprocessing.py  # Clean data & generate typos
│   ├── 02_model_training.py      # Train LSTM model
│   ├── 03_model_conversion.py    # Convert to TensorFlow.js
│   │
│   ├── data/
│   │   ├── corpus.txt            # Raw training text (you provide)
│   │   ├── train_clean.txt       # Generated: clean sentences
│   │   ├── train_noisy.txt       # Generated: noisy sentences
│   │   └── tokenizer_config.json # Generated: vocabulary
│   │
│   └── autocorrect_model.h5      # Generated: trained model
│
└── extension/
    └── model/
        ├── model.json            # Generated: TF.js model
        └── group1-shard*.bin     # Generated: weights
```

---

## 🎯 Next Steps

After successful conversion:

1. **Test the model** in the browser extension
2. **Tune hyperparameters** for better accuracy
3. **Train on more data** for improved performance
4. **Experiment with architecture** (deeper LSTMs, attention, etc.)

---

## 💡 Tips for Better Results

### Improve Accuracy
- **More data**: Train on 1M+ sentences
- **More epochs**: Increase to 20-50 epochs
- **Larger model**: Increase `LATENT_DIM` to 512
- **Bidirectional LSTM**: Add `Bidirectional()` wrapper

### Reduce Model Size
- **Smaller embeddings**: Decrease `EMBEDDING_DIM` to 64
- **Quantization**: Use `tfjs.converters.convert()` with `--weight_shard_size_bytes`
- **Pruning**: Remove low-weight connections

### Speed Up Training
- **GPU**: Always use GPU for large datasets
- **Mixed precision**: Add `tf.keras.mixed_precision.set_global_policy('mixed_float16')`
- **Data caching**: Use `tf.data.Dataset` for large datasets

---

## 📚 Additional Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [TensorFlow.js Guide](https://www.tensorflow.org/js/guide)
- [Keras Sequential Models](https://keras.io/guides/sequential_model/)
- [LSTM Theory](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

---

## 🆘 Getting Help

If you encounter issues:

1. Check this guide's troubleshooting section
2. Review error messages carefully
3. Verify environment setup: `conda list`
4. Check GitHub Issues for similar problems

---

**Happy Training! 🎉**
