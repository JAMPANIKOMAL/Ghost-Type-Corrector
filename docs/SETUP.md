# Ghost Type Corrector - Installation and Training Guide# Ghost Type Corrector - Setup and Training Guide



## Table of Contents## Overview



1. [Overview](#overview)This guide walks you through setting up your environment and training the AI autocorrection model.

2. [Prerequisites](#prerequisites)

3. [Environment Setup](#environment-setup)---

4. [Training Pipeline](#training-pipeline)

5. [Configuration](#configuration)## 📋 Prerequisites

6. [Troubleshooting](#troubleshooting)

7. [Performance Benchmarks](#performance-benchmarks)- **Anaconda** or **Miniconda** installed

- For GPU training: **NVIDIA GPU** with CUDA support + latest drivers

---- At least **8GB RAM** (16GB+ recommended for GPU training)

- **10GB free disk space**

## Overview

---

This guide provides comprehensive instructions for setting up your development environment and training the Ghost Type Corrector AI model. The project supports both GPU-accelerated and CPU-only workflows.

## 🚀 Quick Start

---

### Step 1: Choose Your Environment

## Prerequisites

**Option A: GPU Training (Recommended for Speed)**

### Required Software```powershell

- **Anaconda** or **Miniconda** (3.8+)# Create environment

- **Git** (for version control)conda env create -f environment-gpu.yml

- Minimum **8GB RAM** (16GB+ recommended for GPU training)

- **10GB free disk space**# Activate

conda activate ghost-corrector-gpu

### For GPU Training

- **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher# Install CUDA support (CRITICAL!)

- **NVIDIA GPU drivers** (latest version recommended)conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

- **Windows 10/11**, **Linux**, or **macOS** (with compatible GPU)

# Verify GPU

---python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"

```

## Environment Setup

**Option B: CPU Training (Universal)**

### Option A: GPU-Accelerated Training (Recommended)```powershell

# Create environment

#### Step 1: Create Base Environmentconda env create -f environment-cpu.yml



```powershell# Activate

conda env create -f environment-gpu.ymlconda activate ghost-corrector-cpu

```

# Verify installation

This creates an environment named `ghost-corrector-gpu` with Python 3.10 and all required dependencies.python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"

```

#### Step 2: Activate Environment

---

```powershell

conda activate ghost-corrector-gpu### Step 2: Prepare Your Data

```

Place your corpus file in the data directory:

#### Step 3: Install CUDA Support```

Ghost Type Corrector/

**CRITICAL:** TensorFlow 2.10 requires specific CUDA and cuDNN versions:└── ai_model/

    └── data/

```powershell        └── corpus.txt  ← Your training text here

conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0```

```

The corpus should be a text file with one sentence per line.

#### Step 4: Verify GPU Detection

---

```powershell

python -c "import tensorflow as tf; print('GPU Devices:', tf.config.list_physical_devices('GPU'))"### Step 3: Run the Training Pipeline

```

Execute the scripts in order:

**Expected Output:**

```**A. Preprocess Data**

GPU Devices: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]```powershell

```cd ai_model

python 01_data_preprocessing.py

If no GPU is detected, refer to the [Troubleshooting](#troubleshooting) section.```

This creates:

---- `data/train_clean.txt` - Clean sentences

- `data/train_noisy.txt` - Same sentences with synthetic typos

### Option B: CPU-Only Training (Universal)

**B. Train Model**

#### Step 1: Create Environment```powershell

python 02_model_training.py

```powershell```

conda env create -f environment-cpu.ymlThis creates:

```- `autocorrect_model.h5` - Trained Keras model

- `data/tokenizer_config.json` - Character vocabulary

This creates an environment named `ghost-corrector-cpu` with CPU-optimized TensorFlow.

**C. Convert to TensorFlow.js**

#### Step 2: Activate Environment```powershell

python 03_model_conversion.py

```powershell```

conda activate ghost-corrector-cpuThis creates:

```- `extension/model/model.json` - Model architecture

- `extension/model/group*.bin` - Model weights

#### Step 3: Verify Installation

---

```powershell

python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"## ⚙️ Configuration

```

Edit the configuration sections in each script:

**Expected Output:**

```### 01_data_preprocessing.py

TensorFlow version: 2.10.0```python

```NOISE_LEVEL = 0.15           # 15% of words get typos

MIN_SENTENCE_LENGTH = 3      # Minimum words per sentence

---RANDOM_SEED = 42             # For reproducibility

```

## Training Pipeline

### 02_model_training.py

### Data Preparation```python

NUM_SAMPLES = 100000         # Training samples (None = all)

Place your training corpus in the designated location:EMBEDDING_DIM = 128          # Character embedding size

LATENT_DIM = 256             # LSTM hidden units

```EPOCHS = 10                  # Training epochs

ai_model/data/corpus.txtBATCH_SIZE = 64              # Batch size (increase for GPU)

``````



**Format Requirements:**---

- Plain text file

- One sentence per line## 📊 Expected Training Time

- UTF-8 encoding

- Minimum 10,000 sentences recommendedWith 100,000 samples:



**Example:**| Setup | Time per Epoch | Total (10 epochs) |

```|-------|----------------|-------------------|

the quick brown fox jumps over the lazy dog| **NVIDIA RTX 3080** | ~30s | ~5 minutes |

machine learning is transforming technology| **NVIDIA GTX 1060** | ~60s | ~10 minutes |

natural language processing enables better communication| **CPU (8 cores)** | ~5min | ~50 minutes |

```

---

### Pipeline Execution

## 🔧 Troubleshooting

Navigate to the source directory:

### GPU Not Detected

```powershell```powershell

cd ai_model/src# Check CUDA installation

```python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"



#### Step 1: Data Preprocessing# Should show: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

```

```powershell

python 01_data_preprocessing.pyIf empty:

```1. Verify NVIDIA drivers are installed: `nvidia-smi`

2. Reinstall CUDA support:

**Output:**   ```powershell

- `../data/train_clean.txt` - Cleaned sentences   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

- `../data/train_noisy.txt` - Sentences with synthetic typos   ```



**Duration:** 1-2 minutes for 100,000 sentences### Out of Memory (OOM) Error



#### Step 2: Model TrainingReduce batch size in `02_model_training.py`:

```python

```powershellBATCH_SIZE = 32  # or 16 for very limited GPU memory

python 02_model_training.py```

```

### Conversion Failed

**Output:**

- `../autocorrect_model.h5` - Trained Keras modelMake sure you're in the correct environment:

- `../data/tokenizer_config.json` - Character vocabulary configuration```powershell

conda activate ghost-corrector-gpu

**Duration:** 5-50 minutes depending on hardware (see [Performance Benchmarks](#performance-benchmarks))pip install tensorflowjs==4.4.0 protobuf<3.20

```

#### Step 3: Model Conversion

---

```powershell

python 03_model_conversion.py## 📁 Project Structure

```

```

**Output:**Ghost Type Corrector/

- `../../extension/model/model.json` - Model architecture├── environment-gpu.yml           # GPU environment config

- `../../extension/model/group*.bin` - Model weights├── environment-cpu.yml           # CPU environment config

├── SETUP.md                      # This file

**Duration:** 30-60 seconds├── README.md                     # Project overview

│

---├── ai_model/

│   ├── 01_data_preprocessing.py  # Clean data & generate typos

## Configuration│   ├── 02_model_training.py      # Train LSTM model

│   ├── 03_model_conversion.py    # Convert to TensorFlow.js

### Data Preprocessing Parameters│   │

│   ├── data/

Edit `ai_model/src/01_data_preprocessing.py`:│   │   ├── corpus.txt            # Raw training text (you provide)

│   │   ├── train_clean.txt       # Generated: clean sentences

```python│   │   ├── train_noisy.txt       # Generated: noisy sentences

NOISE_LEVEL = 0.15              # Probability of typo per word (0.0-1.0)│   │   └── tokenizer_config.json # Generated: vocabulary

MIN_SENTENCE_LENGTH = 3         # Minimum words per sentence│   │

RANDOM_SEED = 42                # Random seed for reproducibility│   └── autocorrect_model.h5      # Generated: trained model

```│

└── extension/

### Model Training Parameters    └── model/

        ├── model.json            # Generated: TF.js model

Edit `ai_model/src/02_model_training.py`:        └── group1-shard*.bin     # Generated: weights

```

```python

NUM_SAMPLES = 100000            # Training samples (None = all available)---

MAX_SENTENCE_LENGTH = 100       # Maximum characters per sentence

## 🎯 Next Steps

# Model Architecture

EMBEDDING_DIM = 128             # Character embedding dimensionAfter successful conversion:

LATENT_DIM = 256                # LSTM hidden state dimension

1. **Test the model** in the browser extension

# Training Configuration2. **Tune hyperparameters** for better accuracy

EPOCHS = 10                     # Number of training iterations3. **Train on more data** for improved performance

BATCH_SIZE = 64                 # Batch size (increase for GPU)4. **Experiment with architecture** (deeper LSTMs, attention, etc.)

VALIDATION_SPLIT = 0.2          # Fraction of data for validation

```---



**Recommendations:**## 💡 Tips for Better Results

- **GPU users:** Increase `BATCH_SIZE` to 128 or 256

- **CPU users:** Decrease `BATCH_SIZE` to 32 if memory limited### Improve Accuracy

- **Better accuracy:** Increase `EPOCHS` to 20-50- **More data**: Train on 1M+ sentences

- **Larger model:** Increase `LATENT_DIM` to 512- **More epochs**: Increase to 20-50 epochs

- **Larger model**: Increase `LATENT_DIM` to 512

---- **Bidirectional LSTM**: Add `Bidirectional()` wrapper



## Troubleshooting### Reduce Model Size

- **Smaller embeddings**: Decrease `EMBEDDING_DIM` to 64

### GPU Not Detected- **Quantization**: Use `tfjs.converters.convert()` with `--weight_shard_size_bytes`

- **Pruning**: Remove low-weight connections

**Symptoms:**

```### Speed Up Training

GPU Devices: []- **GPU**: Always use GPU for large datasets

```- **Mixed precision**: Add `tf.keras.mixed_precision.set_global_policy('mixed_float16')`

- **Data caching**: Use `tf.data.Dataset` for large datasets

**Solutions:**

---

1. **Verify NVIDIA drivers:**

   ```powershell## 📚 Additional Resources

   nvidia-smi

   ```- [TensorFlow Documentation](https://www.tensorflow.org/)

   Should display GPU information. If not, update drivers from NVIDIA website.- [TensorFlow.js Guide](https://www.tensorflow.org/js/guide)

- [Keras Sequential Models](https://keras.io/guides/sequential_model/)

2. **Reinstall CUDA support:**- [LSTM Theory](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

   ```powershell

   conda activate ghost-corrector-gpu---

   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --force-reinstall

   ```## 🆘 Getting Help



3. **Check TensorFlow GPU support:**If you encounter issues:

   ```powershell

   python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"1. Check this guide's troubleshooting section

   ```2. Review error messages carefully

   Should return `True`.3. Verify environment setup: `conda list`

4. Check GitHub Issues for similar problems

### Out of Memory (OOM) Errors

---

**Symptoms:**

```**Happy Training! 🎉**

ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

1. **Reduce batch size** in `02_model_training.py`:
   ```python
   BATCH_SIZE = 32  # or 16 for very limited memory
   ```

2. **Reduce model size:**
   ```python
   LATENT_DIM = 128  # instead of 256
   ```

3. **Limit training data:**
   ```python
   NUM_SAMPLES = 50000  # instead of 100000
   ```

### Conversion Failures

**Symptoms:**
```
ModuleNotFoundError: No module named 'tensorflowjs'
```

**Solutions:**

1. **Verify correct environment:**
   ```powershell
   conda activate ghost-corrector-gpu
   conda list | findstr tensorflow
   ```

2. **Reinstall tensorflowjs:**
   ```powershell
   pip install tensorflowjs==4.4.0 protobuf<3.20 --force-reinstall
   ```

---

## Performance Benchmarks

### Training Speed

Based on 100,000 samples, 10 epochs:

| Hardware Configuration | Time per Epoch | Total Time (10 epochs) |
|------------------------|----------------|------------------------|
| NVIDIA RTX 3080 (10GB) | 30 seconds     | 5 minutes              |
| NVIDIA RTX 2070 (8GB)  | 45 seconds     | 7.5 minutes            |
| NVIDIA GTX 1060 (6GB)  | 60 seconds     | 10 minutes             |
| Intel i7-10700K (CPU)  | 5 minutes      | 50 minutes             |
| AMD Ryzen 7 3700X (CPU)| 4 minutes      | 40 minutes             |

### Model Size

| Format              | File Size | Description                    |
|---------------------|-----------|--------------------------------|
| Keras (.h5)         | 5-10 MB   | Original trained model         |
| TensorFlow.js       | 3-8 MB    | Converted for browser (sharded)|

### Expected Accuracy

With default configuration (100K samples, 10 epochs):

| Metric                  | Value      |
|-------------------------|------------|
| Training Accuracy       | 85-95%     |
| Validation Accuracy     | 80-90%     |
| Training Loss           | 0.2-0.4    |
| Validation Loss         | 0.3-0.5    |

**Note:** Accuracy depends on corpus quality and diversity.

---

## Project Structure

```
Ghost-Type-Corrector/
├── environment-gpu.yml           # GPU environment configuration
├── environment-cpu.yml           # CPU environment configuration
├── README.md                     # Project overview
├── LICENSE                       # MIT License
│
├── docs/                         # Documentation
│   ├── SETUP.md                  # This file
│   ├── QUICKSTART.md             # Quick reference
│   └── WORKFLOW.md               # Visual workflow guide
│
├── ai_model/                     # Model development
│   ├── src/                      # Source code
│   │   ├── 01_data_preprocessing.py
│   │   ├── 02_model_training.py
│   │   └── 03_model_conversion.py
│   │
│   ├── data/                     # Training data
│   │   ├── corpus.txt            # User provides
│   │   ├── train_clean.txt       # Generated
│   │   ├── train_noisy.txt       # Generated
│   │   └── tokenizer_config.json # Generated
│   │
│   ├── notebooks/                # Jupyter notebooks
│   └── autocorrect_model.h5      # Generated
│
└── extension/                    # Browser extension
    ├── model/                    # TensorFlow.js model
    │   ├── model.json            # Generated
    │   └── *.bin                 # Generated
    ├── assets/
    └── js/
```

---

**Last Updated:** October 2025  
**Version:** 1.0.0
