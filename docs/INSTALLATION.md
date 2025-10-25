# Installation Guide

Complete setup instructions for Ghost Type Corrector from environment creation to model deployment.

## Prerequisites

- Anaconda or Miniconda
- NVIDIA GPU with CUDA support (optional, recommended for faster training)
- 10GB free disk space
- Text corpus data (100K+ sentences recommended)

## Installation Steps

### 1. Clone Repository

```powershell
git clone https://github.com/JAMPANIKOMAL/Ghost-Type-Corrector.git
cd Ghost-Type-Corrector
```

### 2. Create Conda Environment

**For GPU Training (Recommended):**

```powershell
# Create environment
conda env create -f environment-gpu.yml

# Activate
conda activate ghost-corrector-gpu

# Install CUDA support (CRITICAL!)
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y

# Install TensorFlowJS (avoid dependency conflicts)
pip install tensorflowjs==3.18.0
```

**For CPU Training:**

```powershell
# Create environment
conda env create -f environment-cpu.yml

# Activate
conda activate ghost-corrector-cpu

# Install TensorFlowJS
pip install tensorflowjs==3.18.0
```

### 3. Verify GPU Detection (GPU only)

```powershell
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Expected output:
```
GPU: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### 4. Prepare Training Data

Place your corpus file at:
```
ai_model/data/corpus.txt
```

**Format:**
- Plain text file
- One sentence per line
- UTF-8 encoding
- Minimum 10,000 sentences (100,000+ recommended)

**Example corpus.txt:**
```
the quick brown fox jumps over the lazy dog
machine learning is transforming technology
natural language processing enables communication
```

**Where to get corpus data:**
- Leipzig Corpora Collection: https://wortschatz.uni-leipzig.de/en/download
- Download "English (GB) - 1M sentences" dataset
- Extract and save as `corpus.txt`

### 5. Run Training Pipeline

```powershell
cd ai_model\src

# Step 1: Preprocess data (1-2 minutes)
python 01_data_preprocessing.py

# Step 2: Train model (5-50 minutes depending on hardware)
python 02_model_training.py

# Step 3: Convert to TensorFlow.js (30-60 seconds)
python convert_direct.py
```

### 6. Verify Output

Check that these files were created:

```
ai_model/
├── autocorrect_model.h5              # Trained Keras model (9 MB)
├── data/
│   ├── train_clean.txt               # Cleaned sentences
│   ├── train_noisy.txt               # Sentences with typos
│   └── tokenizer_config.json         # Character vocabulary

extension/model/
├── model.json                         # TensorFlow.js architecture
└── group1-shard*.bin                  # Model weights (10 files)
```

---

## Training Configuration

### Data Preprocessing (`01_data_preprocessing.py`)

```python
NOISE_LEVEL = 0.15              # Typo probability (0.0-1.0)
MIN_SENTENCE_LENGTH = 3         # Minimum words per sentence
RANDOM_SEED = 42                # For reproducibility
```

### Model Training (`02_model_training.py`)

```python
NUM_SAMPLES = 100000            # Training samples (None = all)
MAX_SENTENCE_LENGTH = 100       # Max characters per sentence
EMBEDDING_DIM = 128             # Character embedding size
LATENT_DIM = 256                # LSTM hidden units
EPOCHS = 10                     # Training iterations
BATCH_SIZE = 64                 # Batch size (128-256 for GPU)
VALIDATION_SPLIT = 0.2          # Validation data fraction
```

**Recommended Settings:**

| Hardware | BATCH_SIZE | EPOCHS | LATENT_DIM | Training Time |
|----------|------------|--------|------------|---------------|
| RTX 3080 | 256        | 20     | 512        | 10-15 min     |
| RTX 3050 | 128        | 10     | 256        | 7-10 min      |
| GTX 1060 | 64         | 10     | 256        | 15-20 min     |
| CPU i7   | 32         | 10     | 128        | 45-60 min     |

---

## Troubleshooting

### GPU Not Detected

```powershell
# 1. Check NVIDIA driver
nvidia-smi

# 2. Reinstall CUDA
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --force-reinstall

# 3. Verify TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
```

### Out of Memory Error

Edit `02_model_training.py`:
```python
BATCH_SIZE = 32        # Reduce from 64
LATENT_DIM = 128       # Reduce from 256
NUM_SAMPLES = 50000    # Reduce from 100000
```

### Conversion Fails

```powershell
# Use the direct conversion script (more reliable)
python convert_direct.py

# If still fails, check tensorflowjs version
pip list | findstr tensorflow
pip install tensorflowjs==3.18.0 --force-reinstall
```

### Model Won't Load in Browser

The conversion creates a **Keras LayersModel** format. Load it with:

```javascript
const model = await tf.loadLayersModel('model/model.json');
```

---

## Performance Benchmarks

### Training Speed (100K samples, 10 epochs)

| Hardware           | Time/Epoch | Total Time |
|--------------------|------------|------------|
| RTX 3080 (10GB)    | 30s        | 5 min      |
| RTX 3050 (6GB)     | 40s        | 7 min      |
| GTX 1060 (6GB)     | 60s        | 10 min     |
| Intel i7 (CPU)     | 5 min      | 50 min     |

### Expected Accuracy

With default settings (100K samples, 10 epochs):

| Metric              | Value   |
|---------------------|---------|
| Training Accuracy   | 60-70%  |
| Validation Accuracy | 55-65%  |
| Training Loss       | 0.7-0.8 |
| Validation Loss     | 0.8-0.9 |

### Model Size

| Format          | Size     |
|-----------------|----------|
| Keras (.h5)     | 9-10 MB  |
| TensorFlow.js   | 3-4 MB   |

---

## Common Issues & Solutions

### Issue: `tensorflow-decision-forests` dependency error

**Solution:** Use TensorFlowJS 3.18.0 instead of 4.4.0:
```powershell
pip uninstall tensorflowjs -y
pip install tensorflowjs==3.18.0
```

### Issue: NumPy compatibility warnings

**Solution:** Already handled by `convert_direct.py` script (patches NumPy automatically)

### Issue: Permission denied on `saved_model_temp`

**Solution:**
```powershell
cd ai_model
rmdir /s /q saved_model_temp
cd src
python convert_direct.py
```

### Issue: CUDA version mismatch

**Solution:** TensorFlow 2.10 requires CUDA 11.2:
```powershell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
```

---

## Project Structure

```
Ghost-Type-Corrector/
├── environment-gpu.yml              # GPU environment config
├── environment-cpu.yml              # CPU environment config
├── README.md                        # Project overview
├── REPRODUCTION.md                  # This file
│
├── docs/                            # Documentation
│   ├── SETUP.md                     # Detailed setup guide
│   ├── QUICKSTART.md                # Quick reference
│   └── WORKFLOW.md                  # Visual workflows
│
├── ai_model/                        # Model development
│   ├── src/                         # Source scripts
│   │   ├── 01_data_preprocessing.py # Data cleaning & typo generation
│   │   ├── 02_model_training.py     # Seq2seq LSTM training
│   │   └── convert_direct.py        # TensorFlow.js conversion
│   │
│   ├── data/                        # Training data (gitignored)
│   │   ├── corpus.txt               # User provides
│   │   ├── train_clean.txt          # Generated
│   │   ├── train_noisy.txt          # Generated
│   │   └── tokenizer_config.json    # Generated
│   │
│   ├── notebooks/                   # Jupyter notebooks
│   └── autocorrect_model.h5         # Trained model (gitignored)
│
└── extension/                       # Browser extension
    ├── model/                       # TensorFlow.js model (gitignored)
    │   ├── model.json               # Generated
    │   └── *.bin                    # Generated
    ├── assets/
    ├── js/
    └── manifest.json
```

---

## Quick Reproduction Commands

**Complete setup in one go:**

```powershell
# 1. Create and activate environment
conda env create -f environment-gpu.yml
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install tensorflowjs==3.18.0

# 2. Verify GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# 3. Run pipeline (after adding corpus.txt)
cd ai_model\src
python 01_data_preprocessing.py
python 02_model_training.py
python convert_direct.py
```

---

## Notes

- **Python 3.10** is required (3.13 has compatibility issues)
- **TensorFlow 2.10** is the last version with native Windows GPU support
- **TensorFlowJS 3.18.0** works without `tensorflow-decision-forests` dependency
- **CUDA 11.2 + cuDNN 8.1.0** are specifically required for TensorFlow 2.10
- Training is **5-10x faster on GPU** compared to CPU
- Model conversion uses **manual weight extraction** (most reliable method)

---

## Support

For issues or questions:
1. Check [SETUP.md](docs/SETUP.md) for detailed instructions
2. Check [QUICKSTART.md](docs/QUICKSTART.md) for quick commands
3. Open an issue on GitHub

---

**Last Updated:** October 25, 2025  
**Tested On:** Windows 11, NVIDIA RTX 3050, Python 3.10.13
