# Quick Start Guide

Fast setup for experienced users. For detailed instructions, see [INSTALLATION.md](INSTALLATION.md).

## Prerequisites

- Anaconda/Miniconda installed
- Corpus data at `ai_model/data/corpus.txt`

## Setup Commands

### GPU Training (Recommended)

```powershell
# Create environment
conda env create -f environment-gpu.yml
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install tensorflowjs==3.18.0

# Verify GPU
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"

# Train model
cd ai_model\src
python 01_data_preprocessing.py
python 02_model_training.py
python convert_direct.py
```

### CPU Training

```powershell
# Create environment
conda env create -f environment-cpu.yml
conda activate ghost-corrector-cpu
pip install tensorflowjs==3.18.0

# Train model
cd ai_model\src
python 01_data_preprocessing.py
python 02_model_training.py
python convert_direct.py
```

## Expected Output

```
ai_model/
├── autocorrect_model.h5              (9 MB)
├── data/
│   ├── train_clean.txt
│   ├── train_noisy.txt
│   └── tokenizer_config.json

extension/model/
├── model.json
└── group1-shard*.bin                 (10 files, 3 MB total)
```

## Training Time

| Hardware    | Time     |
|-------------|----------|
| RTX 3080    | 5 min    |
| RTX 3050    | 7 min    |
| GTX 1060    | 10 min   |
| Intel i7    | 50 min   |

## Common Issues

### GPU not detected
```powershell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 --force-reinstall
```

### Out of memory
Edit `02_model_training.py`:
```python
BATCH_SIZE = 32        # Reduce from 64
LATENT_DIM = 128       # Reduce from 256
```

### Conversion fails
```powershell
pip install tensorflowjs==3.18.0 --force-reinstall
python convert_direct.py
```

## Next Steps

- See [INSTALLATION.md](INSTALLATION.md) for detailed setup
- See [CONFIGURATION.md](CONFIGURATION.md) for tuning options
- See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
