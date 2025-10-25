# 🎯 Ghost Type Corrector - Visual Workflow Guide

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GHOST TYPE CORRECTOR WORKFLOW                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ STEP 1: ENVIRONMENT SETUP (One-time)                                │
└─────────────────────────────────────────────────────────────────────┘

    Choose Your Path:
    
    ┌──────────────────────┐              ┌──────────────────────┐
    │   GPU Training       │              │   CPU Training       │
    │   (Recommended)      │              │   (Universal)        │
    └──────────────────────┘              └──────────────────────┘
             │                                     │
             ├─ Fast training (~5 min)             ├─ Slower (~50 min)
             ├─ Requires NVIDIA GPU                ├─ Works anywhere
             └─ CUDA 11.2 + cuDNN 8.1              └─ No GPU needed
             │                                     │
             ▼                                     ▼
    
    conda env create -f          conda env create -f
    environment-gpu.yml          environment-cpu.yml
             │                                     │
             ▼                                     ▼
    conda activate                 conda activate
    ghost-corrector-gpu          ghost-corrector-cpu
             │                                     │
             ▼                                     │
    conda install -c                              │
    conda-forge                                   │
    cudatoolkit=11.2                              │
    cudnn=8.1.0                                   │
             │                                     │
             └──────────────┬──────────────────────┘
                            │
                            ▼
                    ENVIRONMENT READY! ✅


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 2: DATA PIPELINE                                               │
└─────────────────────────────────────────────────────────────────────┘

    📄 corpus.txt (you provide)
         │
         │  One sentence per line
         │  Example:
         │    "the quick brown fox jumps"
         │    "machine learning is amazing"
         │
         ▼
    ┌─────────────────────────────────┐
    │ 01_data_preprocessing.py        │
    │                                 │
    │ ✓ Clean text                    │
    │ ✓ Lowercase                     │
    │ ✓ Remove punctuation            │
    │ ✓ Generate synthetic typos      │
    └─────────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         │              │              │
         ▼              ▼              ▼
    train_clean.txt  train_noisy.txt
         │                  │
         │  "the quick     │  "the qiuck
         │   brown fox"    │   borwn fox"
         │                 │
         └────────┬────────┘
                  │
                  ▼
            DATA READY! ✅


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 3: MODEL TRAINING                                              │
└─────────────────────────────────────────────────────────────────────┘

    Clean + Noisy Data
         │
         ▼
    ┌─────────────────────────────────┐
    │ 02_model_training.py            │
    │                                 │
    │ ┌─────────────────────┐         │
    │ │  Character Tokenizer│         │
    │ │  a→1, b→2, c→3...   │         │
    │ └─────────────────────┘         │
    │          │                      │
    │          ▼                      │
    │ ┌─────────────────────┐         │
    │ │   Seq2Seq LSTM      │         │
    │ │   Encoder-Decoder   │         │
    │ │                     │         │
    │ │  Noisy → [LSTM] →   │         │
    │ │          [LSTM] →   │         │
    │ │          [Dense] →  │         │
    │ │          Clean      │         │
    │ └─────────────────────┘         │
    │          │                      │
    │   Train 10 epochs               │
    │   Batch size: 64                │
    │   GPU Accelerated 🚀            │
    └─────────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         │              │              │
         ▼              ▼              ▼
    autocorrect_   tokenizer_
    model.h5       config.json
         │              │
         └──────┬───────┘
                │
                ▼
         MODEL TRAINED! ✅


┌─────────────────────────────────────────────────────────────────────┐
│ STEP 4: MODEL CONVERSION                                            │
└─────────────────────────────────────────────────────────────────────┘

    autocorrect_model.h5
         │
         │  Keras Format (Python)
         │
         ▼
    ┌─────────────────────────────────┐
    │ 03_model_conversion.py          │
    │                                 │
    │ TensorFlowJS Converter          │
    │                                 │
    │ Keras → TensorFlow.js           │
    └─────────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         │              │              │
         ▼              ▼              ▼
    model.json    group1-shard1.bin  ...
         │              │
         │  JavaScript  │  Weights
         │  Format      │
         │              │
         └──────┬───────┘
                │
                ▼
    BROWSER-READY MODEL! ✅


┌─────────────────────────────────────────────────────────────────────┐
│ FINAL: DEPLOY TO BROWSER                                            │
└─────────────────────────────────────────────────────────────────────┘

    extension/model/
    ├── model.json
    └── *.bin
         │
         ▼
    ┌─────────────────────────────────┐
    │  Browser Extension              │
    │                                 │
    │  TensorFlow.js Runtime          │
    │                                 │
    │  User types: "helllo"           │
    │  Model predicts: "hello"        │
    │  Autocorrect! ✨                │
    └─────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════

📊 PERFORMANCE METRICS

Training Time (100,000 samples, 10 epochs):
┌──────────────────┬─────────────┐
│ Hardware         │ Time        │
├──────────────────┼─────────────┤
│ RTX 3080 (GPU)   │ ~5 minutes  │ ⚡ Recommended
│ GTX 1060 (GPU)   │ ~10 minutes │ ✓ Good
│ CPU (8-core)     │ ~50 minutes │ ✓ OK
└──────────────────┴─────────────┘

Model Size:
┌──────────────────┬─────────────┐
│ Format           │ Size        │
├──────────────────┼─────────────┤
│ Keras (.h5)      │ 5-10 MB     │
│ TensorFlow.js    │ 3-8 MB      │
└──────────────────┴─────────────┘

═══════════════════════════════════════════════════════════════════════

🎛️  CONFIGURATION CHEATSHEET

Data Preprocessing (01_data_preprocessing.py):
┌────────────────────┬─────────┬─────────────────────────┐
│ Parameter          │ Default │ Purpose                 │
├────────────────────┼─────────┼─────────────────────────┤
│ NOISE_LEVEL        │ 0.15    │ 15% words get typos     │
│ MIN_SENTENCE_LEN   │ 3       │ Filter short sentences  │
│ RANDOM_SEED        │ 42      │ Reproducibility         │
└────────────────────┴─────────┴─────────────────────────┘

Model Training (02_model_training.py):
┌────────────────────┬─────────┬─────────────────────────┐
│ Parameter          │ Default │ Purpose                 │
├────────────────────┼─────────┼─────────────────────────┤
│ NUM_SAMPLES        │ 100000  │ Training data size      │
│ EMBEDDING_DIM      │ 128     │ Char vector size        │
│ LATENT_DIM         │ 256     │ LSTM hidden units       │
│ EPOCHS             │ 10      │ Training iterations     │
│ BATCH_SIZE         │ 64      │ GPU: ↑, CPU/OOM: ↓      │
│ VALIDATION_SPLIT   │ 0.2     │ 20% for validation      │
└────────────────────┴─────────┴─────────────────────────┘

═══════════════════════════════════════════════════════════════════════

🚨 TROUBLESHOOTING

GPU Not Detected?
  → nvidia-smi (check drivers)
  → conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

Out of Memory?
  → Reduce BATCH_SIZE (try 32 or 16)
  → Reduce NUM_SAMPLES
  → Close other GPU applications

Conversion Fails?
  → pip install tensorflowjs==4.4.0 protobuf<3.20
  → Verify correct conda environment is active

═══════════════════════════════════════════════════════════════════════
```
