# Ghost Type Corrector - Workflow Guide

## Table of Contents

1. [Complete Training Pipeline](#complete-training-pipeline)
2. [Environment Selection](#environment-selection)
3. [Data Flow Diagram](#data-flow-diagram)
4. [File Dependencies](#file-dependencies)
5. [Decision Trees](#decision-trees)

---

## Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     GHOST TYPE CORRECTOR                        │
│                   AI Model Training Pipeline                    │
└─────────────────────────────────────────────────────────────────┘

[START]
   │
   ├──> Environment Setup
   │    ├─ GPU Available?
   │    │  ├─ YES: conda env create -f environment-gpu.yml
   │    │  │       conda install cudatoolkit=11.2 cudnn=8.1.0
   │    │  │
   │    │  └─ NO:  conda env create -f environment-cpu.yml
   │    │
   │    └─ conda activate ghost-corrector-{gpu|cpu}
   │
   ├──> Data Preparation
   │    ├─ INPUT:  ai_model/data/corpus.txt
   │    ├─ SCRIPT: python ai_model/src/01_data_preprocessing.py
   │    └─ OUTPUT: ai_model/data/train_clean.txt
   │                ai_model/data/train_noisy.txt
   │
   ├──> Model Training
   │    ├─ INPUT:  train_clean.txt, train_noisy.txt
   │    ├─ SCRIPT: python ai_model/src/02_model_training.py
   │    └─ OUTPUT: ai_model/autocorrect_model.h5
   │                ai_model/data/tokenizer_config.json
   │
   ├──> Model Conversion
   │    ├─ INPUT:  autocorrect_model.h5, tokenizer_config.json
   │    ├─ SCRIPT: python ai_model/src/03_model_conversion.py
   │    └─ OUTPUT: extension/model/model.json
   │                extension/model/*.bin
   │
   └──> Browser Integration
        └─ USE: TensorFlow.js model in extension

[END]
```

---

## Environment Selection

```
┌───────────────────────────────────────────────────────┐
│         Which Environment Should I Use?               │
└───────────────────────────────────────────────────────┘

                    [START]
                       │
                       ▼
            ┌──────────────────────┐
            │ Do you have NVIDIA   │
            │ GPU with CUDA?       │
            └──────────────────────┘
                   /        \
                YES          NO
                 /            \
                ▼              ▼
    ┌──────────────────┐  ┌──────────────────┐
    │ Is GPU driver    │  │ Use CPU          │
    │ up to date?      │  │ Environment      │
    └──────────────────┘  └──────────────────┘
           /    \                  │
        YES      NO                │
         /        \                │
        ▼          ▼               ▼
 ┌────────────┐ ┌──────────┐ ┌────────────────┐
 │ GPU Ready  │ │ Update   │ │ environment-   │
 │            │ │ Driver   │ │ cpu.yml        │
 └────────────┘ └──────────┘ └────────────────┘
       │              │              │
       ▼              ▼              ▼
 ┌────────────────────────────┐ ┌────────────────┐
 │ environment-gpu.yml        │ │ BATCH_SIZE=32  │
 │ + cudatoolkit=11.2         │ │ Slower training│
 │ + cudnn=8.1.0              │ └────────────────┘
 └────────────────────────────┘         │
       │                                │
       ▼                                │
 ┌────────────────┐                    │
 │ BATCH_SIZE=128 │                    │
 │ Fast training  │                    │
 └────────────────┘                    │
       │                                │
       └────────────┬───────────────────┘
                    ▼
              [CONTINUE TO
               DATA PREP]
```

---

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA FLOW ARCHITECTURE                    │
└──────────────────────────────────────────────────────────────────┘

USER INPUT:
┌──────────────────────┐
│ corpus.txt           │  Raw training sentences (100K+ lines)
│                      │  Format: One sentence per line
│ Example:             │  Encoding: UTF-8
│ "The quick brown fox"│
└──────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│ 01_data_preprocessing.py                                    │
│ ─────────────────────────────────────────────────────────   │
│ 1. Read corpus.txt                                          │
│ 2. Clean text (lowercase, remove punctuation)              │
│ 3. Filter short sentences (< MIN_SENTENCE_LENGTH)          │
│ 4. Generate noisy versions (add typos)                     │
│    - Delete chars     ("hello" → "helo")                   │
│    - Insert chars     ("hello" → "helxlo")                 │
│    - Substitute chars ("hello" → "jello")                  │
│    - Swap chars       ("hello" → "hlelo")                  │
│ 5. Write parallel files                                     │
└─────────────────────────────────────────────────────────────┘
          │
          ├──────────────┬──────────────┐
          ▼              ▼              ▼
┌──────────────────┐ ┌──────────────────┐
│ train_clean.txt  │ │ train_noisy.txt  │
│                  │ │                  │
│ "the quick brown"│ │ "teh qick browm" │
│ "machine learn"  │ │ "machin lern"    │
│ "neural network" │ │ "nerual netwok"  │
└──────────────────┘ └──────────────────┘
          │                   │
          └─────────┬─────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 02_model_training.py                                        │
│ ─────────────────────────────────────────────────────────   │
│ 1. Load paired sentences                                   │
│ 2. Build character tokenizer                               │
│    - Unique characters: ['a'-'z', ' ', special tokens]     │
│    - Create char → index mapping                           │
│ 3. Convert text to sequences                               │
│    - "hello" → [8, 5, 12, 12, 15]                          │
│ 4. Pad sequences to MAX_SENTENCE_LENGTH                    │
│ 5. Build seq2seq model                                     │
│    ┌─────────────────────────────────────────────┐         │
│    │ ENCODER                                      │         │
│    │ Input (noisy) → Embedding → LSTM → State    │         │
│    └─────────────────────────────────────────────┘         │
│    ┌─────────────────────────────────────────────┐         │
│    │ DECODER                                      │         │
│    │ State → LSTM → Dense → Output (clean)       │         │
│    └─────────────────────────────────────────────┘         │
│ 6. Train with Adam optimizer                               │
│ 7. Save model and tokenizer                                │
└─────────────────────────────────────────────────────────────┘
          │
          ├────────────────────┬──────────────────────┐
          ▼                    ▼                      ▼
┌────────────────────┐ ┌──────────────────────┐
│ autocorrect_model  │ │ tokenizer_config.json│
│     .h5            │ │                      │
│ (Keras format)     │ │ {                    │
│ - Model arch       │ │   "char_index": {...}│
│ - Trained weights  │ │   "max_len": 100     │
│ - 5-10 MB          │ │ }                    │
└────────────────────┘ └──────────────────────┘
          │                    │
          └─────────┬──────────┘
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ 03_model_conversion.py                                      │
│ ─────────────────────────────────────────────────────────   │
│ 1. Load Keras model (.h5)                                  │
│ 2. Call tensorflowjs_converter via subprocess              │
│    - Convert architecture to JSON                          │
│    - Convert weights to binary shards                      │
│    - Optimize for browser execution                        │
│ 3. Save to extension/model/                                │
└─────────────────────────────────────────────────────────────┘
          │
          ├────────────────┬──────────────────┐
          ▼                ▼                  ▼
┌──────────────┐  ┌───────────────┐  ┌───────────────┐
│ model.json   │  │ group1-shard  │  │ group2-shard  │
│              │  │  1of1.bin     │  │  1of1.bin     │
│ {            │  │               │  │               │
│  "format":   │  │ Weight data   │  │ Weight data   │
│   "graph",   │  │ (binary)      │  │ (binary)      │
│  "layers":   │  └───────────────┘  └───────────────┘
│   [...]      │
│ }            │
└──────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────────┐
│ BROWSER EXTENSION                                            │
│ ─────────────────────────────────────────────────────────    │
│ 1. Load TensorFlow.js model                                 │
│ 2. Capture user typing                                      │
│ 3. Predict corrections                                      │
│ 4. Apply invisibly                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## File Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                    FILE DEPENDENCY GRAPH                    │
└─────────────────────────────────────────────────────────────┘

CONFIGURATION FILES:
┌────────────────────┐     ┌────────────────────┐
│ environment-gpu.yml│     │ environment-cpu.yml│
│                    │     │                    │
│ - python=3.10      │     │ - python=3.10      │
│ - tensorflow-gpu   │     │ - tensorflow-cpu   │
│ - cudatoolkit      │     │ - numpy            │
│ - cudnn            │     │ - tqdm             │
│ - tensorflowjs     │     │ - tensorflowjs     │
└────────────────────┘     └────────────────────┘
         │                          │
         └────────┬─────────────────┘
                  ▼
         [CONDA ENVIRONMENT]
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                      PROCESSING SCRIPTS                     │
└─────────────────────────────────────────────────────────────┘

01_data_preprocessing.py
├─ READS:  ai_model/data/corpus.txt
├─ WRITES: ai_model/data/train_clean.txt
└─ WRITES: ai_model/data/train_noisy.txt
           │
           ▼
02_model_training.py
├─ READS:  ai_model/data/train_clean.txt
├─ READS:  ai_model/data/train_noisy.txt
├─ WRITES: ai_model/autocorrect_model.h5
└─ WRITES: ai_model/data/tokenizer_config.json
           │
           ▼
03_model_conversion.py
├─ READS:  ai_model/autocorrect_model.h5
├─ READS:  ai_model/data/tokenizer_config.json
├─ WRITES: extension/model/model.json
└─ WRITES: extension/model/*.bin

┌─────────────────────────────────────────────────────────────┐
│                      OUTPUT STRUCTURE                       │
└─────────────────────────────────────────────────────────────┘

ai_model/
├── data/
│   ├── corpus.txt               [USER PROVIDED]
│   ├── train_clean.txt          [GENERATED: Step 1]
│   ├── train_noisy.txt          [GENERATED: Step 1]
│   └── tokenizer_config.json   [GENERATED: Step 2]
│
├── autocorrect_model.h5        [GENERATED: Step 2]
│
extension/
└── model/
    ├── model.json               [GENERATED: Step 3]
    └── *.bin                    [GENERATED: Step 3]
```

---

## Decision Trees

### When to Retrain vs. Reconvert

```
┌───────────────────────────────────────────────────────┐
│         Problem: Model Not Working                    │
└───────────────────────────────────────────────────────┘

                [START]
                   │
                   ▼
        ┌────────────────────────┐
        │ What is the symptom?   │
        └────────────────────────┘
               /    |    \
              /     │     \
             /      │      \
            ▼       ▼       ▼
    ┌──────────┐ ┌──────┐ ┌────────────┐
    │ Low      │ │ Won't│ │ Browser    │
    │ Accuracy │ │ Load │ │ Error      │
    └──────────┘ └──────┘ └────────────┘
         │           │           │
         ▼           ▼           ▼
    ┌──────────┐ ┌──────────┐ ┌──────────────┐
    │ RETRAIN  │ │ Check    │ │ RECONVERT    │
    │          │ │ Keras    │ │              │
    │ Actions: │ │ Version  │ │ Actions:     │
    │ - More   │ │          │ │ - Reinstall  │
    │   epochs │ │ TF 2.10? │ │   tfjs       │
    │ - More   │ │   │      │ │ - Check      │
    │   data   │ │   ▼      │ │   paths      │
    │ - Bigger │ │ ┌──────┐ │ │ - Verify     │
    │   model  │ │ │RETRAIN││ │   .h5 exists │
    └──────────┘ │ └──────┘ │ └──────────────┘
                 │          │
                 └────┬─────┘
                      ▼
               [RESOLUTION]
```

### Configuration Selection

```
┌───────────────────────────────────────────────────────┐
│      How Should I Configure Training?                │
└───────────────────────────────────────────────────────┘

                    [START]
                       │
                       ▼
            ┌──────────────────────┐
            │ What is your goal?   │
            └──────────────────────┘
               /         │         \
              /          │          \
             ▼           ▼           ▼
    ┌────────────┐ ┌─────────┐ ┌──────────┐
    │ Quick Test │ │ Quality │ │ Memory   │
    │            │ │ Model   │ │ Limited  │
    └────────────┘ └─────────┘ └──────────┘
         │              │            │
         ▼              ▼            ▼
    EPOCHS=5       EPOCHS=20    BATCH=16
    BATCH=32       BATCH=128    LATENT=128
    SAMPLES=10K    LATENT=512   SAMPLES=50K
    LATENT=128     SAMPLES=MAX
         │              │            │
         └──────┬───────┴────────────┘
                ▼
         [START TRAINING]
```

---

## Timeline Estimates

```
┌─────────────────────────────────────────────────────────────┐
│                   TYPICAL PROJECT TIMELINE                  │
└─────────────────────────────────────────────────────────────┘

SETUP PHASE:
├─ Install Anaconda/Miniconda            [15 minutes]
├─ Create conda environment              [5 minutes]
├─ Install CUDA/cuDNN (GPU only)         [10 minutes]
└─ Verify installation                   [2 minutes]
   ─────────────────────────────────────────────────────────
   TOTAL: ~30 minutes (one-time setup)

DATA PREPARATION PHASE:
├─ Obtain corpus (download)              [10-60 minutes]
├─ Format corpus (if needed)             [5-30 minutes]
├─ Run preprocessing script              [1-2 minutes]
└─ Verify outputs                        [1 minute]
   ─────────────────────────────────────────────────────────
   TOTAL: ~20-90 minutes

TRAINING PHASE:
├─ Configure parameters                  [5 minutes]
├─ Run training script                   [5-50 minutes]
│  ├─ GPU (NVIDIA RTX 3080): ~5 min
│  ├─ GPU (NVIDIA GTX 1060): ~10 min
│  └─ CPU (Intel i7): ~50 min
└─ Verify model saved                    [1 minute]
   ─────────────────────────────────────────────────────────
   TOTAL: ~10-60 minutes

CONVERSION PHASE:
├─ Run conversion script                 [30-60 seconds]
├─ Verify TensorFlow.js outputs          [1 minute]
└─ Test in browser (manual)              [5 minutes]
   ─────────────────────────────────────────────────────────
   TOTAL: ~7 minutes

═══════════════════════════════════════════════════════════
TOTAL PROJECT TIME (first run): 1-3 hours
TOTAL PROJECT TIME (subsequent): 20-60 minutes (training only)
```

---

**Reference:** See [SETUP.md](SETUP.md) for detailed instructions  
**Quick Commands:** See [QUICKSTART.md](QUICKSTART.md) for command reference  
**Version:** 1.0.0
