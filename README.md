# Ghost Type Corrector

A next-generation, AI-powered browser extension that provides truly invisible, contextual autocorrection, bringing the seamless "mobile keyboard" experience to your desktop browser.

This project is an advanced rebuild of an initial prototype: https://github.com/JAMPANIKOMAL/invisible-autocorrect-extension, replacing the original frequency-dictionary "brain" with a character-level, sequence-to-sequence (seq2seq) neural network.

## Core Features

- **Invisible Correction:** No distracting popups, underlines, or suggestions. Typos are corrected instantly and silently the moment you press the spacebar.

- **Contextual AI:** The model understands context, allowing it to fix both spelling mistakes ("wrod") and grammatical/contextual errors ("I went too the store").

- **Mobile-Style Undo:** The signature feature: if the AI makes an unwanted correction, simply press Backspace once to instantly undo the correction and revert to your original typing.

- **Browser-Native Feel:** The extension disables the browser's default red-squiggle spellcheck, providing a clean, seamless, and native-feeling typing experience.

## Quick Start

### 1. Setup Environment

Choose GPU (faster) or CPU (universal):

**GPU Training (Recommended):**
```powershell
conda env create -f environment-gpu.yml
conda activate ghost-corrector-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

**CPU Training:**
```powershell
conda env create -f environment-cpu.yml
conda activate ghost-corrector-cpu
```

### 2. Train the Model

```powershell
cd ai_model/src

# Step 1: Prepare data
python 01_data_preprocessing.py

# Step 2: Train model (5-50 minutes depending on hardware)
python 02_model_training.py

# Step 3: Convert to TensorFlow.js
python 03_model_conversion.py
```

**For detailed instructions, see [docs/SETUP.md](docs/SETUP.md)**

---

## Project Status

**Complete:**
- Data preprocessing pipeline
- Seq2seq LSTM model architecture  
- GPU-accelerated training support
- TensorFlow.js conversion
- Conda environment configurations

**In Development:**
- Browser extension JavaScript
- Content script for autocorrection
- User interface and settings

---

## Project Structure

```
Ghost Type Corrector/
├── environment-gpu.yml              # Conda env for GPU training
├── environment-cpu.yml              # Conda env for CPU training
├── README.md                        # This file
├── LICENSE                          # MIT License
│
├── docs/                            # Documentation
│   ├── SETUP.md                     # Complete setup and training guide
│   ├── QUICKSTART.md                # Quick reference commands
│   ├── WORKFLOW.md                  # Visual workflow diagrams
│   └── CLEANUP_SUMMARY.md           # Project reorganization notes
│
├── ai_model/                        # AI model development
│   ├── src/                         # Source code
│   │   ├── 01_data_preprocessing.py # Clean data & generate typos
│   │   ├── 02_model_training.py     # Train seq2seq LSTM model
│   │   └── 03_model_conversion.py   # Convert to TensorFlow.js
│   │
│   ├── data/                        # Training data
│   │   ├── corpus.txt               # Raw text (you provide)
│   │   ├── train_clean.txt          # Clean sentences (generated)
│   │   ├── train_noisy.txt          # Noisy sentences (generated)
│   │   └── tokenizer_config.json    # Vocabulary (generated)
│   │
│   ├── notebooks/                   # Jupyter notebooks for research
│   │   ├── 01_data_exploration.ipynb
│   │   └── 02_model_building.ipynb
│   │
│   └── autocorrect_model.h5         # Trained model (generated)
│
└── extension/                       # Browser extension
    ├── model/                       # TensorFlow.js model (generated)
    │   ├── model.json               # Model architecture
    │   └── *.bin                    # Model weights
    ├── assets/                      # Icons and static files
    │   └── icons/
    ├── js/                          # Extension JavaScript
    │   └── lib/
    └── manifest.json                # Extension configuration
```

## Acknowledgements

- Development: This project is being developed with the assistance of Google's Gemini.
- Dataset: The AI model is trained on the British English (en_GB) corpus provided by the Leipzig Corpora Collection, Leipzig University. We are using the eng-uk_web-public_2018_1M (1 million sentences) dataset.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
