# Documentation Index

Professional documentation for Ghost Type Corrector project.

## Quick Navigation

### Getting Started
1. [INSTALLATION.md](INSTALLATION.md) - Complete setup from scratch
2. [QUICKSTART.md](QUICKSTART.md) - Fast commands for experienced users

### Configuration and Troubleshooting
3. [CONFIGURATION.md](CONFIGURATION.md) - Training parameters and performance tuning
4. [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions

### Technical Reference
5. [API.md](API.md) - Script functions, data formats, and code reference

## Document Overview

### INSTALLATION.md
Complete installation and setup guide covering:
- Prerequisites and system requirements
- Environment creation (GPU and CPU)
- Data preparation and corpus setup
- Training pipeline execution
- Output verification
- Performance benchmarks
- Common issues during setup

**Target Audience:** New users, first-time setup  
**Length:** Comprehensive (500+ lines)

### QUICKSTART.md
Condensed reference for rapid setup:
- Copy-paste command sequences
- Minimal explanation
- Common troubleshooting
- Expected outputs
- Training time estimates

**Target Audience:** Experienced users, repeat deployments  
**Length:** Concise (100 lines)

### CONFIGURATION.md
Parameter tuning and optimization:
- Data preprocessing settings
- Model architecture parameters
- Hardware-specific configurations
- Memory usage calculations
- Performance tuning strategies
- Advanced customization options

**Target Audience:** Users optimizing performance  
**Length:** Technical (400+ lines)

### TROUBLESHOOTING.md
Solutions for common problems:
- Environment issues (GPU detection, CUDA, conda)
- Training issues (OOM, accuracy, NaN loss)
- Conversion issues (TensorFlowJS, NumPy compatibility)
- Data issues (encoding, missing files)
- Performance issues (slow training, conversion)
- Validation issues (model loading)

**Target Audience:** Users encountering errors  
**Length:** Problem-solution format (500+ lines)

### API.md
Technical code reference:
- Script purposes and workflows
- Function signatures and examples
- Configuration constants
- Model architecture details
- Data format specifications
- Error handling patterns
- Performance metrics
- Dependencies and versions

**Target Audience:** Developers, code contributors  
**Length:** Technical reference (600+ lines)

## Recommended Reading Path

### First-Time Users
1. README.md (project overview)
2. INSTALLATION.md (complete setup)
3. TROUBLESHOOTING.md (if issues occur)

### Quick Setup
1. README.md (verify prerequisites)
2. QUICKSTART.md (run commands)

### Performance Optimization
1. CONFIGURATION.md (tune parameters)
2. TROUBLESHOOTING.md (resolve bottlenecks)

### Development
1. API.md (understand code structure)
2. CONFIGURATION.md (parameter effects)
3. Source files in `ai_model/src/`

## Project Structure Reference

```
Ghost-Type-Corrector/
├── README.md                        # Project overview
├── LICENSE                          # MIT License
│
├── docs/                            # This folder
│   ├── README.md                    # This file (index)
│   ├── INSTALLATION.md              # Complete setup guide
│   ├── QUICKSTART.md                # Fast reference
│   ├── CONFIGURATION.md             # Parameter tuning
│   ├── TROUBLESHOOTING.md           # Common issues
│   └── API.md                       # Technical reference
│
├── ai_model/
│   ├── src/
│   │   ├── 01_data_preprocessing.py # Clean corpus, generate typos
│   │   ├── 02_model_training.py     # Train seq2seq LSTM
│   │   └── convert_direct.py        # Convert to TensorFlow.js
│   ├── data/
│   │   ├── corpus.txt               # User provides
│   │   ├── train_clean.txt          # Generated
│   │   ├── train_noisy.txt          # Generated
│   │   └── tokenizer_config.json    # Generated
│   ├── notebooks/                   # Jupyter research
│   └── autocorrect_model.h5         # Generated Keras model
│
├── extension/
│   ├── model/
│   │   ├── model.json               # Generated TensorFlow.js
│   │   └── *.bin                    # Generated weights
│   ├── assets/
│   ├── js/
│   └── manifest.json
│
├── environment-gpu.yml              # GPU training setup
└── environment-cpu.yml              # CPU training setup
```

## Code Files Reference

### Source Scripts

| File | Lines | Purpose |
|------|-------|---------|
| `01_data_preprocessing.py` | ~150 | Clean corpus and generate synthetic typos |
| `02_model_training.py` | ~200 | Train encoder-decoder LSTM model |
| `convert_direct.py` | ~120 | Convert Keras to TensorFlow.js with patches |

### Configuration Files

| File | Purpose |
|------|---------|
| `environment-gpu.yml` | Conda environment for GPU training |
| `environment-cpu.yml` | Conda environment for CPU training |
| `.gitignore` | Version control exclusions |

## External Resources

### Dataset
- **Source:** Leipzig Corpora Collection
- **URL:** https://wortschatz.uni-leipzig.de/en/download
- **File:** eng-uk_web-public_2018_1M
- **Size:** 1 million sentences, British English

### Dependencies
- **TensorFlow:** https://www.tensorflow.org/install
- **TensorFlow.js:** https://www.tensorflow.org/js
- **Conda:** https://docs.conda.io/en/latest/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-toolkit
- **cuDNN:** https://developer.nvidia.com/cudnn

## Version Information

- **Python:** 3.10.13
- **TensorFlow:** 2.10.1
- **TensorFlow.js:** 3.18.0
- **NumPy:** 1.24.3
- **CUDA:** 11.2
- **cuDNN:** 8.1.0

## Support

For issues or questions:
1. Check relevant documentation file
2. Review TROUBLESHOOTING.md
3. Verify all prerequisites installed
4. Open GitHub issue with error details

## Contributing

When contributing documentation:
- Follow professional tone (no emojis)
- Use clear section headers
- Include code examples
- Test all commands
- Update this index

---

**Last Updated:** October 25, 2025  
**Documentation Version:** 1.0
