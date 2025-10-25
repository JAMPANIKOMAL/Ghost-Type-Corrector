# API Reference

Technical reference for Ghost Type Corrector scripts and modules.

## Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_data_preprocessing.py` | Clean corpus and generate typos | `corpus.txt` | `train_clean.txt`, `train_noisy.txt` |
| `02_model_training.py` | Train seq2seq LSTM model | Training text files | `autocorrect_model.h5`, `tokenizer_config.json` |
| `convert_direct.py` | Convert to TensorFlow.js (recommended) | Keras model | `model.json`, weight files |
| `03_model_conversion.py` | Alternative conversion method | Keras model | `model.json`, weight files |

**Note:** Use `convert_direct.py` for conversion. It includes NumPy compatibility patches and is more reliable.

## 01_data_preprocessing.py

### Functions

#### `clean_line(line: str) -> str`

Cleans raw text line by removing extra whitespace and normalizing.

**Parameters:**
- `line` (str): Raw text line from corpus

**Returns:**
- `str`: Cleaned text

**Example:**
```python
clean_line("  Hello   world  ")  # Returns: "hello world"
```

#### `add_noise_to_sentence(sentence: str, noise_level: float, seed: int) -> str`

Generates realistic typos in a sentence.

**Parameters:**
- `sentence` (str): Clean input sentence
- `noise_level` (float): Probability of typo per character (0.0-1.0)
- `seed` (int): Random seed for reproducibility

**Returns:**
- `str`: Sentence with synthetic typos

**Typo Types:**
- Delete character (30%): "hello" → "helo"
- Insert character (30%): "hello" → "helloo"
- Substitute character (30%): "hello" → "helli"
- Swap characters (10%): "hello" → "hlelo"

**Example:**
```python
add_noise_to_sentence("hello world", 0.15, 42)  
# Returns: "helo worold" (example)
```

### Configuration Constants

```python
NOISE_LEVEL = 0.15              # Typo probability
MIN_SENTENCE_LENGTH = 3         # Minimum words
RANDOM_SEED = 42                # Reproducibility seed
```

### File Paths

```python
corpus_file = script_dir.parent / 'data' / 'corpus.txt'
clean_output = script_dir.parent / 'data' / 'train_clean.txt'
noisy_output = script_dir.parent / 'data' / 'train_noisy.txt'
```

## 02_model_training.py

### Model Architecture

#### Encoder

```python
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(num_encoder_tokens, EMBEDDING_DIM)
encoder_lstm = LSTM(LATENT_DIM, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding(encoder_inputs))
encoder_states = [state_h, state_c]
```

**Components:**
- Input: Variable-length character sequences
- Embedding: 128-dimensional character vectors
- LSTM: 256-dimensional hidden states
- Output: Context vectors (state_h, state_c)

#### Decoder

```python
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(num_decoder_tokens, EMBEDDING_DIM)
decoder_lstm = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
```

**Components:**
- Input: Target sequence with start token
- Embedding: Shared 128-dimensional space
- LSTM: 256-dimensional, sequences output
- Dense: Softmax over vocabulary

### Configuration Constants

```python
NUM_SAMPLES = 100000            # Training samples (None = all)
MAX_SENTENCE_LENGTH = 100       # Max characters
EMBEDDING_DIM = 128             # Embedding dimension
LATENT_DIM = 256                # LSTM hidden units
EPOCHS = 10                     # Training iterations
BATCH_SIZE = 64                 # Batch size
VALIDATION_SPLIT = 0.2          # Validation fraction
```

### GPU Configuration

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

Enables dynamic GPU memory allocation to prevent OOM errors.

### Data Preparation

#### Tokenization

```python
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(all_texts)
```

Creates character-level vocabulary from combined clean and noisy text.

#### Sequence Encoding

```python
encoder_input_data = np.zeros(
    (num_samples, max_encoder_seq_length),
    dtype='float32'
)
decoder_input_data = np.zeros(
    (num_samples, max_decoder_seq_length),
    dtype='float32'
)
decoder_target_data = np.zeros(
    (num_samples, max_decoder_seq_length, num_decoder_tokens),
    dtype='float32'
)
```

**Shapes:**
- `encoder_input_data`: (samples, max_length)
- `decoder_input_data`: (samples, max_length)
- `decoder_target_data`: (samples, max_length, vocab_size)

### Training

```python
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)
```

**Inputs:**
- Noisy text (encoder)
- Clean text with start token (decoder input)

**Targets:**
- Clean text with end token (decoder target)

### Output Files

```python
model.save(model_path)  # autocorrect_model.h5
with open(tokenizer_path, 'w') as f:
    json.dump(tokenizer_config, f)  # tokenizer_config.json
```

## convert_direct.py

### NumPy Compatibility Patches

```python
import numpy
numpy.object = object
numpy.bool = bool
```

Patches deprecated NumPy attributes for TensorFlowJS 3.18.0 compatibility.

### Conversion Process

#### 1. Load Model

```python
model = tf.keras.models.load_model(model_path)
```

#### 2. Extract Weights Manually

```python
weights_manifest = []
for i, layer in enumerate(model.layers):
    layer_weights = layer.get_weights()
    # Save each weight array to binary file
```

**Benefits:**
- More reliable than SavedModel
- Avoids permission issues
- Direct weight access

#### 3. Create model.json

```python
model_json = {
    'modelTopology': json.loads(model.to_json()),
    'weightsManifest': weights_manifest,
    'format': 'layers-model',
    'generatedBy': 'keras v' + tf.keras.__version__
}
```

### Output Structure

```
extension/model/
├── model.json                      # Architecture + metadata
├── group1-shard1of10.bin          # Weight file 1
├── group1-shard2of10.bin          # Weight file 2
├── ...
└── group1-shard10of10.bin         # Weight file 10
```

### Weight Sharding

Weights are automatically split into multiple files for efficient loading:

```python
BYTES_PER_SHARD = 4 * 1024 * 1024  # 4 MB per file
```

## Model Inference (JavaScript)

### Loading Model

```javascript
const model = await tf.loadLayersModel('model/model.json');
```

### Character-Level Prediction

```javascript
// Tokenize input
const inputSeq = tokenizeText(noisyText);

// Run encoder
const encoderOutput = encoderModel.predict(inputSeq);

// Run decoder (autoregressive)
let decodedText = '';
let currentToken = START_TOKEN;

for (let i = 0; i < maxLength; i++) {
    const decoderInput = tf.tensor2d([[currentToken]]);
    const prediction = decoderModel.predict([decoderInput, encoderOutput]);
    
    currentToken = prediction.argMax(-1).dataSync()[0];
    if (currentToken === END_TOKEN) break;
    
    decodedText += indexToChar[currentToken];
}
```

## Data Formats

### corpus.txt

```
Plain text file
One sentence per line
UTF-8 encoding
No special formatting required
```

### train_clean.txt

```
Cleaned sentences
Lowercase
Normalized whitespace
One sentence per line
```

### train_noisy.txt

```
Sentences with synthetic typos
Parallel to train_clean.txt
Same number of lines
Character-level corruption
```

### tokenizer_config.json

```json
{
    "word_index": {
        "a": 1,
        "b": 2,
        "c": 3,
        ...
    },
    "index_word": {
        "1": "a",
        "2": "b",
        "3": "c",
        ...
    }
}
```

### autocorrect_model.h5

Binary Keras model file containing:
- Model architecture
- Layer weights
- Optimizer state
- Training configuration

### model.json

```json
{
    "modelTopology": {
        "class_name": "Functional",
        "config": { ... },
        "keras_version": "2.10.0"
    },
    "weightsManifest": [
        {
            "paths": ["group1-shard1of10.bin"],
            "weights": [ ... ]
        }
    ],
    "format": "layers-model",
    "generatedBy": "keras v2.10.0"
}
```

## Error Handling

### Preprocessing

```python
try:
    clean_text = clean_line(line)
    if len(clean_text.split()) >= MIN_SENTENCE_LENGTH:
        # Process
except Exception as e:
    print(f"Error processing line: {e}")
    continue
```

### Training

```python
try:
    model.fit(...)
except tf.errors.ResourceExhaustedError:
    print("Out of memory. Reduce BATCH_SIZE or LATENT_DIM")
```

### Conversion

```python
try:
    model = tf.keras.models.load_model(model_path)
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)
```

## Performance Metrics

### Memory Usage

```python
# Approximate GPU memory (GB)
memory_gb = (BATCH_SIZE * MAX_SENTENCE_LENGTH * LATENT_DIM * 4) / 1e9
```

### Training Speed

```python
# Samples per second (GPU)
samples_per_sec = BATCH_SIZE / time_per_batch
```

### Model Size

```python
# Parameter count
total_params = sum([np.prod(w.shape) for layer in model.layers 
                    for w in layer.get_weights()])
```

## Dependencies

### Required Packages

```python
import tensorflow as tf           # 2.10.1
import numpy as np               # 1.24.3
import json                      # stdlib
from pathlib import Path         # stdlib
from tqdm import tqdm           # progress bars
import random                    # stdlib
import string                    # stdlib
```

### Version Requirements

- Python: 3.10.x
- TensorFlow: 2.10.1
- NumPy: 1.24.3
- CUDA: 11.2 (GPU only)
- cuDNN: 8.1.0 (GPU only)
- TensorFlowJS: 3.18.0

---

**Last Updated:** October 25, 2025
