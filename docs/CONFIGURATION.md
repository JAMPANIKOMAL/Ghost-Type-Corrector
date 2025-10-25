# Configuration Guide

Customize training parameters and model architecture for optimal performance.

## Data Preprocessing Configuration

File: `ai_model/src/01_data_preprocessing.py`

### Key Parameters

```python
NOISE_LEVEL = 0.15              # Probability of typo per character (0.0-1.0)
MIN_SENTENCE_LENGTH = 3         # Minimum words per sentence
RANDOM_SEED = 42                # For reproducible results
```

### Typo Types

The preprocessing script generates four types of realistic typos:

1. **Character Deletion** (30%): "hello" → "helo"
2. **Character Insertion** (30%): "hello" → "helloo"
3. **Character Substitution** (30%): "hello" → "helli"
4. **Character Swap** (10%): "hello" → "hlelo"

### Recommendations

| Corpus Size | NOISE_LEVEL | Result |
|-------------|-------------|--------|
| 10K-50K     | 0.10        | Conservative, fewer typos |
| 50K-200K    | 0.15        | Balanced (default) |
| 200K+       | 0.20        | Aggressive, more variations |

## Model Training Configuration

File: `ai_model/src/02_model_training.py`

### Architecture Parameters

```python
NUM_SAMPLES = 100000            # Training samples (None = use all)
MAX_SENTENCE_LENGTH = 100       # Maximum characters per sentence
EMBEDDING_DIM = 128             # Character embedding dimension
LATENT_DIM = 256                # LSTM hidden state dimension
```

### Training Parameters

```python
EPOCHS = 10                     # Training iterations
BATCH_SIZE = 64                 # Samples per batch
VALIDATION_SPLIT = 0.2          # Validation data percentage
```

### Hardware-Specific Settings

| GPU Model      | BATCH_SIZE | LATENT_DIM | EMBEDDING_DIM | Notes |
|----------------|------------|------------|---------------|-------|
| RTX 4090       | 256        | 512        | 256           | Maximum performance |
| RTX 3080       | 256        | 512        | 128           | High performance |
| RTX 3060/3050  | 128        | 256        | 128           | Balanced (default) |
| GTX 1660       | 64         | 256        | 128           | Memory constrained |
| GTX 1060       | 64         | 128        | 64            | Low memory |
| CPU (i7/i9)    | 32         | 128        | 64            | Limited performance |

### Memory Usage Estimation

```
Memory (GB) = (BATCH_SIZE × MAX_SENTENCE_LENGTH × LATENT_DIM × 4) / 1e9
```

Examples:
- `BATCH_SIZE=64, LATENT_DIM=256`: ~2.5 GB
- `BATCH_SIZE=128, LATENT_DIM=512`: ~10 GB
- `BATCH_SIZE=256, LATENT_DIM=512`: ~20 GB

## Performance Tuning

### For Faster Training

```python
NUM_SAMPLES = 50000             # Reduce dataset size
BATCH_SIZE = 128                # Increase batch size (if GPU allows)
EPOCHS = 5                      # Reduce training iterations
```

### For Better Accuracy

```python
NUM_SAMPLES = None              # Use entire dataset
EPOCHS = 20                     # More training iterations
LATENT_DIM = 512                # Larger model capacity
VALIDATION_SPLIT = 0.1          # More training data
```

### For Lower Memory Usage

```python
BATCH_SIZE = 32                 # Smaller batches
LATENT_DIM = 128                # Smaller hidden states
MAX_SENTENCE_LENGTH = 50        # Shorter sequences
```

## Model Conversion Configuration

File: `ai_model/src/convert_direct.py`

### Output Settings

```python
output_path = script_dir.parent.parent / 'extension' / 'model'
```

The converter automatically:
- Creates weight shards for efficient loading
- Generates model.json with architecture
- Optimizes for browser deployment

## Expected Accuracy

With default configuration (100K samples, 10 epochs):

| Metric              | Target Range |
|---------------------|--------------|
| Training Accuracy   | 60-70%       |
| Validation Accuracy | 55-65%       |
| Training Loss       | 0.7-0.8      |
| Validation Loss     | 0.8-0.9      |

Higher accuracy requires:
- More training data (500K+ samples)
- Longer training (20+ epochs)
- Larger model (LATENT_DIM=512)

## Advanced Customization

### Custom Vocabulary

To use a domain-specific vocabulary, modify the tokenizer in `02_model_training.py`:

```python
# Add custom characters to vocabulary
tokenizer = tf.keras.preprocessing.text.Tokenizer(
    char_level=True,
    filters='',  # Keep all characters
    lower=False   # Preserve case
)
```

### Custom Typo Patterns

Add your own typo generation logic in `01_data_preprocessing.py`:

```python
def add_custom_noise(word):
    # Your custom typo logic
    return modified_word
```

### Model Architecture Changes

Modify the encoder-decoder in `02_model_training.py`:

```python
# Add more LSTM layers
encoder_lstm = LSTM(LATENT_DIM, return_state=True, dropout=0.2, recurrent_dropout=0.2)

# Add attention mechanism (advanced)
# Requires additional implementation
```

## Monitoring Training

Training outputs real-time metrics:

```
Epoch 1/10
1563/1563 [==============================] - 40s 25ms/step
  loss: 0.8234
  accuracy: 0.6102
  val_loss: 0.8876
  val_accuracy: 0.5892
```

Good training indicators:
- Loss decreasing steadily
- Accuracy increasing steadily
- Validation metrics following training metrics (not diverging)

Warning signs:
- Validation loss increasing (overfitting)
- Accuracy not improving (underfitting)
- Loss becoming NaN (training instability)

## Debugging Options

Enable verbose output in training:

```python
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    verbose=2  # More detailed output
)
```

---

**Last Updated:** October 25, 2025
