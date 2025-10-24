import tensorflow as tf
import numpy as np
import os
import re
import json
import time

print(f"--- Starting Model Training Script ---")
print("TensorFlow Version:", tf.__version__)

# --- Configuration ---
# Set NUM_LINES to None to use the full dataset (will take much longer!)
NUM_LINES = 100000  # Use the same number as in the notebook for consistency
# NUM_LINES = None # Uncomment this to use the full dataset for final training

# Model Hyperparameters
embedding_dim = 128
latent_dim = 256

# Training Parameters
epochs = 5  # Use the same number as in the notebook for consistency
# epochs = 20 # Increase for final training if needed
batch_size = 64
validation_split = 0.2

# --- Define File Paths ---
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # ai_model directory
data_dir = os.path.join(base_dir, 'data')
clean_file_path = os.path.join(data_dir, 'train_clean.txt')
noisy_file_path = os.path.join(data_dir, 'train_noisy.txt')
tokenizer_config_path = os.path.join(data_dir, 'tokenizer_config.json')
model_save_path = os.path.join(base_dir, 'autocorrect_model.h5') # Save in ai_model directory

print(f"Clean data file: {clean_file_path}")
print(f"Noisy data file: {noisy_file_path}")
print(f"Tokenizer config: {tokenizer_config_path}")
print(f"Model save path: {model_save_path}")

# --- Load Data ---
start_time = time.time()
print(f"\nLoading {NUM_LINES if NUM_LINES else 'all'} lines from files...")
clean_lines = []
noisy_lines = []
try:
    with open(clean_file_path, 'r', encoding='utf-8') as f_clean, \
         open(noisy_file_path, 'r', encoding='utf-8') as f_noisy:
        line_num = 0
        while True:
            clean_line = f_clean.readline().strip()
            noisy_line = f_noisy.readline().strip()
            if not clean_line or not noisy_line: break
            if len(clean_line) < 100 and len(noisy_line) < 100: # Same filter as notebook
                clean_lines.append(clean_line)
                noisy_lines.append(noisy_line)
            line_num += 1
            if NUM_LINES is not None and line_num >= NUM_LINES: break
    print(f"Loaded {len(clean_lines)} pairs of lines in {time.time() - start_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading data: {e}")
    exit() # Stop if data loading fails

# --- Load Tokenizer Config ---
print("\nLoading tokenizer configuration...")
try:
    with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    char_to_index = tokenizer_config['char_to_index']
    index_to_char = {int(k): v for k, v in tokenizer_config['index_to_char'].items()} # JSON keys are strings
    max_seq_length = tokenizer_config['max_seq_length']
    vocab_size = tokenizer_config['vocab_size']
    START_TOKEN = index_to_char[tokenizer_config['start_token_index']] # Get actual tokens back
    END_TOKEN = index_to_char[tokenizer_config['end_token_index']]
    PAD_TOKEN_INDEX = tokenizer_config['pad_token_index'] # Should be 0

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Max sequence length: {max_seq_length}")
except Exception as e:
    print(f"Error loading tokenizer config: {e}")
    print("Ensure 'tokenizer_config.json' exists in the data directory.")
    exit()

# --- Vectorize and Pad Data ---
start_time = time.time()
print("\nVectorizing and padding data...")
def vectorize_text(text_list):
    vectorized = []
    start_index = tokenizer_config['start_token_index']
    end_index = tokenizer_config['end_token_index']
    for text in text_list:
        tokens = [start_index] + [char_to_index.get(char, PAD_TOKEN_INDEX) for char in text] + [end_index] # Use get for safety
        vectorized.append(tokens)
    return vectorized

noisy_vectors = vectorize_text(noisy_lines)
clean_vectors = vectorize_text(clean_lines)

noisy_padded = tf.keras.preprocessing.sequence.pad_sequences(
    noisy_vectors, maxlen=max_seq_length, padding='post'
)
clean_padded = tf.keras.preprocessing.sequence.pad_sequences(
    clean_vectors, maxlen=max_seq_length, padding='post'
)
print(f"Vectorized and padded data in {time.time() - start_time:.2f} seconds.")
print(f"Shape of noisy_padded (Input X): {noisy_padded.shape}")
print(f"Shape of clean_padded (Input Y base): {clean_padded.shape}")

# --- Prepare Decoder Target Data ---
start_time = time.time()
print("\nPreparing decoder targets...")
decoder_target_data = clean_padded[:, 1:]
padding_column = np.zeros((decoder_target_data.shape[0], 1), dtype=np.int32)
decoder_target_data = np.concatenate([decoder_target_data, padding_column], axis=-1)
print(f"Prepared decoder targets in {time.time() - start_time:.2f} seconds.")
print(f"Shape of decoder_target_data (Target Y): {decoder_target_data.shape}")

# --- Define Model Architecture ---
# (Identical to the notebook)
print("\nDefining model architecture...")
encoder_inputs = tf.keras.layers.Input(shape=(max_seq_length,), name='encoder_input')
encoder_embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
encoder_lstm = tf.keras.layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = tf.keras.layers.Input(shape=(max_seq_length,), name='decoder_input')
decoder_embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True, name='decoder_embedding')
decoder_embedding = decoder_embedding_layer(decoder_inputs)
decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'), name='output_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq_autocorrect')
model.summary()

# --- Compile Model ---
print("\nCompiling model...")
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
print("Model compiled.")

# --- Train Model ---
print(f"\nStarting training for {epochs} epochs...")
start_time = time.time()
history = model.fit([noisy_padded, clean_padded], decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split)
end_time = time.time()
print(f"\n--- Model Training Complete ---")
print(f"Training took {end_time - start_time:.2f} seconds.")

# --- Save Model ---
print(f"\nSaving trained model to {model_save_path}...")
try:
    model.save(model_save_path)
    print("Model successfully saved.")
except Exception as e:
    print(f"Error saving model: {e}")

print(f"\n--- Training Script Finished ---")
