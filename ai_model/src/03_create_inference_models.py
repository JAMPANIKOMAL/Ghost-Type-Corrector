#!/usr/bin/env python3
"""
Create separate encoder and decoder models for inference
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import tensorflowjs as tfjs

print("=" * 70)
print("CREATING ENCODER AND DECODER MODELS FOR INFERENCE")
print("=" * 70)
print()

# Load the trained model
script_dir = Path(__file__).parent
ai_model_dir = script_dir.parent
project_root = ai_model_dir.parent

model_path = ai_model_dir / 'autocorrect_model.h5'
output_dir = project_root / 'extension' / 'model'

print(f"Loading model from: {model_path}")
model = keras.models.load_model(str(model_path))
print(f"✓ Model loaded: {model.name}")
print()

# Display model structure
model.summary()
print()

# Extract encoder
print("Creating encoder model...")
encoder_inputs = model.get_layer('encoder_input').output
encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
encoder_lstm = model.get_layer('encoder_lstm')
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

encoder_model = keras.Model(encoder_inputs, encoder_states, name='encoder')
print("✓ Encoder model created")
encoder_model.summary()
print()

# Extract decoder
print("Creating decoder model...")
decoder_inputs = keras.Input(shape=(101,), name='decoder_input')
decoder_embedding_layer = model.get_layer('decoder_embedding')
decoder_embedding = decoder_embedding_layer(decoder_inputs)

# Decoder LSTM with state inputs
decoder_state_input_h = keras.Input(shape=(512,), name='decoder_state_h')
decoder_state_input_c = keras.Input(shape=(512,), name='decoder_state_c')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm = model.get_layer('decoder_lstm')
# Call the LSTM layer correctly with initial_state parameter
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, 
    initial_state=decoder_states_inputs
)
decoder_states = [state_h, state_c]

decoder_dense = model.get_layer('output_dense')
decoder_outputs = decoder_dense(decoder_outputs)

# Create model with all inputs explicitly listed
decoder_model = keras.Model(
    inputs=[decoder_inputs, decoder_state_input_h, decoder_state_input_c],
    outputs=[decoder_outputs, state_h, state_c],
    name='decoder'
)
print("✓ Decoder model created")
decoder_model.summary()
print()

# Save both models
print("Converting encoder to TensorFlow.js...")
encoder_dir = output_dir / 'encoder'
encoder_dir.mkdir(parents=True, exist_ok=True)
tfjs.converters.save_keras_model(encoder_model, str(encoder_dir))
print(f"✓ Encoder saved to {encoder_dir}")
print()

print("Converting decoder to TensorFlow.js...")
decoder_dir = output_dir / 'decoder'
decoder_dir.mkdir(parents=True, exist_ok=True)
tfjs.converters.save_keras_model(decoder_model, str(decoder_dir))
print(f"✓ Decoder saved to {decoder_dir}")
print()

print("=" * 70)
print("✓ SUCCESS!")
print("=" * 70)
print()
print("Created two separate models:")
print(f"  - Encoder: {encoder_dir}")
print(f"  - Decoder: {decoder_dir}")
print()
print("These models can be used for iterative Seq2Seq inference.")
print()
