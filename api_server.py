#!/usr/bin/env python3
"""
Ghost Type Corrector - Local API Server
Provides autocorrection via HTTP endpoint for the Chrome extension
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Allow requests from Chrome extension

# Load model and tokenizer on startup
print("Loading AI model...")
model_path = Path(__file__).parent / 'ai_model' / 'autocorrect_model.h5'
tokenizer_path = Path(__file__).parent / 'extension' / 'data' / 'tokenizer_config.json'

model = keras.models.load_model(str(model_path))
print(f"✓ Model loaded: {model.name}")

with open(tokenizer_path, 'r') as f:
    tokenizer_config = json.load(f)

char_to_index = tokenizer_config['char_to_index']
index_to_char = {int(k): v for k, v in tokenizer_config['index_to_char'].items()}
max_seq_length = tokenizer_config['max_seq_length']
start_token_index = tokenizer_config['start_token_index']
end_token_index = tokenizer_config['end_token_index']
pad_token_index = tokenizer_config['pad_token_index']

print("✓ Tokenizer loaded")
print(f"Server ready! Listening on http://localhost:5000")
print()


def encode_sequence(text):
    """Encode text to padded sequence"""
    indices = [char_to_index.get(char.lower(), pad_token_index) for char in text]
    indices.insert(0, start_token_index)
    
    # Pad or truncate
    padded = [pad_token_index] * max_seq_length
    for i in range(min(len(indices), max_seq_length)):
        padded[i] = indices[i]
    
    return np.array([padded], dtype=np.float32)


def decode_sequence(input_word):
    """Decode using Seq2Seq inference"""
    # Encode input
    encoder_input = encode_sequence(input_word)
    
    # Start decoder with start token
    decoder_sequence = [start_token_index]
    decoder_input = np.zeros((1, max_seq_length), dtype=np.float32)
    decoder_input[0, 0] = start_token_index
    
    corrected_word = ''
    max_iterations = max_seq_length - 2
    
    # Iterative decoding
    for _ in range(max_iterations):
        # Predict
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        
        # Get next token (at current position)
        predicted_index = np.argmax(predictions[0, len(decoder_sequence) - 1, :])
        
        # Stop if end token
        if predicted_index == end_token_index:
            break
        
        # Add to sequence
        decoder_sequence.append(predicted_index)
        
        # Update decoder input
        if len(decoder_sequence) < max_seq_length:
            decoder_input[0, len(decoder_sequence) - 1] = predicted_index
        
        # Add character to output (skip special tokens)
        if (predicted_index != start_token_index and 
            predicted_index != pad_token_index and 
            predicted_index != end_token_index):
            char = index_to_char.get(predicted_index, '')
            if char and char not in ['\t', '\n']:
                corrected_word += char
    
    return corrected_word.strip()


@app.route('/correct', methods=['POST'])
def correct_word():
    """API endpoint for word correction"""
    try:
        data = request.get_json()
        word = data.get('word', '')
        
        if not word:
            return jsonify({'error': 'No word provided'}), 400
        
        # Perform correction
        corrected = decode_sequence(word)
        
        return jsonify({
            'original': word,
            'corrected': corrected,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'model': model.name})


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=False)
