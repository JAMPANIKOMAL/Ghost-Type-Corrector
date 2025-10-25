// extension/js/sandbox_logic.js
// AI model inference logic running in sandboxed environment

'use strict';

// --- Global AI Variables ---
let model = null;
let tokenizerConfig = null;
let charToIndex = null;
let indexToChar = null;
let maxSeqLength = null;
let startTokenIndex = null;
let endTokenIndex = null;
let padTokenIndex = null;
let vocabSize = null;

// --- Helper Functions ---

/**
 * Converts a raw string (typo) into a padded TensorFlow input tensor.
 * @param {string} text - The input text to encode
 * @returns {tf.Tensor} Padded input sequence tensor
 */
function encodeInput(text) {
    const indices = text.toLowerCase().split('').map(char => charToIndex[char] || padTokenIndex);
    indices.unshift(startTokenIndex); // Add start token

    // Pad or truncate to maxSeqLength
    const paddedIndices = new Array(maxSeqLength).fill(padTokenIndex);
    const length = Math.min(indices.length, maxSeqLength);
    for (let i = 0; i < length; i++) {
        paddedIndices[i] = indices[i];
    }

    // Return as a float32 tensor with shape [1, maxSeqLength]
    return tf.tensor2d([paddedIndices], [1, maxSeqLength], 'float32');
}

/**
 * Generates the corrected word using the loaded TensorFlow.js Graph Model.
 * This is the core Seq2Seq inference function.
 * @param {string} inputWord - The word to correct (typo)
 * @returns {string|null} The predicted correct word, or null if no correction
 */
function predictCorrection(inputWord) {
    if (!model || !inputWord) return null;

    try {
        // Use tf.tidy to automatically clean up intermediate tensors
        return tf.tidy(() => {
            // 1. Prepare both inputs for the model
            const encoderInput = encodeInput(inputWord);
            
            // 2. Prepare decoder input (starts with <start> token, padded to maxSeqLength)
            const decoderSequence = new Array(maxSeqLength).fill(padTokenIndex);
            decoderSequence[0] = startTokenIndex;
            const decoderInput = tf.tensor2d([decoderSequence], [1, maxSeqLength], 'float32');
            
            // 3. Run the model (Graph model uses execute, not predict)
            // The model expects two inputs: encoder_input and decoder_input
            const prediction = model.execute({
                'encoder_input': encoderInput,
                'decoder_input': decoderInput
            });
            
            // 4. Extract the output sequence
            // The output should be shape [1, maxSeqLength, vocabSize]
            const outputArray = prediction.arraySync();
            const outputSequence = outputArray[0]; // Get first batch
            
            // 5. Decode the output sequence to characters
            let correctedWord = '';
            for (let i = 0; i < outputSequence.length; i++) {
                // Get the index with highest probability at each time step
                const tokenProbs = outputSequence[i];
                const tokenIndex = tokenProbs.indexOf(Math.max(...tokenProbs));
                
                // Stop if we hit the end token
                if (tokenIndex === endTokenIndex) {
                    break;
                }
                
                // Skip special tokens (start, pad)
                if (tokenIndex !== startTokenIndex && tokenIndex !== padTokenIndex) {
                    const char = indexToChar[tokenIndex];
                    if (char) {
                        correctedWord += char;
                    }
                }
            }
            
            // Clean up and return
            return correctedWord.trim();
        });
    } catch (error) {
        console.error("Sandbox: Prediction error:", error);
        console.error("  Error details:", error.stack);
        return null;
    }
}

// --- Initialization ---

async function initialize(tokenizerUrl, modelUrl) {
    console.log("Sandbox: Initializing AI resources...");
    console.log("Sandbox: Tokenizer URL:", tokenizerUrl);
    console.log("Sandbox: Model URL:", modelUrl);
    
    try {
        // 1. Load Tokenizer Configuration
        console.log("Sandbox: Fetching tokenizer config...");
        
        const tokenizerResponse = await fetch(tokenizerUrl);
        if (!tokenizerResponse.ok) {
            throw new Error(`Tokenizer fetch failed: ${tokenizerResponse.status} ${tokenizerResponse.statusText}`);
        }
        
        tokenizerConfig = await tokenizerResponse.json();
        
        // Populate global variables from tokenizer config
        charToIndex = tokenizerConfig.char_to_index;
        indexToChar = Object.fromEntries(
            Object.entries(tokenizerConfig.index_to_char).map(([k, v]) => [parseInt(k), v])
        );
        maxSeqLength = tokenizerConfig.max_seq_length;
        startTokenIndex = tokenizerConfig.start_token_index;
        endTokenIndex = tokenizerConfig.end_token_index;
        padTokenIndex = tokenizerConfig.pad_token_index;
        vocabSize = tokenizerConfig.vocab_size;
        
        console.log("Sandbox: Tokenizer config loaded successfully");
        console.log(`  Max sequence length: ${maxSeqLength}`);
        console.log(`  Vocabulary size: ${vocabSize}`);

        // 2. Load TensorFlow.js Model
        if (typeof tf === 'undefined') {
            throw new Error("TensorFlow.js library (tf) not found in sandbox. Check that tf.min.js is loaded.");
        }
        
        console.log("Sandbox: Loading TensorFlow.js model as GraphModel...");
        
        // Load as graph model (converted from SavedModel)
        model = await tf.loadGraphModel(modelUrl);
        console.log("Sandbox: Model loaded successfully");
        
        // Log model signature for debugging
        console.log("Sandbox: Model signature:", model.signature);

        // 3. Model Warmup (optional but recommended for performance)
        console.log("Sandbox: Warming up model...");
        tf.tidy(() => {
            try {
                const warmupEncoderInput = tf.zeros([1, maxSeqLength], 'float32');
                const warmupDecoderInput = tf.zeros([1, maxSeqLength], 'float32');
                model.execute({
                    'encoder_input': warmupEncoderInput,
                    'decoder_input': warmupDecoderInput
                });
                console.log("Sandbox: Model warmup complete");
            } catch (e) {
                console.warn("Sandbox: Warmup prediction failed:", e.message);
            }
        });

        // 4. Signal readiness to the content script
        window.parent.postMessage({ type: 'GTC_READY' }, '*');
        console.log("Sandbox: Ready signal sent to content script");

    } catch (error) {
        console.error("Sandbox: Initialization failed:", error);
        window.parent.postMessage({ 
            type: 'GTC_ERROR', 
            message: `AI initialization failed: ${error.message}` 
        }, '*');
    }
}

// --- Communication Listener ---

/**
 * Listen for messages from the content script (parent window)
 * to perform AI prediction on user input
 */
window.addEventListener('message', async (event) => {
    // Security: Only process messages from the parent frame with correct type
    if (event.source !== window.parent || !event.data) {
        return;
    }

    if (event.data.type === 'GTC_INIT') {
        // Receive initialization URLs from content script
        const { tokenizerUrl, modelUrl } = event.data;
        console.log("Sandbox: Received init message from content script");
        initialize(tokenizerUrl, modelUrl);
    } else if (event.data.type === 'GTC_PREDICT') {
        const originalWord = event.data.word;
        
        if (!originalWord || typeof originalWord !== 'string') {
            console.warn("Sandbox: Invalid word received for prediction");
            return;
        }
        
        // Perform the AI correction
        const correctedWord = predictCorrection(originalWord);

        // Send the result back to the content script
        window.parent.postMessage({ 
            type: 'GTC_RESULT', 
            originalWord: originalWord, 
            correctedWord: correctedWord // Will be null if no correction found
        }, '*');
    }
});

// Signal to parent that sandbox is loaded and ready to receive init message
console.log("Sandbox: Script loaded, waiting for initialization message...");
window.parent.postMessage({ type: 'GTC_SANDBOX_LOADED' }, '*');