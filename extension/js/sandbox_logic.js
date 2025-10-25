// extension/js/sandbox_logic.js
// AI model inference logic running in sandboxed environment

'use strict';

// --- Global AI Variables ---
let encoderModel = null;
let decoderModel = null;
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
 * Generates the corrected word using separate encoder and decoder models.
 * This is the proper Seq2Seq inference with iterative decoding.
 * @param {string} inputWord - The word to correct (typo)
 * @returns {Promise<string|null>} The predicted correct word, or null if no correction
 */
async function predictCorrection(inputWord) {
    if (!encoderModel || !decoderModel || !inputWord) return null;

    try {
        // 1. Encode the input word
        const encoderInput = encodeInput(inputWord);
        
        console.log(`Sandbox: Predicting correction for "${inputWord}"`);
        
        // 2. Get encoder states
        const encoderStates = encoderModel.predict(encoderInput);
        let stateH = encoderStates[0];
        let stateC = encoderStates[1];
        
        // 3. Start with the start token
        let decoderSequence = [startTokenIndex];
        let correctedWord = '';
        let maxIterations = maxSeqLength - 2;
        
        // 4. Iterative decoding
        for (let i = 0; i < maxIterations; i++) {
            // Prepare decoder input (current sequence)
            const decoderInputArray = new Array(maxSeqLength).fill(padTokenIndex);
            for (let j = 0; j < Math.min(decoderSequence.length, maxSeqLength); j++) {
                decoderInputArray[j] = decoderSequence[j];
            }
            const decoderInput = tf.tensor2d([decoderInputArray], [1, maxSeqLength], 'float32');
            
            // Predict next token - use object notation with input names
            const inputDict = {
                'decoder_input': decoderInput,
                'decoder_state_h': stateH,
                'decoder_state_c': stateC
            };
            
            const decoderOutputs = decoderModel.execute(inputDict);
            const outputTokens = decoderOutputs[0];  // [1, maxSeqLength, vocabSize]
            const newStateH = decoderOutputs[1];
            const newStateC = decoderOutputs[2];
            
            // Get the prediction for the last generated token
            const lastTokenProbs = outputTokens.slice([0, decoderSequence.length - 1, 0], [1, 1, vocabSize]).dataSync();
            const tokenIndex = Array.from(lastTokenProbs).indexOf(Math.max(...lastTokenProbs));
            
            console.log(`  Step ${i}: token=${tokenIndex}, char="${indexToChar[tokenIndex]}"`);
            
            // Check for end token
            if (tokenIndex === endTokenIndex) {
                console.log("  Found end token, stopping");
                break;
            }
            
            // Add token to sequence
            decoderSequence.push(tokenIndex);
            
            // Add character to output (skip special tokens)
            if (tokenIndex !== startTokenIndex && 
                tokenIndex !== padTokenIndex && 
                tokenIndex !== endTokenIndex) {
                const char = indexToChar[tokenIndex];
                if (char && char !== '\t' && char !== '\n') {
                    correctedWord += char;
                }
            }
            
            // Update states for next iteration
            stateH.dispose();
            stateC.dispose();
            stateH = newStateH;
            stateC = newStateC;
            
            // Cleanup
            decoderInput.dispose();
            outputTokens.dispose();
        }
        
        // Cleanup
        encoderInput.dispose();
        stateH.dispose();
        stateC.dispose();
        
        console.log(`Sandbox: Correction result: "${inputWord}" -> "${correctedWord}"`);
        
        return correctedWord.trim();
    } catch (error) {
        console.error("Sandbox: Prediction error:", error);
        console.error("  Error details:", error.stack);
        return null;
    }
}

// --- Initialization ---

async function initialize(tokenizerUrl, encoderUrl, decoderUrl) {
    console.log("Sandbox: Initializing AI resources...");
    console.log("Sandbox: Tokenizer URL:", tokenizerUrl);
    console.log("Sandbox: Encoder URL:", encoderUrl);
    console.log("Sandbox: Decoder URL:", decoderUrl);
    
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

        // 2. Load TensorFlow.js Models
        if (typeof tf === 'undefined') {
            throw new Error("TensorFlow.js library (tf) not found in sandbox. Check that tf.min.js is loaded.");
        }
        
        console.log("Sandbox: Loading encoder model...");
        encoderModel = await tf.loadLayersModel(encoderUrl);
        console.log("Sandbox: ✓ Encoder model loaded");
        
        console.log("Sandbox: Loading decoder model...");
        decoderModel = await tf.loadLayersModel(decoderUrl);
        console.log("Sandbox: ✓ Decoder model loaded");

        // 3. Model Warmup
        console.log("Sandbox: Warming up models...");
        try {
            const warmupInput = tf.zeros([1, maxSeqLength], 'float32');
            const warmupStates = encoderModel.predict(warmupInput);
            const warmupStateH = warmupStates[0];
            const warmupStateC = warmupStates[1];
            
            decoderModel.execute({
                'decoder_input': warmupInput,
                'decoder_state_h': warmupStateH,
                'decoder_state_c': warmupStateC
            });
            
            warmupInput.dispose();
            warmupStateH.dispose();
            warmupStateC.dispose();
            console.log("Sandbox: Model warmup complete");
        } catch (e) {
            console.warn("Sandbox: Warmup failed:", e.message);
        }

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
        const { tokenizerUrl, encoderUrl, decoderUrl } = event.data;
        console.log("Sandbox: Received init message from content script");
        initialize(tokenizerUrl, encoderUrl, decoderUrl);
    } else if (event.data.type === 'GTC_PREDICT') {
        const originalWord = event.data.word;
        
        if (!originalWord || typeof originalWord !== 'string') {
            console.warn("Sandbox: Invalid word received for prediction");
            return;
        }
        
        // Perform the AI correction (now async)
        const correctedWord = await predictCorrection(originalWord);

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