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
 * Generates the corrected word using the loaded Keras model.
 * This is the core Seq2Seq inference loop.
 * @param {string} inputWord - The word to correct (typo)
 * @returns {string|null} The predicted correct word, or null if no correction
 */
function predictCorrection(inputWord) {
    if (!model || !inputWord) return null;

    try {
        // Use tf.tidy to automatically clean up intermediate tensors
        return tf.tidy(() => {
            // 1. Prepare Encoder Input (the typo)
            const encoderInput = encodeInput(inputWord);
            
            // 2. Get initial decoder state from encoder
            const [encoderOutput, stateH, stateC] = model.predict(encoderInput);

            let decodedSequence = [startTokenIndex]; // Start with <start> token
            let stopCondition = false;
            let correctedWord = '';
            let step = 0;
            const maxSteps = maxSeqLength - 2; // Prevent infinite loops
            
            let decoderInput = tf.tensor2d([decodedSequence.map(i => i)], [1, maxSeqLength], 'float32');
            let states = [stateH, stateC];

            // 3. Decoding Loop: Generate one character at a time
            while (!stopCondition && step < maxSteps) {
                // Prediction step
                const [outputTokens, h, c] = model.predict([decoderInput, ...states]);
                
                // Get the last predicted token's probabilities
                const lastTimeStep = outputTokens.slice([0, step, 0], [1, 1, vocabSize]);
                const nextTokenIndex = lastTimeStep.argMax(-1).dataSync()[0];

                // Append to the sequence
                decodedSequence.push(nextTokenIndex);
                
                // Check for stop conditions
                if (nextTokenIndex === endTokenIndex || correctedWord.length >= maxSteps) {
                    stopCondition = true;
                }

                // Update states for next iteration
                states = [h, c];
                step++;
                
                // Prepare the decoder input for the next step
                const nextDecoderInput = new Array(maxSeqLength).fill(padTokenIndex);
                const currentDecodedLength = Math.min(decodedSequence.length, maxSeqLength);
                for (let i = 0; i < currentDecodedLength; i++) {
                    nextDecoderInput[i] = decodedSequence[i];
                }
                decoderInput = tf.tensor2d([nextDecoderInput], [1, maxSeqLength], 'float32');

                // Map index to character for the output word (excluding special tokens)
                if (nextTokenIndex !== startTokenIndex && 
                    nextTokenIndex !== endTokenIndex && 
                    nextTokenIndex !== padTokenIndex &&
                    correctedWord.length < maxSteps) {
                    const char = indexToChar[nextTokenIndex];
                    if (char) {
                        correctedWord += char;
                    }
                }
            }
            
            // Clean up the word by removing any remaining special tokens
            return correctedWord.replace(/<start>|<end>|<pad>/g, '').trim();
        });
    } catch (error) {
        console.error("Sandbox: Prediction error:", error);
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
        
        console.log("Sandbox: Loading TensorFlow.js model...");
        
        // Try to load as a graph model first (more compatible with converted Keras models)
        try {
            model = await tf.loadGraphModel(modelUrl);
            console.log("Sandbox: Model loaded as GraphModel");
        } catch (graphError) {
            console.log("Sandbox: GraphModel loading failed, trying LayersModel...");
            console.log("  Error:", graphError.message);
            
            try {
                model = await tf.loadLayersModel(modelUrl);
                console.log("Sandbox: Model loaded as LayersModel");
            } catch (layersError) {
                throw new Error(`Model loading failed: ${layersError.message}`);
            }
        }
        
        console.log("Sandbox: TensorFlow.js model loaded successfully");

        // 3. Model Warmup (optional but recommended for performance)
        console.log("Sandbox: Warming up model...");
        tf.tidy(() => {
            try {
                const warmupInput = tf.zeros([1, maxSeqLength], 'float32');
                model.predict(warmupInput);
                console.log("Sandbox: Model warmup complete");
            } catch (e) {
                console.warn("Sandbox: Warmup prediction skipped (Seq2Seq model):", e.message);
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