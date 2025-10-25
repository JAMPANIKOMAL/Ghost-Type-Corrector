// extension/js/sandbox_logic.js

// --- Global AI Variables (Moved from content.js) ---
let model = null;
let tokenizerConfig = null;
let charToIndex = null;
let indexToChar = null;
let maxSeqLength = null;
let startTokenIndex = null;
let endTokenIndex = null;
let padTokenIndex = null;
let vocabSize = null;

// --- Helper Functions (From content.js, adapted for Sandbox environment) ---

/**
 * Converts a raw string (typo) into a padded TensorFlow input tensor.
 * @param {string} text 
 * @returns {tf.Tensor} Padded input sequence tensor.
 */
function encodeInput(text) {
    const indices = text.toLowerCase().split('').map(char => charToIndex[char] || padTokenIndex);
    indices.unshift(startTokenIndex); // Add start token

    // Pad or truncate
    const paddedIndices = new Array(maxSeqLength).fill(padTokenIndex);
    const length = Math.min(indices.length, maxSeqLength);
    for (let i = 0; i < length; i++) {
        paddedIndices[i] = indices[i];
    }

    // Return as a float32 tensor
    return tf.tensor2d([paddedIndices], [1, maxSeqLength], 'float32');
}

/**
 * Generates the corrected word using the loaded Keras model.
 * This is the core Seq2Seq inference loop.
 * @param {string} inputWord - The word to correct (typo).
 * @returns {string} The predicted correct word, or null/empty string if no correction.
 */
function predictCorrection(inputWord) {
    if (!model) return null;

    // Use tf.tidy to clean up tensors automatically after prediction
    return tf.tidy(() => {
        // 1. Prepare Encoder Input (Typo)
        const encoderInput = encodeInput(inputWord);
        
        // 2. Initial Decoder Input and States (Context from the encoder)
        const [encoderOutput, stateH, stateC] = model.predict(encoderInput);

        let decodedSequence = [startTokenIndex]; // Start with the <start> token
        let stopCondition = false;
        let correctedWord = '';
        let step = 0;
        
        let decoderInput = tf.tensor2d([decodedSequence.map(i => i)], [1, maxSeqLength], 'float32');
        let states = [stateH, stateC];

        // 3. Decoding Loop: Generate one character at a time
        while (!stopCondition) {
            
            // Prediction step
            const [outputTokens, h, c] = model.predict([decoderInput, ...states]);
            
            // Get the last predicted token's probabilities
            const lastTimeStep = outputTokens.slice([0, step, 0], [1, 1, vocabSize]);
            const nextTokenIndex = lastTimeStep.argMax(-1).dataSync()[0];

            // Append to the sequence
            decodedSequence.push(nextTokenIndex);
            
            // Check for stop condition
            if (nextTokenIndex === endTokenIndex || correctedWord.length >= maxSeqLength - 2) {
                stopCondition = true;
            }

            // Update for the next loop
            states = [h, c];
            step++;
            
            // Prepare the decoder input for the next step (it's the sequence so far)
            const nextDecoderInput = new Array(maxSeqLength).fill(padTokenIndex);
            const currentDecodedLength = decodedSequence.length;
            for(let i=0; i < currentDecodedLength; i++) {
                nextDecoderInput[i] = decodedSequence[i];
            }
            decoderInput = tf.tensor2d([nextDecoderInput], [1, maxSeqLength], 'float32');

            // Map index to character for the output word (excluding start token)
            if (nextTokenIndex !== startTokenIndex && nextTokenIndex !== endTokenIndex && correctedWord.length < maxSeqLength - 2) {
                correctedWord += indexToChar[nextTokenIndex] || '';
            }
        }
        
        // Clean up the word
        return correctedWord.replace(/<start>|<end>|<pad>/g, '').trim();
    });
}

// --- Initialization ---

async function initialize() {
    console.log("Sandbox: Initializing AI resources...");
    try {
        // 1. Load Tokenizer Configuration
        const tokenizerUrl = chrome.runtime.getURL('data/tokenizer_config.json');
        console.log("Sandbox: Fetching tokenizer config from:", tokenizerUrl);
        const tokenizerResponse = await fetch(tokenizerUrl);
        if (!tokenizerResponse.ok) throw new Error(`Tokenizer fetch failed: ${tokenizerResponse.statusText}`);
        tokenizerConfig = await tokenizerResponse.json();
        
        // Populate global variables
        charToIndex = tokenizerConfig.char_to_index;
        indexToChar = Object.fromEntries(Object.entries(tokenizerConfig.index_to_char).map(([k, v]) => [parseInt(k), v]));
        maxSeqLength = tokenizerConfig.max_seq_length;
        startTokenIndex = tokenizerConfig.start_token_index;
        endTokenIndex = tokenizerConfig.end_token_index;
        padTokenIndex = tokenizerConfig.pad_token_index;
        vocabSize = tokenizerConfig.vocab_size;
        console.log("Sandbox: Tokenizer config loaded. Max length:", maxSeqLength);

        // 2. Load TensorFlow.js Model
        if (typeof tf === 'undefined') throw new Error("TensorFlow.js library (tf) not found in sandbox.");
        const modelUrl = chrome.runtime.getURL('model/model.json');
        console.log("Sandbox: Loading model from:", modelUrl);
        // The sandbox runs the Keras model (encoder and decoder)
        model = await tf.loadLayersModel(modelUrl);
        console.log("Sandbox: TensorFlow.js model loaded successfully.");

        // 3. Model Warmup
        console.log("Sandbox: Warming up model...");
        tf.tidy(() => { 
            const warmupInput = tf.zeros([1, maxSeqLength], 'float32');
            const dummyDecoderInput = tf.zeros([1, maxSeqLength], 'float32');
            try {
                // The actual model prediction logic needs to be here. 
                // Since this is a Seq2Seq model (encoder/decoder), the simple predict might not work,
                // but we run a dummy operation to ensure the WebGL backend is active.
                model.predict(warmupInput); 
            } catch (e) {
                 console.warn("Sandbox: Warmup prediction failed (expected for Seq2Seq):", e);
            }
        });
        console.log("Sandbox: Model warmup complete.");

        // 4. Signal readiness to the content script (parent window)
        window.parent.postMessage({ type: 'GTC_READY' }, '*');
        console.log("Sandbox: Ready signal sent to content script.");

    } catch (error) {            
        console.error("Sandbox: Initialization failed:", error);
        window.parent.postMessage({ type: 'GTC_ERROR', message: `AI initialization failed: ${error.message}` }, '*');
    }
}

// --- Communication Listener ---

// Event listener to receive messages (typos) from content.js (the parent window)
window.addEventListener('message', async (event) => {
    // Only process messages from the parent frame (the content script) and specific type
    if (event.source !== window.parent || !event.data || event.data.type !== 'GTC_PREDICT') {
        return;
    }

    const originalWord = event.data.word;
    
    // Perform the AI correction
    const correctedWord = predictCorrection(originalWord);

    // Send the result back to the content script
    window.parent.postMessage({ 
        type: 'GTC_RESULT', 
        originalWord: originalWord, 
        correctedWord: correctedWord // Will be null if no correction is needed/found
    }, '*');
});

// Start the initialization process
initialize();