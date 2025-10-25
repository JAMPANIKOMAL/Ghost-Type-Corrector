// content.js
// Main script for Ghost Type Corrector: Handles AI model loading, prediction,
// user input monitoring, correction application, and backspace undo.

(function() {
    'use strict';
    console.log("Ghost Type Corrector: Content script loaded.");

    // --- Global Variables ---
    let model = null;                 // To hold the loaded TensorFlow.js model
    let tokenizerConfig = null;       // To hold char_to_index, max_seq_length etc.
    let charToIndex = null;           // Shortcut for the character map
    let indexToChar = null;           // Shortcut for inverse map
    let maxSeqLength = null;          // Shortcut for max sequence length
    let startTokenIndex = null;
    let endTokenIndex = null;
    let padTokenIndex = 0;            // Usually 0

    let lastOriginalWord = null;      // For backspace undo
    let lastCorrectedWord = null;     // For backspace undo
    let lastProcessedElement = null;  // Keep track of the element for undo

    // --- Initialization ---
    // Use an async function to handle loading promises
    async function initialize() {
        console.log("Ghost Type Corrector: Initializing...");
        try {
            // 1. Load Tokenizer Configuration
            // We need the chrome.runtime.getURL to access web_accessible_resources
            const tokenizerUrl = chrome.runtime.getURL('../ai_model/data/tokenizer_config.json');
             // ^^^ NOTE: The path is relative to the *extension root* when using getURL
            console.log("Fetching tokenizer config from:", tokenizerUrl);
            const tokenizerResponse = await fetch(tokenizerUrl);
            if (!tokenizerResponse.ok) {
                throw new Error(`Failed to fetch tokenizer config: ${tokenizerResponse.statusText}`);
            }
            tokenizerConfig = await tokenizerResponse.json();
            charToIndex = tokenizerConfig.char_to_index;
            // Need to rebuild indexToChar correctly as JSON keys are strings
            indexToChar = Object.fromEntries(Object.entries(tokenizerConfig.index_to_char).map(([k,v]) => [parseInt(k), v]));
            maxSeqLength = tokenizerConfig.max_seq_length;
            startTokenIndex = tokenizerConfig.start_token_index;
            endTokenIndex = tokenizerConfig.end_token_index;
            padTokenIndex = tokenizerConfig.pad_token_index; // Should be 0
            console.log("Tokenizer config loaded successfully. Max length:", maxSeqLength);
            // console.log("Char to index sample:", Object.entries(charToIndex).slice(0, 5));

            // 2. Load TensorFlow.js Model
            // Ensure tf object is available (loaded via manifest.json)
            if (typeof tf === 'undefined') {
                 throw new Error("TensorFlow.js library (tf) not found. Check manifest.json.");
            }
            const modelUrl = chrome.runtime.getURL('model/model.json');
             // ^^^ NOTE: Path relative to extension root
            console.log("Loading model from:", modelUrl);
            model = await tf.loadLayersModel(modelUrl);
            console.log("TensorFlow.js model loaded successfully.");

            // Optional: Warm up the model (run a dummy prediction)
            console.log("Warming up model...");
            const warmupInput = tf.zeros([1, maxSeqLength], 'int32');
            // Seq2Seq model needs two inputs during training, but often only encoder input for inference setup.
            // However, our conversion might keep the structure needing two. Let's try both ways.
            try {
                 // Try predicting with just encoder input (common inference setup, might fail if model wasn't saved this way)
                 // model.predict(warmupInput).dispose(); // This structure might not match if saved directly from training graph

                 // Try predicting with dummy encoder AND decoder inputs (matching training structure)
                 const dummyDecoderInput = tf.zeros([1, maxSeqLength], 'int32');
                 model.predict([warmupInput, dummyDecoderInput]).dispose();

            } catch (warmupError) {
                 console.warn("Model warmup prediction failed (might be expected if inference graph differs or needs single input):", warmupError);
                 // If the two-input predict fails, maybe it ONLY wants encoder input? Try that structure explicitly if needed later.
            }
            warmupInput.dispose(); // Clean up tensor
            console.log("Model warmup complete.");


            // 3. Attach Event Listeners (We'll add the actual functions later)
            attachListeners(); // Placeholder for now

            console.log("Ghost Type Corrector: Initialization complete.");

        } catch (error) {
            console.error("Ghost Type Corrector: Initialization failed:", error);
            // Disable functionality if loading fails
            model = null;
            tokenizerConfig = null;
        }
    }

    // --- Event Listeners (Placeholder) ---
    function attachListeners() {
        console.log("Attaching listeners (stub - implementation pending)...");
        // TODO: Add keyup and keydown listeners here later
        // document.body.addEventListener('keyup', handleKeyUp);
        // document.body.addEventListener('keydown', handleKeyDown);
    }

    // --- AI Processing Functions (Placeholders) ---
    function vectorizeInputText(text) {
        console.log("Vectorizing (stub):", text);
        // TODO: Implement conversion of text to padded tensor based on tokenizerConfig
        return null; // Placeholder
    }

    async function predictCorrection(inputTensor) {
        console.log("Predicting (stub)");
        // TODO: Implement model.predict() call
        // IMPORTANT: Seq2Seq inference is more complex than simple model.predict()
        // We need to run the encoder, get states, then run the decoder step-by-step.
        return null; // Placeholder
    }

    function decodeOutputVector(outputVector) {
        console.log("Decoding (stub)");
        // TODO: Implement conversion of output tensor indices back to string
        return null; // Placeholder
    }

    // --- Start Initialization ---
    // Use setTimeout to ensure the page DOM is likely ready, even with document_idle
    // This helps avoid issues with accessing document.body too early in some cases.
    setTimeout(initialize, 500); // Wait 500ms before starting

})(); // IIFE