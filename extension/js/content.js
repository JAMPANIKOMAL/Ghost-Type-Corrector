// content.js
// Main script for Ghost Type Corrector: Handles AI model loading, prediction,
// user input monitoring, correction application, and backspace undo.

(function() {
    'use strict';
    console.log("Ghost Type Corrector: Content script loaded.");

    // --- Global Variables ---
    let model = null;
    let tokenizerConfig = null;
    let charToIndex = null;
    let indexToChar = null;
    let maxSeqLength = null;
    let startTokenIndex = null;
    let endTokenIndex = null;
    let padTokenIndex = 0;
    let vocabSize = null; // Added vocab size

    let lastOriginalPhrase = null; // Store the phrase before correction
    let lastCorrectedPhrase = null;// Store the corrected phrase
    let lastProcessedElement = null;// Keep track of the element for undo
    let lastCursorPosition = null; // Store cursor position before correction

    // --- Inject TensorFlow.js into page context ---
    function injectTensorFlowJS() {
        return new Promise((resolve, reject) => {
            // Check if TensorFlow is already loaded
            if (typeof window.tf !== 'undefined') {
                console.log("TensorFlow.js already loaded in page context.");
                resolve();
                return;
            }

            const script = document.createElement('script');
            script.src = chrome.runtime.getURL('js/lib/tf.min.js');
            
            script.onload = () => {
                console.log("TensorFlow.js script injected successfully.");
                
                // Poll for window.tf to be available (max 5 seconds)
                let attempts = 0;
                const maxAttempts = 50; // 50 * 100ms = 5 seconds
                
                const checkTF = setInterval(() => {
                    attempts++;
                    
                    if (typeof window.tf !== 'undefined') {
                        clearInterval(checkTF);
                        console.log("TensorFlow.js is now available on window object.");
                        resolve();
                    } else if (attempts >= maxAttempts) {
                        clearInterval(checkTF);
                        reject(new Error("TensorFlow.js loaded but 'tf' is not available on window object after 5 seconds."));
                    }
                }, 100);
            };
            
            script.onerror = () => {
                reject(new Error("Failed to inject TensorFlow.js script."));
            };
            
            (document.head || document.documentElement).appendChild(script);
        });
    }

    // --- Initialization ---
    async function initialize() {
        console.log("Ghost Type Corrector: Initializing...");
        try {
            // 0. Inject TensorFlow.js first
            await injectTensorFlowJS();

            // 1. Load Tokenizer Configuration
            const tokenizerUrl = chrome.runtime.getURL('data/tokenizer_config.json');
            console.log("Fetching tokenizer config from:", tokenizerUrl);
            const tokenizerResponse = await fetch(tokenizerUrl);
            if (!tokenizerResponse.ok) throw new Error(`Tokenizer fetch failed: ${tokenizerResponse.statusText}`);
            tokenizerConfig = await tokenizerResponse.json();
            charToIndex = tokenizerConfig.char_to_index;
            indexToChar = Object.fromEntries(Object.entries(tokenizerConfig.index_to_char).map(([k, v]) => [parseInt(k), v]));
            maxSeqLength = tokenizerConfig.max_seq_length;
            startTokenIndex = tokenizerConfig.start_token_index;
            endTokenIndex = tokenizerConfig.end_token_index;
            padTokenIndex = tokenizerConfig.pad_token_index;
            vocabSize = tokenizerConfig.vocab_size; // Added
            console.log("Tokenizer config loaded. Max length:", maxSeqLength, "Vocab size:", vocabSize);

            // 2. Load TensorFlow.js Model
            if (typeof window.tf === 'undefined') throw new Error("TensorFlow.js library (tf) not found on window object.");
            const modelUrl = chrome.runtime.getURL('model/model.json');
            console.log("Loading model from:", modelUrl);
            model = await window.tf.loadLayersModel(modelUrl);
            console.log("TensorFlow.js model loaded successfully.");

            // 3. Model Warmup
            console.log("Warming up model...");
            await window.tf.tidy(() => { // Use tidy to auto-dispose tensors
                const warmupInput = window.tf.zeros([1, maxSeqLength], 'int32');
                const dummyDecoderInput = window.tf.zeros([1, maxSeqLength], 'int32');
                try {
                    // Our model expects two inputs as defined during training
                    model.predict([warmupInput, dummyDecoderInput]);
                } catch (e) {
                     console.warn("Warmup prediction failed (might indicate issue with model structure):", e);
                }
            });
            console.log("Model warmup complete.");

            // 4. Attach Event Listeners
            attachListeners();

            console.log("Ghost Type Corrector: Initialization complete. Ready!");

        } catch (error) {
            console.error("Ghost Type Corrector: Initialization failed:", error);
            model = null; // Disable functionality
        }
    }

    // --- Event Listeners ---
    function attachListeners() {
        console.log("Attaching listeners...");
        // Use event delegation on the body for dynamically added elements
        document.body.addEventListener('keyup', handleKeyUp, true); // Use capture phase
        document.body.addEventListener('keydown', handleKeyDown, true); // Use capture phase
    }

    // --- AI Processing Functions ---

    function vectorizeInputText(text) {
        if (!tokenizerConfig) return null;

        let cleanedText = text.toLowerCase();
        const allowedChars = Object.keys(charToIndex).filter(c => c && c !== '\t' && c !== '\n').join('');
        const regexPattern = new RegExp(`[^${allowedChars}\\s]`, 'g');
        cleanedText = cleanedText.replace(regexPattern, '');
        cleanedText = cleanedText.replace(/\s+/g, ' ').trim();

        let indices = [startTokenIndex];
        for (let i = 0; i < cleanedText.length; i++) {
            indices.push(charToIndex[cleanedText[i]] || padTokenIndex);
        }
        indices.push(endTokenIndex);

        const currentLength = indices.length;
        if (currentLength > maxSeqLength) {
            indices = indices.slice(0, maxSeqLength);
            indices[maxSeqLength - 1] = endTokenIndex;
            console.warn(`Input text truncated to maxSeqLength: ${maxSeqLength}`);
        } else {
            for (let i = currentLength; i < maxSeqLength; i++) {
                indices.push(padTokenIndex);
            }
        }
        // Return tensor within a tidy scope if created here, or manage disposal later
        return window.tf.tensor2d([indices], [1, maxSeqLength], 'int32');
    }

    async function predictCorrection(inputText) {
        if (!model || !tokenizerConfig) return null; // Ensure everything is loaded

        // console.time("Prediction"); // Optional: time the prediction
        const inputTensor = vectorizeInputText(inputText);
        if (!inputTensor) return null;

        let correctedText = '';
        try {
            await window.tf.tidy(async () => { // Auto-dispose intermediate tensors

                // --- Encoder Step ---
                // We need to get the encoder's final states
                // Find the encoder LSTM layer by name
                const encoderLstmLayer = model.getLayer('encoder_lstm');
                if (!encoderLstmLayer) throw new Error("Could not find 'encoder_lstm' layer in the loaded model.");

                // Create an intermediate model to get encoder outputs + states
                const encoderModel = window.tf.model({
                    inputs: model.input[0], // Encoder input
                    outputs: [...encoderLstmLayer.output] // Output sequence AND states [outputs, state_h, state_c]
                });

                // Run the encoder
                const encoderResults = encoderModel.predict(inputTensor);
                const statesValue = [encoderResults[1], encoderResults[2]]; // [state_h, state_c]

                // --- Decoder Step (Iterative Prediction) ---
                // Start with the START_TOKEN index
                let targetSeq = window.tf.buffer([1, maxSeqLength], 'int32');
                targetSeq.set(startTokenIndex, 0, 0); // Set first element to START

                let stopCondition = false;
                let decodedSentence = '';
                const decoderLstmLayer = model.getLayer('decoder_lstm');
                const decoderEmbeddingLayer = model.getLayer('decoder_embedding');
                const outputDenseLayer = model.getLayer('output_dense');

                if (!decoderLstmLayer || !decoderEmbeddingLayer || !outputDenseLayer) {
                     throw new Error("Could not find necessary decoder layers (decoder_lstm, decoder_embedding, output_dense).");
                }


                for (let i = 0; i < maxSeqLength - 1; i++) { // Loop up to max length - 1
                    // Prepare decoder input for this timestep (only the current target sequence matters)
                    const currentTargetTensor = window.tf.tensor(targetSeq.toTensor());

                    // Predict the next character
                    // We need a model that takes encoder states and current decoder input
                    // This requires reconstructing parts of the graph or having separate inference models
                    // For simplicity here, we'll try using the full model, feeding the current sequence
                    // This assumes the model structure handles inference correctly when states are passed implicitly (often true with Keras conversion)
                    // OR we need specific inference models saved/converted.

                    // Let's assume the converted model handles the state passing implicitly for now.
                    // This might be slow or incorrect if the conversion wasn't optimized for inference.
                    const outputTokensTensor = model.predict([inputTensor, currentTargetTensor]);

                    // Get the character index with the highest probability at the current timestep 'i'
                    const sampledTokenIndex = window.tf.argMax(outputTokensTensor.slice([0, i, 0], [1, 1, vocabSize]), -1).dataSync()[0];

                    const sampledChar = indexToChar[sampledTokenIndex];

                    // Append the predicted character
                    decodedSentence += sampledChar;

                    // Exit condition: either END_TOKEN predicted or max length reached
                    if (sampledChar === indexToChar[endTokenIndex] || decodedSentence.length > maxSeqLength) {
                        stopCondition = true;
                    }

                    // Update the target sequence for the next timestep
                    if (i + 1 < maxSeqLength) {
                         targetSeq.set(sampledTokenIndex, 0, i + 1);
                    }

                    // Dispose intermediate tensor
                     currentTargetTensor.dispose();
                     outputTokensTensor.dispose();


                    if (stopCondition) {
                        break;
                    }
                }
                 // Clean up encoder results
                 encoderResults.forEach(t => t.dispose());

                 // Remove potential END_TOKEN character if present at the end
                 correctedText = decodedSentence.replace(/\n$/, '');


            }); // End tf.tidy
        } catch (error) {
             console.error("Prediction failed:", error);
             correctedText = null; // Indicate failure
        } finally {
            inputTensor.dispose(); // Manually dispose input tensor after tidy finishes
           // console.timeEnd("Prediction");
        }

        return correctedText;
    }

    // Decode function is implicitly handled within predictCorrection now
    // function decodeOutputVector(outputVector) { ... } // Not needed separately


    // --- Core Logic ---

    // Debounce mechanism to avoid running prediction on every keystroke
    let debounceTimer;
    const DEBOUNCE_DELAY = 300; // ms to wait after typing stops

    async function handleKeyUp(event) {
        // Only trigger on space or enter, and only if AI is ready
        if (!model || (event.key !== ' ' && event.key !== 'Enter')) {
            clearTimeout(debounceTimer); // Clear timer if it's not space/enter
            return;
        }

        const target = event.target;
        // Check if the target is an editable field
        if (!(target.isContentEditable || target.nodeName === 'TEXTAREA' || (target.nodeName === 'INPUT' && /^(text|search|url|tel|email|password)$/i.test(target.type)))) {
             clearTimeout(debounceTimer);
             return;
        }

        // --- Debounce ---
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(async () => {
            let textToProcess = '';
            let currentCursorPosition = 0;

            // Get text and cursor position based on element type
             if (target.isContentEditable) {
                // For contentEditable, process the text content up to the cursor
                const selection = window.getSelection();
                if (selection.rangeCount > 0) {
                     const range = selection.getRangeAt(0);
                     currentCursorPosition = range.startOffset; // Position within the current text node
                     const container = range.startContainer;
                     if(container.nodeType === Node.TEXT_NODE) {
                         textToProcess = container.textContent.substring(0, currentCursorPosition);
                     } else {
                         // Fallback for complex contentEditable - might need refinement
                         textToProcess = target.textContent;
                         currentCursorPosition = textToProcess.length; // Approximate
                     }
                } else {
                     textToProcess = target.textContent; // Get all text if no selection
                     currentCursorPosition = textToProcess.length;
                }
            } else { // Input or Textarea
                textToProcess = target.value;
                currentCursorPosition = target.selectionStart;
            }


            // Trim whitespace from the end for processing
             const trimmedText = textToProcess.trimEnd();
             if (trimmedText.length === 0) return; // Nothing to process


             // --- Extract last meaningful segment (e.g., sentence or phrase) ---
             // Simple approach: process the text before the cursor
             const textBeforeCursor = trimmedText.substring(0, currentCursorPosition);

            // More robust: Find the last sentence boundary (., !, ?) or start of text
             let lastSentenceStart = Math.max(
                 textBeforeCursor.lastIndexOf('. '),
                 textBeforeCursor.lastIndexOf('! '),
                 textBeforeCursor.lastIndexOf('? '),
                 textBeforeCursor.lastIndexOf('\n'), // Consider newlines as boundaries
                 0 // Start of the text
             );
             // Adjust if the boundary char was found
             if (lastSentenceStart > 0 && ['.', '!', '?','\n'].includes(textBeforeCursor[lastSentenceStart])) {
                 lastSentenceStart += 1; // Start after the boundary character
             }
             // Trim leading space after boundary
             while (lastSentenceStart < textBeforeCursor.length && textBeforeCursor[lastSentenceStart] === ' ') {
                 lastSentenceStart++;
             }

             const phraseToCorrect = textBeforeCursor.substring(lastSentenceStart).trim();


            if (!phraseToCorrect || phraseToCorrect.length < 3) return; // Don't correct very short inputs

            console.log("Phrase to correct:", `"${phraseToCorrect}"`);
            const correctedPhrase = await predictCorrection(phraseToCorrect);

             if (correctedPhrase && correctedPhrase !== phraseToCorrect) {
                 console.log("Correction found:", `"${correctedPhrase}"`);

                 // Store for undo
                 lastOriginalPhrase = phraseToCorrect;
                 lastCorrectedPhrase = correctedPhrase;
                 lastProcessedElement = target;
                 lastCursorPosition = currentCursorPosition; // Store cursor pos BEFORE correction

                 // --- Replace the text ---
                 // This needs to carefully replace only the corrected part
                 const textBeforePhrase = textBeforeCursor.substring(0, lastSentenceStart);
                 const textAfterCursor = textToProcess.substring(currentCursorPosition); // Text originally after cursor

                 const newText = textBeforePhrase + correctedPhrase; // + textAfterCursor; // Decide if text after cursor should be kept
                 const newCursorPosition = textBeforePhrase.length + correctedPhrase.length;

                 if (target.isContentEditable) {
                     // ContentEditable replacement is complex and needs range manipulation
                     // Simple approach (might lose formatting):
                     // target.textContent = newText + textAfterCursor; // Overwrite all
                     // More complex range manipulation needed here for robust solution
                     console.warn("ContentEditable replacement not fully implemented - using basic textContent update.");
                     // Attempt basic update (find the text node and replace relevant part)
                     const selection = window.getSelection();
                     if (selection.rangeCount > 0) {
                         const range = selection.getRangeAt(0);
                         const container = range.startContainer;
                         if(container.nodeType === Node.TEXT_NODE) {
                              const originalContent = container.textContent;
                              // Replace the segment within the text node
                              container.textContent = originalContent.substring(0, lastSentenceStart) + correctedPhrase + originalContent.substring(currentCursorPosition);
                              // Try to restore cursor (approximate)
                               const newRange = document.createRange();
                               newRange.setStart(container, Math.min(newCursorPosition, container.textContent.length));
                               newRange.collapse(true);
                               selection.removeAllRanges();
                               selection.addRange(newRange);
                         } else {
                              target.textContent = newText + textAfterCursor; // Fallback
                         }
                     } else {
                          target.textContent = newText + textAfterCursor; // Fallback
                     }


                 } else { // Input or Textarea
                     target.value = newText + textAfterCursor; // Replace the value
                     // Restore cursor position
                     target.setSelectionRange(newCursorPosition, newCursorPosition);
                 }


             } else if (correctedPhrase === phraseToCorrect) {
                // console.log("No correction needed.");
             } else {
                console.log("Prediction failed or returned null.");
             }

        }, DEBOUNCE_DELAY); // End debounce setTimeout
    }

    function handleKeyDown(event) {
        if (!model || event.key !== 'Backspace') {
            lastOriginalPhrase = null; // Clear undo state if any other key is pressed
            lastCorrectedPhrase = null;
            lastProcessedElement = null;
            lastCursorPosition = null;
            return;
        }

        const target = event.target;
        // Check if undo is possible
        if (target === lastProcessedElement && lastOriginalPhrase && lastCorrectedPhrase) {
            let textBeforeCursor = '';
            let currentCursorPosition = 0;

            // Get current text and cursor position
            if (target.isContentEditable) {
                const selection = window.getSelection();
                if (selection.rangeCount > 0) {
                     const range = selection.getRangeAt(0);
                     currentCursorPosition = range.startOffset;
                     const container = range.startContainer;
                     if (container.nodeType === Node.TEXT_NODE) {
                         textBeforeCursor = container.textContent.substring(0, currentCursorPosition);
                     } else { // Fallback needed
                          textBeforeCursor = target.textContent.substring(0, lastCursorPosition); // Use stored pos as approx
                          currentCursorPosition = lastCursorPosition;
                     }
                } else return; // Cannot undo without selection info

            } else { // Input or Textarea
                currentCursorPosition = target.selectionStart;
                textBeforeCursor = target.value.substring(0, currentCursorPosition);
            }


            // Check if the text immediately preceding the cursor matches the corrected phrase
            if (textBeforeCursor.endsWith(lastCorrectedPhrase)) {
                // Prevent the default backspace action
                event.preventDefault();

                // Calculate where the original phrase should start
                const replaceStartIndex = currentCursorPosition - lastCorrectedPhrase.length;

                // Restore the original phrase
                const textBeforeOriginal = textBeforeCursor.substring(0, replaceStartIndex);
                let textAfterOriginal = '';

                if (target.isContentEditable) {
                     const selection = window.getSelection();
                     const range = selection.getRangeAt(0);
                     const container = range.startContainer;
                     if(container.nodeType === Node.TEXT_NODE) {
                         textAfterOriginal = container.textContent.substring(currentCursorPosition);
                         container.textContent = textBeforeOriginal + lastOriginalPhrase + textAfterOriginal;
                         // Set cursor position after the restored original phrase
                         const newCursorPos = replaceStartIndex + lastOriginalPhrase.length;
                         const newRange = document.createRange();
                         newRange.setStart(container, Math.min(newCursorPos, container.textContent.length));
                         newRange.collapse(true);
                         selection.removeAllRanges();
                         selection.addRange(newRange);

                     } else { // Fallback needed
                          target.textContent = textBeforeOriginal + lastOriginalPhrase + target.textContent.substring(lastCursorPosition);
                     }

                } else { // Input or Textarea
                    textAfterOriginal = target.value.substring(currentCursorPosition);
                    target.value = textBeforeOriginal + lastOriginalPhrase + textAfterOriginal;
                    // Set cursor position after the restored original phrase
                    const newCursorPos = replaceStartIndex + lastOriginalPhrase.length;
                    target.setSelectionRange(newCursorPos, newCursorPos);
                }


                console.log("Undo applied:", `"${lastCorrectedPhrase}" -> "${lastOriginalPhrase}"`);

                // Clear undo state after applying
                lastOriginalPhrase = null;
                lastCorrectedPhrase = null;
                lastProcessedElement = null;
                lastCursorPosition = null;

            } else {
                 // Cursor isn't right after the corrected word, allow normal backspace
                 lastOriginalPhrase = null;
                 lastCorrectedPhrase = null;
                 lastProcessedElement = null;
                 lastCursorPosition = null;
            }
        }
    }


    // --- Start Initialization ---
    setTimeout(initialize, 500); // Wait 500ms

})(); // IIFE