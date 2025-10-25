// page-script.js
// Runs in page context and handles all TensorFlow.js operations
// Communicates with content script via window.postMessage

(function() {
    'use strict';
    
    console.log("Page Script: Starting...");
    
    let model = null;
    let tokenizerConfig = null;
    let charToIndex = null;
    let indexToChar = null;
    let maxSeqLength = null;
    let startTokenIndex = null;
    let endTokenIndex = null;
    let padTokenIndex = 0;
    let vocabSize = null;
    
    // Wait for TensorFlow.js to be available
    function waitForTF() {
        return new Promise((resolve) => {
            if (typeof tf !== 'undefined') {
                console.log("Page Script: TensorFlow.js already available");
                resolve();
                return;
            }
            
            const checkInterval = setInterval(() => {
                if (typeof tf !== 'undefined') {
                    clearInterval(checkInterval);
                    console.log("Page Script: TensorFlow.js is now available");
                    resolve();
                }
            }, 100);
        });
    }
    
    // Initialize the model
    async function initialize(config) {
        try {
            await waitForTF();
            
            console.log("Page Script: Loading tokenizer config...");
            tokenizerConfig = config.tokenizerConfig;
            charToIndex = tokenizerConfig.char_to_index;
            indexToChar = Object.fromEntries(Object.entries(tokenizerConfig.index_to_char).map(([k, v]) => [parseInt(k), v]));
            maxSeqLength = tokenizerConfig.max_seq_length;
            startTokenIndex = tokenizerConfig.start_token_index;
            endTokenIndex = tokenizerConfig.end_token_index;
            padTokenIndex = tokenizerConfig.pad_token_index;
            vocabSize = tokenizerConfig.vocab_size;
            
            console.log("Page Script: Loading TensorFlow model...");
            model = await tf.loadLayersModel(config.modelUrl);
            console.log("Page Script: Model loaded successfully!");
            
            // Warmup
            console.log("Page Script: Warming up model...");
            await tf.tidy(() => {
                const warmupInput = tf.zeros([1, maxSeqLength], 'int32');
                const dummyDecoderInput = tf.zeros([1, maxSeqLength], 'int32');
                try {
                    model.predict([warmupInput, dummyDecoderInput]);
                } catch (e) {
                    console.warn("Page Script: Warmup failed:", e);
                }
            });
            console.log("Page Script: Warmup complete!");
            
            return { success: true, message: "Model initialized successfully" };
        } catch (error) {
            console.error("Page Script: Initialization failed:", error);
            return { success: false, error: error.message };
        }
    }
    
    // Vectorize input text
    function vectorizeInputText(text) {
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
        } else {
            for (let i = currentLength; i < maxSeqLength; i++) {
                indices.push(padTokenIndex);
            }
        }
        
        return tf.tensor2d([indices], [1, maxSeqLength], 'int32');
    }
    
    // Predict correction
    async function predictCorrection(inputText) {
        if (!model || !tokenizerConfig) {
            return { success: false, error: "Model not initialized" };
        }
        
        try {
            const inputTensor = vectorizeInputText(inputText);
            let correctedText = '';
            
            await tf.tidy(async () => {
                const encoderLstmLayer = model.getLayer('encoder_lstm');
                if (!encoderLstmLayer) throw new Error("Could not find 'encoder_lstm' layer");

                const encoderModel = tf.model({
                    inputs: model.input[0],
                    outputs: [...encoderLstmLayer.output]
                });

                const encoderResults = encoderModel.predict(inputTensor);
                let targetSeq = tf.buffer([1, maxSeqLength], 'int32');
                targetSeq.set(startTokenIndex, 0, 0);

                let decodedSentence = '';

                for (let i = 0; i < maxSeqLength - 1; i++) {
                    const currentTargetTensor = tf.tensor(targetSeq.toTensor());
                    const outputTokensTensor = model.predict([inputTensor, currentTargetTensor]);
                    const sampledTokenIndex = tf.argMax(outputTokensTensor.slice([0, i, 0], [1, 1, vocabSize]), -1).dataSync()[0];
                    const sampledChar = indexToChar[sampledTokenIndex];
                    decodedSentence += sampledChar;

                    if (sampledChar === indexToChar[endTokenIndex] || decodedSentence.length > maxSeqLength) {
                        break;
                    }

                    if (i + 1 < maxSeqLength) {
                        targetSeq.set(sampledTokenIndex, 0, i + 1);
                    }

                    currentTargetTensor.dispose();
                    outputTokensTensor.dispose();
                }
                
                encoderResults.forEach(t => t.dispose());
                correctedText = decodedSentence.replace(/\n$/, '');
            });
            
            inputTensor.dispose();
            return { success: true, correctedText: correctedText };
        } catch (error) {
            console.error("Page Script: Prediction error:", error);
            return { success: false, error: error.message };
        }
    }
    
    // Listen for messages from content script
    window.addEventListener('message', async (event) => {
        // Only accept messages from same origin
        if (event.source !== window) return;
        
        const message = event.data;
        if (!message.type || !message.type.startsWith('GHOST_TYPE_')) return;
        
        console.log("Page Script: Received message:", message.type);
        
        let response;
        switch (message.type) {
            case 'GHOST_TYPE_INIT':
                response = await initialize(message.data);
                window.postMessage({
                    type: 'GHOST_TYPE_INIT_RESPONSE',
                    id: message.id,
                    data: response
                }, '*');
                break;
                
            case 'GHOST_TYPE_PREDICT':
                response = await predictCorrection(message.data.text);
                window.postMessage({
                    type: 'GHOST_TYPE_PREDICT_RESPONSE',
                    id: message.id,
                    data: response
                }, '*');
                break;
        }
    });
    
    // Signal ready
    console.log("Page Script: Ready to receive messages");
    window.postMessage({ type: 'GHOST_TYPE_PAGE_SCRIPT_READY' }, '*');
    
})();
