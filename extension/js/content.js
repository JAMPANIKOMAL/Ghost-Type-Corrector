// extension/js/content.js
// Main script for Ghost Type Corrector: Sets up the sandboxed AI environment 
// and handles communication, user input, and text replacement.

(function() {
    'use strict';
    console.log("Ghost Type Corrector: Content script loaded (Sandbox Mode).");

    // --- Global Variables ---
    let sandboxReady = false;
    let sandboxIframe = null; 

    // Variables for Text Correction (Keeping these on the content side is key)
    let lastOriginalWord = null;
    let lastCorrectedWord = null;
    let lastProcessedElement = null; // Keep track of the element for undo

    // --- Helper Function: Case Matching (Kept here for speed) ---
    function matchCase(originalWord, correctedWord) {
        if (!originalWord || !correctedWord) return correctedWord;

        // All caps
        if (originalWord === originalWord.toUpperCase()) {
            return correctedWord.toUpperCase();
        }
        
        // Title case (First letter capitalized)
        if (originalWord[0] === originalWord[0].toUpperCase()) {
            return correctedWord.charAt(0).toUpperCase() + correctedWord.slice(1);
        }

        // Default to lowercase
        return correctedWord;
    }

    // --- Helper Function: Move Cursor ---
    function moveCursorToEnd(element) {
        if (element.value !== undefined) {
            // For textarea/input
            element.selectionStart = element.selectionEnd = element.value.length;
            element.focus();
        } else if (element.isContentEditable) {
            // For contentEditable elements
            const range = document.createRange();
            const selection = window.getSelection();
            range.selectNodeContents(element);
            range.collapse(false); 
            selection.removeAllRanges();
            selection.addRange(range);
        }
    }

    // --- Sandbox Communication Setup ---

    function setupSandbox() {
        sandboxIframe = document.createElement('iframe');
        // Load the sandboxed HTML page
        sandboxIframe.src = chrome.runtime.getURL('sandbox.html');
        sandboxIframe.style.display = 'none'; // Keep it invisible
        document.body.appendChild(sandboxIframe);

        // Listener to receive messages from the sandbox
        window.addEventListener('message', handleSandboxMessage);
        
        console.log("Content Script: Sandbox iframe created and listener attached.");
    }

    function handleSandboxMessage(event) {
        // Security check: Only accept messages from the origin of the content script itself
        if (event.origin !== window.location.origin || !event.data || event.source !== sandboxIframe.contentWindow) {
            return; 
        }

        if (event.data.type === 'GTC_READY') {
            // AI model is loaded and ready to receive prediction requests
            sandboxReady = true;
            console.log("Content Script: Sandbox is ready! AI model loaded.");
            // 4. Attach Event Listeners once everything is ready
            attachListeners(); 
        } else if (event.data.type === 'GTC_ERROR') {
            console.error("Content Script: Sandbox Error:", event.data.message);
        } else if (event.data.type === 'GTC_RESULT') {
            // Correction result received from the sandbox
            applyCorrection(event.data.originalWord, event.data.correctedWord);
        }
    }

    // --- Autocorrect Logic ---

    function attachListeners() {
        document.body.addEventListener('keyup', handleKeyUp);
        document.body.addEventListener('keydown', handleKeyDownForUndo); // For undo feature
    }

    function handleKeyUp(event) {
        // Trigger only on spacebar AND if sandbox is ready
        if (event.key !== ' ' || !sandboxReady) {
            return;
        }

        const activeElement = event.target;
        if (activeElement.tagName.toLowerCase() !== 'textarea' && 
            activeElement.type !== 'text' && 
            activeElement.type !== 'search' && 
            !activeElement.isContentEditable) {
            return;
        }

        const text = activeElement.value || activeElement.textContent;
        const words = text.trim().split(/\s+/);
        if (words.length === 0) {
            return;
        }

        const wordToCheck = words[words.length - 1];
        lastProcessedElement = activeElement; // Store the element for later correction/undo

        // 1. Send word to the sandbox for prediction
        sandboxIframe.contentWindow.postMessage({
            type: 'GTC_PREDICT',
            word: wordToCheck
        }, '*');
    }

    // --- Apply Correction from Sandbox Result ---

    function applyCorrection(originalWord, rawCorrection) {
        if (!rawCorrection || originalWord.toLowerCase() === rawCorrection.toLowerCase() || !lastProcessedElement) {
            return;
        }
        
        const activeElement = lastProcessedElement;
        const finalCorrection = matchCase(originalWord, rawCorrection);
        
        const text = activeElement.value || activeElement.textContent;
        const words = text.trim().split(/\s+/);
        
        // Ensure the word hasn't changed since we sent the request
        if (words[words.length - 1].toLowerCase() !== originalWord.toLowerCase()) {
             // User kept typing, ignore stale correction
            return;
        }
        
        // Perform replacement
        words[words.length - 1] = finalCorrection;
        const newText = words.join(' ') + ' ';

        if (activeElement.value !== undefined) {
            activeElement.value = newText;
        } else {
            activeElement.textContent = newText;
            moveCursorToEnd(activeElement);
        }

        // Track autocorrect for undo feature
        lastOriginalWord = originalWord;
        lastCorrectedWord = finalCorrection;
    }


    // --- Undo Autocorrect Feature ---
    function handleKeyDownForUndo(event) {
        if (event.key === 'Backspace' && lastCorrectedWord && lastProcessedElement) {
            const activeElement = document.activeElement;
            // Check if the user is in the corrected element
            if (activeElement === lastProcessedElement) {
                const text = activeElement.value || activeElement.textContent;
                const words = text.trim().split(/\s+/);
                
                // Check if the last word matches the corrected word
                if (words.length > 0 && words[words.length - 1] === lastCorrectedWord) {
                    
                    // Undo autocorrect
                    words[words.length - 1] = lastOriginalWord;
                    const newText = words.join(' ') + ' ';
                    
                    if (activeElement.value !== undefined) {
                        activeElement.value = newText;
                    } else {
                        activeElement.textContent = newText;
                        moveCursorToEnd(activeElement);
                    }
                    
                    // Clear last correction data so it only works once
                    lastOriginalWord = null;
                    lastCorrectedWord = null;
                    lastProcessedElement = null;

                    // Prevent default backspace action, since we handled the correction
                    event.preventDefault();
                }
            }
        }
    }
    
    // --- Initialization Entry Point ---
    async function initialize() {
        console.log("Ghost Type Corrector: Initializing...");
        
        // --- ONLY Sandbox Setup Here ---
        // This is the CRITICAL change: We do NOT try to inject TF.js or poll for window.tf.
        setupSandbox();
        
        console.log("Ghost Type Corrector: Waiting for Sandbox AI model to load...");
    }

    // Start the extension setup
    initialize();

})();