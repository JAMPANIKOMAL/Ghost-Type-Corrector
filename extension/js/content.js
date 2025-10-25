// content.js - Simplified version using message passing
(function() {
    'use strict';
    console.log("Ghost Type Corrector: Content script loaded.");

    let isModelReady = false;
    let messageId = 0;
    let pendingMessages = new Map();
    
    let lastOriginalPhrase = null;
    let lastCorrectedPhrase = null;
    let lastProcessedElement = null;

    function sendMessageToPage(type, data) {
        return new Promise((resolve, reject) => {
            const id = ++messageId;
            pendingMessages.set(id, { resolve, reject });
            window.postMessage({ type, id, data }, '*');
            setTimeout(() => {
                if (pendingMessages.has(id)) {
                    pendingMessages.delete(id);
                    reject(new Error('Timeout'));
                }
            }, 30000);
        });
    }

    window.addEventListener('message', (event) => {
        if (event.source !== window) return;
        const message = event.data;
        if (!message.type) return;
        
        if (message.type === 'GHOST_TYPE_PAGE_SCRIPT_READY') {
            console.log("Content: Page script ready");
            initialize();
        } else if (message.type === 'GHOST_TYPE_INIT_RESPONSE' || message.type === 'GHOST_TYPE_PREDICT_RESPONSE') {
            const pending = pendingMessages.get(message.id);
            if (pending) {
                pendingMessages.delete(message.id);
                pending.resolve(message.data);
            }
        }
    });

    function injectScripts() {
        const tfScript = document.createElement('script');
        tfScript.src = chrome.runtime.getURL('js/lib/tf.min.js');
        (document.head || document.documentElement).appendChild(tfScript);
        
        tfScript.onload = () => {
            const pageScript = document.createElement('script');
            pageScript.src = chrome.runtime.getURL('js/page-script.js');
            (document.head || document.documentElement).appendChild(pageScript);
        };
    }

    async function initialize() {
        try {
            const tokenizerUrl = chrome.runtime.getURL('data/tokenizer_config.json');
            const tokenizerResponse = await fetch(tokenizerUrl);
            const tokenizerConfig = await tokenizerResponse.json();
            
            const modelUrl = chrome.runtime.getURL('model/model.json');
            const response = await sendMessageToPage('GHOST_TYPE_INIT', { tokenizerConfig, modelUrl });
            
            if (response.success) {
                isModelReady = true;
                console.log("Ghost Type Corrector: Ready!");
                attachListeners();
            }
        } catch (error) {
            console.error("Init failed:", error);
        }
    }

    function attachListeners() {
        document.body.addEventListener('keyup', handleKeyUp, true);
        document.body.addEventListener('keydown', handleKeyDown, true);
    }

    async function predictCorrection(text) {
        const response = await sendMessageToPage('GHOST_TYPE_PREDICT', { text });
        return response.success ? response.correctedText : null;
    }

    let keyUpTimeout;
    async function handleKeyUp(event) {
        const el = event.target;
        if (!isModelReady || !((el.tagName === 'INPUT' && el.type === 'text') || el.tagName === 'TEXTAREA')) return;
        
        if (event.key === ' ' || event.key === 'Enter') {
            clearTimeout(keyUpTimeout);
            keyUpTimeout = setTimeout(async () => {
                const text = el.value;
                const pos = el.selectionStart;
                const before = text.substring(0, pos);
                const start = Math.max(0, before.lastIndexOf(' ') + 1);
                const after = text.substring(pos);
                const end = after.indexOf(' ');
                const endPos = end === -1 ? text.length : pos + end;
                const phrase = text.substring(start, endPos);
                
                if (phrase.trim().length < 2) return;
                
                const corrected = await predictCorrection(phrase);
                if (corrected && corrected.trim() !== phrase.trim()) {
                    lastOriginalPhrase = phrase;
                    lastCorrectedPhrase = corrected.trim();
                    lastProcessedElement = el;
                    
                    el.value = text.substring(0, start) + corrected.trim() + text.substring(endPos);
                    el.setSelectionRange(start + corrected.trim().length, start + corrected.trim().length);
                    console.log(""  "");
                }
            }, 300);
        }
    }

    function handleKeyDown(event) {
        if (event.key === 'Backspace' && lastOriginalPhrase && lastProcessedElement === event.target) {
            const el = event.target;
            if (el.value.includes(lastCorrectedPhrase)) {
                const idx = el.value.indexOf(lastCorrectedPhrase);
                if (Math.abs(el.selectionStart - (idx + lastCorrectedPhrase.length)) <= 2) {
                    event.preventDefault();
                    el.value = el.value.substring(0, idx) + lastOriginalPhrase + el.value.substring(idx + lastCorrectedPhrase.length);
                    el.setSelectionRange(idx + lastOriginalPhrase.length, idx + lastOriginalPhrase.length);
                    console.log(Undo: ""  "");
                    lastOriginalPhrase = null;
                    lastCorrectedPhrase = null;
                    lastProcessedElement = null;
                }
            }
        }
    }

    injectScripts();
})();
