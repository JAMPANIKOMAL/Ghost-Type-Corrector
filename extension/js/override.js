// override.js
// Disables the browser's default spellcheck UI on text input elements

(function() {
    'use strict';

    // Function to disable spellcheck on an element
    function disableSpellcheck(element) {
        if (element && typeof element.setAttribute === 'function') {
            element.setAttribute('spellcheck', 'false');
            element.setAttribute('autocomplete', 'off'); // Often helpful too
            element.setAttribute('autocorrect', 'off');  // iOS specific, but doesn't hurt
            element.setAttribute('autocapitalize', 'off'); // iOS specific
        }
    }

    // Apply to existing elements on page load
    function applyToExistingElements() {
        const inputs = document.querySelectorAll('input[type="text"], textarea, [contenteditable="true"]');
        inputs.forEach(disableSpellcheck);
        // console.log("GhostType: Disabled spellcheck on initial elements.");
    }

    // Use MutationObserver to apply to dynamically added elements
    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(node => {
                    // Check if the added node itself is an input/textarea/contenteditable
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (node.matches('input[type="text"], textarea, [contenteditable="true"]')) {
                            disableSpellcheck(node);
                        }
                        // Also check children of the added node
                        const inputs = node.querySelectorAll('input[type="text"], textarea, [contenteditable="true"]');
                        inputs.forEach(disableSpellcheck);
                    }
                });
            }
        }
    });

    // Start observing the document body for added nodes
    observer.observe(document.body, { childList: true, subtree: true });

    // Apply initially when the script runs (document_idle)
    applyToExistingElements();

    // Re-apply on window load just in case some elements load late
    window.addEventListener('load', applyToExistingElements);

})(); // Immediately Invoked Function Expression (IIFE) to avoid polluting global scope