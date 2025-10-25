// extension/js/override.js
// Disables the browser's default spellcheck UI on text input elements
// for a clean typing experience with Ghost Type Corrector

(function() {
    'use strict';

    /**
     * Disable spellcheck and autocomplete on an element
     * @param {HTMLElement} element - The element to disable spellcheck on
     */
    function disableSpellcheck(element) {
        if (element && typeof element.setAttribute === 'function') {
            element.setAttribute('spellcheck', 'false');
            element.setAttribute('autocomplete', 'off');
            element.setAttribute('autocorrect', 'off');  // iOS/Safari specific
            element.setAttribute('autocapitalize', 'off'); // iOS/Safari specific
        }
    }

    /**
     * Apply spellcheck disable to existing elements on page load
     */
    function applyToExistingElements() {
        const selectors = [
            'input[type="text"]',
            'input[type="email"]',
            'input[type="search"]',
            'input[type="url"]',
            'textarea',
            '[contenteditable="true"]'
        ];
        
        const inputs = document.querySelectorAll(selectors.join(', '));
        inputs.forEach(disableSpellcheck);
        
        if (inputs.length > 0) {
            console.log(`Ghost Type Corrector: Disabled spellcheck on ${inputs.length} input elements`);
        }
    }

    /**
     * Use MutationObserver to apply spellcheck disable to dynamically added elements
     */
    function observeDynamicElements() {
        const observer = new MutationObserver((mutationsList) => {
            for (const mutation of mutationsList) {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach(node => {
                        // Check if the added node itself is an input/textarea/contenteditable
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            const selectors = [
                                'input[type="text"]',
                                'input[type="email"]',
                                'input[type="search"]',
                                'input[type="url"]',
                                'textarea',
                                '[contenteditable="true"]'
                            ];
                            
                            const selector = selectors.join(', ');
                            
                            if (node.matches && node.matches(selector)) {
                                disableSpellcheck(node);
                            }
                            
                            // Also check children of the added node
                            if (node.querySelectorAll) {
                                const inputs = node.querySelectorAll(selector);
                                inputs.forEach(disableSpellcheck);
                            }
                        }
                    });
                }
            }
        });

        // Start observing the document body for added nodes
        observer.observe(document.body, { 
            childList: true, 
            subtree: true 
        });

        console.log("Ghost Type Corrector: MutationObserver started for dynamic elements");
    }

    /**
     * Initialize the spellcheck override
     */
    function initialize() {
        // Apply to existing elements when the script runs
        applyToExistingElements();
        
        // Observe dynamic elements
        observeDynamicElements();
        
        // Re-apply on window load just in case some elements load late
        window.addEventListener('load', applyToExistingElements);
    }

    // Start initialization when DOM is ready
    if (document.body) {
        initialize();
    } else {
        // Wait for body to be available
        const observer = new MutationObserver((mutations, obs) => {
            if (document.body) {
                initialize();
                obs.disconnect();
            }
        });
        observer.observe(document.documentElement, { childList: true });
    }

})();