// tf-bridge.js
// This script runs in the page context and provides access to TensorFlow.js
// It communicates with the content script via custom events

(function() {
    'use strict';
    
    console.log("TF Bridge: Waiting for TensorFlow.js to load...");
    
    // Wait for TensorFlow.js to be available
    function waitForTF() {
        return new Promise((resolve) => {
            if (typeof tf !== 'undefined') {
                resolve();
                return;
            }
            
            const checkInterval = setInterval(() => {
                if (typeof tf !== 'undefined') {
                    clearInterval(checkInterval);
                    resolve();
                }
            }, 100);
        });
    }
    
    waitForTF().then(() => {
        console.log("TF Bridge: TensorFlow.js is ready!");
        
        // Expose tf to window for content script access
        window.tfReady = true;
        window.tfVersion = tf.version.tfjs;
        
        // Dispatch custom event to notify content script
        window.dispatchEvent(new CustomEvent('tensorflowReady', {
            detail: { version: tf.version.tfjs }
        }));
        
        console.log("TF Bridge: Notified content script. TensorFlow.js version:", tf.version.tfjs);
    });
})();
