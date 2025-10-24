Ghost Type Corrector: AI-Powered Contextual Autocorrect

Ghost Type Corrector is a browser extension that provides an invisible, "native-feel" autocorrection experience. Unlike other correctors, it features no popups or underlines. It silently corrects text as you type and includes a "backspace-to-undo" feature, mimicking the feel of a mobile keyboard.

This project replaces a simple dictionary-based prototype with a true, lightweight, in-browser neural network (RNN/LSTM) to provide contextual grammar and spelling corrections.

Core Features

Invisible Correction: No popups or distracting UI.

Context-Aware AI: Corrects both spelling (stor -> store) and context (too -> to).

Backspace Undo: Instantly revert any automatic correction by pressing backspace.

Lightweight: Runs 100% in the browser using TensorFlow.js. No server or internet connection required for inference.

Project Structure

/ai_model/: Contains the Python source code, data, and notebooks used to train and convert the AI model.

/extension/: Contains the loadable, unpacked browser extension (JavaScript, HTML, CSS).

Development Setup

(We will fill this in as we build the project)