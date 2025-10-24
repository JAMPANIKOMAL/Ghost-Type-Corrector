# Ghost Type Corrector

**AI-Powered Contextual Autocorrect Extension**

## About the project
Ghost Type Corrector is a browser extension that provides a phone‑like, invisible autocorrect experience. Instead of underlines or popups, it silently fixes typing errors as you type and lets you undo a correction by pressing Backspace immediately after it occurs.

This is a complete rewrite of the original "Invisible Autocorrect" prototype, replacing a frequency‑dictionary lookup with an in‑browser neural network (RNN/LSTM) for genuine contextual corrections (e.g., "I went too the store").

## Core features
- Truly invisible: no popups, no underlines, no distractions.
- Context‑aware AI: corrects spelling and contextual errors.
- Backspace‑to‑undo: Backspace immediately after a correction reverts to your original text.
- Lightweight & private: runs 100% in the browser with TensorFlow.js — no server required.

## Project structure
- `/ai_model/` — Python source, training data, and notebooks used to train and convert the model.  
- `/extension/` — Loadable unpacked browser extension (JavaScript, TF.js model, etc.).

## Original prototype
The original dictionary‑based prototype is available at:
https://github.com/JAMPANIKOMAL/invisible-autocorrect-extension

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements
Developed with assistance from Google's Gemini.  
The original prototype's frequency ideas were inspired by the SymSpell project.
