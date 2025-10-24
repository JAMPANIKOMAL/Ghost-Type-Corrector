# Ghost Type Corrector

A next-generation, AI-powered browser extension that provides truly invisible, contextual autocorrection, bringing the seamless "mobile keyboard" experience to your desktop browser.

This project is an advanced rebuild of an initial prototype: https://github.com/JAMPANIKOMAL/invisible-autocorrect-extension, replacing the original frequency-dictionary "brain" with a character-level, sequence-to-sequence (seq2seq) neural network.

## Core Features

- **Invisible Correction:** No distracting popups, underlines, or suggestions. Typos are corrected instantly and silently the moment you press the spacebar.

- **Contextual AI:** The model understands context, allowing it to fix both spelling mistakes ("wrod") and grammatical/contextual errors ("I went too the store").

- **Mobile-Style Undo:** The signature feature: if the AI makes an unwanted correction, simply press Backspace once to instantly undo the correction and revert to your original typing.

- **Browser-Native Feel:** The extension disables the browser's default red-squiggle spellcheck, providing a clean, seamless, and native-feeling typing experience.

## Project Status

This project is currently in development. The core focus is on building a lightweight, on-device (TensorFlow.js) neural network that is small enough and fast enough to run entirely within the browser.

## Project Structure

- /ai_model/: Contains all the Python code for developing the AI.
- /data/: Holds the raw text data for training.
- /notebooks/: Jupyter Notebooks for data exploration and model prototyping.
- /src/: Final, clean .py scripts for data processing, training, and conversion.
- /.venv/: The isolated Python virtual environment.
- /extension/: Contains the JavaScript, HTML, and JSON files for the Chrome extension.
- /assets/: Icons and other static files.
- /js/: The extension's logic.
- /lib/: JavaScript libraries (e.g., TensorFlow.js).
- content.js: The main script that runs on pages.
- override.js: The script that disables browser spellcheck.
- /model/: The final, converted TensorFlow.js model (model.json, etc.).
- manifest.json: The extension's configuration file.

## Acknowledgements

- Development: This project is being developed with the assistance of Google's Gemini.
- Dataset: The AI model is trained on the British English (en_GB) corpus provided by the Leipzig Corpora Collection, Leipzig University. We are using the eng-uk_web-public_2018_1M (1 million sentences) dataset.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
