# Ghost Type Corrector - Browser Extension

AI-powered Chrome extension providing invisible, contextual autocorrection.

## Installation

### Load Unpacked Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable **Developer mode** (toggle in top-right corner)
3. Click **Load unpacked**
4. Select the `extension` folder from this project
5. The extension icon should appear in your browser toolbar

### Verify Installation

1. Open the Chrome DevTools Console (F12)
2. Navigate to any webpage with text inputs
3. Check for these console messages:
   - `"Ghost Type Corrector: Content script loaded (Sandbox Mode)"`
   - `"Sandbox: Initializing AI resources..."`
   - `"Sandbox: Ready signal sent to content script"`
   - `"Content Script: Sandbox is ready! AI model loaded"`

## File Structure

```
extension/
├── manifest.json              # Extension configuration (Manifest V3)
├── sandbox.html               # Sandboxed page for AI model
├── README.md                  # This file
│
├── assets/
│   └── icons/                 # Extension icons (16, 48, 128 px)
│
├── data/
│   └── tokenizer_config.json  # Character-to-index mapping (from training)
│
├── js/
│   ├── content.js             # Main content script (handles UI interaction)
│   ├── override.js            # Disables browser spellcheck
│   └── sandbox_logic.js       # AI model inference logic (runs in sandbox)
│
├── js/lib/
│   └── tf.min.js              # TensorFlow.js library
│
└── model/
    ├── model.json             # TensorFlow.js model architecture
    └── group1-shard*.bin      # Model weights (10 files)
```

## Technical Architecture

### Manifest V3 Compliance

The extension uses Chrome's Manifest V3 with the following key features:

- **Content Scripts**: `content.js` and `override.js` inject into all web pages
- **Sandboxed Page**: `sandbox.html` runs the TensorFlow.js model in isolation
- **Web Accessible Resources**: Model files and scripts accessible to sandboxed page
- **Host Permissions**: `<all_urls>` to work on any website

### Sandbox Security

Chrome extensions use Content Security Policy (CSP) which prevents inline scripts and `eval()`. TensorFlow.js requires these features, so we:

1. Load the model in a **sandboxed iframe** (`sandbox.html`)
2. Use **postMessage API** for communication between content script and sandbox
3. Keep UI logic in content script, AI logic in sandbox

### Communication Flow

```
User types → content.js detects spacebar → sends word to sandbox
                                            ↓
                                    sandbox_logic.js runs AI model
                                            ↓
                                    returns corrected word
                                            ↓
content.js applies correction with case matching
```

## Key Features Implemented

### 1. Invisible Autocorrection
- Triggers on spacebar press
- No popup UI or underlines
- Preserves original case (uppercase, title case, lowercase)

### 2. Undo Feature
- Press Backspace once after correction to revert
- Only works immediately after correction
- Mobile keyboard-style behavior

### 3. Smart Element Detection
- Works on `<textarea>`, `<input type="text">`, `<input type="email">`, etc.
- Supports `contentEditable` elements
- Validates element before correction

### 4. Spellcheck Override
- Disables browser native spellcheck
- Handles dynamically added elements via MutationObserver
- Cleaner typing experience

## Chrome Extension Manifest V3 Updates

### Changes from Original Code

1. **manifest.json**:
   - Added `host_permissions` for `<all_urls>` (required in MV3)
   - Updated `version` to semantic versioning (1.0.0)
   - Added `override.js` to content_scripts array
   - Expanded web_accessible_resources with explicit file patterns
   - Added proper description

2. **content.js**:
   - Fixed origin validation for sandboxed iframe messages
   - Added explicit sandbox attribute to iframe
   - Improved error handling and null checks
   - Added support for more input types (email, url)
   - Enhanced logging for debugging
   - Added DOM ready check before initialization

3. **sandbox_logic.js**:
   - Added comprehensive error handling
   - Improved input validation
   - Added null/undefined checks throughout
   - Enhanced console logging for debugging
   - Added JSDoc comments for functions
   - Better tensor cleanup with tf.tidy()
   - Added maxSteps guard against infinite loops

4. **override.js**:
   - Added more input types (email, search, url)
   - Better code organization with functions
   - Enhanced logging
   - Added DOM ready check

5. **sandbox.html**:
   - Added proper HTML5 doctype and meta tags
   - Added semantic HTML structure
   - Improved code comments

## Debugging

### Common Issues

#### Extension Not Loading
- Check `chrome://extensions/` for errors
- Ensure all files are present (especially model files)
- Verify `manifest.json` is valid JSON

#### Model Not Loading
- Open DevTools Console and look for errors
- Check Network tab for failed requests
- Verify `model/model.json` and all `.bin` files exist
- Ensure `data/tokenizer_config.json` exists

#### No Corrections Happening
- Check console for "Sandbox is ready!" message
- Verify TensorFlow.js loaded: type `typeof tf` in sandbox console
- Test typing in different input fields
- Check that spacebar triggers the handler

### Console Logging

The extension logs important events:

```javascript
// Initialization
"Ghost Type Corrector: Initializing..."
"Content Script: Sandbox iframe created..."
"Sandbox: Tokenizer config loaded successfully"
"Sandbox: TensorFlow.js model loaded successfully"

// Runtime
"Content Script: Sandbox is ready! AI model loaded"
'Ghost Type Corrector: "teh" → "the"'
'Ghost Type Corrector: Undo - "the" → "teh"'
```

## Performance

- **Model Load Time**: 1-3 seconds (one-time on page load)
- **Inference Time**: 50-200ms per word
- **Memory Usage**: ~50MB (TensorFlow.js + model weights)
- **Model Size**: 3-5 MB total

## Browser Compatibility

- **Chrome**: 88+ (Manifest V3 support)
- **Edge**: 88+ (Chromium-based)
- **Brave**: Latest version
- **Opera**: Latest version

**Note**: Firefox uses a different extension format (Manifest V2) and would require porting.

## Development Tips

### Testing Changes

1. Make code changes
2. Go to `chrome://extensions/`
3. Click the **Reload** button on the extension card
4. Refresh the webpage you're testing on
5. Check DevTools Console for errors

### Modifying the AI Model

If you retrain the model:

1. Run the training scripts in `ai_model/src/`
2. Convert to TensorFlow.js with `convert_direct.py`
3. Copy `model/` folder to `extension/model/`
4. Copy `tokenizer_config.json` to `extension/data/`
5. Reload the extension

### Adding New Features

- **UI Changes**: Edit `content.js`
- **AI Logic**: Edit `sandbox_logic.js`
- **Configuration**: Edit `manifest.json`
- **Styling**: The extension has no UI, so no CSS needed

## Security Considerations

### Content Security Policy

The sandboxed iframe has restricted CSP:
- No access to extension APIs
- No access to page content
- Can only run JavaScript and load specified resources
- Communicates only via postMessage

### Message Validation

Both `content.js` and `sandbox_logic.js` validate messages:
- Check `event.source` matches expected iframe
- Verify message type before processing
- No eval() or dangerous string execution

### Permissions

The extension requests minimal permissions:
- `host_permissions: ["<all_urls>"]` - To inject on all pages
- No storage, cookies, or other sensitive APIs

## License

MIT License - See parent directory LICENSE file

## Support

For issues related to:
- **Extension not loading**: Check Chrome version and manifest.json
- **Model errors**: Verify model files are present and valid
- **Corrections not working**: Check console logs and verify model loaded

---

**Last Updated**: October 25, 2025
