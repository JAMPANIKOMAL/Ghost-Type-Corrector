# Quick Testing Guide - Ghost Type Corrector Extension

## Load Extension in Chrome

1. Open Chrome browser
2. Navigate to: `chrome://extensions/`
3. Toggle **Developer mode** ON (top-right corner)
4. Click **Load unpacked**
5. Browse to: `C:\Users\jampa\Videos\Ghost Type Corrector\extension`
6. Click **Select Folder**

## Verify Installation ✅

### In chrome://extensions/ page:
- Extension card shows "Ghost Type Corrector"
- Version shows "1.0.0"
- Status shows "Enabled"
- No red error messages

### Test on a webpage:

1. Navigate to any website (e.g., https://www.google.com)
2. Open Chrome DevTools (Press F12)
3. Go to **Console** tab

**Expected console output:**
```
Ghost Type Corrector: Content script loaded (Sandbox Mode)
Content Script: Sandbox iframe created and listener attached
Sandbox: Initializing AI resources...
Sandbox: Fetching tokenizer config from: chrome-extension://[id]/data/tokenizer_config.json
Sandbox: Tokenizer config loaded successfully
  Max sequence length: [number]
  Vocabulary size: [number]
Sandbox: Loading TensorFlow.js model from: chrome-extension://[id]/model/model.json
Sandbox: TensorFlow.js model loaded successfully
Sandbox: Warming up model...
Sandbox: Ready signal sent to content script
Content Script: Sandbox is ready! AI model loaded
Ghost Type Corrector: MutationObserver started for dynamic elements
Ghost Type Corrector: Disabled spellcheck on [n] input elements
```

## Test Functionality

### Test 1: Basic Autocorrect
1. Click in the Google search box
2. Type: `teh` (with a space after)
3. **Expected**: Word changes to `the`
4. **Console shows**: `Ghost Type Corrector: "teh" → "the"`

### Test 2: Undo Feature
1. Immediately after a correction, press Backspace once
2. **Expected**: Word reverts to original typo
3. **Console shows**: `Ghost Type Corrector: Undo - "the" → "teh"`

### Test 3: Case Preservation
1. Type: `TEH ` (all caps)
2. **Expected**: Corrects to `THE` (preserves case)

1. Type: `Teh ` (title case)
2. **Expected**: Corrects to `The`

### Test 4: Different Input Types
Try typing in:
- Textarea elements
- Email input fields
- Search boxes
- Content editable divs

### Test 5: Spellcheck Disabled
1. Look at any text input field
2. **Expected**: No red underlines for misspelled words
3. Browser's native spellcheck should be disabled

## Common Test Websites

- **Google Search**: https://www.google.com
- **Twitter Compose**: https://twitter.com/compose/tweet
- **Gmail**: https://mail.google.com
- **GitHub Issues**: https://github.com (create new issue)
- **Reddit**: https://reddit.com (create post)

## Troubleshooting

### Extension doesn't load
- Check that all files are present in the extension folder
- Verify `model/` folder contains model.json and 10 .bin files
- Verify `data/tokenizer_config.json` exists

### Model not loading
- Check Console for error messages
- Look for Network errors in DevTools Network tab
- Verify TensorFlow.js loaded: in Console type `chrome.runtime.getURL('js/lib/tf.min.js')`

### No corrections happening
- Ensure "Sandbox is ready!" message appeared in console
- Try typing in different input fields
- Check that you're pressing spacebar after the word
- Verify the word is actually a common typo the model knows

### Console errors about CSP
- This is normal for the main page
- The sandbox should load without CSP errors
- Check the iframe's console separately

## Performance Benchmarks

- **Initial Load**: 1-3 seconds (one-time per page)
- **Per-word Prediction**: 50-200ms
- **Memory Usage**: ~50MB
- **Model Size**: 3-5 MB

## Debug Mode

To see more detailed logging, open DevTools Console and type:
```javascript
// Filter to only Ghost Type Corrector messages
console.log("Filtering: Ghost Type Corrector")
```

## Unload/Reload Extension

After making changes:
1. Go to `chrome://extensions/`
2. Click the **Reload** icon on the extension card
3. Refresh any open webpages to reload the content script

---

**Happy Testing!** 🚀

If you encounter any issues, check:
1. Console for error messages
2. `CHANGES.md` for known fixes
3. `extension/README.md` for detailed documentation
