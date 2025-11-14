# API Key Configuration Feature

## Overview

Users can now configure their API keys directly through the TraceMind MCP Server UI. These user-provided keys will override environment variables for the current session.

## What's New

### 1. Settings Tab (⚙️)
A new Settings tab has been added as the first tab in the UI where users can:
- Enter their **Google Gemini API Key**
- Enter their **HuggingFace Token**
- Save keys for the current session
- Clear session keys and revert to environment variables

### 2. Session-Only Storage
- API keys are stored in Gradio's session state
- Keys are **NOT** persisted to disk or cookies
- Keys are automatically cleared when the browser session ends
- Each user session has its own isolated key storage

### 3. Automatic Key Validation
- Gemini API keys are validated when saved by creating a test client
- Invalid keys are rejected with clear error messages
- Users receive immediate feedback on key validity

## How to Use

### Option 1: Configure via UI (Recommended)

1. **Navigate to Settings Tab**
   - Open the TraceMind MCP Server app
   - Click on the "⚙️ Settings" tab (first tab)

2. **Enter Your Keys**
   - **Gemini API Key**: Get from https://aistudio.google.com/app/apikey
   - **HuggingFace Token**: Get from https://huggingface.co/settings/tokens

3. **Save Keys**
   - Click "💾 Save API Keys for This Session"
   - Wait for validation confirmation
   - Keys are now active for all tools

4. **Use Any Tool**
   - Navigate to any other tab (Analyze Leaderboard, Debug Trace, etc.)
   - Tools will automatically use your configured keys
   - No additional configuration needed

### Option 2: Environment Variables (Still Supported)

You can still use environment variables as before:

```bash
export GEMINI_API_KEY="your-key-here"
export HF_TOKEN="your-token-here"
```

**Note**: UI-configured keys always override environment variables.

## Technical Details

### Architecture Changes

#### 1. UI Layer (`app.py`)
- Added Settings tab with key input forms
- Implemented session state management with `gr.State()`
- Updated all tool functions to accept API keys as parameters
- Added key validation and error handling

#### 2. Tool Layer (`mcp_tools.py`)
- Updated all functions to accept optional `hf_token` parameter
- Modified `load_dataset()` calls to use user-provided tokens
- Added fallback to environment variables when no token provided
- Functions updated:
  - `analyze_leaderboard()`
  - `debug_trace()`
  - `compare_runs()`
  - `get_dataset()`
  - `get_leaderboard_data()` (MCP Resource)
  - `get_trace_data()` (MCP Resource)

#### 3. Client Layer (`gemini_client.py`)
- `GeminiClient.__init__()` already supported optional `api_key` parameter
- No changes needed - already designed for key override

### Key Features

1. **Priority Order**:
   ```
   User-provided key (UI) > Environment variable > Error
   ```

2. **Validation**:
   - Gemini keys: Validated by creating test `GeminiClient`
   - HF tokens: Accepted without validation (validated on first use)

3. **Error Handling**:
   - Clear error messages when keys are missing
   - Helpful prompts to configure keys in Settings tab
   - Validation errors shown immediately

4. **Session Management**:
   - Keys stored in `gr.State()` (Gradio session state)
   - Isolated per-user in multi-user environments
   - Automatically cleared on session end

## Security Considerations

### ✅ Secure Practices

1. **No Persistence**: Keys are never written to disk
2. **Session Isolation**: Each user has isolated key storage
3. **Password Fields**: Keys displayed as `type="password"` (hidden)
4. **No Logging**: Keys not logged or exposed in error messages

### ⚠️ Security Notes

- **HTTPS Required**: Always use HTTPS in production to protect keys in transit
- **Public Spaces**: Be cautious using on public HuggingFace Spaces
- **Shared Environments**: Each browser session is isolated, but server has access
- **Recommendation**: Use environment variables for production deployments

## Examples

### Example 1: First-Time User

```
1. User opens app (no env vars set)
2. User sees "⚠️ Status: No API key configured" in Settings
3. User enters Gemini API key and HF token
4. User clicks "Save API Keys"
5. User sees "✅ Gemini API key validated and saved"
6. User switches to "Analyze Leaderboard" tab
7. Tool works using user-provided keys
```

### Example 2: Overriding Environment Variables

```
1. User has GEMINI_API_KEY set in environment
2. User wants to test with a different key
3. User enters new key in Settings tab
4. User clicks "Save API Keys"
5. All tools now use the new key (not the env var)
6. User clicks "Clear Session Keys" to revert
7. Tools now use environment variable again
```

### Example 3: Error Handling

```
1. User enters invalid Gemini API key
2. User clicks "Save API Keys"
3. User sees "❌ Gemini API key invalid: [error message]"
4. User corrects the key and tries again
5. User sees "✅ Gemini API key validated and saved"
```

## API Changes

### Function Signatures

All tool functions now accept optional API key parameters:

```python
# Before
async def analyze_leaderboard(
    gemini_client: GeminiClient,
    leaderboard_repo: str = "...",
    ...
) -> str:

# After
async def analyze_leaderboard(
    gemini_client: GeminiClient,
    leaderboard_repo: str = "...",
    ...,
    hf_token: Optional[str] = None  # NEW
) -> str:
```

### Backward Compatibility

- ✅ All existing code continues to work
- ✅ Environment variables still supported
- ✅ No breaking changes to MCP protocol
- ✅ Optional parameters have sensible defaults

## Testing Checklist

- [x] UI renders Settings tab correctly
- [x] Gemini API key input works (password field)
- [x] HF token input works (password field)
- [x] Save button validates and stores keys
- [x] Clear button reverts to environment variables
- [ ] All tools use user-provided Gemini key
- [ ] All tools use user-provided HF token
- [ ] Invalid Gemini key shows error
- [ ] Missing keys show helpful error messages
- [ ] Session isolation works in multi-user scenario
- [ ] Keys cleared on browser close

## Future Enhancements

1. **Key Persistence** (Optional):
   - Add opt-in browser localStorage support
   - Warning about security implications

2. **Multiple Key Profiles**:
   - Save multiple key configurations
   - Quick switch between profiles

3. **Usage Tracking**:
   - Show API usage per session
   - Cost estimation based on usage

4. **Token Expiration**:
   - Detect expired HF tokens
   - Prompt for refresh

## Troubleshooting

### Keys Not Working

**Problem**: Tools show "No API key configured" error

**Solutions**:
1. Check you clicked "Save API Keys" button
2. Look for validation success message
3. Try refreshing the page and re-entering keys
4. Check browser console for errors

### Validation Fails

**Problem**: "❌ Gemini API key invalid" error

**Solutions**:
1. Verify key copied correctly (no extra spaces)
2. Check key is active at https://aistudio.google.com/app/apikey
3. Ensure you have API quota remaining
4. Try generating a new key

### Dataset Access Denied

**Problem**: "Error loading dataset: Access denied"

**Solutions**:
1. Verify HF token is correct
2. Check token has read permissions
3. Ensure dataset is public or you have access
4. Try using a new token

## Support

For issues or questions:
- Check the Settings tab for status messages
- Review error messages in tool outputs
- Open an issue on GitHub with:
  - Steps to reproduce
  - Error messages (DO NOT include actual API keys)
  - Browser and OS information
