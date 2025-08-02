# Bedrock Proxy API Documentation

## Bedrock Proxy API Endpoints

When using the Bedrock proxy with custom URLs (e.g., Regeneron's air-api), the expected URL format is:

- **Non-streaming**: `https://air-api.regeneron.com/v1.0/model/bedrock/model/claude-35-sonnet/converse`
- **Streaming**: `https://air-api.regeneron.com/v1.0/model/bedrock/model/claude-35-sonnet/converse-stream`

The proxy transforms model IDs as follows:
- `us.anthropic.claude-3-5-sonnet-20241022-v2:0` → `claude-35-sonnet`
- `anthropic.claude-3-5-sonnet-20241022-v2:0` → `claude-35-sonnet`

## Regeneron API Expected Payload Format

According to Regeneron documentation, a successful request should follow this structure:

```bash
curl --location 'https://air-api.regeneron.com/v1.0/model/bedrock/model/claude-35-sonnet/converse' \
  --header 'accept: application/json' \
  --header 'authorization-token: ApplicationAccessToken or UserAccessToken' \
  --header 'Content-Type: application/json' \
  --data '{
    "system": [
      {
        "text": "You are a coding expert"
      }
    ],
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "text": "Write me a python code to print hello world"
          }
        ]
      }
    ],
    "inferenceConfig": {
      "maxTokens": 1000,
      "temperature": 0.5
    }
  }'
```

### Key Payload Requirements:

1. **Messages Structure**: Messages must have a `content` array containing objects with `text` fields
   ```json
   "messages": [
     {
       "role": "user",
       "content": [
         {
           "text": "Your message here"
         }
       ]
     }
   ]
   ```

2. **System Prompts**: System prompts go in a `system` array with objects containing `text` fields
   ```json
   "system": [
     {
       "text": "System prompt here"
     }
   ]
   ```

3. **Inference Configuration**: Parameters go in `inferenceConfig` with camelCase keys
   ```json
   "inferenceConfig": {
     "maxTokens": 1000,
     "temperature": 0.5
   }
   ```

4. **Headers Required**:
   - `accept: application/json`
   - `authorization-token: <token>`
   - `Content-Type: application/json`