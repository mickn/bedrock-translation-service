#!/bin/bash
# Start LiteLLM with custom Bedrock Converse provider

echo "Starting LiteLLM with custom Bedrock Converse provider..."

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Use our Python wrapper that properly loads the custom provider
python3 litellm_server.py --config litellm_config_minimal.yaml --port 4000