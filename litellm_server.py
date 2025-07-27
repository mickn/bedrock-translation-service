#!/usr/bin/env python3
"""
LiteLLM server with custom Bedrock Converse provider
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import and register our custom provider BEFORE importing litellm
import custom_bedrock_provider

# Now import litellm to verify provider is registered
import litellm

# Start LiteLLM server
from litellm.proxy.proxy_cli import run_server
import argparse

if __name__ == "__main__":
    # Create parser for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="litellm_config_minimal.yaml")
    parser.add_argument("--port", type=int, default=4000)
    args = parser.parse_args()
    
    # Verify custom provider is registered
    print(f"Starting LiteLLM server with custom Bedrock Converse provider")
    print(f"Config: {args.config}")
    print(f"Port: {args.port}")
    print(f"Custom providers registered: {litellm.custom_provider_map}")
    
    # Set up sys.argv for LiteLLM
    sys.argv = ['litellm', '--config', args.config, '--port', str(args.port)]
    
    # Run the server
    run_server()