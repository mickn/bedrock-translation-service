#!/usr/bin/env python3
"""
Test script for custom Bedrock provider with LiteLLM
"""

import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_litellm_proxy():
    """Test LiteLLM proxy with custom provider"""
    
    # LiteLLM proxy endpoint
    url = "http://localhost:4000/chat/completions"
    
    # Request in OpenAI format (which LiteLLM expects)
    request_data = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": "Say 'Hello from custom Bedrock provider!' and nothing else."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.5
    }
    
    print("Testing LiteLLM with custom Bedrock provider...")
    print(f"URL: {url}")
    print(f"Request: {json.dumps(request_data, indent=2)}")
    
    try:
        response = requests.post(
            url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"\nResponse status: {response.status_code}")
        
        if response.status_code == 200:
            response_data = response.json()
            print("\n✅ SUCCESS!")
            print(f"Response: {json.dumps(response_data, indent=2)}")
            
            # Extract the actual message
            if 'choices' in response_data and len(response_data['choices']) > 0:
                message = response_data['choices'][0]['message']['content']
                print(f"\nAssistant said: {message}")
        else:
            print(f"\n❌ FAILED!")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to LiteLLM proxy.")
        print("Make sure it's running: ./start_litellm_with_custom_provider.sh")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    test_litellm_proxy()