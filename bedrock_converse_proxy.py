#!/usr/bin/env python3
"""
Simple proxy server that intercepts Claude API calls and forwards them to AWS Bedrock Converse API
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# AWS Bedrock client
bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
)

class BedrockConverseProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Bedrock Converse proxy"""
    
    def do_POST(self):
        """Handle POST requests to /v1/messages"""
        path = urlparse(self.path).path
        
        # Only handle /v1/messages endpoint (Claude API format)
        if path != '/v1/messages':
            self.send_error(404, f"Endpoint {path} not found")
            return
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length)
        
        try:
            # Parse Claude API request
            claude_request = json.loads(request_body)
            
            # Convert to Bedrock Converse format
            bedrock_messages = []
            system_content = None
            
            # Handle system message if present
            if 'system' in claude_request:
                system_text = claude_request['system']
                # Only add system if it's a non-empty string
                if isinstance(system_text, str) and system_text.strip():
                    system_content = [{'text': system_text}]
            
            # Convert messages
            for msg in claude_request.get('messages', []):
                role = msg['role']
                content = msg['content']
                
                # Handle content format
                if isinstance(content, str):
                    bedrock_content = [{'text': content}]
                elif isinstance(content, list):
                    bedrock_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            bedrock_content.append({'text': item['text']})
                        elif isinstance(item, str):
                            bedrock_content.append({'text': item})
                else:
                    bedrock_content = [{'text': str(content)}]
                
                bedrock_messages.append({
                    'role': role,
                    'content': bedrock_content
                })
            
            # Build Bedrock request
            bedrock_request = {
                'modelId': 'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
                'messages': bedrock_messages,
                'inferenceConfig': {
                    'maxTokens': claude_request.get('max_tokens', 4096),
                    'temperature': claude_request.get('temperature', 0.7)
                }
            }
            
            if system_content:
                bedrock_request['system'] = system_content
            
            # Debug logging
            print(f"\nClaude request received:")
            print(json.dumps(claude_request, indent=2))
            print(f"\nBedrock request to send:")
            print(json.dumps(bedrock_request, indent=2))
            
            # Call Bedrock Converse API
            response = bedrock_client.converse(**bedrock_request)
            
            # Convert response back to Claude format
            content = response['output']['message']['content'][0]['text']
            
            claude_response = {
                'id': f"msg_{os.urandom(12).hex()}",
                'type': 'message',
                'role': 'assistant',
                'content': content,
                'model': claude_request.get('model', 'claude-3-5-sonnet-20241022'),
                'stop_reason': response.get('stopReason', 'end_turn'),
                'stop_sequence': None,
                'usage': {
                    'input_tokens': response['usage'].get('inputTokens', 0),
                    'output_tokens': response['usage'].get('outputTokens', 0)
                }
            }
            
            # Send response
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(claude_response).encode())
            
        except Exception as e:
            # Log the error for debugging
            print(f"Error: {e}")
            print(f"Request was: {json.dumps(claude_request, indent=2)}")
            
            # Send error response
            error_response = {
                'type': 'error',
                'error': {
                    'type': 'api_error',
                    'message': str(e)
                }
            }
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override to reduce logging"""
        return

def run_proxy(port=8000):
    """Run the proxy server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, BedrockConverseProxyHandler)
    
    print(f"Bedrock Converse proxy server running on port {port}")
    print(f"Claude API endpoint: http://localhost:{port}/v1/messages")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()

if __name__ == "__main__":
    port = int(os.environ.get('PROXY_PORT', 8000))
    run_proxy(port)