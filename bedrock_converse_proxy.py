#!/usr/bin/env python3
"""
Simple proxy server that intercepts Claude API calls and forwards them to AWS Bedrock Converse API
"""

import json
import os
import re
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import boto3
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if using custom Bedrock URL
BEDROCK_CUSTOM_URL = os.environ.get('BEDROCK_CUSTOM_URL')
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')

# AWS Bedrock client (only used if no custom URL)
bedrock_client = None
if not BEDROCK_CUSTOM_URL:
    bedrock_client = boto3.client(
        'bedrock-runtime',
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
    )

class BedrockConverseProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the Bedrock Converse proxy"""
    
    def do_POST(self):
        """Handle POST requests to /v1/messages and /model/*/invoke endpoints"""
        path = urlparse(self.path).path
        
        # Handle different endpoint patterns
        if path == '/v1/messages':
            self.handle_messages_endpoint()
        elif re.match(r'^/model/.+/(invoke|invoke-with-response-stream)$', path):
            self.handle_invoke_endpoint(path)
        else:
            self.send_error(404, f"Endpoint {path} not found")
            return
        
    def handle_messages_endpoint(self):
        """Handle /v1/messages endpoint"""
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
            if BEDROCK_CUSTOM_URL:
                # Use custom URL with authorization header
                headers = {
                    'Content-Type': 'application/json',
                    'authorization-token': ACCESS_TOKEN
                }
                api_response = requests.post(
                    f"{BEDROCK_CUSTOM_URL.rstrip('/')}/converse",
                    json=bedrock_request,
                    headers=headers
                )
                api_response.raise_for_status()
                response = api_response.json()
            else:
                # Use standard boto3 client
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
    
    def handle_invoke_endpoint(self, path):
        """Handle /model/*/invoke and /model/*/invoke-with-response-stream endpoints"""
        # Extract model ID from path
        match = re.match(r'^/model/(.+)/(invoke|invoke-with-response-stream)$', path)
        if not match:
            self.send_error(400, "Invalid invoke endpoint")
            return
        
        model_id = match.group(1)
        is_streaming = match.group(2) == 'invoke-with-response-stream'
        
        # Read request body
        content_length = int(self.headers.get('Content-Length', 0))
        request_body = self.rfile.read(content_length)
        
        try:
            # Parse Anthropic format request (used by invoke endpoints)
            anthropic_request = json.loads(request_body)
            
            # Convert Anthropic format to Bedrock Converse format
            bedrock_messages = []
            system_content = None
            
            # Handle system prompt if present
            if 'system' in anthropic_request:
                system_text = anthropic_request['system']
                if isinstance(system_text, str) and system_text.strip():
                    system_content = [{'text': system_text}]
            
            # Convert messages from Anthropic format
            for msg in anthropic_request.get('messages', []):
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
            
            # Build Bedrock Converse request
            bedrock_request = {
                'modelId': model_id,
                'messages': bedrock_messages,
                'inferenceConfig': {
                    'maxTokens': anthropic_request.get('max_tokens', 4096),
                    'temperature': anthropic_request.get('temperature', 0.7)
                }
            }
            
            if system_content:
                bedrock_request['system'] = system_content
            
            # Add stop sequences if present
            if 'stop_sequences' in anthropic_request:
                bedrock_request['inferenceConfig']['stopSequences'] = anthropic_request['stop_sequences']
            
            # Debug logging
            print(f"\nInvoke request received on {path}:")
            print(json.dumps(anthropic_request, indent=2))
            print(f"\nBedrock Converse request to send:")
            print(json.dumps(bedrock_request, indent=2))
            
            if is_streaming:
                # Handle streaming response
                self.handle_streaming_response(bedrock_request, anthropic_request)
            else:
                # Handle non-streaming response
                if BEDROCK_CUSTOM_URL:
                    # Use custom URL with authorization header
                    headers = {
                        'Content-Type': 'application/json',
                        'authorization-token': ACCESS_TOKEN
                    }
                    api_response = requests.post(
                        f"{BEDROCK_CUSTOM_URL.rstrip('/')}/converse",
                        json=bedrock_request,
                        headers=headers
                    )
                    api_response.raise_for_status()
                    response = api_response.json()
                else:
                    # Use standard boto3 client
                    response = bedrock_client.converse(**bedrock_request)
                
                # Convert response back to Anthropic format expected by invoke
                content_blocks = []
                for content_item in response['output']['message']['content']:
                    if 'text' in content_item:
                        content_blocks.append({
                            'type': 'text',
                            'text': content_item['text']
                        })
                
                anthropic_response = {
                    'id': f"msg_{os.urandom(12).hex()}",
                    'type': 'message',
                    'role': 'assistant',
                    'content': content_blocks,
                    'model': model_id,
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
                self.wfile.write(json.dumps(anthropic_response).encode())
            
        except Exception as e:
            # Log the error for debugging
            print(f"Error in invoke endpoint: {e}")
            print(f"Request was: {request_body.decode()}")
            
            # Send error response in Anthropic format
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
    
    def handle_streaming_response(self, bedrock_request, anthropic_request):
        """Handle streaming response for invoke-with-response-stream"""
        try:
            # Call Bedrock ConverseStream API
            if BEDROCK_CUSTOM_URL:
                # Use custom URL with authorization header for streaming
                headers = {
                    'Content-Type': 'application/json',
                    'authorization-token': ACCESS_TOKEN,
                    'Accept': 'text/event-stream',
                    'Accept-Encoding': 'gzip, deflate'
                }
                api_response = requests.post(
                    f"{BEDROCK_CUSTOM_URL.rstrip('/')}/converse-stream",
                    json=bedrock_request,
                    headers=headers,
                    stream=True
                )
                api_response.raise_for_status()
                
                # Parse streaming response
                response = {'stream': []}
                buffer = ""
                
                # Handle SSE (Server-Sent Events) format
                for chunk in api_response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        buffer += chunk
                        lines = buffer.split('\n')
                        
                        # Process complete lines
                        for i in range(len(lines) - 1):
                            line = lines[i].strip()
                            if line.startswith('data: '):
                                try:
                                    event_data = json.loads(line[6:])
                                    response['stream'].append(event_data)
                                except json.JSONDecodeError:
                                    # Skip malformed lines
                                    print(f"Warning: Could not parse event data: {line}")
                                    continue
                        
                        # Keep the last incomplete line in buffer
                        buffer = lines[-1]
                
                # Process any remaining data in buffer
                if buffer.strip().startswith('data: '):
                    try:
                        event_data = json.loads(buffer.strip()[6:])
                        response['stream'].append(event_data)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse final event data: {buffer}")
            else:
                # Use standard boto3 client
                response = bedrock_client.converse_stream(**bedrock_request)
            
            # Send headers for streaming response
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()
            
            # Stream events
            message_start_sent = False
            content_blocks = []
            current_index = 0
            total_text = ""
            
            for event in response['stream']:
                if 'messageStart' in event:
                    # Send message_start event
                    message_start = {
                        'type': 'message_start',
                        'message': {
                            'id': f"msg_{os.urandom(12).hex()}",
                            'type': 'message',
                            'role': 'assistant',
                            'content': [],
                            'model': bedrock_request['modelId'],
                            'stop_reason': None,
                            'stop_sequence': None,
                            'usage': {
                                'input_tokens': 0,
                                'output_tokens': 0
                            }
                        }
                    }
                    self.wfile.write(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode())
                    message_start_sent = True
                    
                elif 'contentBlockStart' in event:
                    # Send content_block_start event
                    content_block_start = {
                        'type': 'content_block_start',
                        'index': current_index,
                        'content_block': {
                            'type': 'text',
                            'text': ''
                        }
                    }
                    self.wfile.write(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())
                    
                elif 'contentBlockDelta' in event:
                    # Send content_block_delta event
                    delta_text = event['contentBlockDelta']['delta'].get('text', '')
                    total_text += delta_text
                    content_block_delta = {
                        'type': 'content_block_delta',
                        'index': current_index,
                        'delta': {
                            'type': 'text_delta',
                            'text': delta_text
                        }
                    }
                    self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(content_block_delta)}\n\n".encode())
                    
                elif 'contentBlockStop' in event:
                    # Send content_block_stop event
                    content_block_stop = {
                        'type': 'content_block_stop',
                        'index': current_index
                    }
                    self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())
                    current_index += 1
                    
                elif 'messageStop' in event:
                    # Send message_delta and message_stop events
                    stop_reason = event['messageStop'].get('stopReason', 'end_turn')
                    
                    # Send message_delta with final usage
                    if 'metadata' in event and 'usage' in event['metadata']:
                        usage = event['metadata']['usage']
                        message_delta = {
                            'type': 'message_delta',
                            'delta': {
                                'stop_reason': stop_reason,
                                'stop_sequence': None
                            },
                            'usage': {
                                'output_tokens': usage.get('outputTokens', 0)
                            }
                        }
                        self.wfile.write(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode())
                    
                    # Send message_stop
                    message_stop = {
                        'type': 'message_stop'
                    }
                    self.wfile.write(f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode())
                    
                elif 'metadata' in event:
                    # Handle metadata events (usage information)
                    if 'usage' in event['metadata']:
                        usage = event['metadata']['usage']
                        # Usage is typically included in message_delta or message_stop
            
            # Ensure the response is flushed
            self.wfile.flush()
            
        except Exception as e:
            print(f"Error in streaming response: {e}")
            # Try to send error event if possible
            try:
                error_event = {
                    'type': 'error',
                    'error': {
                        'type': 'api_error',
                        'message': str(e)
                    }
                }
                self.wfile.write(f"event: error\ndata: {json.dumps(error_event)}\n\n".encode())
                self.wfile.flush()
            except:
                pass
    
    def log_message(self, format, *args):
        """Override to reduce logging"""
        return

def run_proxy(port=8000):
    """Run the proxy server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, BedrockConverseProxyHandler)
    
    print(f"Bedrock Converse proxy server running on port {port}")
    print(f"Supported endpoints:")
    print(f"  - Claude API: http://localhost:{port}/v1/messages")
    print(f"  - Bedrock invoke: http://localhost:{port}/model/{{model-id}}/invoke")
    print(f"  - Bedrock invoke-stream: http://localhost:{port}/model/{{model-id}}/invoke-with-response-stream")
    print("Press Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()

if __name__ == "__main__":
    port = int(os.environ.get('PROXY_PORT', 8000))
    run_proxy(port)