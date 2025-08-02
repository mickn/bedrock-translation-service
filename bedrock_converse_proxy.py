#!/usr/bin/env python3
"""
Bedrock proxy that transparently maps:
  • POST /v1/messages   -> AWS Bedrock Converse / ConverseStream
  • POST /model/{id}/invoke[‑with‑response‑stream]
      – with "anthropic_version" -> InvokeModel / InvokeModelWithResponseStream
      – otherwise                -> Converse / ConverseStream
"""

import json, os, re, uuid, traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import boto3, requests
from dotenv import load_dotenv

load_dotenv()

REGION          = os.getenv("AWS_DEFAULT_REGION", "us‑east‑1")
CUSTOM_URL      = os.getenv("BEDROCK_CUSTOM_URL")   # e.g. a private VPC endpoint
ACCESS_TOKEN    = os.getenv("ACCESS_TOKEN")         # optional bearer for custom URL
DEFAULT_MODEL   = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"

client = boto3.client("bedrock-runtime", region_name=REGION) if not CUSTOM_URL else None


# --------------------------------------------------------------------------- #
# Helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _uuid() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"

def _write_json(handler, code, obj):
    data = json.dumps(obj).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)

def verify_regeneron_payload_structure(payload: dict) -> None:
    """
    Verify the payload matches Regeneron's expected structure.
    Raises ValueError if structure is incorrect.
    """
    # Check messages structure
    if "messages" in payload:
        for i, msg in enumerate(payload["messages"]):
            if "role" not in msg:
                raise ValueError(f"Message {i} missing 'role'")
            if "content" not in msg:
                raise ValueError(f"Message {i} missing 'content' array")
            if not isinstance(msg["content"], list):
                raise ValueError(f"Message {i} 'content' must be an array")
            for j, content_item in enumerate(msg["content"]):
                if not isinstance(content_item, dict) or "text" not in content_item:
                    raise ValueError(f"Message {i} content[{j}] must be an object with 'text' field")
    
    # Check system structure if present
    if "system" in payload and payload["system"] is not None:
        if not isinstance(payload["system"], list):
            raise ValueError("'system' must be an array")
        for i, sys_item in enumerate(payload["system"]):
            if not isinstance(sys_item, dict) or "text" not in sys_item:
                raise ValueError(f"system[{i}] must be an object with 'text' field")
    
    # Check inferenceConfig structure if present
    if "inferenceConfig" in payload and payload["inferenceConfig"] is not None:
        if not isinstance(payload["inferenceConfig"], dict):
            raise ValueError("'inferenceConfig' must be an object")
        # Check for correct camelCase keys
        valid_keys = {"maxTokens", "temperature", "topP", "topK", "stopSequences"}
        for key in payload["inferenceConfig"]:
            if key not in valid_keys:
                raise ValueError(f"Invalid inferenceConfig key: {key}. Valid keys: {valid_keys}")

def _bedrock_http(path: str, payload: dict, stream: bool = False):
    """
    Call a private Bedrock data‑plane reverse proxy – if one is configured
    (needed in some regulated environments).  Otherwise we rely on boto3.
    """
    # Headers as per Regeneron documentation
    headers = {
        "Content-Type": "application/json",
        "accept": "application/json", 
        "authorization-token": ACCESS_TOKEN
    }
    url = CUSTOM_URL.rstrip("/") + path

    # Log the request details for debugging
    print(f"\n=== Bedrock HTTP Request ===")
    print(f"URL: {url}")
    print(f"Path argument: {path}")
    print(f"CUSTOM_URL base: {CUSTOM_URL}")
    print(f"Headers: {headers}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print(f"Stream: {stream}")
    
    # Extra debug - check message structure
    if "messages" in payload and payload["messages"]:
        first_msg = payload["messages"][0]
        print(f"First message structure: {json.dumps(first_msg, indent=2)}")
        print(f"First message keys: {list(first_msg.keys())}")
    
    # Verify payload structure matches Regeneron expectations
    try:
        verify_regeneron_payload_structure(payload)
        print("✓ Payload structure verified")
    except ValueError as e:
        print(f"✗ Payload structure error: {e}")
        
    print("===========================\n")

    resp = requests.post(url, json=payload, headers=headers, stream=stream, timeout=90)
    
    print(f"Response status code: {resp.status_code}")
    print(f"Response headers: {dict(resp.headers)}")

    # If we get a 400, log the response body
    if resp.status_code == 400:
        print(f"\n=== 400 Bad Request Response ===")
        print(f"Status: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        try:
            print(f"Body: {resp.text}")
        except:
            print("Could not read response body")
        print("================================\n")

    resp.raise_for_status()
    return resp


def anthropic_to_bedrock_messages(msgs):
    """
    Normalise Anthropic message blocks into Bedrock's message schema
    (list of {role, content:[{text:..}]})
    """
    out = []
    for m in msgs:
        role, content = m["role"], m["content"]
        blocks = []
        if isinstance(content, str):
            blocks.append({"text": content})
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    blocks.append({"text": item})
                elif isinstance(item, dict) and item.get("type") == "text":
                    blocks.append({"text": item["text"]})
        else:                                   # fallback
            blocks.append({"text": str(content)})
        out.append({"role": role, "content": blocks})
    return out


def build_converse_request(body):
    """
    Turn an Anthropic‑style body into a Bedrock Converse request.
    Forward ALL recognised inference params.
    """
    infer_defaults = {
        "maxTokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "topP": body.get("top_p"),
        "topK": body.get("top_k"),
        "stopSequences": body.get("stop_sequences"),
    }
    # prune Nones so we don’t expose nulls
    infer = {k: v for k, v in infer_defaults.items() if v is not None}

    req = {
        "modelId": body.get("model", DEFAULT_MODEL),
        "messages": anthropic_to_bedrock_messages(body.get("messages", [])),
    }
    
    # Only add inferenceConfig if there are actual values
    if infer:
        req["inferenceConfig"] = infer
    
    # Only add system if it's not empty
    if sys := body.get("system"):
        # Handle system as either string or list
        if isinstance(sys, str):
            # Skip empty string system prompts
            if sys.strip():
                req["system"] = [{"text": sys}]
        elif isinstance(sys, list):
            # System is already a list, use as-is if not empty
            if sys:
                req["system"] = sys
    
    return req


def build_invoke_request(body):
    """
    Claude v1/v2 invoke_model request – retain anthropic_version AND camel‑case
    content list because invoke_model expects the *original* snake‑case JSON.
    """
    invoke_body = {
        "anthropic_version": body["anthropic_version"],
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "top_k": body.get("top_k"),
        "stop_sequences": body.get("stop_sequences"),
        "system": body.get("system"),
        "messages": body["messages"],
    }
    # prune nulls
    invoke_body = {k: v for k, v in invoke_body.items() if v is not None}
    return invoke_body


# --------------------------------------------------------------------------- #
# HTTP handler                                                                #
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    # -------- high‑level router ------------------------------------------- #
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw or "{}")

            path = urlparse(self.path).path
            if path == "/v1/messages":
                self._handle_messages(body)
            else:
                m = re.fullmatch(r"/model/([^/]+)/(?P<op>invoke|invoke-with-response-stream)", path)
                if not m:
                    raise ValueError(f"Unsupported endpoint {path}")
                model_id = m.group(1)
                streaming = m.group("op") == "invoke-with-response-stream"
                self._handle_invoke(model_id, body, streaming)
        except Exception as exc:
            traceback.print_exc()
            _write_json(self, 500, {"type": "error", "error": {"type": "api_error", "message": str(exc)}})

    # -------- v1/messages  ------------------------------------------------- #
    def _handle_messages(self, body):
        bed_req = build_converse_request(body)
        # choose transport
        if CUSTOM_URL:
            # Extract model name from the modelId in the request
            model_id = bed_req.get("modelId", DEFAULT_MODEL)
            model_name = model_id
            
            if '.' in model_name:
                # Remove region/provider prefix
                model_name = model_name.split('.', 1)[-1]
            if ':' in model_name:
                # Remove version suffix
                model_name = model_name.split(':', 1)[0]
            # Remove date suffix if present (e.g., -20241022-v2)
            model_name = re.sub(r'-\d{8}-v\d+$', '', model_name)
            # Shorten claude-3-5-sonnet to claude-35-sonnet for the API
            model_name = model_name.replace('claude-3-5-sonnet', 'claude-35-sonnet')
            
            bedrock_path = f'/model/{model_name}/converse'
            
            print(f"\n=== _handle_messages Debug ===")
            print(f"Model name: {model_name}")
            print(f"Bedrock path: {bedrock_path}")
            print(f"Payload being sent:")
            print(json.dumps(bed_req, indent=2))
            print("=============================\n")
            
            resp = _bedrock_http(bedrock_path, bed_req).json()
        else:
            resp = client.converse(**bed_req)
        # build Anthropic‑style response
        out_txt = resp["output"]["message"]["content"][0]["text"]
        result = {
            "id": _uuid(),
            "type": "message",
            "role": "assistant",
            "content": out_txt,
            "model": bed_req["modelId"],
            "stop_reason": resp.get("stopReason", "end_turn"),
            "usage": {
                "input_tokens": resp["usage"].get("inputTokens", 0),
                "output_tokens": resp["usage"].get("outputTokens", 0),
            },
        }
        _write_json(self, 200, result)

    # -------- /model/{id}/invoke  ----------------------------------------- #
    def _handle_invoke(self, model_id, body, streaming: bool):
        """
        ALWAYS use Converse API, regardless of anthropic_version presence.
        """
        print(f"\n=== Handle Invoke Request ===")
        print(f"Model ID: {model_id}")
        print(f"Streaming: {streaming}")
        print(f"Incoming body: {json.dumps(body, indent=2)}")
        print("===========================\n")

        # Always build a Converse request
        payload = build_converse_request({**body, "model": model_id})
        
        print(f"\n=== Debug: Converse Payload Structure ===")
        print(f"Payload modelId: {payload.get('modelId')}")
        print(f"Payload keys: {list(payload.keys())}")
        if 'messages' in payload and payload['messages']:
            print(f"First message: {json.dumps(payload['messages'][0], indent=2)}")
        print("==============================\n")

        if CUSTOM_URL:
            # Debug: log the original path
            print(f"Original self.path: {self.path}")
            print(f"Model ID from path: {model_id}")
            
            # Extract just the model name from the full model ID
            # e.g., "us.anthropic.claude-3-5-sonnet-20241022-v2:0" -> "claude-3-5-sonnet"
            # or "anthropic.claude-3-5-sonnet-20241022-v2:0" -> "claude-3-5-sonnet"
            model_name = model_id
            print(f"Step 1 - Original model_id: {model_name}")
            
            if '.' in model_name:
                # Remove region/provider prefix
                model_name = model_name.split('.', 1)[-1]
                print(f"Step 2 - After removing prefix: {model_name}")
                
            if ':' in model_name:
                # Remove version suffix
                model_name = model_name.split(':', 1)[0]
                print(f"Step 3 - After removing version suffix: {model_name}")
                
            # Remove date suffix if present (e.g., -20241022-v2)
            model_name = re.sub(r'-\d{8}-v\d+$', '', model_name)
            print(f"Step 4 - After removing date suffix: {model_name}")
            
            # Shorten claude-3-5-sonnet to claude-35-sonnet for the API
            model_name = model_name.replace('claude-3-5-sonnet', 'claude-35-sonnet')
            
            print(f"Step 5 - Final extracted model name: {model_name}")
            
            # Determine the correct Bedrock endpoint with model in path
            # The custom URL already includes /model/bedrock, so we just append /model/{model-name}/converse
            if streaming:
                # For streaming, use /model/{model}/converse-stream
                bedrock_path = f'/model/{model_name}/converse-stream'
                print(f"Transforming streaming invoke to model-specific converse-stream")
            else:
                # For non-streaming, use /model/{model}/converse
                bedrock_path = f'/model/{model_name}/converse'
                print(f"Transforming non-streaming invoke to model-specific converse")
            
            print(f"Bedrock API path: {bedrock_path}")
            print(f"About to call _bedrock_http with path: {bedrock_path}")
            
            # For Regeneron API, remove modelId from payload as it's in the URL path
            if "modelId" in payload:
                payload_for_api = {k: v for k, v in payload.items() if k != "modelId"}
                print(f"Removed 'modelId' from payload for Regeneron API")
            else:
                payload_for_api = payload
            
            # For debugging: log the exact payload we're about to send
            print(f"\n=== Payload being sent to Bedrock ===")
            print(json.dumps(payload_for_api, indent=2))
            print("=====================================\n")
            
            resp = _bedrock_http(bedrock_path, payload_for_api, stream=streaming)
        else:
            resp = client.converse_stream(**payload) if streaming else client.converse(**payload)

        # Check if this was a legacy request (for response formatting)
        is_legacy = "anthropic_version" in body

        # -------- stream or eager return ---------------------------------- #
        if streaming:
            self._stream_response(resp, is_legacy)
        else:
            self._return_eager(resp, is_legacy)

    # -------- helpers ----------------------------------------------------- #
    def _return_eager(self, resp, legacy=None):
        # Always using Converse response format now
        assistant_msg = resp["output"]["message"]
        content_blocks = [{"type": "text", "text": blk["text"]}
                          for blk in assistant_msg["content"]]
        out = {
            "id": _uuid(),
            "type": "message",
            "role": "assistant",
            "content": content_blocks,
            "model": assistant_msg.get("modelId", DEFAULT_MODEL),
            "stop_reason": resp.get("stopReason", "end_turn"),
            "usage": {
                "input_tokens": resp["usage"].get("inputTokens", 0),
                "output_tokens": resp["usage"].get("outputTokens", 0),
            },
        }
        _write_json(self, 200, out)

    def _stream_response(self, resp, legacy=None):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        # Always using converse_stream now, so always use the same format
        # converse_stream camel‑case -> snake‑case mapping
        mapping = {
            "messageStart": "message_start",
            "contentBlockStart": "content_block_start",
            "contentBlockDelta": "content_block_delta",
            "contentBlockStop": "content_block_stop",
            "messageStop": "message_stop",
            "metadata": "metadata",
        }
        
        # Handle streaming differently for CUSTOM_URL vs boto3
        if CUSTOM_URL:
            # For CUSTOM_URL, Regeneron uses a binary protocol, not standard SSE
            print(f"\n=== Streaming Response Debug ===")
            print(f"Starting to read binary stream...")
            
            # Read the raw binary stream
            buffer = b""
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    buffer += chunk
                    
                    # Look for JSON messages in the buffer
                    # The format appears to be: :message-type\x07\x00\x05event{JSON}
                    while b':message-type\x07\x00\x05event{' in buffer:
                        # Find the start of JSON
                        start_idx = buffer.find(b':message-type\x07\x00\x05event{') + len(b':message-type\x07\x00\x05event')
                        
                        # Find the end of JSON by looking for the closing brace
                        # This is a simple approach - may need refinement
                        brace_count = 0
                        end_idx = start_idx
                        in_string = False
                        escape_next = False
                        
                        for i in range(start_idx, len(buffer)):
                            char = buffer[i:i+1]
                            
                            if escape_next:
                                escape_next = False
                                continue
                                
                            if char == b'\\':
                                escape_next = True
                                continue
                                
                            if char == b'"' and not escape_next:
                                in_string = not in_string
                                continue
                                
                            if not in_string:
                                if char == b'{':
                                    brace_count += 1
                                elif char == b'}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        end_idx = i + 1
                                        break
                        
                        if end_idx > start_idx:
                            # Extract and parse the JSON
                            json_bytes = buffer[start_idx:end_idx]
                            try:
                                json_str = json_bytes.decode('utf-8')
                                data = json.loads(json_str)
                                print(f"Parsed event: {data}")
                                
                                # Convert to Bedrock Converse streaming format
                                if "role" in data and data["role"] == "assistant":
                                    # This is a messageStart event - Bedrock format
                                    event_data = {
                                        "type": "message_start",
                                        "role": "assistant"
                                    }
                                    print(f"Sending event: message_start - {event_data}")
                                    self.wfile.write(f"event: message_start\ndata: {json.dumps(event_data)}\n\n".encode())
                                    
                                    # Also send contentBlockStart
                                    block_start = {
                                        "type": "content_block_start",
                                        "contentBlockIndex": 0,
                                        "contentBlock": {
                                            "text": ""
                                        }
                                    }
                                    print(f"Sending event: content_block_start - {block_start}")
                                    self.wfile.write(f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n".encode())
                                    
                                elif "delta" in data and "text" in data["delta"]:
                                    # This is a content block delta - Bedrock format
                                    event_data = {
                                        "type": "content_block_delta",
                                        "delta": {
                                            "text": data["delta"]["text"]
                                        },
                                        "contentBlockIndex": data.get("contentBlockIndex", 0)
                                    }
                                    print(f"Sending event: content_block_delta - text: '{data['delta']['text']}'")
                                    self.wfile.write(f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n".encode())
                                    
                                elif "contentBlockIndex" in data and not "delta" in data and len(data) <= 2:
                                    # This appears to be a contentBlockStop event (only has contentBlockIndex and p)
                                    block_stop = {
                                        "type": "content_block_stop",
                                        "contentBlockIndex": data["contentBlockIndex"]
                                    }
                                    print(f"Sending event: content_block_stop - {block_stop}")
                                    self.wfile.write(f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n".encode())
                                    
                                elif "stopReason" in data:
                                    # This is a message stop event
                                    stop_event = {
                                        "type": "message_stop",
                                        "stopReason": data["stopReason"]
                                    }
                                    print(f"Sending event: message_stop - {stop_event}")
                                    self.wfile.write(f"event: message_stop\ndata: {json.dumps(stop_event)}\n\n".encode())
                                    
                                elif "metrics" in data and "usage" in data:
                                    # This is metadata event with usage stats
                                    metadata_event = {
                                        "type": "metadata",
                                        "usage": {
                                            "inputTokens": data["usage"]["inputTokens"],
                                            "outputTokens": data["usage"]["outputTokens"],
                                            "totalTokens": data["usage"]["totalTokens"]
                                        }
                                    }
                                    print(f"Sending event: metadata - {metadata_event}")
                                    self.wfile.write(f"event: metadata\ndata: {json.dumps(metadata_event)}\n\n".encode())
                                
                                self.wfile.flush()
                                
                            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                                print(f"Failed to parse JSON: {e}")
                                print(f"JSON bytes: {json_bytes[:100]}...")
                            
                            # Remove processed data from buffer
                            buffer = buffer[end_idx:]
                        else:
                            # No complete JSON found yet, wait for more data
                            break
            
            print(f"Finished processing binary stream")
        else:
            # For boto3, resp is already parsed
            for evt in resp:
                k = next(iter(evt))
                snake = mapping.get(k, k)
                payload = {**evt[k], "type": snake} if k != "metadata" else {"type": snake, **evt[k]}
                self.wfile.write(f"event: {snake}\ndata: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

    # silence BaseHTTPRequestHandler logging
    def log_message(self, *_):
        pass


def main():
    port = int(os.getenv("PROXY_PORT", 8000))
    srv = HTTPServer(("", port), Handler)
    print(f"⇢ Bedrock proxy listening on http://localhost:{port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down…")


if __name__ == "__main__":
    main()
