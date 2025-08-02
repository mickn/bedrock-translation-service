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

def _bedrock_http(path: str, payload: dict, stream: bool = False):
    """
    Call a private Bedrock data‑plane reverse proxy – if one is configured
    (needed in some regulated environments).  Otherwise we rely on boto3.
    """
    headers = {"Content-Type": "application/json", "authorization-token": ACCESS_TOKEN}
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
    print("===========================\n")

    resp = requests.post(url, json=payload, headers=headers, stream=stream, timeout=90)

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
        "inferenceConfig": infer or None,
    }
    if sys := body.get("system"):
        req["system"] = [{"text": sys}]
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
            
            # For debugging: log the exact payload we're about to send
            print(f"\n=== Payload being sent to Bedrock ===")
            print(json.dumps(payload, indent=2))
            print("=====================================\n")
            
            resp = _bedrock_http(bedrock_path, payload, stream=streaming)
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
        for evt in resp["stream"] if CUSTOM_URL else resp:
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
