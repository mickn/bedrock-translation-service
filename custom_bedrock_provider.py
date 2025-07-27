#!/usr/bin/env python3
"""
Custom LiteLLM provider for AWS Bedrock that handles endpoint translation
Converts /invoke-with-response-stream to /converse-stream
"""

import json
import os
import httpx
from typing import Optional, Dict, Any, AsyncGenerator, Generator
from litellm.llms.custom_httpx.http_handler import CustomHTTPHandler, AsyncCustomHTTPHandler
from litellm.types.utils import ModelResponse, Message, Usage, Choices
from litellm.utils import CustomStreamWrapper
import litellm

class BedrockConverseProvider(CustomHTTPHandler):
    """Custom provider for AWS Bedrock Converse API"""
    
    def __init__(self):
        super().__init__()
        
    def _translate_endpoint(self, api_base: str) -> str:
        """Translate LiteLLM's expected endpoint to Bedrock's Converse endpoint"""
        if '/invoke-with-response-stream' in api_base:
            return api_base.replace('/invoke-with-response-stream', '/converse-stream')
        elif '/invoke' in api_base:
            return api_base.replace('/invoke', '/converse')
        return api_base
    
    def _prepare_bedrock_messages(self, messages: list) -> list:
        """Convert LiteLLM messages to Bedrock Converse format"""
        bedrock_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                # Handle content format
                if isinstance(content, str):
                    bedrock_content = [{'text': content}]
                elif isinstance(content, list):
                    bedrock_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'text':
                            bedrock_content.append({'text': item.get('text', '')})
                        elif isinstance(item, str):
                            bedrock_content.append({'text': item})
                else:
                    bedrock_content = [{'text': str(content)}]
                
                bedrock_messages.append({
                    'role': role,
                    'content': bedrock_content
                })
        
        return bedrock_messages
    
    def _prepare_bedrock_request(self, messages: list, **kwargs) -> dict:
        """Prepare the full Bedrock Converse API request"""
        # Extract system message if present
        system_messages = []
        user_messages = []
        
        for msg in messages:
            if msg.get('role') == 'system':
                system_messages.append(msg)
            else:
                user_messages.append(msg)
        
        # Build request
        request_body = {
            'messages': self._prepare_bedrock_messages(user_messages),
            'inferenceConfig': {
                'maxTokens': kwargs.get('max_tokens', 4096),
                'temperature': kwargs.get('temperature', 0.7),
            }
        }
        
        # Add system message if present
        if system_messages:
            system_content = ' '.join(msg.get('content', '') for msg in system_messages)
            request_body['system'] = [{'text': system_content}]
        
        return request_body
    
    def _process_response(self, response: httpx.Response) -> ModelResponse:
        """Process Bedrock response into LiteLLM format"""
        try:
            data = response.json()
            
            # Extract content from Bedrock response
            output_message = data.get('output', {}).get('message', {})
            content = output_message.get('content', [{}])[0].get('text', '')
            
            # Build LiteLLM response
            model_response = ModelResponse(
                id=f"chatcmpl-{os.urandom(12).hex()}",
                choices=[
                    Choices(
                        finish_reason=data.get('stopReason', 'stop'),
                        index=0,
                        message=Message(
                            content=content,
                            role='assistant'
                        )
                    )
                ],
                model=data.get('model', 'claude-3-5-sonnet'),
                usage=Usage(
                    prompt_tokens=data.get('usage', {}).get('inputTokens', 0),
                    completion_tokens=data.get('usage', {}).get('outputTokens', 0),
                    total_tokens=data.get('usage', {}).get('totalTokens', 0)
                )
            )
            
            return model_response
            
        except Exception as e:
            raise Exception(f"Error processing Bedrock response: {str(e)}")
    
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        headers: Optional[Dict[str, str]] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        logger_fn: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        custom_prompt_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """Handle non-streaming completion"""
        # Translate the endpoint
        api_base = self._translate_endpoint(api_base)
        
        # Prepare the request
        request_body = self._prepare_bedrock_request(messages, **kwargs)
        
        # Make the request
        headers = headers or {}
        headers['Content-Type'] = 'application/json'
        
        if print_verbose:
            print_verbose(f"Making request to: {api_base}")
            print_verbose(f"Request body: {json.dumps(request_body, indent=2)}")
        
        response = httpx.post(
            url=api_base,
            json=request_body,
            headers=headers,
            timeout=kwargs.get('timeout', 600.0)
        )
        
        if response.status_code != 200:
            raise Exception(f"Bedrock API error: {response.status_code} - {response.text}")
        
        return self._process_response(response)
    
    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        headers: Optional[Dict[str, str]] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        logger_fn: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        custom_prompt_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> CustomStreamWrapper:
        """Handle streaming completion"""
        # Translate the endpoint
        api_base = self._translate_endpoint(api_base)
        
        # Prepare the request
        request_body = self._prepare_bedrock_request(messages, **kwargs)
        
        # Make the streaming request
        headers = headers or {}
        headers['Content-Type'] = 'application/json'
        
        if print_verbose:
            print_verbose(f"Making streaming request to: {api_base}")
            print_verbose(f"Request body: {json.dumps(request_body, indent=2)}")
        
        response = httpx.post(
            url=api_base,
            json=request_body,
            headers=headers,
            timeout=kwargs.get('timeout', 600.0),
            stream=True
        )
        
        if response.status_code != 200:
            raise Exception(f"Bedrock API error: {response.status_code} - {response.text}")
        
        return CustomStreamWrapper(
            completion_stream=response.iter_lines(),
            model=model,
            custom_llm_provider="bedrock_converse",
            logging_obj=logger_fn
        )


class AsyncBedrockConverseProvider(AsyncCustomHTTPHandler):
    """Async version of the custom Bedrock provider"""
    
    def __init__(self):
        super().__init__()
        self.sync_handler = BedrockConverseProvider()
    
    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        headers: Optional[Dict[str, str]] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        logger_fn: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        custom_prompt_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> ModelResponse:
        """Handle async non-streaming completion"""
        # Use sync handler's methods for translation
        api_base = self.sync_handler._translate_endpoint(api_base)
        request_body = self.sync_handler._prepare_bedrock_request(messages, **kwargs)
        
        headers = headers or {}
        headers['Content-Type'] = 'application/json'
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=api_base,
                json=request_body,
                headers=headers,
                timeout=kwargs.get('timeout', 600.0)
            )
        
        if response.status_code != 200:
            raise Exception(f"Bedrock API error: {response.status_code} - {response.text}")
        
        return self.sync_handler._process_response(response)
    
    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        headers: Optional[Dict[str, str]] = None,
        model_response: Optional[ModelResponse] = None,
        print_verbose: Optional[callable] = None,
        optional_params: Optional[Dict[str, Any]] = None,
        litellm_params: Optional[Dict[str, Any]] = None,
        logger_fn: Optional[callable] = None,
        encoding: Optional[Any] = None,
        api_key: Optional[str] = None,
        custom_prompt_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> AsyncGenerator:
        """Handle async streaming completion"""
        # Use sync handler's methods for translation
        api_base = self.sync_handler._translate_endpoint(api_base)
        request_body = self.sync_handler._prepare_bedrock_request(messages, **kwargs)
        
        headers = headers or {}
        headers['Content-Type'] = 'application/json'
        
        async with httpx.AsyncClient() as client:
            async with client.stream(
                'POST',
                url=api_base,
                json=request_body,
                headers=headers,
                timeout=kwargs.get('timeout', 600.0)
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Bedrock API error: {response.status_code}")
                
                async for line in response.aiter_lines():
                    if line:
                        yield line


# Register the provider with LiteLLM
def register_provider():
    """Register this provider with LiteLLM"""
    import litellm
    litellm.bedrock_converse_provider = BedrockConverseProvider
    litellm.async_bedrock_converse_provider = AsyncBedrockConverseProvider

# Auto-register when module is imported
register_provider()