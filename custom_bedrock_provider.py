#!/usr/bin/env python3
"""
Custom LiteLLM provider for AWS Bedrock that handles endpoint translation
Converts /invoke-with-response-stream to /converse-stream
"""

import json
import os
import boto3
from typing import Optional, Dict, Any, Iterator, AsyncIterator
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ModelResponse, Message, Usage, Choices
from litellm.types.llms.openai import ChatCompletionMessageChoice
from litellm.utils import CustomStreamWrapper
import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class BedrockConverseProvider(CustomLLM):
    """Custom provider for AWS Bedrock Converse API"""
    
    def __init__(self):
        super().__init__()
        self.bedrock_client = None
        
    def get_bedrock_client(self):
        """Get or create Bedrock client"""
        if not self.bedrock_client:
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1')
            )
        return self.bedrock_client
    
    def _prepare_bedrock_messages(self, messages: list) -> tuple[list, Optional[list]]:
        """Convert LiteLLM messages to Bedrock Converse format"""
        bedrock_messages = []
        system_messages = []
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                
                if role == 'system':
                    system_messages.append(content)
                    continue
                
                # Convert role names
                if role == 'assistant':
                    bedrock_role = 'assistant'
                else:
                    bedrock_role = 'user'
                
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
                    'role': bedrock_role,
                    'content': bedrock_content
                })
        
        # Combine system messages
        system_content = None
        if system_messages:
            system_content = [{'text': ' '.join(system_messages)}]
            
        return bedrock_messages, system_content
    
    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: callable,
        encoding,
        logging_obj,
        optional_params: Dict,
        custom_prompt_dict: Dict,
        timeout: Optional[float] = None,
        litellm_params: Dict = None,
        logger_fn=None,
        **kwargs
    ) -> ModelResponse:
        """Handle non-streaming completion"""
        
        # Extract the actual model ID from the model string
        # Format: bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0
        if model.startswith('bedrock/'):
            model_id = model.replace('bedrock/', '')
        else:
            model_id = model
        
        # Prepare messages for Bedrock
        bedrock_messages, system_content = self._prepare_bedrock_messages(messages)
        
        # Build Bedrock request
        bedrock_request = {
            'messages': bedrock_messages,
            'inferenceConfig': {
                'maxTokens': optional_params.get('max_tokens', 4096),
                'temperature': optional_params.get('temperature', 0.7),
            }
        }
        
        # Add system message if present
        if system_content:
            bedrock_request['system'] = system_content
        
        # Make request to Bedrock
        client = self.get_bedrock_client()
        
        try:
            # Use the Converse API directly
            response = client.converse(
                modelId=model_id,
                **bedrock_request
            )
            
            # Extract content from Bedrock response
            output_message = response.get('output', {}).get('message', {})
            content = output_message.get('content', [{}])[0].get('text', '')
            
            # Update the model_response
            model_response.choices[0].message.content = content
            model_response.choices[0].finish_reason = response.get('stopReason', 'stop')
            
            # Update usage if available
            if 'usage' in response:
                model_response.usage = Usage(
                    prompt_tokens=response['usage'].get('inputTokens', 0),
                    completion_tokens=response['usage'].get('outputTokens', 0),
                    total_tokens=response['usage'].get('totalTokens', 0)
                )
            
            return model_response
            
        except Exception as e:
            raise litellm.APIError(
                message=f"Bedrock API error: {str(e)}",
                status_code=500,
                request=bedrock_request
            )
    
    def streaming(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: callable,
        encoding,
        logging_obj,
        optional_params: Dict,
        custom_prompt_dict: Dict,
        timeout: Optional[float] = None,
        litellm_params: Dict = None,
        logger_fn=None,
        **kwargs
    ) -> Iterator:
        """Handle streaming completion"""
        
        # Extract the actual model ID
        if model.startswith('bedrock/'):
            model_id = model.replace('bedrock/', '')
        else:
            model_id = model
        
        # Prepare messages for Bedrock
        bedrock_messages, system_content = self._prepare_bedrock_messages(messages)
        
        # Build Bedrock request
        bedrock_request = {
            'messages': bedrock_messages,
            'inferenceConfig': {
                'maxTokens': optional_params.get('max_tokens', 4096),
                'temperature': optional_params.get('temperature', 0.7),
            }
        }
        
        # Add system message if present
        if system_content:
            bedrock_request['system'] = system_content
        
        # Make streaming request to Bedrock
        client = self.get_bedrock_client()
        
        try:
            # Use the ConverseStream API
            response = client.converse_stream(
                modelId=model_id,
                **bedrock_request
            )
            
            # Stream the response
            for event in response['stream']:
                if 'contentBlockDelta' in event:
                    delta = event['contentBlockDelta']['delta']
                    if 'text' in delta:
                        # Create a chunk that mimics OpenAI format
                        chunk = {
                            "choices": [{
                                "delta": {
                                    "content": delta['text']
                                },
                                "index": 0
                            }]
                        }
                        yield chunk
                        
                elif 'messageStop' in event:
                    # Final chunk
                    chunk = {
                        "choices": [{
                            "delta": {},
                            "finish_reason": event['messageStop'].get('stopReason', 'stop'),
                            "index": 0
                        }]
                    }
                    yield chunk
                    
        except Exception as e:
            raise litellm.APIError(
                message=f"Bedrock streaming error: {str(e)}",
                status_code=500,
                request=bedrock_request
            )
    
    async def acompletion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: callable,
        encoding,
        logging_obj,
        optional_params: Dict,
        custom_prompt_dict: Dict,
        timeout: Optional[float] = None,
        litellm_params: Dict = None,
        logger_fn=None,
        **kwargs
    ) -> ModelResponse:
        """Handle async non-streaming completion"""
        # For simplicity, we'll use the sync version
        # In production, you'd want to use aioboto3
        return self.completion(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            logging_obj=logging_obj,
            optional_params=optional_params,
            custom_prompt_dict=custom_prompt_dict,
            timeout=timeout,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            **kwargs
        )
    
    async def astreaming(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose: callable,
        encoding,
        logging_obj,
        optional_params: Dict,
        custom_prompt_dict: Dict,
        timeout: Optional[float] = None,
        litellm_params: Dict = None,
        logger_fn=None,
        **kwargs
    ) -> AsyncIterator:
        """Handle async streaming completion"""
        # For simplicity, convert sync streaming to async
        for chunk in self.streaming(
            model=model,
            messages=messages,
            model_response=model_response,
            print_verbose=print_verbose,
            encoding=encoding,
            logging_obj=logging_obj,
            optional_params=optional_params,
            custom_prompt_dict=custom_prompt_dict,
            timeout=timeout,
            litellm_params=litellm_params,
            logger_fn=logger_fn,
            **kwargs
        ):
            yield chunk


# Register the provider with LiteLLM
bedrock_converse_provider = BedrockConverseProvider()

# Set up the custom provider map
litellm.custom_provider_map = [
    {"provider": "bedrock_converse", "custom_handler": bedrock_converse_provider}
]