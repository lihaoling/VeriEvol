"""
General-purpose vision-language model API interface
Provides a concise function interface for calling vision-language models for image-text QA
"""
import io
import random
from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import base64
from urllib.parse import urlparse
from openai import OpenAI
import httpx
from loguru import logger


class VisionAPIClient:
    """Vision-language model API client manager"""

    def __init__(
        self,
        api_endpoints: List[str],
        api_key: str = "sk-abc123",
        model_name: str = "qwen_3",
        temperature: float = 0.7,
        max_tokens: int = 32768,
        client_selection: str = "random",
        timeout: int = 3000
    ):
        """
        Initialize the API client

        Args:
            api_endpoints: List of API endpoints, e.g. ["http://ip:port/v1", ...]
            api_key: API key
            model_name: Model name
            temperature: Temperature parameter
            max_tokens: Maximum number of generated tokens
            client_selection: Client selection strategy, "random", "round_robin" or "localhost"
            timeout: Timeout in seconds
        """
        self.api_endpoints = api_endpoints
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.client_selection = client_selection
        self.timeout = timeout
        self.current_index = 0
        self.localhost_index = 0

        # Automatically adjust max_tokens based on the maximum context length of common models
        # to avoid exceeding the model limit when making requests
        self.max_tokens = self._adjust_max_tokens(max_tokens, model_name)

        # Initialize all clients
        # Use an httpx client to set the timeout, compatible with newer versions of the openai library
        self.clients = []
        for url in api_endpoints:
            try:
                # Try setting the timeout via the http_client parameter (newer openai)
                # Use an httpx.Timeout object to avoid passing unexpected parameters
                timeout_obj = httpx.Timeout(timeout, connect=30.0)
                http_client = httpx.Client(timeout=timeout_obj)
                client = OpenAI(
                    base_url=url,
                    api_key=api_key,
                    http_client=http_client
                )
            except (TypeError, ValueError, AttributeError) as e:
                # If it fails, try passing timeout directly (older openai)
                try:
                    client = OpenAI(
                        base_url=url, api_key=api_key, timeout=timeout)
                except (TypeError, ValueError):
                    # If it still fails, do not set timeout
                    logger.warning(f"Unable to set timeout for endpoint {url}, using default: {e}")
                    client = OpenAI(base_url=url, api_key=api_key)
            self.clients.append(client)

        # Filter out localhost clients
        self.localhost_clients = []
        for i, url in enumerate(api_endpoints):
            if self._is_localhost(url):
                self.localhost_clients.append(self.clients[i])

        if client_selection == "localhost":
            if not self.localhost_clients:
                logger.warning("No localhost endpoint found, will use all clients")
                self.localhost_clients = self.clients
            else:
                logger.info(f'Found {len(self.localhost_clients)} localhost endpoints')

        logger.info(
            f'Initialized {len(self.clients)} API clients, model: {model_name}, '
            f'max_tokens: {self.max_tokens}'
        )

    def _is_localhost(self, url: str) -> bool:
        """
        Determine whether a URL is localhost

        Args:
            url: API endpoint URL

        Returns:
            Returns True if it is localhost, otherwise False
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname or ""
            # Check whether it is a localhost-related address
            localhost_indicators = ['localhost', '127.0.0.1', '0.0.0.0', '::1']
            return any(indicator in hostname.lower() for indicator in localhost_indicators)
        except Exception:
            # If parsing fails, fall back to simple string matching
            url_lower = url.lower()
            return any(indicator in url_lower for indicator in ['localhost', '127.0.0.1', '0.0.0.0'])

    def _adjust_max_tokens(self, max_tokens: int, model_name: str) -> int:
        """
        Automatically adjust max_tokens based on the model's maximum context length to avoid exceeding the limit

        Args:
            max_tokens: The max_tokens requested by the user
            model_name: Model name

        Returns:
            The adjusted max_tokens
        """
        # Mapping of maximum context lengths for common models (conservative estimates)
        # If the model name contains these keywords, use the corresponding maximum context length
        model_max_contexts = {
            # 32K context models
            '32768': 32768,
            '32k': 32768,
            '32K': 32768,
            # Other common values
            '8192': 8192,
            '4096': 4096,
            '2048': 2048,
        }

        # Try to infer the maximum context length from the model name
        estimated_max_context = None
        model_lower = model_name.lower()

        # Check whether the model name contains context length information
        for key, context_len in model_max_contexts.items():
            if key.lower() in model_lower:
                estimated_max_context = context_len
                break

        # If it cannot be inferred from the model name, use a conservative default
        # Based on the error logs, many models have a maximum context length of 32768
        if estimated_max_context is None:
            # If max_tokens is very large (>= 30000), it is likely a 32K model
            if max_tokens >= 30000:
                estimated_max_context = 32768
            else:
                # For smaller max_tokens, do not adjust
                return max_tokens

        # Compute a safety margin: reserve 15% of space for the input message
        # This avoids exceeding the limit when the input message is long
        safe_max_tokens = int(estimated_max_context * 0.85)

        # If the user-requested max_tokens exceeds the safe value, adjust automatically
        if max_tokens > safe_max_tokens:
            logger.info(
                f"Automatically adjusting max_tokens: {max_tokens} -> {safe_max_tokens} "
                f"(model maximum context length: {estimated_max_context}, reserving 15% safety margin)"
            )
            return safe_max_tokens

        return max_tokens

    def _get_client(self):
        """Get a client instance"""
        if self.client_selection == "random":
            return random.choice(self.clients)
        elif self.client_selection == "round_robin":
            client = self.clients[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.clients)
            return client
        elif self.client_selection == "localhost":
            if not self.localhost_clients:
                # If there are no localhost clients, fall back to the first client
                return self.clients[0]
            client = self.localhost_clients[self.localhost_index]
            self.localhost_index = (
                self.localhost_index + 1) % len(self.localhost_clients)
            return client
        else:
            return self.clients[0]

    def call(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 5
    ) -> Optional[str]:
        """
        Call the API to get a response

        Args:
            messages: List of messages in OpenAI format
            max_retries: Maximum number of retries

        Returns:
            Model response text, or None on failure
        """
        client = self._get_client()
        # Dynamically adjust max_tokens to handle token-limit-exceeded errors
        current_max_tokens = self.max_tokens

        for attempt in range(max_retries):
            try:
                # Debug: log request parameters (only on the first attempt)
                if attempt == 0:
                    logger.debug(f"Sending API request:")
                    logger.debug(f"  Model name: {self.model_name}")
                    logger.debug(f"  Temperature: {self.temperature}")
                    logger.debug(f"  Max tokens: {current_max_tokens}")
                    logger.debug(f"  Timeout: {self.timeout}s")
                    logger.debug(f"  Number of messages: {len(messages)}")
                    # Inspect message content
                    for i, msg in enumerate(messages):
                        logger.debug(
                            f"  Message {i}: role={msg.get('role')}, content type={type(msg.get('content'))}")
                        if isinstance(msg.get('content'), list):
                            for j, item in enumerate(msg.get('content', [])):
                                logger.debug(
                                    f"    Content item {j}: type={item.get('type')}")
                                if item.get('type') == 'image_url':
                                    url = item.get(
                                        'image_url', {}).get('url', '')
                                    logger.debug(
                                        f"      Image URL length: {len(url)}")
                                    logger.debug(
                                        f"      Image URL prefix: {url[:100] if len(url) > 100 else url}")

                completion = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=current_max_tokens,
                    messages=messages
                )

                # Debug: inspect the full response structure
                if not completion.choices:
                    logger.error(
                        f"API call returned empty choices (attempt {attempt + 1}/{max_retries})"
                    )
                    logger.debug(f"Full response: {completion}")
                    if attempt < max_retries - 1:
                        if self.client_selection == "random":
                            client = self._get_client()
                        continue
                    else:
                        return None

                message = completion.choices[0].message
                content = message.content

                # Prefer content; if content is None, then consider using reasoning_content
                if content is None:
                    if hasattr(message, 'reasoning_content'):
                        reasoning_content = message.reasoning_content
                        if reasoning_content:
                            logger.debug(
                                f"content is None, using reasoning_content as the return value")
                            content = reasoning_content

                if content is None:
                    # Print more debug information (use INFO level so it is visible)
                    logger.warning(
                        f"API call returned content as None (attempt {attempt + 1}/{max_retries})"
                    )
                    logger.info(f"Model name: {self.model_name}")
                    logger.info(f"API endpoint: {client.base_url}")
                    logger.info(f"Number of choices: {len(completion.choices)}")
                    if hasattr(message, 'role'):
                        logger.info(f"Message role: {message.role}")
                    if hasattr(completion, 'usage'):
                        logger.info(f"Usage: {completion.usage}")
                    # Print the full response object (only on the first attempt)
                    if attempt == 0:
                        logger.info(f"Full response object: {completion}")
                        logger.info(f"Message object: {message}")
                        logger.info(f"Message object attributes: {dir(message)}")
                        # Try to check whether other fields contain content
                        if hasattr(message, '__dict__'):
                            logger.info(f"Message object dict: {message.__dict__}")
                        # Check finish_reason
                        if hasattr(completion.choices[0], 'finish_reason'):
                            finish_reason = completion.choices[0].finish_reason
                            logger.info(f"Finish reason: {finish_reason}")
                            if finish_reason and finish_reason != 'stop':
                                logger.warning(f"Abnormal finish reason: {finish_reason}")

                        # Check whether there is a reasoning_content field
                        if hasattr(message, 'reasoning_content'):
                            reasoning_content = message.reasoning_content
                            logger.info(
                                f"Reasoning content: {reasoning_content}")
                            if reasoning_content:
                                logger.info(
                                    f"Attempted to use reasoning_content, but content is still None"
                                )

                    if attempt < max_retries - 1:
                        if self.client_selection == "random":
                            client = self._get_client()
                        continue
                    else:
                        logger.error(
                            f"API call ultimately failed: returned content is None. "
                            f"Please check: 1) whether the model name is correct ({self.model_name}); "
                            f"2) whether the API server supports this model; "
                            f"3) whether the request format meets the server's requirements"
                        )
                        return None
                return content
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): [{error_type}] {error_msg}"
                )
                logger.warning(
                    f"  Model name: {self.model_name}, API endpoint: {client.base_url}"
                )

                # Check whether it is a token-limit-exceeded error
                is_token_limit_error = (
                    error_type == 'BadRequestError' and
                    ('maximum context length' in error_msg.lower() or
                     'tokens' in error_msg.lower() and 'requested' in error_msg.lower())
                )

                if is_token_limit_error and attempt < max_retries - 1:
                    # Try to extract the maximum context length from the error message
                    import re
                    max_context_match = re.search(
                        r'maximum context length is (\d+)', error_msg, re.IGNORECASE)
                    if max_context_match:
                        max_context = int(max_context_match.group(1))
                        # Reduce max_tokens, reserving a 10% safety margin
                        new_max_tokens = int(max_context * 0.9)
                        if new_max_tokens < current_max_tokens:
                            current_max_tokens = new_max_tokens
                            logger.warning(
                                f"Detected token limit exceeded, automatically reducing max_tokens to {current_max_tokens} and retrying"
                            )
                        else:
                            # If the computed value is larger, reduce the current value by 20%
                            current_max_tokens = int(current_max_tokens * 0.8)
                            logger.warning(
                                f"Detected token limit exceeded, automatically reducing max_tokens to {current_max_tokens} and retrying"
                            )
                    else:
                        # If it cannot be extracted, reduce the current value by 20%
                        current_max_tokens = int(current_max_tokens * 0.8)
                        logger.warning(
                            f"Detected token limit exceeded, automatically reducing max_tokens to {current_max_tokens} and retrying"
                        )

                if attempt < max_retries - 1:
                    # Choose a different client on the next attempt (if in random mode)
                    if self.client_selection == "random":
                        client = self._get_client()
                    # Wait a short while before retrying
                    import time
                    time.sleep(0.5)
                else:
                    logger.error(
                        f"API call ultimately failed after {max_retries} retries: [{error_type}] {error_msg}"
                    )
                    logger.error(
                        f"  Model name: {self.model_name}, API endpoint: {client.base_url}"
                    )
                    if is_token_limit_error:
                        logger.error(
                            f"  Suggestion: reduce the max_tokens parameter (current attempted value: {current_max_tokens}) or reduce the input message length"
                        )
                    return None

        return None

    def call_with_reasoning(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Call the API to get a response, returning both reasoning_content and content

        Args:
            messages: List of messages in OpenAI format
            max_retries: Maximum number of retries

        Returns:
            A dict containing 'reasoning_content' and 'content', or None on failure
            Format: {"reasoning_content": "...", "content": "..."}
        """
        client = self._get_client()
        # Dynamically adjust max_tokens to handle token-limit-exceeded errors
        current_max_tokens = self.max_tokens

        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model_name,
                    temperature=self.temperature,
                    max_tokens=current_max_tokens,
                    messages=messages
                )

                if not completion.choices:
                    logger.error(
                        f"API call returned empty choices (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        if self.client_selection == "random":
                            client = self._get_client()
                        continue
                    else:
                        return None

                message = completion.choices[0].message
                content = message.content
                reasoning_content = None

                # Get reasoning_content
                if hasattr(message, 'reasoning_content'):
                    reasoning_content = message.reasoning_content

                # If content is None, try using reasoning_content
                if content is None and reasoning_content:
                    logger.debug(
                        "content is None, using reasoning_content as content")
                    content = reasoning_content

                if content is None:
                    logger.warning(
                        f"API call returned content as None (attempt {attempt + 1}/{max_retries})"
                    )
                    if attempt < max_retries - 1:
                        if self.client_selection == "random":
                            client = self._get_client()
                        continue
                    else:
                        return None

                # Return a dict containing both fields
                return {
                    "reasoning_content": reasoning_content if reasoning_content else "",
                    "content": content if content else ""
                }
            except Exception as e:
                error_msg = str(e)
                error_type = type(e).__name__
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{max_retries}): [{error_type}] {error_msg}"
                )
                logger.warning(
                    f"  Model name: {self.model_name}, API endpoint: {client.base_url}"
                )

                # Check whether it is a token-limit-exceeded error
                is_token_limit_error = (
                    error_type == 'BadRequestError' and
                    ('maximum context length' in error_msg.lower() or
                     'tokens' in error_msg.lower() and 'requested' in error_msg.lower())
                )

                if is_token_limit_error and attempt < max_retries - 1:
                    # Try to extract the maximum context length from the error message
                    import re
                    max_context_match = re.search(
                        r'maximum context length is (\d+)', error_msg, re.IGNORECASE)
                    if max_context_match:
                        max_context = int(max_context_match.group(1))
                        # Reduce max_tokens, reserving a 10% safety margin
                        new_max_tokens = int(max_context * 0.9)
                        if new_max_tokens < current_max_tokens:
                            current_max_tokens = new_max_tokens
                            logger.warning(
                                f"Detected token limit exceeded, automatically reducing max_tokens to {current_max_tokens} and retrying"
                            )
                        else:
                            # If the computed value is larger, reduce the current value by 20%
                            current_max_tokens = int(current_max_tokens * 0.8)
                            logger.warning(
                                f"Detected token limit exceeded, automatically reducing max_tokens to {current_max_tokens} and retrying"
                            )
                    else:
                        # If it cannot be extracted, reduce the current value by 20%
                        current_max_tokens = int(current_max_tokens * 0.8)
                        logger.warning(
                            f"Detected token limit exceeded, automatically reducing max_tokens to {current_max_tokens} and retrying"
                        )

                if attempt < max_retries - 1:
                    if self.client_selection == "random":
                        client = self._get_client()
                    import time
                    time.sleep(0.5)
                else:
                    logger.error(
                        f"API call ultimately failed after {max_retries} retries: [{error_type}] {error_msg}"
                    )
                    logger.error(
                        f"  Model name: {self.model_name}, API endpoint: {client.base_url}"
                    )
                    if is_token_limit_error:
                        logger.error(
                            f"  Suggestion: reduce the max_tokens parameter (current attempted value: {current_max_tokens}) or reduce the input message length"
                        )
                    return None

        return None


def base64_encode_image(image_info: Dict[str, Any]) -> Tuple[str, str]:
    """
    Read from an image file and encode it as base64

    Args:
        image_info: Image info dict, containing:
            - patch: Image file path
            - start_num: Start position in the file (bytes)
            - size: Image data size (bytes)

    Returns:
        (base64_encoded_string, image_format) tuple

    Raises:
        FileNotFoundError: The image file does not exist
        IOError: Failed to read the image file
        ValueError: Invalid image format
    """
    file_path = image_info.get('patch')
    start = image_info.get('start_num', 0)
    size = image_info.get('size')

    if not file_path:
        raise ValueError("Image info is missing the 'patch' field")
    if size is None:
        raise ValueError("Image info is missing the 'size' field")

    try:
        with open(file_path, 'rb') as f:
            f.seek(start)
            image_bytes = f.read(size)

        if len(image_bytes) != size:
            raise IOError(
                f"Incomplete image file read: expected {size} bytes, actually read {len(image_bytes)} bytes"
            )

        # Validate the image format
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_format = image.format.lower() if image.format else 'jpeg'
            image.close()  # Close the image to release resources
        except Exception as e:
            raise ValueError(f"Unable to parse image format: {e}")

        image_encode = base64.b64encode(image_bytes).decode('utf-8')
        return image_encode, image_format
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file does not exist: {file_path}")
    except Exception as e:
        raise IOError(f"Failed to read image file ({file_path}): {e}")


def prepare_image_message(image_info: Dict[str, Any]) -> str:
    """
    Convert image info into a data URI formatted message

    Args:
        image_info: Image info dict (same format as base64_encode_image)

    Returns:
        A data URI formatted string, e.g. "data:image/jpeg;base64,..."

    Raises:
        FileNotFoundError: The image file does not exist
        IOError: Failed to read the image file
        ValueError: Invalid image format
    """
    try:
        base64_image, image_format = base64_encode_image(image_info)
        return f"data:image/{image_format};base64,{base64_image}"
    except Exception as e:
        logger.error(f"Failed to prepare image message: {e}")
        raise


def call_vision_api(
    api_client: VisionAPIClient,
    image: Union[Dict[str, Any], str],
    question: str,
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Call the vision-language model API for image-text QA (main interface function)

    Args:
        api_client: VisionAPIClient instance
        image: Image info, which can be:
            - dict format: {"patch": "...", "start_num": ..., "size": ...}
            - string format: an already-encoded data URI (e.g. "data:image/jpeg;base64,...")
        question: Question text
        system_prompt: Optional system prompt

    Returns:
        Model response text, or None on failure

    Example:
        >>> from vision_api import VisionAPIClient, call_vision_api
        >>> 
        >>> # Initialize the client
        >>> client = VisionAPIClient(
        ...     api_endpoints=["http://ip:port/v1"],
        ...     model_name="qwen_3"
        ... )
        >>> 
        >>> # Prepare image info
        >>> image_info = {
        ...     "patch": "/path/to/image",
        ...     "start_num": 0,
        ...     "size": 1024
        ... }
        >>> 
        >>> # Call the API
        >>> response = call_vision_api(
        ...     api_client=client,
        ...     image=image_info,
        ...     question="What is in this image?"
        ... )
        >>> print(response)
    """
    # Build the message list
    messages = []

    # Add the system prompt (if any)
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Process the image
    try:
        if isinstance(image, dict):
            # Image info dict, needs to be converted to a data URI
            logger.debug(f"Processing image dict: {image.get('patch', 'unknown')}")
            image_url = prepare_image_message(image)
            logger.debug(f"Image encoded successfully, data URI length: {len(image_url)}")
        elif isinstance(image, str):
            # Already a data URI format
            image_url = image
            logger.debug(f"Using already-encoded image data URI, length: {len(image_url)}")
        else:
            raise ValueError(f"Unsupported image format: {type(image)}")
    except (FileNotFoundError, IOError, ValueError) as e:
        logger.error(f"Image processing failed: {e}")
        import traceback
        logger.error(f"Error stack: {traceback.format_exc()}")
        return None
    except Exception as e:
        logger.error(f"Unexpected image processing error: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Error stack: {traceback.format_exc()}")
        return None

    # Build the user message
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": image_url,
                },
            }
        ]
    }
    messages.append(user_message)

    # Debug: log the message structure (only the first time)
    logger.debug(f"Number of messages sent: {len(messages)}")
    logger.debug(f"Question text length: {len(question)}")
    logger.debug(
        f"Image data URI prefix: {image_url[:50] if len(image_url) > 50 else image_url}...")

    # Call the API
    return api_client.call(messages)


# Convenience functions: create a client with default config and call it
_default_client: Optional[VisionAPIClient] = None


def init_default_client(
    api_endpoints: List[str],
    api_key: str = "sk-abc123",
    model_name: str = "qwen_3",
    **kwargs
):
    """
    Initialize the default global client

    Args:
        api_endpoints: List of API endpoints
        api_key: API key
        model_name: Model name
        **kwargs: Other VisionAPIClient parameters
    """
    global _default_client
    _default_client = VisionAPIClient(
        api_endpoints=api_endpoints,
        api_key=api_key,
        model_name=model_name,
        **kwargs
    )


def call_vision_api_simple(
    image: Union[Dict[str, Any], str],
    question: str,
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Simplified interface that calls the API using the default client

    Note: Before use, you must first call init_default_client() to initialize the client

    Args:
        image: Image info (dict or data URI string)
        question: Question text
        system_prompt: Optional system prompt

    Returns:
        Model response text, or None on failure
    """
    if _default_client is None:
        raise RuntimeError(
            "The default client is not initialized. Please call init_default_client() first "
            "or use the call_vision_api(api_client, ...) function"
        )

    return call_vision_api(
        api_client=_default_client,
        image=image,
        question=question,
        system_prompt=system_prompt
    )


def call_text_api(
    api_client: VisionAPIClient,
    question: str,
    system_prompt: Optional[str] = None
) -> Optional[str]:
    """
    Call a text-only language model API (no image required)

    Args:
        api_client: VisionAPIClient instance
        question: Question/prompt text
        system_prompt: Optional system prompt

    Returns:
        Model response text, or None on failure

    Example:
        >>> from vision_api import VisionAPIClient, call_text_api
        >>> 
        >>> # Initialize the client
        >>> client = VisionAPIClient(
        ...     api_endpoints=["http://ip:port/v1"],
        ...     model_name="qwen_3"
        ... )
        >>> 
        >>> # Call the text-only API
        >>> response = call_text_api(
        ...     api_client=client,
        ...     question="Please help me evolve this question..."
        ... )
        >>> print(response)
    """
    # Build the message list
    messages = []

    # Add the system prompt (if any)
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    # Build the user message (text only, no image)
    user_message = {
        "role": "user",
        "content": question
    }
    messages.append(user_message)

    # Debug logging
    logger.debug(f"Sending text-only request, number of messages: {len(messages)}")
    logger.debug(f"Question text length: {len(question)}")

    # Call the API
    return api_client.call(messages)


def rollout_vision_api(
    image: Union[Dict[str, Any], str],
    question: str,
    api_endpoints: List[str],
    n: int = 1,
    api_key: str = "sk-abc123",
    model_name: str = "qwen_3",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 32768,
    client_selection: str = "random",
    timeout: int = 3000,
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> List[Optional[str]]:
    """
    Rollout strategy: call the API n times for the given image, question and api list

    Args:
        image: Image info, which can be:
            - dict format: {"patch": "...", "start_num": ..., "size": ...}
            - string format: an already-encoded data URI (e.g. "data:image/jpeg;base64,...")
        question: Question text
        api_endpoints: List of API endpoints, e.g. ["http://ip:port/v1", ...]
        n: Number of calls
        api_key: API key
        model_name: Model name
        system_prompt: Optional system prompt
        temperature: Temperature parameter
        max_tokens: Maximum number of generated tokens
        client_selection: Client selection strategy, "random", "round_robin" or "localhost"
        timeout: Timeout in seconds
        parallel: Whether to call in parallel (using multiple threads)
        max_workers: Maximum number of worker threads for parallel calls, None means use the default

    Returns:
        A list containing the results of n calls, each element is the model response text (None on failure)

    Example:
        >>> from vision_api import rollout_vision_api
        >>> 
        >>> # Prepare image info
        >>> image_info = {
        ...     "patch": "/path/to/image",
        ...     "start_num": 0,
        ...     "size": 1024
        ... }
        >>> 
        >>> # Call the API 5 times
        >>> results = rollout_vision_api(
        ...     image=image_info,
        ...     question="What is in this image?",
        ...     api_endpoints=["http://ip1:port1/v1", "http://ip2:port2/v1"],
        ...     n=5
        ... )
        >>> 
        >>> # Inspect the results
        >>> for i, result in enumerate(results):
        ...     print(f"Call {i+1}: {result}")
    """
    if n <= 0:
        logger.warning(f"Invalid number of calls n={n}, returning an empty list")
        return []

    # Initialize the API client
    api_client = VisionAPIClient(
        api_endpoints=api_endpoints,
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        client_selection=client_selection,
        timeout=timeout
    )

    if parallel and n > 1:
        # Parallel calls
        from concurrent.futures import ThreadPoolExecutor

        def _call_once(_):
            """Wrapper function for a single call"""
            return call_vision_api(
                api_client=api_client,
                image=image,
                question=question,
                system_prompt=system_prompt
            )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(_call_once, i) for i in range(n)]

            # Collect results in submission order (preserve order)
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Parallel call {i+1}/{n} failed: {e}")
                    results.append(None)

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Parallel calls completed, {success_count}/{n} succeeded")
        return results
    else:
        # Serial calls
        results = []
        for i in range(n):
            logger.debug(f"Starting call {i+1}/{n}")
            result = call_vision_api(
                api_client=api_client,
                image=image,
                question=question,
                system_prompt=system_prompt
            )
            results.append(result)
            if result is None:
                logger.warning(f"Call {i+1}/{n} failed")
            else:
                logger.debug(f"Call {i+1}/{n} succeeded, response length: {len(result)}")

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Serial calls completed, {success_count}/{n} succeeded")
        return results


# # Simplest verification: send a simple request and inspect the HTTP status code
# curl -s -o /dev/null -w "HTTP status code: %{http_code}\n" \
#   -X POST "http://<VLLM_ENDPOINT>/v1/chat/completions" \
#   -H "Authorization: Bearer sk-abc123" \
#   -H "Content-Type: application/json" \
#   -d '{"model":"/path/to/model/Qwen3-VL-235B-A22B-Thinking-FP8","messages":[{"role":"user","content":"test"}],"max_tokens":8192}'
