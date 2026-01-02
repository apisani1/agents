import os

from anthropic import Anthropic
from openai import OpenAI


OPENAI_CLIENT_MAP = {
    "openai": {"base_url": "https://api.openai.com/v1", "env_var": "OPENAI_API_KEY"},
    "google": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "env_var": "GOOGLE_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com/v1", "env_var": "DEEPSEEK_API_KEY"},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "env_var": "GROQ_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "env_var": "OLLAMA_API_KEY"},
}


class ChatModel:
    def __init__(self, model_name, provider="openai", api_key=None, **kwargs):
        self.model_name = model_name
        if provider in OPENAI_CLIENT_MAP:
            api_key = api_key or os.getenv(OPENAI_CLIENT_MAP[provider]["env_var"])
            if not api_key:
                raise ValueError(
                    f"Missing API key for {provider} and {OPENAI_CLIENT_MAP[provider]['env_var']} not found inenvironment variables either."
                )
            self.client = OpenAI(base_url=OPENAI_CLIENT_MAP[provider]["base_url"], api_key=api_key, **kwargs)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key, **kwargs)
        else:
            raise ValueError("Unsupported provider")

    @staticmethod
    def _extract_system_messages(messages):
        """
        Extract consecutive system messages from the beginning of a message list.

        Anthropic's API requires system messages to be passed via a separate 'system'
        parameter rather than in the messages array.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys

        Returns:
            tuple: (system_content, remaining_messages)
                - system_content: Combined system message string or None
                - remaining_messages: Messages list without system messages
        """
        system_messages = []
        remaining_messages = messages

        while remaining_messages and remaining_messages[0].get("role") == "system":
            system_messages.append(remaining_messages[0]["content"])
            remaining_messages = remaining_messages[1:]

        system_content = "\n\n".join(system_messages) if system_messages else None
        return system_content, remaining_messages

    @staticmethod
    def _prepare_tool_params(structured_response, response_format):
        """
        Prepare tool parameters for Anthropic's structured response via tool use.

        Args:
            structured_response: Whether to request a structured response
            response_format: Pydantic model defining the expected response structure

        Returns:
            dict: Tool parameters for the API call, or empty dict if not using structured response
        """
        if not structured_response:
            return {}

        return {
            "tools": [
                {
                    "name": "structured_response",
                    "description": "Return a structured response",
                    "input_schema": response_format.model_json_schema(),
                }
            ],
            "tool_choice": {"type": "tool", "name": "structured_response"},
        }

    def generate_response(
        self,
        messages,
        *,
        max_tokens=10000,
        structured_response=False,
        response_format=None,
        print_response=False,
        **kwargs,
    ):
        """
        Generate a response from the LLM using the configured provider.

        Supports both text and structured responses across multiple LLM providers
        (OpenAI, Anthropic, Google, DeepSeek, Groq, Ollama).

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                     Standard roles: 'system', 'user', 'assistant'.
            max_tokens: Maximum tokens in the response (default: 10000).
                       Only used for Anthropic; OpenAI uses default from API.
            structured_response: If True, return a structured Pydantic model instance
                               instead of text (default: False).
            response_format: Pydantic model class for structured responses.
                           Required when structured_response=True.
            print_response: If True, display the response (markdown in notebooks,
                          plain text in scripts) (default: False).
            **kwargs: Additional provider-specific parameters passed to the API.

        Returns:
            str | BaseModel: Response text (str) or Pydantic model instance if
                           structured_response=True.

        Raises:
            ValueError: If structured_response=True but response_format=None.

        Examples:
            Basic text response:
                >>> model = ChatModel("gpt-4o-mini")
                >>> response = model.generate_response([
                ...     {"role": "user", "content": "Hello!"}
                ... ])

            Structured response:
                >>> class Answer(BaseModel):
                ...     text: str
                ...     confidence: float
                >>> response = model.generate_response(
                ...     messages=[{"role": "user", "content": "What is 2+2?"}],
                ...     structured_response=True,
                ...     response_format=Answer
                ... )

        Note:
            - OpenAI: Uses native parse API for structured responses
            - Anthropic: Uses tool calling for structured responses
            - System messages: Automatically handled per provider requirements
        """
        if structured_response and response_format is None:
            raise ValueError("response_format must be provided when structured_response=True")

        if isinstance(self.client, OpenAI):
            if not structured_response:
                response = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
                answer = response.choices[0].message.content
            else:
                response = self.client.beta.chat.completions.parse(
                    model="gpt-4o-mini", messages=messages, response_format=response_format, **kwargs  # type: ignore
                )
                answer = response.choices[0].message.parsed

        elif isinstance(self.client, Anthropic):
            # Anthropic API differences:
            # 1. Uses separate 'system' parameter instead of system messages in the array
            # 2. Uses tool calling (function calling) for structured output instead of a native parse API
            system_content, anthropic_messages = self._extract_system_messages(messages)

            # Prepare tool parameters only if structured response is requested
            tool_params = self._prepare_tool_params(structured_response, response_format)

            # Single API call with conditional tool parameters
            response = self.client.messages.create(
                model=self.model_name,
                messages=anthropic_messages,
                max_tokens=max_tokens,
                system=system_content,  # type: ignore
                **tool_params,  # Only includes tools/tool_choice if structured_response=True
                **kwargs,
            )

            # Extract answer based on response type
            if not structured_response:
                answer = response.content[0].text
            else:
                # Extract structured data from tool use: find the tool_use block and instantiate the Pydantic model
                tool_use = next(block for block in response.content if block.type == "tool_use")
                answer = response_format(**tool_use.input)  # type: ignore

        if print_response and isinstance(answer, str):
            try:
                from IPython.display import (
                    Markdown,
                    display,
                )

                get_ipython()  # type: ignore # Not defined if not in a IPython environment
                display(Markdown(answer))  # Use IPython display only if in a notebook
            except (ImportError, NameError):
                print(answer)

        return answer
