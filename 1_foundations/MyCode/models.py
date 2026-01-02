import os
from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display


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
            # Anthropic uses a separate 'system' parameter instead of system messages in the array
            system_content = None
            anthropic_messages = messages
            if messages and messages[0].get("role") == "system":
                system_content = messages[0]["content"]
                anthropic_messages = messages[1:]  # Remove system message from messages array

            if not structured_response:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=anthropic_messages,
                    max_tokens=max_tokens,
                    system=system_content,  # type: ignore
                    **kwargs,
                )
                answer = response.content[0].text
            else:
                # Use tool use for structured output with Anthropic
                tools = [
                    {
                        "name": "structured_response",
                        "description": "Return a structured response",
                        "input_schema": response_format.model_json_schema(),  # type: ignore
                    }
                ]
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=anthropic_messages,
                    max_tokens=max_tokens,
                    system=system_content,  # type: ignore
                    tools=tools,  # type: ignore
                    tool_choice={"type": "tool", "name": "structured_response"},
                    **kwargs,
                )
                # Extract the tool use result and parse into Pydantic model
                tool_use = next(block for block in response.content if block.type == "tool_use")
                answer = response_format(**tool_use.input)  # type: ignore
        if print_response and isinstance(answer, str):
            try:
                get_ipython()  # type: ignore
                display(Markdown(answer))
            except NameError:
                print(answer)
        return answer
