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
        self, messages, *, max_tokens=10000, structured_response=False, response_format=None, print_response=False, **kwargs
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
            response = self.client.messages.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                **kwargs,
            )
            answer = response.content[0].text
        if print_response:
            display(Markdown(answer))
        return answer
