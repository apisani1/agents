from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display

OPENAI_CLIENT_ENDPOINT_MAP = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "deepseek": "https://api.deepseek.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}


class ChatModel:
    def __init__(self, model_name, provider="openai", api_key=None, **kwargs):
        self.model_name = model_name
        if provider in OPENAI_CLIENT_ENDPOINT_MAP:
            self.client = OpenAI(base_url=OPENAI_CLIENT_ENDPOINT_MAP[provider], api_key=api_key, **kwargs)
        elif provider == "anthropic":
            self.client = Anthropic(api_key=api_key, **kwargs)
        else:
            raise ValueError("Unsupported provider")

    def generate_response(self, messages, *, max_tokens=10000, print_response=False, **kwargs):
        if isinstance(self.client, OpenAI):
            response = self.client.chat.completions.create(model=self.model_name, messages=messages, **kwargs)
            answer = response.choices[0].message.content
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
