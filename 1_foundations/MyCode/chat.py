import gradio as gr
from chat_factory import chat_factory
from models import ChatModel
from pypdf import PdfReader
from dotenv import (
    find_dotenv,
    load_dotenv,
)


reader = PdfReader("../me/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

with open("../me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Ed Donner"

ED_GENERATOR_PROMPT = f"""You are acting as {name}. You are answering questions on {name}'s website,
particularly questions related to {name}'s career, background, skills and experience.
Your responsibility is to represent {name} for interactions on the website as faithfully as possible.
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions.
Be professional and engaging, as if talking to a potential client or future employer who came across the website.
If you don't know the answer, say so.
## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n
With this context, please chat with the user, always staying in character as {name}."""

ED_EVALUATOR_PROMPT = f"""You are an evaluator that decides whether a response to a question is acceptable.
You are provided with a conversation between a User and an Agent. Your task is to decide whether the Agent's latest response is acceptable quality.
The Agent is playing the role of {name} and is representing {name} on their website.
The Agent has been instructed to be professional and engaging, as if talking to a potential client or future employer who came across the website.
The Agent has been provided with context on {name} in the form of their summary and LinkedIn details. Here's the information:"
## Summary:\n{summary}\n\n## LinkedIn Profile:\n{linkedin}\n"
With this context, please evaluate the latest response, replying with whether the response is acceptable and your feedback."""


def main():
    generator = ChatModel(model_name="gpt-4o-mini", provider="openai")
    evaluator = ChatModel(model_name="claude-sonnet-4-5", provider="anthropic")
    gr.ChatInterface(
        chat_factory(
            generator_model=generator,
            system_prompt=ED_GENERATOR_PROMPT,
            evaluator_model=evaluator,
            evaluator_system_prompt=ED_EVALUATOR_PROMPT,
        ),
        type="messages",
    ).launch()


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    main()
