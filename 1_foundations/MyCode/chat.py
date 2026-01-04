import os
import requests

from dotenv import (
    find_dotenv,
    load_dotenv,
)
import gradio as gr
from pypdf import PdfReader

from chat_factory import chat_factory
from models import ChatModel


load_dotenv(find_dotenv(), override=True)


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        },
    )


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    """Record user details for follow-up contact.

    Args:
        email: The email address of this user
        name: The user's name, if they provided it
        notes: Any additional information about the conversation

    Returns:
        dict: Confirmation of recording
    """
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question: str):
    """Record questions that couldn't be answered.

    Args:
        question: The question that couldn't be answered

    Returns:
        dict: Confirmation of recording
    """
    push(f"Recording {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "function": record_user_details,
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "function": record_unknown_question,
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

# Tool Definition Options:
# Option 1: Simplest - just pass functions (auto-generates schema from type hints and docstrings)
# tools = [record_user_details, record_unknown_question]

# Option 2: Auto-generate with description override
# tools = [
#     {"function": record_user_details, "description": "Custom description"},
#     {"function": record_unknown_question}
# ]

# Option 3: Manual schema (backward compatible - current approach)
tools = [
    record_user_details_json,
    record_unknown_question_json,
]


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
            tools=tools,
        ),
        type="messages",
    ).launch()


if __name__ == "__main__":
    main()
