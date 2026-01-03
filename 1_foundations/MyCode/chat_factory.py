import json
from typing import (
    List,
    Optional,
)

from pydantic import BaseModel

from dotenv import (
    find_dotenv,
    load_dotenv,
)
from models import ChatModel


class Evaluation(BaseModel):
    is_acceptable: bool
    feedback: str


GENERATOR_PROMPT = """You are a helpful AI assistant.
Your responsibility is to provide accurate, professional, and engaging responses to user questions.
Be clear and concise in your answers.
If you don't know the answer to something, say so honestly rather than making up information."""

EVALUATOR_PROMPT = """You are an evaluator that decides whether a response to a question is acceptable quality.
You are provided with a conversation between a User and an Agent.
Your task is to decide whether the Agent's latest response is acceptable.
The Agent should be helpful, accurate, professional, and appropriate.
Consider whether the response:
- Accurately addresses the user's question
- Is professional and well-written
- Is complete and helpful
- Avoids making up information or being misleading
Please evaluate the latest response and provide feedback."""


def _convert_custom_tools_to_openai(custom_tools):
    """
    Convert custom tool format to OpenAI format.

    Custom format: {"function": callable, "description": str, "parameters": dict}
    OpenAI format: {"type": "function", "function": {"name": str, "description": str, "parameters": dict}}

    Args:
        custom_tools: List of tools in custom format with function references

    Returns:
        tuple: (openai_tools, tool_map)
            - openai_tools: List of tools in OpenAI format
            - tool_map: Dict mapping function names to actual functions
    """
    if not custom_tools:
        return [], {}

    openai_tools = []
    tool_map = {}

    for tool in custom_tools:
        func = tool["function"]
        func_name = func.__name__

        openai_tools.append({
            "type": "function",
            "function": {
                "name": func_name,
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        })

        tool_map[func_name] = func

    return openai_tools, tool_map


def chat_factory(
    generator_model: Optional[ChatModel] = None,
    system_prompt: str = GENERATOR_PROMPT,
    evaluator_model: Optional[ChatModel] = None,
    evaluator_system_prompt: str = EVALUATOR_PROMPT,
    response_limit: int = 5,
    tools: Optional[List] = None,
):
    load_dotenv(find_dotenv(), override=True)

    generator_model = generator_model or ChatModel(model_name="gpt-4o-mini")
    evaluator_model = evaluator_model or ChatModel(model_name="gpt-4o-mini")

    # Convert custom tools to OpenAI format and create tool map
    openai_tools, tool_map = _convert_custom_tools_to_openai(tools)

    def chat(message, history):

        def evaluator_user_prompt(user_message, agent_reply, extended_history) -> str:
            user_prompt = f"Here's the conversation between the User and the Agent: \n\n{extended_history}\n\n"
            user_prompt += f"Here's the latest message from the User: \n\n{user_message}\n\n"
            user_prompt += f"Here's the latest response from the Agent: \n\n{agent_reply}\n\n"
            user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
            return user_prompt

        def evaluate(user_message, agent_reply, extended_history) -> Evaluation:
            messages = [{"role": "system", "content": evaluator_system_prompt}] + [
                {"role": "user", "content": evaluator_user_prompt(user_message, agent_reply, extended_history)}
            ]
            return evaluator_model.generate_response(messages=messages, response_format=Evaluation)  # type: ignore

        def sanitize_messages(messages):
            """Remove extra fields from messages that may be added by Gradio or other UIs."""
            return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

        def handle_tool_call(tool_calls):
            results = []
            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                print(f"Tool called: {tool_name}", flush=True)
                tool = tool_map.get(tool_name)
                result = tool(**arguments) if tool else {}
                print(f"Tool result: {result}", flush=True)
                results.append(generator_model.format_tool_result(tool_call_id=tool_call.id, result=result))
            return results

        def get_reply(extended_history):
            messages = extended_history.copy()
            reply = generator_model.generate_response(
                messages=messages,
                tools=openai_tools,
            )
            while isinstance(reply, list):
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": reply
                })
                messages += handle_tool_call(reply)
                reply = generator_model.generate_response(
                    messages=messages,
                    tools=openai_tools,
                )
            return reply, messages

        def rerun(reply, feedback, extended_history):
            updated_system_prompt = (
                system_prompt
                + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
            )
            updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
            updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
            messages = (
                [{"role": "system", "content": updated_system_prompt}]
                + extended_history[1:]  # exclude previous system prompt
                # + [{"role": "user", "content": message}]
            )
            return get_reply(messages)

        messages = (
            [{"role": "system", "content": system_prompt}]
            + sanitize_messages(history)
            + [{"role": "user", "content": message}]
        )
        reply, extended_history = get_reply(messages)

        responses = 1
        while responses < response_limit:

            evaluation = evaluate(message, reply, extended_history)

            if evaluation.is_acceptable:
                print("Passed evaluation - returning reply")
                break
            else:
                print("Failed evaluation - retrying")
                print(evaluation.feedback)
                reply, extended_history = rerun(reply, evaluation.feedback, extended_history)
                responses += 1

        print(f"****Final response after {responses} attempt(s).")
        return reply

    return chat
