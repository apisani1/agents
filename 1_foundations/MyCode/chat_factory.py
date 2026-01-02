from typing import Optional

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


def chat_factory(
    generator_model: Optional[ChatModel] = None,
    system_prompt: str = GENERATOR_PROMPT,
    evaluator_model: Optional[ChatModel] = None,
    evaluator_system_prompt: str = EVALUATOR_PROMPT,
    response_limit: int = 5,
):
    load_dotenv(find_dotenv(), override=True)

    generator_model = generator_model or ChatModel(model_name="gpt-4o-mini")
    evaluator_model = evaluator_model or ChatModel(model_name="gpt-4o-mini")

    def chat(message, history):

        def evaluator_user_prompt(reply):
            user_prompt = f"Here's the conversation between the User and the Agent: \n\n{history}\n\n"
            user_prompt += f"Here's the latest message from the User: \n\n{message}\n\n"
            user_prompt += f"Here's the latest response from the Agent: \n\n{reply}\n\n"
            user_prompt += "Please evaluate the response, replying with whether it is acceptable and your feedback."
            return user_prompt

        def evaluate(reply) -> Evaluation:
            messages = [{"role": "system", "content": evaluator_system_prompt}] + [
                {"role": "user", "content": evaluator_user_prompt(reply)}
            ]
            return evaluator_model.generate_response(
                messages=messages, structured_response=True, response_format=Evaluation
            )  # type: ignore

        def rerun(reply, feedback):
            updated_system_prompt = (
                system_prompt
                + "\n\n## Previous answer rejected\nYou just tried to reply, but the quality control rejected your reply\n"
            )
            updated_system_prompt += f"## Your attempted answer:\n{reply}\n\n"
            updated_system_prompt += f"## Reason for rejection:\n{feedback}\n\n"
            messages = (
                [{"role": "system", "content": updated_system_prompt}]
                + history
                + [{"role": "user", "content": message}]
            )
            return generator_model.generate_response(messages)

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        reply = generator_model.generate_response(messages)

        responses = 1
        while responses < response_limit:

            evaluation = evaluate(reply)

            if evaluation.is_acceptable:
                print("Passed evaluation - returning reply")
                break
            else:
                print("Failed evaluation - retrying")
                print(evaluation.feedback)
                reply = rerun(reply, evaluation.feedback)
                responses += 1

        print(f"****Final response after {responses} attempt(s).")
        return reply

    return chat
