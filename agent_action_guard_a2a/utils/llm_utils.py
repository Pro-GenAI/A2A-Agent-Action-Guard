import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI


load_dotenv()

# Initialize Cohere client
model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise Exception("OPENAI_MODEL is not set in the environment variables")
print(f"Model: {model}")
model_key = model.split("/")[-1].lower()

llm = ChatOpenAI(
    model=model,
    # api_key=lambda: api_key,
    # base_url=base_url,
    reasoning_effort="low",
    max_retries=3,
)

# Create the prompt
SYSTEM_PROMPT = "You are a helpful assistant that can use tools."

# Create the agent
agent_tools = {}
custom_agent = create_agent(llm, tools=[], system_prompt=SYSTEM_PROMPT)


def reset_agent(tools: Optional[List[Any]] = None) -> None:
    if tools is None:
        tools = []
    global custom_agent
    custom_agent = create_agent(llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def get_agent_response(
    messages: str | List[BaseMessage] | List[Dict[str, str]],
) -> str:
    """Get response from an OpenAI-compatible API with automatic caching."""
    if not messages:
        raise ValueError("Messages cannot be empty")

    if isinstance(messages, str):
        messages = [HumanMessage(content=messages)]

    conversation_data = custom_agent.invoke(
        {"messages": messages},  # type: ignore
        # context={"user_role": "expert"}
    )
    if not conversation_data:
        raise Exception("Empty response from the bot")

    # Extract the final response text
    response_data: AIMessage = conversation_data["messages"][-1]
    if isinstance(response_data.content, list):
        response_text = "\n".join(str(response_data.content))
    else:
        response_text = str(response_data.content)

    return response_text
