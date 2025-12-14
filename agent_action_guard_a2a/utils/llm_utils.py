import json
import logging
import os
from typing import Any, Callable, Dict

from dotenv import load_dotenv
import openai


load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s %(levelname)s %(name)s: %(message)s')


# Initialize Cohere client
model = os.getenv("OPENAI_MODEL", "")
if not model:
    raise Exception("OPENAI_MODEL is not set in the environment variables")
print(f"Model: {model}")
model_key = model.split("/")[-1].lower()


# Create the prompt
SYSTEM_PROMPT = "You are a helpful assistant that can use tools."


client = openai.OpenAI()

agent_tools = {}  # tool_name: tool_details
execution_counts = {}  # tool_name: count


def call_tool(tool_name: str, *args, **kwargs) -> Any:
    """Wrapper to call a tool and track execution counts."""
    tool = agent_tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool {tool_name} not found.")

    if tool_name not in execution_counts:
        execution_counts[tool_name] = 0
    execution_counts[tool_name] += 1

    print(f"Called {tool_name} with args: {args}, kwargs: {kwargs}")
    return f"Successfully called tool {tool_name} with kwargs: {kwargs}"


def call_agent(prompt: str) -> str:
    """Get API response, using cache if available."""
    try:
        # Create tools_param from agent_tools
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            tools=list(agent_tools.values()),
            # reasoning_effort="low" if "gpt-oss" in model_key else "minimal",  # type: ignore
            tool_choice="required",
            max_completion_tokens=1000,
        )
        result = response.choices[0]
        # If we're in local/emulation mode, execute any function tool calls ourselves

        msg_obj = result.message or {}
        if not msg_obj.tool_calls:
            raise Exception("The model has returned empty tool calls.")

        tool_calls_done = set()
        for action in msg_obj.tool_calls:
            if action.type != "function":
                raise Exception(f"Unknown tool call type: {action.type}")

            fn = action.function or {}
            raw_args = fn.arguments or "{}"
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args or {}
            result = call_tool(fn.name, **args)
            tool_calls_done.add(f"{fn.name} returned: {result}")

        return "\n".join(tool_calls_done)

    except openai.RateLimitError as e:
        logger.warning(f"--->WARNING:Rate limit error calling API for prompt: {prompt[:50]}...: {e}")
        raise e
    except Exception as e:
        raise e


if __name__ == "__main__":
    # Test the agent

    agent_tools["get_weather"] = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The location to get the weather for.",
                    }
                },
                "required": ["location"],
            },
        },
    }

    response = call_agent("What's the weather in New York?")
    print("Agent response:", response)
