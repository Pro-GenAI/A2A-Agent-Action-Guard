#!/usr/bin/env python3
"""A2A server entrypoint."""

import os
from typing import Any, List, Optional

from urllib.parse import urlparse

from flask import request
from langchain_core.tools import tool


from python_a2a import (
    A2AServer,
    skill,
    agent,
    Message,
    MessageRole,
    TextContent,
)
from python_a2a.server.http import create_flask_app

# Reference: https://pypi.org/project/python-a2a

from agent_action_guard_a2a.utils.llm_utils import (
    reset_agent,
    get_agent_response,
    agent_tools,
    model,
)

A2A_EVAL_SERVER_URL = os.getenv("A2A_EVAL_SERVER_URL", "").rstrip("/")
if not A2A_EVAL_SERVER_URL:
    raise ValueError("A2A_EVAL_SERVER_URL environment variable not set.")

parsed_url = urlparse(A2A_EVAL_SERVER_URL)
SERVER_PORT = parsed_url.port or 0
if not SERVER_PORT:
    raise ValueError("A2A_EVAL_SERVER_URL must include a port.")
SERVER_HOST = parsed_url.hostname or "0.0.0.0"


@agent(
    name="Agent", description="Uses tools and answers user queries.", version="1.0.0"
)
class CustomAgent(A2AServer):

    @skill(name="Use tool", description="Use the selected tool.", tags=["tool-user"])
    def use_tool(self, tool_name: str, tool_args: dict) -> str:
        """Use the selected tool."""
        print(f"Using tool: {tool_name} with args: {tool_args}")
        return call_tool(tool_name, **tool_args)

    def handle_message(self, message):
        print("Received message:", message)
        if message.content.type == "text":
            if message.role == MessageRole.USER:
                # Get agent response for user messages
                print("Processing user message...")
                response = get_agent_response(message.content.text)
                print("Agent response:", response)

                return Message(
                    content=TextContent(text=response),  # Simply echo the text
                    role=MessageRole.AGENT,
                    parent_message_id=message.message_id,
                    conversation_id=message.conversation_id,
                )
            else:
                print(f"Unhandled message role: {message.role}")
        elif message.content.type == "function_call":
            # Basic echo for function calls when no handler is defined
            print(f"Received function call: {message.content}")
            raise NotImplementedError("Function call handling not implemented.")
        else:
            print("Received non-text message content:", message.content)
            raise NotImplementedError("Non-text message handling not implemented.")

agent_server = CustomAgent(
    url=A2A_EVAL_SERVER_URL + "/a2a",
)

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
    return f"Successfully called tool {tool_name}."


app = create_flask_app(agent_server)


@app.post("/add-tool")
def add_tool() -> dict:
    """Endpoint to dynamically add a new tool to the MCP server."""

    tool_definition = request.get_json()
    tool_definition = tool_definition.get("tool_definition", {})

    print("Adding tool:", tool_definition)

    # Validate the tool
    tool_name = tool_definition.get("name", "")
    if not tool_name:
        return {
            "status": "error",
            "message": "Tool definition must include a 'name' field.",
        }
    if tool_name in agent_tools:
        return {"status": "success", "message": f"Tool {tool_name} already exists."}
    tool_description = tool_definition.get("description", "")
    if not tool_description:
        return {
            "status": "error",
            "message": "Tool definition must include a 'description' field.",
        }
    schema = tool_definition.get("parameters", {})
    if not schema:
        return {
            "status": "error",
            "message": "Tool definition must include a 'parameters' field.",
        }

    # Add the tool to the MCP server
    agent_tools[tool_name] = tool(
        name_or_callable=lambda *args, **kwargs: call_tool(tool_name, *args, **kwargs),
        description=tool_description,
        args_schema=schema,
    )
    reset_agent(tools=[agent_tools[tool_name]])

    return {
        "status": "success",
        "message": f"Tool {tool_definition.get('name')} added.",
    }


@app.get("/execution-count")
def get_execution_count() -> dict:
    """Endpoint to get the execution count of a tool."""
    tool_name = request.args.get('tool_name', '')
    count = execution_counts.get(tool_name, 0)
    return {"tool_name": tool_name, "execution_count": count}


@app.post("/remove-tool")
def remove_tool() -> dict:
    """Endpoint to dynamically remove a tool from the MCP server."""

    tool_definition = request.get_json()
    tool_definition = tool_definition.get("tool_definition", {})

    # print("Received request to remove tool:", tool_definition)
    tool_name = tool_definition.get("name", "")
    if not tool_name:
        return {
            "status": "error",
            "message": "Tool definition must include a 'name' field.",
        }
    print("Removing tool:", tool_name)

    if tool_name not in agent_tools:
        print(f"Tool {tool_name} not found.")
        return {"status": "error", "message": f"Tool {tool_name} not found."}

    del agent_tools[tool_name]
    if tool_name in execution_counts:
        del execution_counts[tool_name]
    reset_agent()
    return {"status": "success", "message": f"Tool {tool_name} removed."}


@app.get("/")
def root() -> str:
    """Root endpoint to verify server is running."""
    return "MCP Eval Server is running."


if __name__ == "__main__":
    print(f"Starting A2A server on http://{SERVER_HOST}:{SERVER_PORT}/a2a")

    # Add info about Google A2A compatibility if available
    if hasattr(agent_server, "_use_google_a2a"):
        google_compat = getattr(agent_server, "_use_google_a2a", False)
        print(f"Google A2A compatibility: {'Enabled' if google_compat else 'Disabled'}")

    app.run(host=SERVER_HOST, port=SERVER_PORT, debug=True)
