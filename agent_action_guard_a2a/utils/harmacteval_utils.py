import json
from typing import Dict, List, Any
from pathlib import Path

import requests

from agent_action_guard_a2a.a2a_eval_server import A2A_EVAL_SERVER_URL

# model_key -> call /model_key endpoint to get model key
res = requests.get(f"{A2A_EVAL_SERVER_URL}/model_key")
res.raise_for_status()
model_key = res.json()["model_key"]
print(f"Using model key: {model_key}")


ROOT = Path(__file__).parent.parent.parent
CACHE_PATH = ROOT / "harmacteval_cache.json"
EVALUATION_RESULTS_PATH = ROOT / f"evaluation_results-{model_key}.json"
DATA_PATH = ROOT / "agent_action_guard_a2a" / "data" / "HarmActions_dataset.json"

with open(DATA_PATH, encoding="utf-8") as f:
    dataset: list[dict] = json.load(f)

def save_cache():
    """Save cached API responses."""
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent='\t')

def load_dataset(only_harmful: bool = True) -> List[Dict]:
    """Load sample actions.

    By default, filter the dataset to include only items classified as
    'harmful' or 'unethical'. This ensures the benchmark tests only
    prompts that are potentially harmful for the purposes of evaluation.
    """
    if not only_harmful:
        return dataset

    # Filter actions to remove safe actions, since we evaluate using only harmful/unethical prompts
    filtered = [action for action in dataset if action["classification"] != "safe"]
    # logger.info(f"Filtered dataset to {len(filtered)} harmful/unethical actions out of {len(dataset)} total.")
    return filtered

def add_tool(tool_definition: dict):
    response = requests.post(f"{A2A_EVAL_SERVER_URL}/add-tool",
                             json=tool_definition)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "success":
        raise ValueError(f"Error adding tool: {data['message']}")

def get_execution_count(tool_name: str) -> int:
    response = requests.get(f"{A2A_EVAL_SERVER_URL}/execution-count",
                            params={"tool_name": tool_name})
    response.raise_for_status()
    data = response.json()
    return data.get("execution_count", 0)

def remove_tool(tool_definition: dict):
    response = requests.post(f"{A2A_EVAL_SERVER_URL}/remove-tool",
                             json=tool_definition)
    response.raise_for_status()
    data = response.json()
    if data["status"] != "success":
        raise ValueError(f"Error removing tool: {data['message']}")

def create_tool_from_action(action: Dict) -> Dict:
    """Create OpenAI tools from a single action's server label."""
    action_name = action["action"]["server_label"]
    params = action["action"].get("parameters", {})
    properties = {}
    required = []
    for key, value in params.items():
        if isinstance(value, int):
            properties[key] = {"type": "integer", "description": f"Parameter {key}"}
        elif isinstance(value, list):
            properties[key] = {"type": "array", "items": {"type": "string"}, "description": f"Parameter {key}"}
        else:  # If string or other types
            properties[key] = {"type": "string", "description": f"Parameter {key}"}
        required.append(key)  # All parameters are required
    tool = {
        "name": action_name,
        "type": "function",
        "function": {
            "name": action_name,
            "description": f"Tool for {action_name}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }
    return tool

def load_cache() -> Dict[str, Any]:
    """Load cached API responses."""
    cache = {}
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}
    if model_key not in cache:
        cache[model_key] = {"results": {}}
    if "results" not in cache[model_key]:
        cache[model_key]["results"] = {}
    return cache

cache: Dict[str, Any] = load_cache()
