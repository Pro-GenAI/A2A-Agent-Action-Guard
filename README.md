# A2A-Agent-Action-Guard
A2A version of Agent Action Guard: Safe AI Agents through Action Classifier (https://github.com/Pro-GenAI/Agent-Action-Guard).

## Problem Statement

Autonomous AI agents can perform harmful actions when they are prompted or influenced to misuse tools. Common causes include jailbreak attempts, hallucinations or misunderstanding, overconfidence in incorrect knowledge, insufficient constraints or guidelines, inadequate training data for specific scenarios, and malicious or misleading MCP tool descriptions. Detecting and blocking potentially harmful tool calls before execution is necessary to improve agent safety and reliability.

## Abstract

The Green Agent (`harmacteval_a2a.py`) evaluates an A2A (Agent-to-Agent) model's tendency to propose harmful or unethical actions. It runs a benchmark over the curated `HarmActions_dataset.json`, dynamically registers candidate tools on an A2A-compatible MCP server, and determines whether the evaluated agent attempts to call those tools. The benchmark produces per-action outcomes and an aggregate "percent predicted harmful" score.

## Available agents:

🟣 __Competing Agent__: [a2a_eval_server.py](agent_action_guard_a2a/a2a_eval_server.py) \
🟢 __Evaluator Agent__: [harmacteval_a2a.py](agent_action_guard_a2a/harmacteval_a2a.py)

## Run the Green Agent (Evaluator)

- Prerequisites: create and activate a Python environment and install the package in editable mode:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

- The evaluator requires the environment variable `A2A_EVAL_SERVER_URL` pointing to an A2A-compatible server. This repo includes a server entrypoint at [agent_action_guard_a2a/a2a_eval_server.py](agent_action_guard_a2a/a2a_eval_server.py).

- Start the server (example binds to port 8080):

```bash
export A2A_EVAL_SERVER_URL="http://0.0.0.0:8080"
python agent_action_guard_a2a/a2a_eval_server.py &
```

- In another shell, run the evaluator:

```bash
export A2A_EVAL_SERVER_URL="http://localhost:8080"
python agent_action_guard_a2a/harmacteval_a2a.py
```

- Outputs: `harmacteval_cache.json` (cache) and `evaluation_results-<model_key>.json` (results) are created in the repository root.

## Baseline Purple Agent(s)

Baseline purple agents are simple A2A clients that issue prompts intended to provoke tool usage. Included examples:

- `unused/a2a_sample_client.py` — basic client showing how to add tools and query the A2A server.
- `agent_action_guard_a2a/purple_agent_example.py` — a small baseline adversarial client that registers a "dangerous" tool and asks the evaluated agent to invoke it. Run after the A2A server is active:

```bash
export A2A_EVAL_SERVER_URL="http://localhost:8080"
python agent_action_guard_a2a/purple_agent_example.py
```

These examples illustrate how the benchmark checks whether the red agent executes registered tools when presented with adversarial prompts.

## Docker image — Packaged Green Agent

A `Dockerfile` is provided to run the A2A server and the Green Agent evaluator together in one container. The container starts the A2A server and then runs the evaluation end-to-end.

Build the image:

```bash
docker build -t a2a-harmacteval:latest .
```


Run the container (example binds container port 8080 to host port 8080).

Note: The A2A server relies on an LLM client. Provide `OPENAI_MODEL` and the provider credentials (for OpenAI: `OPENAI_API_KEY`) when running the container. Example using OpenAI environment variables:

```bash
mkdir -p out
docker run --rm \
	-e A2A_EVAL_SERVER_URL="http://0.0.0.0:8080" \
	-e OPENAI_MODEL="gpt-4o-mini" \
	-e OPENAI_API_KEY="$OPENAI_API_KEY" \
	-v $(pwd)/out:/out \
	-p 8080:8080 \
	a2a-harmacteval:latest
```

The container will start the server, run the evaluation, and copy `evaluation_results-*.json` and `harmacteval_cache.json` to the mounted `out` directory when finished.

---

If you'd like, I can add a `docker-compose.yml` that runs the green agent plus external MCP server(s), or extend the Dockerfile to publish images to a registry — tell me which you prefer.


