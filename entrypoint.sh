#!/usr/bin/env bash
set -euo pipefail

# Start A2A eval server in background
echo "Starting A2A eval server..."
python agent_action_guard_a2a/a2a_eval_server.py &
server_pid=$!

# Normalize A2A_EVAL_SERVER_URL for local checks (replace 0.0.0.0 with 127.0.0.1)
CHECK_URL="${A2A_EVAL_SERVER_URL/http:\/\/0.0.0.0/http://127.0.0.1}"

echo "Waiting for A2A server at $CHECK_URL..."
retries=0
max_retries=30
until curl -sSf "$CHECK_URL/" >/dev/null 2>&1; do
  retries=$((retries+1))
  if [ "$retries" -ge "$max_retries" ]; then
    echo "A2A server did not become ready after $max_retries attempts. Exiting."
    kill "$server_pid" || true
    exit 1
  fi
  sleep 1
done

echo "A2A server is ready. Running evaluation..."

python agent_action_guard_a2a/harmacteval_a2a.py
rc=$?

echo "Evaluation finished with exit code $rc"

# Copy results to /out if mounted
if [ -d "/out" ]; then
  echo "Copying results to /out"
  cp -v evaluation_results-* /out/ 2>/dev/null || true
  cp -v harmacteval_cache.json /out/ 2>/dev/null || true
fi

# Wait for server process to exit (or kill it)
kill "$server_pid" || true
exit "$rc"
