FROM python:3.13-slim

WORKDIR /app

# Copy repository into container
COPY . /app

# Install package and dependencies
RUN pip install --no-cache-dir -e .

# Default A2A server URL (container expects the server to listen on this host:port)
ENV A2A_EVAL_SERVER_URL=http://0.0.0.0:8080

# Expose server port
EXPOSE 8080

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Entrypoint starts the A2A server, waits for readiness, runs evaluation, and copies results to /out if mounted.
CMD ["/app/entrypoint.sh"]
