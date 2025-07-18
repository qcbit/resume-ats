# Use a Python image with uv pre-installed
FROM python:3.12-slim-bookworm

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

ARG PORT=5000
ENV PORT=${PORT:-5000}

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

COPY services/match-scorer-openai /app/services/match-scorer

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY pyproject.toml uv.lock /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

# Reset the entrypoint, don't invoke `uv`
ENTRYPOINT []

# Run the FastAPI application by default
# Use a shell to substitute the PORT environment variable
CMD ["sh", "-c", "cd /app/services/match-scorer; uv run main.py --port $PORT"]