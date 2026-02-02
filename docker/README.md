# Docker Environments

This folder contains Dockerfiles for testing nlolib in multiple deployment environments.

## Layout

- `docker/codex-universal/` - OpenAI Codex universal base image.
- `docker/scripts/` - Shared install/build scripts used by Dockerfiles.

## Build and Run

Example (codex-universal):

```
docker build -t nlolib-codex -f docker/codex-universal/Dockerfile .
docker run --rm -it \
  -e CODEX_ENV_PYTHON_VERSION=3.12 \
  -e CODEX_ENV_NODE_VERSION=20 \
  -e CODEX_ENV_RUST_VERSION=1.87.0 \
  -e CODEX_ENV_GO_VERSION=1.23.8 \
  -e CODEX_ENV_SWIFT_VERSION=6.2 \
  -e CODEX_ENV_RUBY_VERSION=3.4.4 \
  -e CODEX_ENV_PHP_VERSION=8.4 \
  -v $(pwd):/workspace/$(basename $(pwd)) \
  -w /workspace/$(basename $(pwd)) \
  nlolib-codex
```

Note: Build from the repo root (context `.`) so the Dockerfile can access `docker/scripts/*`.

Inside the container, you can run:

```
bash docker/scripts/build_and_test.sh
```
