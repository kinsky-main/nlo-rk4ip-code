FROM ghcr.io/openai/codex-universal:latest

LABEL org.opencontainers.image.title="nlolib-codex-universal"
LABEL org.opencontainers.image.description="Local test container for nlolib SIMD changes"
LABEL org.opencontainers.image.vendor="nlolib"

COPY docker/scripts/install_deps.sh /tmp/install_deps.sh
RUN bash /tmp/install_deps.sh && rm /tmp/install_deps.sh

WORKDIR /workspace/nlolib
