#!/usr/bin/env bash
# Convenience runner for Dockerized CME environment on Linux/macOS
set -euo pipefail

IMAGE_NAME="cme:latest"

usage() {
    cat << EOF
Usage: ./utils/docker_run.sh [command] [options]

Commands:
  build             Build the CME Docker container image
  test              Run full test suite (test_small + test_exhaustive + GPU mock)
  cli <args...>     Run cme_cli with current directory mounted into container
  compose <svc>     Run via docker-compose (cme, test, verify)
  shell             Open an interactive bash shell inside the container

Examples:
  ./utils/docker_run.sh build
  ./utils/docker_run.sh test
  ./utils/docker_run.sh cli examples/michaelis_menten.net --d 4 --reduce --out out/mm
  ./utils/docker_run.sh compose test
EOF
    exit 1
}

CMD="${1:-help}"

case "$CMD" in
  build)
    echo "==> Building CME Docker image ($IMAGE_NAME)..."
    docker build -t "$IMAGE_NAME" .
    ;;
  test)
    echo "==> Running full CME test suite inside Docker..."
    docker run --rm --entrypoint test_exhaustive "$IMAGE_NAME"
    ;;
  cli)
    shift
    mkdir -p out
    docker run --rm \
      -v "$(pwd):/cme/data" \
      -w /cme/data \
      "$IMAGE_NAME" "$@"
    ;;
  compose)
    shift
    docker compose up "${1:-test}"
    ;;
  shell)
    docker run --rm -it \
      -v "$(pwd):/cme/data" \
      -w /cme/data \
      --entrypoint /bin/bash \
      "$IMAGE_NAME"
    ;;
  *)
    usage
    ;;
esac
