#!/usr/bin/env bash
# Maintainer-only: builds this repo's Dockerfile and pushes it to GHCR so
# end users can `docker compose pull` instead of building locally (see
# docker-compose.yml, which points at the image this script publishes).
# Never run automatically -- a human decides when a new version ships.
#
# One-time setup on whatever machine runs this (needs Docker + a GitHub
# Personal Access Token with `write:packages` scope):
#   echo "$GHCR_TOKEN" | docker login ghcr.io -u rdkmedia0 --password-stdin
# Then, on GitHub, set the ghcr.io/rdkmedia0/dream-pipeline package
# visibility to Public (Package settings) so `docker pull` needs no login
# on the consuming side.
set -euo pipefail

IMAGE="ghcr.io/rdkmedia0/dream-pipeline"
TAG="${1:-latest}"

cd "$(dirname "$0")"
docker build -t "${IMAGE}:${TAG}" .
if [ "$TAG" != "latest" ]; then
    docker tag "${IMAGE}:${TAG}" "${IMAGE}:latest"
fi
docker push "${IMAGE}:${TAG}"
if [ "$TAG" != "latest" ]; then
    docker push "${IMAGE}:latest"
fi

echo "Published ${IMAGE}:${TAG}$( [ "$TAG" != "latest" ] && echo " (and :latest)" )."
