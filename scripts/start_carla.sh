#!/usr/bin/env bash
set -euo pipefail
# Official CARLA offscreen mode; expose RPC ports only to the local machine.
CARLA_NAME="${CARLA_NAME:-grounded-drive-carla}"
CARLA_PORT="${CARLA_PORT:-2000}"
CARLA_TOWN="${CARLA_TOWN:-Town01}"
CARLA_END_PORT=$((CARLA_PORT+2))
# A stopped container retains its name. Reuse it on subsequent invocations.
if CARLA_STATUS="$(docker container inspect --format '{{.State.Status}}' "$CARLA_NAME" 2>/dev/null)"; then
  case "$CARLA_STATUS" in
    running)
      echo "CARLA container $CARLA_NAME is already running."
      ;;
    created|exited)
      echo "Starting existing CARLA container $CARLA_NAME with its saved configuration."
      docker start "$CARLA_NAME"
      ;;
    restarting)
      echo "CARLA container $CARLA_NAME is restarting."
      ;;
    *)
      echo "CARLA container $CARLA_NAME is $CARLA_STATUS; resolve that state or choose another CARLA_NAME." >&2
      exit 1
      ;;
  esac
  exit 0
fi
docker run -d --name "$CARLA_NAME" --gpus all \
  -p "127.0.0.1:${CARLA_PORT}-${CARLA_END_PORT}:${CARLA_PORT}-${CARLA_END_PORT}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 bash CarlaUE4.sh "/Game/Carla/Maps/${CARLA_TOWN}" \
  -RenderOffScreen -nosound -quality-level=Low "-carla-rpc-port=${CARLA_PORT}"
