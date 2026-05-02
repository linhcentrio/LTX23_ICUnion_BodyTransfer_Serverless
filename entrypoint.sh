#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export AUX_ANNOTATOR_CKPTS_PATH="${AUX_ANNOTATOR_CKPTS_PATH:-/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts}"

echo "[entrypoint] Khởi động ComfyUI..."
cd /workspace/ComfyUI
python main.py --listen 127.0.0.1 --port "${COMFY_PORT:-8188}" &
COMFY_PID=$!

cleanup() {
  if kill -0 "${COMFY_PID}" 2>/dev/null; then
    kill "${COMFY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[entrypoint] Chờ ComfyUI sẵn sàng..."
for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${COMFY_PORT:-8188}/" >/dev/null; then
    echo "[entrypoint] ComfyUI OK."
    break
  fi
  sleep 1
  if [[ "${i}" -eq 180 ]]; then
    echo "[entrypoint] Timeout chờ ComfyUI." >&2
    exit 1
  fi
done

echo "[entrypoint] Workflow API: ${WORKFLOW_API_DIR:-/app/workflows/api}"
echo "[entrypoint] Chạy RunPod handler..."
exec python /app/handler.py
