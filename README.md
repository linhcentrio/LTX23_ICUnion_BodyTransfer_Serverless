# LTX23 IC-Union Body Transfer — RunPod Serverless

Chạy **ComfyUI + workflow JSON (API)** trên RunPod Serverless. Hai graph:

- `workflows/api/LTX-2.3_-_IV2V_TV2V_transfer_body_movements_IC-Union-Control-lora_DWPose.json`
- `workflows/api/LTX-2.3_-_IV2V_TV2V_transfer_body_movements_IC-Union-Control-lora_SDPose.json`

Export API từ ComfyUI (Save API format) rồi đặt vào `workflows/api/` trước khi build image.

## Cấu trúc

| File | Vai trò |
|------|---------|
| `Dockerfile` | ComfyUI, custom nodes (KJNodes, GGUF, VHS, controlnet_aux, LTXVideo), model weights |
| `entrypoint.sh` | Khởi động ComfyUI → `handler.py` |
| `handler.py` | `/prompt` + WebSocket, patch input, trả base64 hoặc MinIO |

## Input job

- Ảnh nguồn: `source_image_url` \| `source_image_base64` \| `source_image_path`
- Video điều khiển: `control_video_url` \| `control_video_base64` \| `control_video_path`
- Chọn graph: `pose_method` `DWPose` / `SDPose` (hoặc `pose_mode` / `workflow`: `dwpose` / `sdpose`)
- Tùy chọn: `positive_prompt` / `prompt`, `negative_prompt`, `output_format` (`base64` \| `minio`), `output_key`

Đầu vào graph: **ảnh tham chiếu** (LoadImage ~node `2004`) + **video điều khiển** (VHS ~node `5192`). Ghi đè node id qua env nếu export của bạn khác (xem `.env.example`).

## Build RunPod

Build image (model lớn, cần thời gian). Endpoint: `ENTRYPOINT` chạy handler RunPod.
