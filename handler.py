"""
RunPod Serverless: LTX-2.3 IC-Union body transfer (workflow Studio DWPose/SDPose).
Kiến trúc tương tự Infinitetalk: ComfyUI + websocket + workflow API + tùy chọn MinIO.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import runpod
import websocket

try:
    from dotenv import load_dotenv

    load_dotenv("/app/.env", override=False)
except ImportError:
    pass

try:
    from minio import Minio
    from urllib.parse import quote as urlquote
except ImportError:
    Minio = None
    urlquote = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1")
COMFY_PORT = int(os.environ.get("COMFY_PORT", "8188"))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", COMFY_HOST)
INPUT_DIR = Path(os.environ.get("COMFY_INPUT_DIR", "/workspace/ComfyUI/input"))
OUTPUT_DIR = Path(os.environ.get("COMFY_OUTPUT_DIR", "/workspace/ComfyUI/output"))
API_DIR = Path(os.environ.get("WORKFLOW_API_DIR", "/app/workflows/api"))

# Hai file workflow API đã export (mặc định trùng tên trong workflows/api/)
_WORKFLOW_DW_DEFAULT = "LTX-2.3_-_IV2V_TV2V_transfer_body_movements_IC-Union-Control-lora_DWPose.json"
_WORKFLOW_SD_DEFAULT = "LTX-2.3_-_IV2V_TV2V_transfer_body_movements_IC-Union-Control-lora_SDPose.json"
WORKFLOW_PATH_DWPOSE = Path(
    os.environ.get("WORKFLOW_PATH_DWPOSE", str(API_DIR / _WORKFLOW_DW_DEFAULT))
)
WORKFLOW_PATH_SDPOSE = Path(
    os.environ.get("WORKFLOW_PATH_SDPOSE", str(API_DIR / _WORKFLOW_SD_DEFAULT))
)

NODE_LOAD_IMAGE = os.environ.get("NODE_LOAD_IMAGE", "2004")
NODE_CONTROL_VIDEO = os.environ.get("NODE_CONTROL_VIDEO", "5192")
NODE_CLIP_POSITIVE = os.environ.get("NODE_CLIP_POSITIVE", "2483")
NODE_CLIP_NEGATIVE = os.environ.get("NODE_CLIP_NEGATIVE", "2612")

CLIENT_ID = str(uuid.uuid4())


def _normalize_pose_mode(raw: Any) -> str:
    """
    Trả về 'dwpose' hoặc 'sdpose'.
    Chấp nhận: DWPose, dwpose, SDPose, sdpose, pose_method từ Studio, v.v.
    """
    if raw is None:
        return "dwpose"
    s = str(raw).strip().lower()
    if s in ("sdpose", "sd-pose", "sd_pose") or s.startswith("sdpose"):
        return "sdpose"
    if s in ("dwpose", "dw-pose", "dw_pose") or s.startswith("dwpose"):
        return "dwpose"
    if "sdpose" in s.replace("-", "").replace("_", ""):
        return "sdpose"
    return "dwpose"


def workflow_path_for_mode(mode: str) -> Path:
    return WORKFLOW_PATH_SDPOSE if mode == "sdpose" else WORKFLOW_PATH_DWPOSE


def resolve_pose_mode_from_job(job_input: Dict[str, Any]) -> str:
    """Ưu tiên: pose_method -> pose_mode -> workflow (chỉ nhận dwpose|sdpose)."""
    if "pose_method" in job_input:
        return _normalize_pose_mode(job_input.get("pose_method"))
    if "pose_mode" in job_input:
        return _normalize_pose_mode(job_input.get("pose_mode"))
    w = job_input.get("workflow")
    if isinstance(w, str) and w.strip().lower() in ("dwpose", "sdpose"):
        return w.strip().lower()
    return "dwpose"


def _minio_client():
    if Minio is None:
        return None
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    if not endpoint:
        return None
    access = os.environ.get("MINIO_ACCESS_KEY", "")
    secret = os.environ.get("MINIO_SECRET_KEY", "")
    secure = os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes")
    try:
        return Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
    except Exception as exc:
        logger.warning("MinIO không khởi tạo được: %s", exc)
        return None


MINIO_CLIENT = _minio_client()
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "video")


def download_file_from_url(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["wget", "-q", "-O", str(output_path), "--timeout=60", url],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if r.returncode != 0:
        raise RuntimeError(f"wget thất bại: {url} {r.stderr}")
    return output_path


def save_base64_to_file(b64: str, output_path: Path) -> Path:
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    pad = len(b64) % 4
    if pad:
        b64 += "=" * (4 - pad)
    data = base64.b64decode(b64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)
    return output_path


def fetch_media(job: Dict[str, Any], url_key: str, b64_key: str, path_key: str, dest: Path) -> None:
    if job.get(path_key):
        p = Path(job[path_key])
        if not p.is_file():
            raise FileNotFoundError(f"Không có file: {p}")
        shutil.copy2(p, dest)
        return
    if job.get(url_key):
        download_file_from_url(job[url_key], dest)
        return
    if job.get(b64_key):
        save_base64_to_file(job[b64_key], dest)
        return
    raise ValueError(f"Cần một trong: {url_key}, {b64_key}, {path_key}")


def load_workflow_api(pose_mode: str) -> Tuple[Dict[str, Any], Path]:
    path = workflow_path_for_mode(pose_mode)
    if not path.is_file():
        raise FileNotFoundError(
            f"Thiếu workflow API: {path}. Đặt file export vào {API_DIR} hoặc WORKFLOW_PATH_DWPOSE / WORKFLOW_PATH_SDPOSE."
        )
    logger.info("Đang tải workflow API: %s (mode=%s)", path.name, pose_mode)
    return json.loads(path.read_text(encoding="utf-8")), path


def patch_workflow(
    prompt: Dict[str, Any],
    source_image_name: str,
    control_video_name: str,
    positive: Optional[str],
    negative: Optional[str],
) -> Dict[str, Any]:
    """Ghi đè ảnh nguồn, video điều khiển và prompt (theo node id Studio)."""
    out = json.loads(json.dumps(prompt))

    def set_if_present(nid: str, key: str, value: Any) -> None:
        if nid not in out:
            logger.warning("Không thấy node %s trong workflow API", nid)
            return
        if "inputs" not in out[nid]:
            out[nid]["inputs"] = {}
        out[nid]["inputs"][key] = value

    set_if_present(NODE_LOAD_IMAGE, "image", source_image_name)
    set_if_present(NODE_CONTROL_VIDEO, "video", control_video_name)
    if positive is not None:
        set_if_present(NODE_CLIP_POSITIVE, "text", positive)
    if negative is not None:
        set_if_present(NODE_CLIP_NEGATIVE, "text", negative)
    return out


def queue_prompt(prompt: Dict[str, Any]) -> Dict[str, Any]:
    url = f"http://{SERVER_ADDRESS}:{COMFY_PORT}/prompt"
    payload = json.dumps({"prompt": prompt, "client_id": CLIENT_ID}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ComfyUI /prompt lỗi {e.code}: {detail}") from e


def get_history(prompt_id: str) -> Dict[str, Any]:
    url = f"http://{SERVER_ADDRESS}:{COMFY_PORT}/history/{prompt_id}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def collect_output_videos(hist: Dict[str, Any], prompt_id: str) -> List[Tuple[str, str]]:
    """Trả về [(đường_dẫn_local, filename), ...]."""
    found: List[Tuple[str, str]] = []
    chunk = hist.get(prompt_id) if prompt_id in hist else None
    if not chunk:
        return found
    outputs = chunk.get("outputs", {})
    for _node_id, node_out in outputs.items():
        for key in ("gifs", "videos", "images"):
            if key not in node_out:
                continue
            for item in node_out[key]:
                fn = item.get("filename")
                sub = item.get("subfolder", "")
                if not fn:
                    continue
                candidates = [
                    OUTPUT_DIR / sub / fn if sub else OUTPUT_DIR / fn,
                    Path("/workspace/ComfyUI/temp") / fn,
                    Path(fn),
                ]
                for p in candidates:
                    if p.is_file():
                        found.append((str(p), fn))
                        break
    return found


def wait_ws_and_collect(ws_url: str, prompt: Dict[str, Any]) -> List[Tuple[str, str]]:
    ws = websocket.WebSocket()
    for attempt in range(60):
        try:
            ws.connect(ws_url)
            break
        except Exception as e:
            logger.warning("WebSocket thử %s/60: %s", attempt + 1, e)
            time.sleep(2)
    else:
        raise RuntimeError("Không kết nối được WebSocket ComfyUI")

    try:
        q = queue_prompt(prompt)
        if q.get("error"):
            raise RuntimeError(f"ComfyUI từ chối prompt: {q}")
        prompt_id = q.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"Không có prompt_id: {q}")

        while True:
            raw = ws.recv()
            if not isinstance(raw, str):
                continue
            msg = json.loads(raw)
            if msg.get("type") == "executing":
                data = msg.get("data") or {}
                if data.get("node") is None and data.get("prompt_id") == prompt_id:
                    break
    finally:
        ws.close()

    hist = get_history(prompt_id)
    return collect_output_videos(hist, prompt_id)


def upload_minio(local_path: str, object_name: str) -> str:
    if not MINIO_CLIENT:
        raise RuntimeError("MinIO chưa cấu hình")
    MINIO_CLIENT.fput_object(MINIO_BUCKET, object_name, local_path)
    ep = os.environ.get("MINIO_ENDPOINT", "")
    scheme = "https" if os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes") else "http"
    if urlquote:
        return f"{scheme}://{ep}/{MINIO_BUCKET}/{urlquote(object_name)}"
    return f"{scheme}://{ep}/{MINIO_BUCKET}/{object_name}"


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input") or {}
    job_id = job.get("id", "job")
    task_dir = Path("/tmp") / f"ltx_bt_{job_id}_{uuid.uuid4().hex[:8]}"
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        pose_mode = resolve_pose_mode_from_job(job_input)
        prompt_template, workflow_path = load_workflow_api(pose_mode)

        ext_img = Path(job_input.get("source_image_filename", "source.png")).suffix or ".png"
        ext_vid = Path(job_input.get("control_video_filename", "control.mp4")).suffix or ".mp4"
        source_name = f"{job_id}_source{ext_img}"
        control_name = f"{job_id}_control{ext_vid}"

        src_tmp = task_dir / "_source.bin"
        ctl_tmp = task_dir / "_control.bin"
        fetch_media(job_input, "source_image_url", "source_image_base64", "source_image_path", src_tmp)
        fetch_media(job_input, "control_video_url", "control_video_base64", "control_video_path", ctl_tmp)

        shutil.copy2(src_tmp, INPUT_DIR / source_name)
        shutil.copy2(ctl_tmp, INPUT_DIR / control_name)

        positive = job_input.get("positive_prompt") or job_input.get("prompt")
        negative = job_input.get("negative_prompt")

        graph = patch_workflow(
            prompt_template,
            source_image_name=source_name,
            control_video_name=control_name,
            positive=positive,
            negative=negative,
        )

        ws_url = f"ws://{SERVER_ADDRESS}:{COMFY_PORT}/ws?clientId={CLIENT_ID}"
        outputs = wait_ws_and_collect(ws_url, graph)
        if not outputs:
            return {"status": "error", "error": "Không thu được video từ ComfyUI (kiểm tra history / node VHS_VideoCombine)."}

        local_path, filename = outputs[0]
        output_format = (job_input.get("output_format") or "base64").lower()

        meta = {
            "pose_mode": pose_mode,
            "pose_method": "SDPose" if pose_mode == "sdpose" else "DWPose",
            "workflow_api": workflow_path.name,
        }

        if output_format == "minio":
            key = job_input.get("output_key", f"ltx-bodytransfer/{job_id}/{filename}")
            url = upload_minio(local_path, key)
            return {
                "status": "success",
                "output": {"video_url": url, "filename": filename},
                "metadata": meta,
            }

        with open(local_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return {
            "status": "success",
            "output": {"video_base64": b64, "filename": filename},
            "metadata": meta,
        }
    except Exception as exc:
        logger.exception("Job lỗi")
        return {"status": "error", "error": str(exc), "error_type": type(exc).__name__}
    finally:
        shutil.rmtree(task_dir, ignore_errors=True)


runpod.serverless.start({"handler": handler})
