"""
RunPod Serverless: LTX-2.3 IC-Union body transfer (workflow Studio DWPose/SDPose).
Kiến trúc tương tự Infinitetalk: ComfyUI + websocket + workflow API + tùy chọn MinIO.

Feature parity với Colab v5: expose ~20 tham số qua job input.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
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

_WORKFLOW_DW_DEFAULT = "LTX-2.3_-_IV2V_TV2V_transfer_body_movements_IC-Union-Control-lora_DWPose.json"
_WORKFLOW_SD_DEFAULT = "LTX-2.3_-_IV2V_TV2V_transfer_body_movements_IC-Union-Control-lora_SDPose.json"
WORKFLOW_PATH_DWPOSE = Path(os.environ.get("WORKFLOW_PATH_DWPOSE", str(API_DIR / _WORKFLOW_DW_DEFAULT)))
WORKFLOW_PATH_SDPOSE = Path(os.environ.get("WORKFLOW_PATH_SDPOSE", str(API_DIR / _WORKFLOW_SD_DEFAULT)))

CLIENT_ID = str(uuid.uuid4())

# ── Workflow Node ID Mapping (shared between DWPose & SDPose) ──────────────
# Input media
NODE_LOAD_IMAGE = "2004"           # LoadImage — source reference image
NODE_CONTROL_VIDEO = "5192"        # VHS_LoadVideoFFmpeg — control/driving video

# Prompts
NODE_PROMPT_TEXT = "5242"          # PrimitiveStringMultiline — raw prompt value
NODE_CLIP_POSITIVE = "2483"        # CLIPTextEncode (positive) — linked via switch
NODE_CLIP_NEGATIVE = "2612"        # CLIPTextEncode (negative)
NODE_ENABLE_ENHANCER = "5201"      # PrimitiveBoolean — enable prompt enhancer

# Resolution / Duration
NODE_WIDTH = "5206"                # INTConstant — output WIDTH
NODE_HEIGHT = "5207"               # INTConstant — output HEIGHT
NODE_LENGTH_SECONDS = "5205"       # INTConstant — LENGTH in seconds
NODE_FPS = "5199"                  # PrimitiveFloat — FPS

# Sampling
NODE_SEED_PASS1 = "4832"           # RandomNoise (pass 1)
NODE_SEED_PASS2 = "5068"           # RandomNoise (pass 2)
NODE_SAMPLER_PASS1 = "4831"        # KSamplerSelect (pass 1)
NODE_SAMPLER_PASS2 = "5070"        # KSamplerSelect (pass 2)
NODE_SIGMAS_PASS1 = "5025"         # ManualSigmas (pass 1)
NODE_SIGMAS_PASS2 = "5071"         # ManualSigmas (pass 2)
NODE_CFG_PASS1 = "4828"            # CFGGuider (pass 1)
NODE_CFG_PASS2 = "5069"            # CFGGuider (pass 2)

# IC-LoRA / Guide
NODE_IC_LORA = "5011"              # LTXICLoRALoaderModelOnly — ic_strength
NODE_GUIDE_STRENGTH = "5299"       # PrimitiveFloat — POSE STRENGTH
NODE_I2V_INPLACE = "5067"          # LTXVImgToVideoInplace — pass 2 re-anchor
NODE_IMG_PREPROCESS = "3336"       # LTXVPreprocess — img_compression
NODE_T2V_MODE = "5198"             # PrimitiveBoolean — text-to-video mode

# Pose / Depth blend
NODE_BLEND_SWITCH = "5272"         # ComfySwitchNode — BLEND POSE & DEPTH?
NODE_BLEND_FACTOR = "5115"         # ImageBlend — blend_factor

# Audio (API: 5264 lấy audio cho 5208 — switch 5303 False → 5076 LTX decode; True → 5274)
NODE_CUSTOM_AUDIO_SWITCH = "5303"  # PrimitiveBoolean — bật nhánh audio từ input (5274), tắt = LTX decode
NODE_CUSTOM_FILE_SWITCH = "5274"   # ComfySwitchNode — From custom audio file? (True=file 5263, False=5273)
NODE_AUDIO_FROM_VIDEO = "5273"     # ComfySwitchNode — From input video? (True=audio 5192, False=EmptyAudio)
NODE_LOAD_AUDIO = "5263"           # LoadAudio — luôn trỏ file hợp lệ (custom hoặc silent placeholder)

# Tên file silent.wav placeholder để tránh ComfyUI fail validate node 5263
# kể cả khi nhánh custom audio không được kích hoạt.
SILENT_AUDIO_NAME = "silent.wav"

# NAG
NODE_NAG = "5251"                  # LTX2_NAG — nag_scale, nag_alpha, nag_tau

# Video output (VHS_VideoCombine) — kết quả LTX cuối (decode từ 5075). Node 5120 chỉ là preview pose (AnimateDiff), không dùng làm output.
NODE_FINAL_VIDEO_COMBINE = "5208"

# Defaults matching Colab v5
DEFAULT_NEGATIVE = ("low contrast, washed out, text, subtitles, logo, still image, "
                    "still video, blurry, low quality, distorted, bad anatomy, oversaturated, "
                    "pixelated, low resolution, grainy, compression artifacts, jpeg artifacts, "
                    "glitches, watermark, signature, copyright, distortedsound, saturated sound, "
                    "loud sound, deformed facial features, asymmetrical face, missing facial features, "
                    "extra limbs, disfigured hands, blurry teeth, disfigured teeth")
DEFAULT_SIGMAS_PASS1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_SIGMAS_PASS2 = "0.85, 0.7250, 0.4219, 0.0"


# ── Pose mode helpers ──────────────────────────────────────────────────────
def _normalize_pose_mode(raw: Any) -> str:
    if raw is None:
        return "dwpose"
    s = str(raw).strip().lower().replace("-", "").replace("_", "")
    return "sdpose" if "sdpose" in s else "dwpose"


def resolve_pose_mode_from_job(job_input: Dict[str, Any]) -> str:
    for key in ("pose_method", "pose_mode", "workflow"):
        if key in job_input:
            return _normalize_pose_mode(job_input[key])
    return "dwpose"


# ── MinIO ──────────────────────────────────────────────────────────────────
def _minio_client():
    if Minio is None:
        return None
    endpoint = os.environ.get("MINIO_ENDPOINT", "").strip()
    if not endpoint:
        return None
    try:
        return Minio(
            endpoint,
            access_key=os.environ.get("MINIO_ACCESS_KEY", ""),
            secret_key=os.environ.get("MINIO_SECRET_KEY", ""),
            secure=os.environ.get("MINIO_SECURE", "false").lower() in ("1", "true", "yes"),
        )
    except Exception as exc:
        logger.warning("MinIO không khởi tạo được: %s", exc)
        return None


MINIO_CLIENT = _minio_client()
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "video")


# ── File helpers ───────────────────────────────────────────────────────────
def download_file_from_url(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["wget", "-q", "-O", str(output_path), "--timeout=60", url],
        capture_output=True, text=True, timeout=600,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(b64))
    return output_path


def ensure_silent_audio(input_dir: Path, name: str = SILENT_AUDIO_NAME) -> str:
    """Tạo file silent.wav 1 giây stereo 44.1kHz trong INPUT_DIR (lazy, idempotent).

    Mục đích: ép node LoadAudio (5263) trong workflow JSON luôn trỏ tới một file
    hợp lệ để ComfyUI validate qua được, kể cả khi nhánh custom audio không kích hoạt.
    """
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / name
    if target.is_file() and target.stat().st_size > 0:
        return name
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-t", "1", "-c:a", "pcm_s16le", str(target),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0 or not target.is_file():
            raise RuntimeError(r.stderr.strip() or "ffmpeg trả mã != 0")
    except Exception as exc:
        # Fallback: ghi tay header WAV PCM 1 giây stereo (silence)
        logger.warning("ffmpeg tạo %s thất bại (%s) — dùng fallback ghi WAV thủ công", name, exc)
        sample_rate = 44100
        channels = 2
        bits_per_sample = 16
        n_samples = sample_rate  # 1s
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = n_samples * block_align
        header = b"RIFF" + (36 + data_size).to_bytes(4, "little") + b"WAVE"
        fmt_chunk = (
            b"fmt " + (16).to_bytes(4, "little")
            + (1).to_bytes(2, "little")
            + channels.to_bytes(2, "little")
            + sample_rate.to_bytes(4, "little")
            + byte_rate.to_bytes(4, "little")
            + block_align.to_bytes(2, "little")
            + bits_per_sample.to_bytes(2, "little")
        )
        data_chunk = b"data" + data_size.to_bytes(4, "little") + (b"\x00" * data_size)
        target.write_bytes(header + fmt_chunk + data_chunk)
    return name


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


# ── Workflow loading & patching ────────────────────────────────────────────
def load_workflow_api(pose_mode: str) -> Tuple[Dict[str, Any], Path]:
    path = WORKFLOW_PATH_SDPOSE if pose_mode == "sdpose" else WORKFLOW_PATH_DWPOSE
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu workflow API: {path}")
    logger.info("Đang tải workflow API: %s (mode=%s)", path.name, pose_mode)
    return json.loads(path.read_text(encoding="utf-8")), path


def _set(graph: dict, nid: str, key: str, value: Any) -> None:
    """Set a node input value, replacing any existing link or value."""
    if nid not in graph:
        logger.warning("Node %s không có trong workflow", nid)
        return
    if "inputs" not in graph[nid]:
        graph[nid]["inputs"] = {}
    graph[nid]["inputs"][key] = value


def patch_workflow(
    prompt: Dict[str, Any],
    *,
    source_image_name: str,
    control_video_name: str,
    positive: Optional[str],
    negative: Optional[str],
    # Resolution / Duration
    width: Optional[int] = None,
    height: Optional[int] = None,
    length_seconds: Optional[int] = None,
    fps: Optional[float] = None,
    # Sampling
    seed: Optional[int] = None,
    cfg: Optional[float] = None,
    sampler_pass1: Optional[str] = None,
    sampler_pass2: Optional[str] = None,
    sigmas_pass1: Optional[str] = None,
    sigmas_pass2: Optional[str] = None,
    # IC-LoRA / Guide
    ic_strength: Optional[float] = None,
    guide_strength: Optional[float] = None,
    i2v_inplace_strength: Optional[float] = None,
    img_compression: Optional[int] = None,
    # Pose / Depth
    blend_pose_depth: Optional[bool] = None,
    blend_factor: Optional[float] = None,
    # Audio
    use_control_audio: Optional[bool] = None,
    use_ltx_native_audio: Optional[bool] = None,
    custom_audio_name: Optional[str] = None,
    has_custom_audio: bool = False,
    # NAG
    nag_scale: Optional[float] = None,
    nag_alpha: Optional[float] = None,
    nag_tau: Optional[float] = None,
    # Misc
    enable_prompt_enhancer: Optional[bool] = None,
) -> Dict[str, Any]:
    """Patch tất cả tham số tunable vào workflow JSON trước khi gửi ComfyUI."""
    g = json.loads(json.dumps(prompt))  # deep copy

    # ── Input media ──
    _set(g, NODE_LOAD_IMAGE, "image", source_image_name)
    _set(g, NODE_CONTROL_VIDEO, "video", control_video_name)

    # ── Prompts ──
    if positive is not None:
        _set(g, NODE_PROMPT_TEXT, "value", positive)
        # Also set directly on CLIPTextEncode in case prompt enhancer is off
        _set(g, NODE_CLIP_POSITIVE, "text", positive)
    if negative is not None:
        _set(g, NODE_CLIP_NEGATIVE, "text", negative)

    # ── Resolution / Duration ──
    if width is not None:
        _set(g, NODE_WIDTH, "value", int(width))
    if height is not None:
        _set(g, NODE_HEIGHT, "value", int(height))
    if length_seconds is not None:
        _set(g, NODE_LENGTH_SECONDS, "value", int(length_seconds))
    if fps is not None:
        _set(g, NODE_FPS, "value", float(fps))

    # ── Sampling ──
    if seed is not None:
        _set(g, NODE_SEED_PASS1, "noise_seed", int(seed))
        _set(g, NODE_SEED_PASS2, "noise_seed", int(seed) + 1)
    if cfg is not None:
        _set(g, NODE_CFG_PASS1, "cfg", float(cfg))
        _set(g, NODE_CFG_PASS2, "cfg", float(cfg))
    if sampler_pass1 is not None:
        _set(g, NODE_SAMPLER_PASS1, "sampler_name", sampler_pass1)
    if sampler_pass2 is not None:
        _set(g, NODE_SAMPLER_PASS2, "sampler_name", sampler_pass2)
    if sigmas_pass1 is not None:
        _set(g, NODE_SIGMAS_PASS1, "sigmas", sigmas_pass1)
    if sigmas_pass2 is not None:
        _set(g, NODE_SIGMAS_PASS2, "sigmas", sigmas_pass2)

    # ── IC-LoRA / Guide ──
    if ic_strength is not None:
        _set(g, NODE_IC_LORA, "strength_model", float(ic_strength))
    if guide_strength is not None:
        _set(g, NODE_GUIDE_STRENGTH, "value", float(guide_strength))
    if i2v_inplace_strength is not None:
        _set(g, NODE_I2V_INPLACE, "strength", float(i2v_inplace_strength))
    if img_compression is not None:
        _set(g, NODE_IMG_PREPROCESS, "img_compression", int(img_compression))

    # ── Pose / Depth blend ──
    if blend_pose_depth is not None:
        _set(g, NODE_BLEND_SWITCH, "switch", bool(blend_pose_depth))
    if blend_factor is not None:
        _set(g, NODE_BLEND_FACTOR, "blend_factor", float(blend_factor))

    # ── Audio ──
    # 5263.audio LUÔN cần trỏ tới một file hợp lệ (silent.wav placeholder hoặc
    # file user cung cấp), nếu không ComfyUI sẽ fail validate cả workflow dù
    # nhánh custom audio không kích hoạt.
    if custom_audio_name:
        _set(g, NODE_LOAD_AUDIO, "audio", custom_audio_name)
        # Xoá audioUI để tránh ComfyUI tham chiếu ngược tên file cũ trong UI cache
        if NODE_LOAD_AUDIO in g and isinstance(g[NODE_LOAD_AUDIO].get("inputs"), dict):
            g[NODE_LOAD_AUDIO]["inputs"].pop("audioUI", None)

    # 5303 → switch của 5264 (Use Selected Audio): True = nhánh 5274 (input mux),
    # False = 5076 LTXVAudioVAEDecode (audio do LTX tự sinh trong latent).
    # Mặc định: luôn dùng input (video control hoặc file custom), không dùng LTX decode.
    # Có custom audio thì luôn ưu tiên nhánh input (bỏ qua use_ltx_native_audio).
    use_ltx_decode = bool(use_ltx_native_audio) and not has_custom_audio
    _set(g, NODE_CUSTOM_AUDIO_SWITCH, "value", not use_ltx_decode)

    # 5274: True = dùng audio file (5263→Trim); False = dùng 5273 (video vs EmptyAudio).
    _set(g, NODE_CUSTOM_FILE_SWITCH, "switch", bool(has_custom_audio))

    # 5273: use_control_audio=True → audio track từ VHS_LoadVideoFFmpeg (5192);
    # False → EmptyAudio (im lặng). Chỉ áp dụng khi has_custom_audio=False (5274=false).
    if use_control_audio is not None:
        _set(g, NODE_AUDIO_FROM_VIDEO, "switch", bool(use_control_audio))

    # ── NAG ──
    if nag_scale is not None:
        _set(g, NODE_NAG, "nag_scale", float(nag_scale))
    if nag_alpha is not None:
        _set(g, NODE_NAG, "nag_alpha", float(nag_alpha))
    if nag_tau is not None:
        _set(g, NODE_NAG, "nag_tau", float(nag_tau))

    # ── Prompt enhancer ──
    if enable_prompt_enhancer is not None:
        _set(g, NODE_ENABLE_ENHANCER, "value", bool(enable_prompt_enhancer))

    return g


# ── ComfyUI communication ─────────────────────────────────────────────────
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
    """Chỉ trả về video từ node 5208 (pipeline LTX cuối). Không trả preview pose (5120 / AnimateDiff)."""
    chunk = hist.get(prompt_id)
    if not chunk:
        return []
    outputs = chunk.get("outputs", {})
    final: List[Tuple[str, str]] = []
    node_out = None
    for nid, out in outputs.items():
        if str(nid) == NODE_FINAL_VIDEO_COMBINE:
            node_out = out
            break
    if not node_out:
        return []
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
                    final.append((str(p), fn))
                    break
    return final


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


# ── Job parameter extraction ──────────────────────────────────────────────
def _get_float(d: dict, *keys, default=None) -> Optional[float]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except (ValueError, TypeError):
                pass
    return default


def _get_int(d: dict, *keys, default=None) -> Optional[int]:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return int(d[k])
            except (ValueError, TypeError):
                pass
    return default


def _get_bool(d: dict, *keys, default=None) -> Optional[bool]:
    for k in keys:
        if k in d and d[k] is not None:
            v = d[k]
            if isinstance(v, bool):
                return v
            if isinstance(v, str):
                return v.lower() in ("1", "true", "yes")
            return bool(v)
    return default


def _get_str(d: dict, *keys, default=None) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            s = str(d[k]).strip()
            if s:
                return s
    return default


def extract_params(job_input: Dict[str, Any]) -> Dict[str, Any]:
    """Extract tất cả tham số từ job input, trả về dict cho patch_workflow."""
    seed = _get_int(job_input, "seed", "noise_seed", default=-1)
    if seed is None or seed < 0:
        seed = random.randint(0, 2**32 - 1)

    return {
        # Resolution / Duration
        "width": _get_int(job_input, "width", "output_width"),
        "height": _get_int(job_input, "height", "output_height"),
        "length_seconds": _get_int(job_input, "length_seconds", "duration"),
        "fps": _get_float(job_input, "fps", "frame_rate"),
        # Sampling
        "seed": seed,
        "cfg": _get_float(job_input, "cfg"),
        "sampler_pass1": _get_str(job_input, "sampler_pass1", "sampler"),
        "sampler_pass2": _get_str(job_input, "sampler_pass2"),
        "sigmas_pass1": _get_str(job_input, "sigmas_pass1"),
        "sigmas_pass2": _get_str(job_input, "sigmas_pass2"),
        # IC-LoRA / Guide
        "ic_strength": _get_float(job_input, "ic_strength"),
        "guide_strength": _get_float(job_input, "guide_strength", "pose_strength"),
        "i2v_inplace_strength": _get_float(job_input, "i2v_inplace_strength"),
        "img_compression": _get_int(job_input, "img_compression"),
        # Pose / Depth
        "blend_pose_depth": _get_bool(job_input, "blend_pose_depth"),
        "blend_factor": _get_float(job_input, "blend_factor"),
        # Audio
        "use_control_audio": _get_bool(job_input, "use_control_audio"),
        "use_ltx_native_audio": _get_bool(job_input, "use_ltx_native_audio"),
        # NAG
        "nag_scale": _get_float(job_input, "nag_scale"),
        "nag_alpha": _get_float(job_input, "nag_alpha"),
        "nag_tau": _get_float(job_input, "nag_tau"),
        # Misc
        "enable_prompt_enhancer": _get_bool(job_input, "enable_prompt_enhancer"),
    }


# ── Main handler ──────────────────────────────────────────────────────────
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input") or {}
    job_id = job.get("id", "job")
    task_dir = Path("/tmp") / f"ltx_bt_{job_id}_{uuid.uuid4().hex[:8]}"
    task_dir.mkdir(parents=True, exist_ok=True)

    try:
        INPUT_DIR.mkdir(parents=True, exist_ok=True)
        pose_mode = resolve_pose_mode_from_job(job_input)
        prompt_template, workflow_path = load_workflow_api(pose_mode)
        params = extract_params(job_input)

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

        # ── Custom audio (optional) ─────────────────────────────────────────
        # Nếu user truyền audio_url/audio_base64/audio_path → dùng làm custom audio
        # và bật switch 5303. Ngược lại tạo silent.wav placeholder để node 5263
        # vẫn validate được (kể cả khi nhánh custom không kích hoạt).
        has_custom_audio = bool(
            job_input.get("audio_url")
            or job_input.get("audio_base64")
            or job_input.get("audio_path")
        )
        if has_custom_audio:
            ext_aud = Path(job_input.get("audio_filename", "audio.wav")).suffix or ".wav"
            custom_audio_name = f"{job_id}_audio{ext_aud}"
            aud_tmp = task_dir / "_audio.bin"
            fetch_media(job_input, "audio_url", "audio_base64", "audio_path", aud_tmp)
            shutil.copy2(aud_tmp, INPUT_DIR / custom_audio_name)
        else:
            custom_audio_name = ensure_silent_audio(INPUT_DIR)

        positive = job_input.get("positive_prompt") or job_input.get("prompt")
        negative = job_input.get("negative_prompt")

        graph = patch_workflow(
            prompt_template,
            source_image_name=source_name,
            control_video_name=control_name,
            positive=positive,
            negative=negative,
            custom_audio_name=custom_audio_name,
            has_custom_audio=has_custom_audio,
            **params,
        )

        ws_url = f"ws://{SERVER_ADDRESS}:{COMFY_PORT}/ws?clientId={CLIENT_ID}"
        outputs = wait_ws_and_collect(ws_url, graph)
        if not outputs:
            return {
                "status": "error",
                "error": (
                    "Không có video LTX từ node 5208 (pipeline sinh cuối). "
                    "Thường do UNET/LoRA/VAE không load được hoặc prompt lỗi — không dùng preview pose (AnimateDiff)."
                ),
            }

        local_path, filename = outputs[0]
        output_format = (job_input.get("output_format") or "base64").lower()

        meta = {
            "pose_mode": pose_mode,
            "pose_method": "SDPose" if pose_mode == "sdpose" else "DWPose",
            "workflow_api": workflow_path.name,
            "seed": params.get("seed"),
            "width": params.get("width"),
            "height": params.get("height"),
            "fps": params.get("fps"),
            "ic_strength": params.get("ic_strength"),
            "guide_strength": params.get("guide_strength"),
            "cfg": params.get("cfg"),
        }
        # Remove None values from meta
        meta = {k: v for k, v in meta.items() if v is not None}

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
