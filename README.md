# LTX-2.3 IC-Union Body Transfer — RunPod Serverless

ComfyUI-based serverless endpoint for LTX-2.3 body movement transfer using IC-Union Control LoRA.

**Pipeline:** Source image + Control video → DWPose/SDPose extraction → IC-LoRA guided 2-pass LTX sampling → MP4 output.

## API Input Schema

### Required

| Field | Type | Description |
|---|---|---|
| `source_image_url` | string | URL of the source reference image |
| `control_video_url` | string | URL of the control/driving video |

Alternative input methods: `*_base64` (base64-encoded data) or `*_path` (local path on worker).

### Prompts

| Field | Type | Default | Description |
|---|---|---|---|
| `positive_prompt` / `prompt` | string | (from workflow) | Positive prompt text |
| `negative_prompt` | string | (from workflow) | Negative prompt text |
| `enable_prompt_enhancer` | bool | `false` | Use LTX-2 prompt enhancer |

### Resolution / Duration

| Field | Type | Default | Description |
|---|---|---|---|
| `width` | int | `736` | Output video width |
| `height` | int | `1280` | Output video height |
| `length_seconds` | int | `10` | Video duration in seconds |
| `fps` | float | `24` | Frames per second |

### Sampling

| Field | Type | Default | Description |
|---|---|---|---|
| `seed` | int | `-1` (random) | Random seed (-1 = random) |
| `cfg` | float | `1.0` | CFG scale for both passes |
| `sampler_pass1` | string | `euler_ancestral_cfg_pp` | Sampler for pass 1 |
| `sampler_pass2` | string | `euler_cfg_pp` | Sampler for pass 2 |
| `sigmas_pass1` | string | `1.0, 0.99375, ...` | Manual sigmas for pass 1 |
| `sigmas_pass2` | string | `0.85, 0.7250, ...` | Manual sigmas for pass 2 |

### IC-LoRA / Guide

| Field | Type | Default | Description |
|---|---|---|---|
| `ic_strength` | float | `0.71` | IC-LoRA model strength |
| `guide_strength` / `pose_strength` | float | `0.7` | Pose guide strength |
| `i2v_inplace_strength` | float | `0.7` | Image-to-video inplace strength (pass 2) |
| `img_compression` | int | `18` | LTXVPreprocess image compression |

### Pose / Depth

| Field | Type | Default | Description |
|---|---|---|---|
| `pose_method` | string | `DWPose` | `DWPose` or `SDPose` |
| `blend_pose_depth` | bool | `false` | Blend pose with depth map |
| `blend_factor` | float | `0.5` | Pose-depth blend factor |

### Audio

| Field | Type | Default | Description |
|---|---|---|---|
| `use_control_audio` | bool | `true` | Use audio from control video |

### NAG (Normalized Attention Guidance)

| Field | Type | Default | Description |
|---|---|---|---|
| `nag_scale` | float | `11.0` | NAG scale |
| `nag_alpha` | float | `0.25` | NAG alpha |
| `nag_tau` | float | `2.5` | NAG tau |

### Output

| Field | Type | Default | Description |
|---|---|---|---|
| `output_format` | string | `base64` | `base64` or `minio` |
| `output_key` | string | auto | MinIO object key (when format=minio) |

## Example Requests

### 1) Minimal — chỉ cần source + control + prompt

```json
{
  "input": {
    "source_image_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo3_video---b788bf3f-aca7-4297-8ae3-d8808a425689.mp4",
    "source_image_filename": "demo3_video.mp4",
    "control_video_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo2_video---6297b4c0-bebe-428b-b177-0623dc74b11c.mp4",
    "control_video_filename": "demo2_video.mp4",
    "prompt": "A person dancing naturally, same identity and appearance from the source reference, realistic motion transfer, high quality video",
    "negative_prompt": "low quality, blurry, distorted body, deformed hands, bad anatomy, artifacts",
    "pose_method": "DWPose",
    "output_format": "minio"
  }
}
```

### 2) Full parameters — seed, resolution, sampling, IC-LoRA tuning

```json
{
  "input": {
    "source_image_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo3_video---b788bf3f-aca7-4297-8ae3-d8808a425689.mp4",
    "source_image_filename": "demo3_video.mp4",
    "control_video_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo2_video---6297b4c0-bebe-428b-b177-0623dc74b11c.mp4",
    "control_video_filename": "demo2_video.mp4",
    "prompt": "A person dancing naturally, same identity and appearance from the source reference, realistic motion transfer, high quality video",
    "negative_prompt": "low quality, blurry, distorted body, deformed hands, bad anatomy, artifacts",
    "pose_method": "DWPose",
    "seed": 42,
    "width": 768,
    "height": 1280,
    "length_seconds": 10,
    "fps": 24,
    "cfg": 1.0,
    "ic_strength": 0.71,
    "guide_strength": 0.7,
    "i2v_inplace_strength": 0.7,
    "img_compression": 18,
    "sampler_pass1": "euler_ancestral_cfg_pp",
    "sampler_pass2": "euler_cfg_pp",
    "use_control_audio": true,
    "output_format": "minio"
  }
}
```

### 3) SDPose — dùng SDPose thay DWPose

```json
{
  "input": {
    "source_image_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo3_video---b788bf3f-aca7-4297-8ae3-d8808a425689.mp4",
    "source_image_filename": "demo3_video.mp4",
    "control_video_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo2_video---6297b4c0-bebe-428b-b177-0623dc74b11c.mp4",
    "control_video_filename": "demo2_video.mp4",
    "prompt": "A person dancing naturally, same identity and appearance from the source reference, realistic motion transfer, high quality video",
    "pose_method": "SDPose",
    "seed": 123,
    "width": 736,
    "height": 1280,
    "output_format": "minio"
  }
}
```

### 4) Pose + Depth blend — thêm depth map cho guide chính xác hơn

```json
{
  "input": {
    "source_image_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo3_video---b788bf3f-aca7-4297-8ae3-d8808a425689.mp4",
    "source_image_filename": "demo3_video.mp4",
    "control_video_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo2_video---6297b4c0-bebe-428b-b177-0623dc74b11c.mp4",
    "control_video_filename": "demo2_video.mp4",
    "prompt": "A person dancing naturally, same identity and appearance from the source reference, realistic motion transfer, high quality video",
    "pose_method": "DWPose",
    "blend_pose_depth": true,
    "blend_factor": 0.5,
    "seed": -1,
    "output_format": "minio"
  }
}
```

### 5) High quality — landscape, NAG tuning, tăng IC strength

```json
{
  "input": {
    "source_image_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo3_video---b788bf3f-aca7-4297-8ae3-d8808a425689.mp4",
    "source_image_filename": "demo3_video.mp4",
    "control_video_url": "http://media.aiclip.ai/video/body-transfer-input/test-876d3a52-7af3-46c2-86c2-dcad3d3bdeec/demo2_video---6297b4c0-bebe-428b-b177-0623dc74b11c.mp4",
    "control_video_filename": "demo2_video.mp4",
    "prompt": "A person dancing naturally, same identity and appearance from the source reference, realistic motion transfer, cinematic quality",
    "negative_prompt": "low quality, blurry, distorted body, deformed hands, bad anatomy, artifacts, jitter, flicker",
    "pose_method": "DWPose",
    "seed": 777,
    "width": 1280,
    "height": 736,
    "length_seconds": 5,
    "fps": 24,
    "ic_strength": 0.8,
    "guide_strength": 0.75,
    "cfg": 1.0,
    "nag_scale": 11.0,
    "nag_alpha": 0.25,
    "nag_tau": 2.5,
    "use_control_audio": true,
    "enable_prompt_enhancer": true,
    "output_format": "minio"
  }
}
```

## Docker Build

```bash
docker build -t ltx-bodytransfer-serverless .
```

## Custom Nodes Included

- ComfyUI-KJNodes
- ComfyUI-GGUF
- ComfyUI-VideoHelperSuite
- comfyui_controlnet_aux
- ComfyUI-LTXVideo
- ComfyUI-Easy-Use
- rgthree-comfy
