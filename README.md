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

## Example Request

```json
{
  "input": {
    "source_image_url": "https://example.com/person.jpg",
    "control_video_url": "https://example.com/dance.mp4",
    "prompt": "A person dancing energetically with fluid motion",
    "width": 768,
    "height": 1280,
    "fps": 24,
    "seed": 42,
    "ic_strength": 0.71,
    "guide_strength": 0.7,
    "pose_method": "DWPose",
    "use_control_audio": true
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
