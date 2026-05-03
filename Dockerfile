# LTX-2.3 IC-Union body transfer — ComfyUI + RunPod Serverless (workflow Studio DWPose/SDPose).
# Ảnh nguồn + video điều khiển; model khớp file LTX-2.3-Workflows Control-reference.

FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    COMFY_PORT=8188 \
    COMFY_ROOT=/workspace/ComfyUI \
    AUX_ANNOTATOR_CKPTS_PATH=/workspace/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts \
    WORKFLOW_API_DIR=/app/workflows/api

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "${COMFY_ROOT}" \
    && pip install --no-cache-dir -r "${COMFY_ROOT}/requirements.txt" \
    && pip install --no-cache-dir onnxruntime-gpu

# Custom nodes (khớp workflow API / Colab v5)
RUN cd "${COMFY_ROOT}/custom_nodes" \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
    && git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git \
    && git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git \
    && git clone --depth 1 https://github.com/yolain/ComfyUI-Easy-Use.git \
    && git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git

RUN for d in "${COMFY_ROOT}/custom_nodes"/*/; do \
      [ -f "${d}requirements.txt" ] && pip install --no-cache-dir -r "${d}requirements.txt" || true; \
    done

# Model paths khớp workflow Studio (tên file trong JSON)
# UNET fp8 bắt buộc cho node 5130 (IC-LoRA + sampler → video cuối 5208); ~25GB
RUN mkdir -p "${COMFY_ROOT}/models/unet/LTXVideo/v2" \
    && wget -q -O "${COMFY_ROOT}/models/unet/LTXVideo/v2/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors" \
      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors"

# UNET GGUF (node 5229 UnetLoaderGGUF): mặc định không tải để giảm kích thước image — workflow chính dùng fp8 (5130).
# Khi cần: thêm RUN wget vào models/unet/LTXvideo/LTX-2/quantstack/LTX-2.3-distilled-Q4_K_S.gguf (QuantStack/LTX-2.3-GGUF).

RUN mkdir -p "${COMFY_ROOT}/models/clip" \
    && wget -q -O "${COMFY_ROOT}/models/clip/gemma_3_12B_it_fp8_scaled.safetensors" \
      "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    && wget -q -O "${COMFY_ROOT}/models/clip/ltx-2.3_text_projection_bf16.safetensors" \
      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors"

RUN mkdir -p "${COMFY_ROOT}/models/vae" "${COMFY_ROOT}/models/vae/vae_approx" \
    && wget -q -O "${COMFY_ROOT}/models/vae/LTX23_video_vae_bf16.safetensors" \
      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors" \
    && wget -q -O "${COMFY_ROOT}/models/vae/LTX23_audio_vae_bf16.safetensors" \
      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors" \
    && wget -q -O "${COMFY_ROOT}/models/vae/vae_approx/taeltx2_3.safetensors" \
      "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/vae/taeltx2_3.safetensors" \
    && ln -sf "LTX23_video_vae_bf16.safetensors" "${COMFY_ROOT}/models/vae/LTX23_video_vae_bf16_KJ.safetensors" \
    && ln -sf "LTX23_audio_vae_bf16.safetensors" "${COMFY_ROOT}/models/vae/LTX23_audio_vae_bf16_KJ.safetensors"

RUN mkdir -p "${COMFY_ROOT}/models/loras/LTX/LTX-2/IC-Lora" \
    && wget -q -O "${COMFY_ROOT}/models/loras/LTX/LTX-2/IC-Lora/ltx-2.3-22b-v1.1-ic-lora-union-control-ref0.5.safetensors" \
      "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"

RUN mkdir -p "${COMFY_ROOT}/models/latent_upscale_models" \
    && wget -q -O "${COMFY_ROOT}/models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors" \
      "https://huggingface.co/Lightricks/LTX-2.3/resolve/main/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"

RUN mkdir -p "${COMFY_ROOT}/models/checkpoints/SDPose" \
    && wget -q -O "${COMFY_ROOT}/models/checkpoints/SDPose/sdpose_wholebody_fp16.safetensors" \
      "https://huggingface.co/Comfy-Org/SDPose/resolve/main/checkpoints/sdpose_wholebody_fp16.safetensors"

RUN mkdir -p "${AUX_ANNOTATOR_CKPTS_PATH}/yzd-v/DWPose" \
    && wget -q -O "${AUX_ANNOTATOR_CKPTS_PATH}/yzd-v/DWPose/yolox_l.onnx" \
      "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx" \
    && wget -q -O "${AUX_ANNOTATOR_CKPTS_PATH}/yzd-v/DWPose/dw-ll_ucoco_384.onnx" \
      "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"

# TorchScript pose (workflow DWPreprocessor: dw-ll_ucoco_384_bs5.torchscript.pt) — tránh tải lúc chạy
RUN mkdir -p "${AUX_ANNOTATOR_CKPTS_PATH}/hr16/DWPose-TorchScript-BatchSize5" \
    && wget -q -O "${AUX_ANNOTATOR_CKPTS_PATH}/hr16/DWPose-TorchScript-BatchSize5/dw-ll_ucoco_384_bs5.torchscript.pt" \
      "https://huggingface.co/hr16/DWPose-TorchScript-BatchSize5/resolve/main/dw-ll_ucoco_384_bs5.torchscript.pt"

RUN mkdir -p "${AUX_ANNOTATOR_CKPTS_PATH}/lllyasviel/Annotators" \
    && wget -q -O "${AUX_ANNOTATOR_CKPTS_PATH}/lllyasviel/Annotators/depth_anything_vitl14.pth" \
      "https://huggingface.co/licyk/controlnet_v1.1_annotator/resolve/main/depth_anything/depth_anything_vitl14.pth"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py entrypoint.sh /app/
COPY workflows/api /app/workflows/api

RUN chmod +x /app/entrypoint.sh

EXPOSE 8188

ENTRYPOINT ["/app/entrypoint.sh"]
