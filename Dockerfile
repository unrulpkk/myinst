# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base
# Install libGL.so.1

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1 

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget

# Clean up to reduce image size
# RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir xformers==0.0.21 \
    && pip3 install -r requirements.txt

# Install runpod
RUN pip3 install runpod requests

# Download checkpoints/vae/LoRA to include in image
RUN wget -O models/checkpoints/YFilter_SDXL_RealMix_V2.safetensors https://liblibai-online.vibrou.com/web/model/ab5ff53669bd442bc06562882744ebceb25b9a709bcf18344d5c2ed6d156f01a.safetensors?attname=YFilter_SDXL_RealMix_V2.safetensors
RUN wget -O models/vae/sdxl_vae.safetensors https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
RUN wget -O models/vae/sdxl-vae-fp16-fix.safetensors https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors
#RUN wget -O models/loras/xl_more_art-full_v1.safetensors https://civitai.com/api/download/models/152309

# Example for adding specific models into image
# ADD models/checkpoints/sd_xl_base_1.0.safetensors models/checkpoints/
# ADD models/vae/sdxl_vae.safetensors models/vae/

# 安装ComfyUI_InstantID插件
## 下载ComfyUI_InstantID项目
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git custom_nodes/ComfyUI_InstantID 
## 安装ComfyUI_InstantID项目需要的python库
RUN pip3 install insightface==0.7.3 onnxruntime onnxruntime-gpu
RUN apt-get update && apt-get install -y libglib2.0-dev
RUN mkdir models/insightface
RUN mkdir models/insightface/models
RUN mkdir models/instantid
RUN apt-get update && apt-get install -y unzip
## 下载并解压InsightFace模型antelopev2
RUN wget -O models/insightface/models/antelopev2.zip "https://drive.usercontent.google.com/download?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8&export=download&authuser=0&confirm=t&uuid=a36df47b-aefd-477a-ba78-db4ddff8db8d&at=APZUnTVtyBL8TV8T3CVRCXiPUxhQ%3A1713112758615"
RUN unzip models/insightface/models/antelopev2.zip -d models/insightface/models/
## 下载ip_adapter.bin
RUN wget -O models/instantid/ip-adapter.bin "https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true"
## 下载controlnet
RUN wget -O models/controlnet/instantid-controlnet.safetensors "https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors?download=true"
#WORKDIR /comfyui/custom_nodes/ComfyUI_InstantID
#RUN pip3 install -r requirements.txt

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
