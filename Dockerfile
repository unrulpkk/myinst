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
RUN wget -O models/loras/xl_more_art-full_v1.safetensors https://civitai.com/api/download/models/152309

# Example for adding specific models into image
# ADD models/checkpoints/sd_xl_base_1.0.safetensors models/checkpoints/
# ADD models/vae/sdxl_vae.safetensors models/vae/

# 安装ComfyUI_InstantID插件
## 下载ComfyUI_InstantID项目
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git custom_nodes/ComfyUI_InstantID 
## 安装ComfyUI_InstantID项目需要的python库
RUN pip3 install insightface onnxruntime onnxruntime-gpu
RUN apt-get update && apt-get install -y libglib2.0-dev
RUN mkdir models/insightface
RUN mkdir models/insightface/models
RUN mkdir models/instantid
RUN apt-get update && apt-get install -y unzip
## 下载并解压InsightFace模型antelopev2
RUN wget -O models/insightface/models/antelopev2.zip "https://drive.usercontent.google.com/download?id=18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8&export=download&authuser=0&confirm=t&uuid=a36df47b-aefd-477a-ba78-db4ddff8db8d&at=APZUnTVtyBL8TV8T3CVRCXiPUxhQ%3A1713112758615"
RUN unzip models/insightface/models/antelopev2.zip -d models/insightface/models/
## 下载ip_adapter.bin
RUN wget -O models/instantid/ip-adapter.bin "https://cdn-lfs-us-1.huggingface.co/repos/e7/87/e787a7a4c89b83529c9661aaedad7af97ed194e375a36d1b2fc30f893aa849f5/02b3618e36d803784166660520098089a81388e61a93ef8002aa79a5b1c546e1?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27ip-adapter.bin%3B+filename%3D%22ip-adapter.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1713372047&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzM3MjA0N319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2U3Lzg3L2U3ODdhN2E0Yzg5YjgzNTI5Yzk2NjFhYWVkYWQ3YWY5N2VkMTk0ZTM3NWEzNmQxYjJmYzMwZjg5M2FhODQ5ZjUvMDJiMzYxOGUzNmQ4MDM3ODQxNjY2NjA1MjAwOTgwODlhODEzODhlNjFhOTNlZjgwMDJhYTc5YTViMWM1NDZlMT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=LGSXKN%7E9cMqlv3Cq5esldJCcahZyIL870yqU4O6Lvn7icTGS73%7Eux0En-JScyyhzj7YwZDlGmQjwWMCvetzRgdmR0%7EFXXVY5WZ2CEmjws7wwCNPJmGNicINR-tLzC4yvYblFyEXL2fQnYqI2%7EbqCficagwn0MS5onHK4xQqKQCqprGcD8nMW5wMHG3ZojBWMTpwlCZVf-Ado7Zss82ooTFkDZwb%7E4L6cs4ZqjMqCDKPz7TZt3shKFJ2q0xveLbyI7QgsA7oof7rQtpzVvpBGe5offItFsPczlNT%7EhsYnvnrTQnDqJm-xujIHi4LFuXT9AFuBoVN9nLY7vkAkVW%7Efkw__&Key-Pair-Id=KCD77M1F0VK2B"
## 下载controlnet
RUN wget -O models/controlnet/diffusion_pytorch_model.safetensors "https://cdn-lfs-us-1.huggingface.co/repos/e7/87/e787a7a4c89b83529c9661aaedad7af97ed194e375a36d1b2fc30f893aa849f5/c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27diffusion_pytorch_model.safetensors%3B+filename%3D%22diffusion_pytorch_model.safetensors%22%3B&Expires=1713372464&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzM3MjQ2NH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2U3Lzg3L2U3ODdhN2E0Yzg5YjgzNTI5Yzk2NjFhYWVkYWQ3YWY5N2VkMTk0ZTM3NWEzNmQxYjJmYzMwZjg5M2FhODQ5ZjUvYzgxMjdiZTlmMTc0MTAxZWJkYWZlZTk5NjRkODU2YjQ5YjYzNDQzNWNmNmRhYTM5NmQzZjU5M2NmMGJiYmIwNT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=eh8n-GgiwjpYngnaPkB5aVihLzywEqmvpGPVqV8u4nUWWN2ukOT-xZwjJO7aRw0kXzF7sXKQUvIAzEH6qNPutGI1wyxBC-qtnhz73AiJvNFF0o3m3cC-pb-jcuWWIvnsktrBBeUX1O1bEuEV%7EuiQFc5Srj%7Exfyp79rRB7bnzYL2jTxKRA5fZYYM4QyWT7VunqJdepzz9v6Ia7lLBpVhSTcQmzO92EbSgm0afZTcYH-lyIEpTiDEGomu7ffZ7M%7EsK460QNdQDq0Wk165uhegmWThpu%7EDzTPvmS%7E0dUTtu9%7E9uAJ-QKcE6LtHdw1osZgQeRX7lq5w54XEqStQyG-M0QA__&Key-Pair-Id=KCD77M1F0VK2B"
#WORKDIR /comfyui/custom_nodes/ComfyUI_InstantID
#RUN pip3 install -r requirements.txt

# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
