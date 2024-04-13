# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base
# Install libGL.so.1
RUN apt-get update && apt-get install -y libgl1-mesa-glx
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
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

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
RUN mkdir models/insightface
RUN mkdir models/insightface/models
RUN mkdir models/instantid
RUN apt-get update && apt-get install -y unzip
## 下载并解压InsightFace模型antelopev2
RUN wget -O models/insightface/models/antelopev2.zip "https://cdn-lfs-us-1.huggingface.co/repos/6f/d8/6fd848a602725b2910983354623eef0f05f4a45a142fc63dfdaa418a10547d33/8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27antelopev2.zip%3B+filename%3D%22antelopev2.zip%22%3B&response-content-type=application%2Fzip&Expires=1713074259&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzA3NDI1OX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzZmL2Q4LzZmZDg0OGE2MDI3MjViMjkxMDk4MzM1NDYyM2VlZjBmMDVmNGE0NWExNDJmYzYzZGZkYWE0MThhMTA1NDdkMzMvOGUxODJmMTRmYzZlODBiM2JmYTM3NWIzM2ViNmNmZjdlZTA1ZDhlZjc2MzNlNzM4ZDFjODkwMjFkY2YwYzVjNT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=LhgyZzgruzu%7EWA8pm9abStKYMqaVmDnrYugGikJ5pRSZ2OJBlXmS26E78bhl1fPGmUEXUC4%7EGW0b5VXZWDc74OngPrSDTVLM5qJ2wPfghT6Imw0UE0XiD4V72CD%7EvuL9nl3f8DaAhpiqKffUTyNiqvciPk0RqdBFmLukyeY209I6YVViIzMi%7EiCT%7EBXASoQF4vMrs8G4kQdMESiLc2ad375jK6OVMaM4NNWaBZ%7Eg6LSAfbT%7EswrqtlyTTvBEUruOY53CguqOMJp97WkffGpc5kG7V%7EbEyJ4ql-QRDkIMQyM2A6eQ%7E6EzrPUCUCIMFcxeOjb-B9BxPVC59irZHvKAZA__&Key-Pair-Id=KCD77M1F0VK2B"
RUN unzip models/insightface/models/antelopev2.zip -d models/insightface/models/
## 下载ip_adapter.bin
RUN wget -O models/instantid/ip_adapter.bin "https://cdn-lfs-us-1.huggingface.co/repos/e7/87/e787a7a4c89b83529c9661aaedad7af97ed194e375a36d1b2fc30f893aa849f5/02b3618e36d803784166660520098089a81388e61a93ef8002aa79a5b1c546e1?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27ip-adapter.bin%3B+filename%3D%22ip-adapter.bin%22%3B&response-content-type=application%2Foctet-stream&Expires=1713074805&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzA3NDgwNX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2U3Lzg3L2U3ODdhN2E0Yzg5YjgzNTI5Yzk2NjFhYWVkYWQ3YWY5N2VkMTk0ZTM3NWEzNmQxYjJmYzMwZjg5M2FhODQ5ZjUvMDJiMzYxOGUzNmQ4MDM3ODQxNjY2NjA1MjAwOTgwODlhODEzODhlNjFhOTNlZjgwMDJhYTc5YTViMWM1NDZlMT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSomcmVzcG9uc2UtY29udGVudC10eXBlPSoifV19&Signature=K%7EPDXstHRtxp9iv%7E5o4HIfNGIcFZiBYAv4uIMOIUq6ab1ftbfL1zOFUwXCu%7EABpI3yctDg8hZeKwP5qwc1nAFdz3khZ1-WO8ifgF95mtpG3PYsrRz28lKUD3CGC4D1qXskL4%7EP-NlglG6aSycuMTaku9lb7pdOFmknpIBR8bwsWcBqfymxn0IGOzNMaa3MvIhmQVqyGej2ouyOy2tJFZDCbVjcMVo7CLpX6QY1n2JZe1636bWQ4lKyYDN1DnG86W8wtpRDOCPLsiWA6Dq8ZU2iRXxoSh4x7AN14YTxHjDbXpO-ENTIXm7Sp1dhXp6Ri-wwLNNNQzLnMWcN2JFrlWQA__&Key-Pair-Id=KCD77M1F0VK2B"
## 下载controlnet
RUN wget -O models/controlnet/diffusion_pytorch_model.safetensors "https://cdn-lfs-us-1.huggingface.co/repos/e7/87/e787a7a4c89b83529c9661aaedad7af97ed194e375a36d1b2fc30f893aa849f5/c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27diffusion_pytorch_model.safetensors%3B+filename%3D%22diffusion_pytorch_model.safetensors%22%3B&Expires=1713074852&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxMzA3NDg1Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zL2U3Lzg3L2U3ODdhN2E0Yzg5YjgzNTI5Yzk2NjFhYWVkYWQ3YWY5N2VkMTk0ZTM3NWEzNmQxYjJmYzMwZjg5M2FhODQ5ZjUvYzgxMjdiZTlmMTc0MTAxZWJkYWZlZTk5NjRkODU2YjQ5YjYzNDQzNWNmNmRhYTM5NmQzZjU5M2NmMGJiYmIwNT9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=JsWpYaQyQynipw5RzAqY3t%7ECRkEsAwQ9JaCPi6NgQl-9bsGrfn%7EDQSzjtrwtlGUObM%7EU50Nlyda8Nw9bJUtISdkd6ZcmlSEHmnmQEjUrezjZC8zPgeTvh0HKsOATJKc3%7EGb4G%7EsbPgaYj4wk8zAmY73G7YuWg5A5RPg3mQpvsrZPLcuFnroVcQXDo%7EUtoqYEmrsLizEly0iVNyDWW%7EgQAPQvnD0XJf4z-4An9MapNDlWHULrb4FRfn7sSBvTua4DvZLINsn9728fhq%7Eam9FkZKaXLBodMLdNlXN0dfnJqAxSVeCVRpHtDPnXUtyGkAfDMY1JqK963tZJJUw0WXlzPA__&Key-Pair-Id=KCD77M1F0VK2B"


# Go back to the root
WORKDIR /

# Add the start and the handler
ADD src/start.sh src/rp_handler.py test_input.json ./
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
