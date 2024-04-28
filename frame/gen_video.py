# @File    : gen_video.py
# @Description : 生成视频

import torch
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

# 加载SDXL的base和refiner模型
t2i_base = DiffusionPipeline.from_pretrained(
    "../model/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
t2i_base.to("cuda")
t2i_refiner = DiffusionPipeline.from_pretrained(
    "../model/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=t2i_base.text_encoder_2,
    vae=t2i_base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
t2i_refiner.to("cuda")

# 加载SVD模型
i2v_pipe = StableVideoDiffusionPipeline.from_pretrained(
    "../model/stable-video-diffusion-img2vid-xt",
    torch_dtype=torch.float16,
    variant="fp16"
)
i2v_pipe.to("cuda")

# 设置去噪步骤数和refine比例
n_steps = 40
high_noise_frac = 0.8

# 输入prompt
prompt = input("user prompt: ")

# 生成图像
image = t2i_base(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
).images
image = t2i_refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=image,
).images[0]
image = image.resize((1024, 576))

# 生成视频
frames = i2v_pipe(image, decode_chunk_size=8).frames[0]
export_to_video(frames, "../results/svd_video.mp4", fps=8)  # 保存视频
