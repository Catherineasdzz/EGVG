import os
import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from moviepy.editor import VideoFileClip
from PIL import Image

# # 大语言模型
# MODEL_PATH = os.environ.get('MODEL_PATH', '/root/autodl-tmp/chatglm3-6b')
# TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置页面标题、图标和布局
st.set_page_config(
    page_title="情感引导视频生成",
    # page_icon=
    layout="wide"
)

# @st.cache_resource
# def get_model():
#     tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
#     if 'cuda' in DEVICE:  # AMD, NVIDIA GPU can use Half Precision
#         model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).to(DEVICE).eval()
#     else:  # CPU, Intel GPU and other GPU can use Float16 Precision Only
#         model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()
#     return tokenizer, model

# # 加载Chatglm3的model和tokenizer
# tokenizer, model = get_model()

# T2V模型
pipe1 = DiffusionPipeline.from_pretrained("/root/autodl-tmp/zeroscope_v2_576w", torch_dtype=torch.float16)
pipe1.scheduler = DPMSolverMultistepScheduler.from_config(pipe1.scheduler.config)
pipe1.enable_model_cpu_offload()
pipe1.enable_vae_slicing()
pipe1.unet.enable_forward_chunking(chunk_size=1, dim=1) # disable if enough memory as this slows down significantly

pipe2 = DiffusionPipeline.from_pretrained("/root/autodl-tmp/zeroscope_v2_XL", torch_dtype=torch.float16)
pipe2.scheduler = DPMSolverMultistepScheduler.from_config(pipe2.scheduler.config)
pipe2.enable_model_cpu_offload()
pipe2.enable_vae_slicing()

# web可视化
# 设置侧边栏
genre = st.sidebar.selectbox("请选择输入文本体裁", ["短句", "短文", "诗歌"])
if genre == "短句":
    expand_scene = st.sidebar.selectbox("请选择是否扩充视频场景", ["是", "否"])
emo_aug = st.sidebar.selectbox("请选择是否增强视频情感", ["是", "否"])

# 获取用户输入
text_input = st.text_area("请输入文本", key="user_input")
# buttonScene = st.button("生成视频描述预览", key="t2s")
buttonVideo = st.button("生成视频", key="t2v")

# if text_input:
    # prompt_text = "我想根据以下内容生成一个视频，请全部使用中文给出我视频每一幕的画面描述（不包括旁白），生成文本结构为“画面编号：画面内容”，编号按顺序用数字表示，不要生成其他东西：" + str(text_input)
    # response, history = model.chat(tokenizer, prompt_text, history=[])
    # 生成视频场景描述
# if buttonScene:
#     response, history = model.chat(tokenizer, text_input, history=[])
#     modified_text = st.text_area("视频描述预览", response, height=None, key="modified_text")
#     buttonVideo = st.button("生成视频", key="t2v")
#     if buttonVideo:
#     # prompt = "Darth Vader is surfing on waves"
#         video_frames = pipe(text_input, num_inference_steps=40, height=320, width=576, num_frames=24).frames
#         video_path = export_to_video(video_frames, output_video_path="/root/autodl-tmp/video.mp4")
#         st.write(video_path)
#         st.video('/root/autodl-tmp/video.mp4')

if buttonVideo:
    video_frames = pipe1(str(text_input), num_inference_steps=40, height=320, width=576, num_frames=24).frames
    video = [Image.fromarray(frame).resize((1024, 576)) for frame in video_frames]
    video_frames = pipe2(str(text_input), video=video, strength=0.6).frames
    video_path = export_to_video(video_frames, output_video_path="/root/autodl-tmp/video.mp4")
    output_video_path = "/root/autodl-tmp/output.mp4"
    # 转换生成视频格式为H.264以显示在网页中
    clip = VideoFileClip(video_path)
    clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", preset="slow", bitrate="192k")
    st.video(output_video_path)
