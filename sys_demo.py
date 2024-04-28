# @File    : sys_demo.py
# @Description : 情感引导视频生成系统的界面实现

import os
import torch
import pandas as pd
import altair as alt
import streamlit as st
from diffusers import DiffusionPipeline, StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from moviepy.editor import VideoFileClip
from frame.cal_func_vad import read_lexicon, analyze_emotion, filter_prompt, get_afw_random, get_afw_similarity
from frame.gen_details import expand_details
from frame.post_structured import structured_text

# 设置页面标题、图标和布局
st.set_page_config(
    page_title="情感引导的微视频生成",
    layout="wide"
)
st.markdown("### 情感引导的微视频生成")


# 加载模型
@st.cache_resource
def get_model():
    # T2I模型
    t2i_base = DiffusionPipeline.from_pretrained(
        "model/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    t2i_base.to("cuda")
    t2i_refiner = DiffusionPipeline.from_pretrained(
        "model/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=t2i_base.text_encoder_2,
        vae=t2i_base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    t2i_refiner.to("cuda")
    # I2V模型
    i2v_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "model/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    i2v_pipe.to("cuda")
    return t2i_base, t2i_refiner, i2v_pipe


# 情感分析
@st.cache_data
def ana_emotion(raw_prompt):
    # 切换工作目录
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    frame_dir = os.path.join(cur_dir, 'frame')
    os.chdir(frame_dir)
    ldf = read_lexicon()  # 读取VAD词典
    sp = analyze_emotion(ldf, raw_prompt)  # 情感分析
    df = pd.DataFrame(sp, columns=['Emotion', 'Percentage'])
    # 创建水平柱状图
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Percentage:Q', axis=alt.Axis(format='%'), title='Percentage'),
        y=alt.Y('Emotion:N', title='Emotion'),
        color=alt.Color('Emotion:N', scale=alt.Scale(scheme='category20')),
        tooltip=['Emotion:N', 'Percentage:Q']
    ).properties(width=500, height=300)
    return chart


# 情感极性过滤
@st.cache_data
def fil_prompt(prompt, emotion):
    # 切换工作目录
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    frame_dir = os.path.join(cur_dir, 'frame')
    os.chdir(frame_dir)
    ldf = read_lexicon()  # 读取VAD词典
    filtered_prompt = filter_prompt(ldf, prompt, emotion)  # 情感极性过滤
    return filtered_prompt


# 情感引导词匹配
@st.cache_data
def get_afw(prompt, emotion, radioAfw):
    # 切换工作目录
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    frame_dir = os.path.join(cur_dir, 'frame')
    os.chdir(frame_dir)
    afw = ""
    if radioAfw == "随机匹配":
        afw = get_afw_random(emotion)
    elif radioAfw == "检索匹配":
        afw = get_afw_similarity(prompt, emotion)
    return afw


# 结构化情感细节扩写
@st.cache_data
def exp_details(prompt, emotion):
    # 切换工作目录
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    frame_dir = os.path.join(cur_dir, 'frame')
    os.chdir(frame_dir)
    details = expand_details(prompt, emotion)  # 细节扩写
    structured_prompt = structured_text(str(details))  # 结构化后处理
    structured_prompt = ", " + structured_prompt
    return structured_prompt


# 生成视频
@st.cache_data
def gen_video(prompt, n_steps=40, high_noise_frac=0.8):
    # 切换工作目录
    cur_dir = os.getcwd()
    tgt_dir = os.path.abspath(os.path.join(cur_dir, ".."))
    os.chdir(tgt_dir)
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
    image = image.resize((1024, 576))  # 调节分辨率
    # 生成视频
    frames = i2v_pipe(image, decode_chunk_size=8).frames[0]
    video_path = export_to_video(frames, "results/demo_video.mp4", fps=8)  # 保存视频
    # 转换生成视频格式为H.264，以显示在网页中
    output_path = "results/demo_video_H264.mp4"
    clip = VideoFileClip(video_path)
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac",
                         preset="slow", bitrate="1000k", fps=30)
    return output_path


# 加载模型
t2i_base, t2i_refiner, i2v_pipe = get_model()

# 设置状态变量
if "prompt" not in st.session_state:  # prompt
    st.session_state.prompt = ""
if "emotion" not in st.session_state:  # 情感类别
    st.session_state.emotion = "neutral"  # 默认为中性
if "chart" not in st.session_state:  # 情感分布图
    st.session_state.chart = None
if "radioAfw" not in st.session_state:  # 情感引导词匹配方式
    st.session_state.radioAfw = ""
if "buttonPrompt" not in st.session_state:  # 细节扩写选项
    st.session_state.buttonPrompt = False
if "filtered_prompt" not in st.session_state:  # 过滤prompt
    st.session_state.filtered_prompt = ""
if "afw" not in st.session_state:  # 情感引导词
    st.session_state.afw = ""

# 设置侧边栏
with st.sidebar:
    with st.container(border=True):  # 参数设置框
        radioAfw = st.radio("请选择情感引导词匹配方式", ("检索匹配", "随机匹配"))
        sliderNsteps = st.slider("请给定模型的去噪步骤数", 0, 100, 40)
    with st.container(border=True):  # 使用说明框
        st.markdown("**使用说明**")
        st.markdown("1. 请在输入文本提示框内输入想要生成的内容")
        st.markdown("2. 点击“情感分析”获取输入的情感分布，你可以根据该情感分布在右侧选择你想要生成的情感类别")
        st.markdown("3. 点击“细化提示”为你的输入添加情感视觉细节，你可以根据需要对其进行修改")
        st.markdown("4. 点击“生成视频”获取相应情感类别的视频")
        st.markdown("注：你可以在侧边栏选择情感引导和视频生成的相关参数")

# 若所选匹配方式发生变化，则更新相关状态变量
if radioAfw != st.session_state.radioAfw:
    st.session_state.afw = ""
    st.session_state.radioAfw = radioAfw

# 设置双栏系统界面
left_col, right_col = st.columns(2)

# 用户输入与情感分析
with left_col:
    raw_prompt = st.text_area("请输入文本提示", key="user_input")  # 文本指令输入框
    st.session_state.prompt = str(raw_prompt)  # 更新prompt状态变量
    # 设置3个功能按钮
    col1, col2, col3 = st.columns(3)
    buttonEmotion = col1.button("情感分析", key="emo_vad")  # 情感分析按钮
    buttonPrompt = col2.button("细化提示", key="emo_details")  # 细化提示按钮
    buttonVideo = col3.button("生成视频", key="t2v")  # 生成视频按钮
    # 若点击情感分析按钮，则输出情感分布图
    if buttonEmotion:
        chart = ana_emotion(raw_prompt)  # 获取情感分布图
        st.session_state.chart = chart  # 更新情感分布图状态变量
    if st.session_state.chart is not None:
        st.altair_chart(st.session_state.chart)  # 在页面中显示情感分布图

# 情感选择与引导词匹配
with right_col:
    # 情感类别选择下拉框
    emotion = st.selectbox("请选择情感（中性即不进行情感引导）",
                           ("中性neutral", "愤怒anger", "厌恶disgust", "恐惧fear", "喜悦joy", "悲伤sadness", "惊喜surprise"))
    emotion = emotion[2:]  # 取英文部分为具体情感类别
    # 若所选情感类别发生变化，则更新相关状态变量
    if emotion != st.session_state.emotion:
        st.session_state.afw = ""  # 重置情感引导词状态变量
        st.session_state.emotion = emotion  # 更新情感类别状态变量
        filtered_prompt = fil_prompt(raw_prompt, emotion)  # 情感极性过滤
        st.session_state.prompt = filtered_prompt  # 更新prompt状态变量
        st.session_state.filtered_prompt = filtered_prompt  # 更新过滤prompt状态变量
    afw = get_afw(st.session_state.prompt, emotion, st.session_state.radioAfw)  # 情感引导词匹配
    st.session_state.afw = afw  # 更新情感引导词状态变量
    afw_prompt = st.session_state.prompt + st.session_state.afw  # 获取氛围引导prompt
    st.session_state.prompt = afw_prompt  # 更新prompt状态变量

# 细节扩写
with left_col:
    # 若点击细化提示按钮，则更新相关状态变量
    if buttonPrompt:
        st.session_state.buttonPrompt = buttonPrompt
    if st.session_state.buttonPrompt is False:
        new_prompt = st.text_area("当前提示如下：", st.session_state.prompt, key="new_prompt")  # 实时提示框
        st.session_state.prompt = new_prompt  # 更新prompt状态变量
    else:
        temp_prompt = st.session_state.filtered_prompt  # 扩写前去除引导词
        structured_prompt = exp_details(temp_prompt, st.session_state.emotion)  # 细节扩写
        st.session_state.prompt = temp_prompt + structured_prompt + st.session_state.afw  # 更新prompt状态变量
        new_prompt = st.text_area("当前提示如下：", st.session_state.prompt, key="new_prompt")  # 实时提示框
        st.session_state.prompt = new_prompt  # 更新prompt状态变量

# 视频生成
with right_col:
    # 若点击生成视频按钮，则输出生成的视频
    if buttonVideo:
        output_path = gen_video(st.session_state.prompt, sliderNsteps)  # 生成视频
        st.video(output_path)  # 在页面中显示视频
