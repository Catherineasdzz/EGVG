# 情感引导的微视频生成

## 环境配置

```shell
conda create -n EGVG python=3.9
conda activate EGVG
cd EGVG
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 模型下载

```shell
# 使用huggingface-cli和hf_transfer加速下载模型 
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
# LLM和Embedding模型
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat-hf --local-dir /EGVG/model/llama-2-7b-chat-hf --local-dir-use-symlinks False --token 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
huggingface-cli download --resume-download BAAI/bge-large-en --local-dir /EGVG/model/bge-large-en --local-dir-use-symlinks False
# 视频生成模型
huggingface-cli download --resume-download stabilityai/stable-video-diffusion-img2vid-xt --local-dir /EGVG/model/stable-video-diffusion-img2vid-xt --local-dir-use-symlinks False
huggingface-cli download --resume-download stabilityai/stable-diffusion-xl-base-1.0 --local-dir /EGVG/model/stable-diffusion-xl-base-1.0 --local-dir-use-symlinks False
huggingface-cli download --resume-download stabilityai/stable-diffusion-xl-refiner-1.0 --local-dir /EGVG/model/stable-diffusion-xl-refiner-1.0 --local-dir-use-symlinks False
```

