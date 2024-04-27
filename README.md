# 情感引导的微视频生成

## 配置与安装

### 1.克隆代码

```shell
git clone https://github.com/Catherineasdzz/EGVG.git
```
### 2.环境配置

```shell
conda create -n EGVG python=3.9
conda activate EGVG
cd EGVG
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 模型及数据

### 1.模型下载

```shell
# 使用huggingface-cli和hf_transfer加速下载模型 
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1
# LLM和Embedding模型
huggingface-cli download --resume-download meta-llama/Llama-2-7b-chat-hf --local-dir model/llama-2-7b-chat-hf --local-dir-use-symlinks False --token <your_hf_token>
huggingface-cli download --resume-download BAAI/bge-large-en --local-dir model/bge-large-en --local-dir-use-symlinks False
# 视频生成模型
huggingface-cli download --resume-download stabilityai/stable-video-diffusion-img2vid-xt --local-dir model/stable-video-diffusion-img2vid-xt --local-dir-use-symlinks False
huggingface-cli download --resume-download stabilityai/stable-diffusion-xl-base-1.0 --local-dir model/stable-diffusion-xl-base-1.0 --local-dir-use-symlinks False
huggingface-cli download --resume-download stabilityai/stable-diffusion-xl-refiner-1.0 --local-dir model/stable-diffusion-xl-refiner-1.0 --local-dir-use-symlinks False
```
### 2.数据准备

直接下载预处理后的数据 [EGVG_data](https://pan.baidu.com/s/1-k18nxSsqyPtXUT1NYQ7jg?pwd=0lfn) 或下载原始数据集重新进行预处理
<table>
<thead>
  <tr>
    <th> 数据集 </th>
    <th> 下载链接 </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td> NRC-VAD </td>
    <th> [<a href="https://saifmohammad.com/WebPages/nrc-vad.html">官方</a>][<a href="https://pan.baidu.com/s/1H_wllFhZuwjtaqbJHrixoQ?pwd=ft4u">百度云</a>] </th>
  </tr>
  <tr>
    <td> DiffusionDB </td>
    <th> [<a href="https://huggingface.co/datasets/poloclub/diffusiondb/">官方</a>][<a href="https://pan.baidu.com/s/1QuL6frk3r751bMCivm9jDQ?pwd=vfmi">百度云</a>] </th>
  </tr>
  <tr>
    <td> ArtEmis </td>
    <th> [<a href="https://www.artemisdataset.org/">官方</a>][<a href="https://pan.baidu.com/s/1gonUkwZPxeNWk_f72p5zRQ?pwd=v1cx">百度云</a>] </th>
  </tr>
  <tr>
    <td> ArtEmis v2 </td>
    <th> [<a href="https://www.artemisdataset-v2.org/">官方</a>][<a href="https://pan.baidu.com/s/1EByzR215NRMxjdj14cANNQ?pwd=iiz3">百度云</a>] </th>
  </tr>  
  <tr>
    <td> Affection </td>
    <th> [<a href="https://affective-explanations.org/">官方</a>][<a href="https://pan.baidu.com/s/1zmyJKs2UiWrf46yLOepmjQ?pwd=bycp">百度云</a>] </th>
  </tr>
</tbody>
</table>

模型与数据准备完毕后，代码的组织结构如下：
```
EGVG
|
├── data
|    ├── rag
|    |    ├── anger
|    |    |    └── anger_data.txt
|    |    ├── ...
|    |    └── surprise
|    |         └── surprise_data.txt
|    ├── 7EMO-VAD.csv
|    ├── affective_dictionary.csv
|    ├── ...
|    └── NRC-VAD-Lexicon.csv
|
├── model
|    ├── bge-large-en
|    ├── llama-2-7b-chat-hf
|    ├── stable-diffusion-xl-base-1.0
|    ├── stable-diffusion-xl-refiner-1.0
|    └── stable-video-diffusion-img2vid-xt
|    
├── preprocess
├── frame
├── eval
├── results
├── requirements.txt
└── sys_demo.py
```

## 快速推理

```shell
streamlit run sys_demo.py
```

