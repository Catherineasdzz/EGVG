# @File    : cal_fid.py
# @Description : 计算FID

import torchvision
from pytorch_fid import fid_score

# 真实视觉内容与生成视觉内容路径
real_folder = "target"
generated_folder = "../results"

# 加载预训练的Inception-v3模型
inception_model = torchvision.models.inception_v3(pretrained=True)

# 计算FID值
fid_value = fid_score.calculate_fid_given_paths([real_folder, generated_folder],
                                                device='cuda', dims=2048, batch_size=1, num_workers=0)
print("FID: ", fid_value)