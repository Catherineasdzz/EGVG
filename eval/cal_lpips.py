# @File    : cal_lpips.py
# @Description : 计算LPIPS

import os
import lpips
from PIL import Image
from torchvision import transforms


# 加载并预处理
def load_and_preprocess(image_path):
    try:
        image = Image.open(image_path).convert("RGB")  # 打开图像
        transform = transforms.Compose([
            transforms.Resize((1024, 576)),  # 调整图像尺寸
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])
        image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


# 计算LPIPS距离
def cal_lpips_distance(image_folder1, image_folder2):
    # 获取文件夹中的图像文件列表
    images1 = os.listdir(image_folder1)
    images2 = os.listdir(image_folder2)
    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net='vgg', version='0.1')
    distances = []
    # 对应加载图像并计算LPIPS距离
    for img1, img2 in zip(images1, images2):
        image1_path = os.path.join(image_folder1, img1)
        image2_path = os.path.join(image_folder2, img2)
        image1_tensor = load_and_preprocess(image1_path)
        image2_tensor = load_and_preprocess(image2_path)
        # 如果有任何一张图像加载失败，则跳过计算该图像的LPIPS距离
        if image1_tensor is None or image2_tensor is None:
            continue
        # 使用LPIPS模型计算距离
        distance = lpips_model(image1_tensor, image2_tensor)
        distances.append(distance.item())
    return distances


# 计算LPIPS均值
def cal_mean_lpips(image_folder1, image_folder2):
    distances = cal_lpips_distance(image_folder1, image_folder2)
    mean_distance = sum(distances) / len(distances)
    return mean_distance


# 两个文件夹路径
image_folder1 = "target"
image_folder2 = "../results"

# 计算平均LPIPS距离
mean_lpips_distance = cal_mean_lpips(image_folder1, image_folder2)
print("LPIPS: ", mean_lpips_distance)