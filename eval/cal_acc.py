# @File    : cal_acc.py
# @Description : 计算情感准确率

import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# 情感分类器类
class EmoClassifier(nn.Module):
    def __init__(self):
        super(EmoClassifier, self).__init__()
        self.layer = nn.Linear(768, 6)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.layer(x)
        x2 = self.softmax(x1)
        return x1, x2


# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = EmoClassifier()  # 实例化
model.load_state_dict(
    torch.load("checkpoint/EmoClassifier-08.pkl", map_location=device))  # 加载模型参数
model = model.to(device)

# 加载CLIP处理器和模型
CLIPmodel = CLIPModel.from_pretrained("../model/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("../model/clip-vit-large-patch14")

# 图片文件夹路径
image_folder = "../results"

# 初始化总数和正确预测数
total_images = 0  # 6分类总数
correct_predictions = 0  # 6分类正确预测数
total_images_binary = 0  # 2分类总数
correct_predictions_binary = 0  # 2分类正确预测数

# 遍历图片文件夹中的所有图片
for filename in os.listdir(image_folder):
    if filename.endswith((".png", ".jpg")):
        # 解析文件名获取情感标签
        true_label = int(filename.split(".")[0].split("_")[1])
        # 图片路径
        image_path = os.path.join(image_folder, filename)
        # 加载图像并提取特征
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        imgs = inputs["pixel_values"].to(device)
        features = CLIPmodel.get_image_features(pixel_values=imgs)
        # 使用模型进行预测
        with torch.no_grad():
            outputs1, outputs2 = model(features.to(device))
            predicted_class = torch.argmax(outputs2, dim=1).item()
        # 更新正确预测数
        if predicted_class == true_label:
            correct_predictions += 1
        total_images += 1  # 更新总数
        # 将情感标签映射到两个类别
        if true_label in [0, 1, 2, 4]:
            true_binary_label = 0
        else:
            true_binary_label = 1
        # 将预测标签映射到两个类别
        if predicted_class in [0, 1, 2, 4]:
            predicted_binary_class = 0
        else:
            predicted_binary_class = 1
        # 更新2分类正确预测数
        if predicted_binary_class == true_binary_label:
            correct_predictions_binary += 1
        total_images_binary += 1  # 更新2分类总数


# 2分类情感准确率
binary_accuracy = correct_predictions_binary / total_images_binary if total_images_binary != 0 else 0
print("Emo Acc-2: ", binary_accuracy)

# 6分类情感准确率
accuracy = correct_predictions / total_images if total_images != 0 else 0
print("Emo Acc-6: ", accuracy)