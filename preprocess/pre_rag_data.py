# @File    : pre_rag_data.py
# @Description : 构建类人情感-视觉数据集

import os
import pandas as pd

df1 = pd.read_csv("../data/artemis_dataset_release_v0.csv")  # ArtEmis
df2 = pd.read_csv("../data/affection_raw_public_version_0.csv")  # Affection
df3 = pd.read_csv("../data/Contrastive.csv")  # ArtEmis v2
df = pd.concat([df1[["emotion", "utterance"]], df2[["emotion", "utterance"]], df3[["emotion", "utterance"]]], axis=0)  # 合并数据
df = df.drop_duplicates(subset=["utterance"])  # 删除重复值
valid_emotions = ["amusement", "awe", "anger", "disgust", "fear", "sadness"]  # 保留的情感类别
replace_dict = {"amusement": "joy", "awe": "surprise"}  # 情感类别替换

# 获得所需的六分类情感数据
df = df[df["emotion"].isin(valid_emotions)]
df["emotion"] = df["emotion"].replace(replace_dict)
emotion_label = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
for label in emotion_label:
    texts = df.loc[df["emotion"] == label, "utterance"]
    output_folder = f"../data/rag/{label}"
    output_file = f"{output_folder}/{label}_data.txt"
    # 检查文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_file, "w", encoding="utf-8") as f:
        for text in texts:
            f.write(text + "\n")
    print(f"{label} data have been written to:", output_file)
