# @File    : pre_lexicon.py
# @Description : 将VAD词典写入csv文件并归一化情感映射量表

import csv
import pandas as pd

# 定义txt文件路径和csv文件路径
txt_path = "../data/NRC-VAD-Lexicon.txt"
csv_path = "../data/NRC-VAD-Lexicon.csv"

# 打开txt文件进行读取
with open(txt_path, "r", encoding="utf-8") as txt_file:
    txt_content = txt_file.read()

# 将txt内容按行分割
rows = txt_content.strip().split("\n")

header = ["word", "valence", "arousal", "dominance"]
# 写入csv文件
with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
    # 创建csv写入器
    csv_writer = csv.writer(csv_file, delimiter=",")
    # 写入表头
    csv_writer.writerow(header)
    # 逐行写入csv文件
    for row in rows:
        # 使用\t分割每一行的列
        columns = row.split("\t")
        # 将列写入csv文件
        csv_writer.writerow(columns)

# 归一化情感映射量表
origin_path = "../data/7EMO-VAD-origin.csv"
emo_vad = pd.read_csv(origin_path)

min_val, max_val = -1, 1
for column in ["valence", "arousal", "dominance"]:
    emo_vad[column] = (emo_vad[column] - min_val) / (max_val - min_val)  # 归一化

emo_vad[["emotion", "valence", "arousal", "dominance"]].to_csv("../data/7EMO-VAD.csv", index=False)  # 写入文件