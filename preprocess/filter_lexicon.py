# @File    : filter_lexicon.py
# @Description : 筛选出VAD词典中能够作为情感引导词的词汇

import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm


def read_lexicon(csvpath):
    """
    读取VAD词典并将词汇映射到不同情感类别
    :param csvpath: VAD词典csv路径
    :return: 包含词汇、VAD值和情感类别的DataFrame
    """
    df = pd.read_csv(csvpath)
    df = df[~df["word"].isna()]  # 删除空值
    df["word"] = [x.lower() for x in df["word"]]  # 转换为小写
    # 情感类别映射
    cols_vad = ["valence", "arousal", "dominance"]
    df[cols_vad] = df[cols_vad].astype(float)  # 将情感值转换为浮点数
    emo_df = pd.read_csv("../data/7EMO-VAD.csv")  # 情感类别映射量表
    emo_centers = emo_df[["valence", "arousal", "dominance"]].values  # 定义情感类别vad中心
    word_vad = df[["valence", "arousal", "dominance"]].values  # 获取情感值
    distances = np.linalg.norm(word_vad[:, np.newaxis, :] - emo_centers, axis=2)  # 计算每个样本到每个类别中心的距离
    predicted_labels = np.argmin(distances, axis=1)  # 获取每个样本所属的类别（距离最小的类别）
    df["emotion"] = emo_df["emotion"][predicted_labels].values  # 情感类别映射
    return df


def filter_neutral(df):
    """
    删除VAD词典中的中性词汇
    :param df: 对应情感类别的VAD词典DataFrame
    :return: 删除中性词汇的VAD词典DataFrame
    """
    df = df[df["emotion"] != "neutral"]
    return df


def select_pos(words):
    """
    筛选具有能作为情感引导词的词性的词汇
    """
    # 筛选的词性标签
    selected_tag = ["ADJ", "ADV"]
    model = spacy.load("en_core_web_lg")  # 加载模型
    doc = model(words)  # 文本处理
    # 筛选词汇
    pos_selected = [token.pos_ for token in doc if token.pos_ in selected_tag]
    return pos_selected if pos_selected else None


def select_affective_words(df):
    """
    筛选情感引导词
    :param df: VAD词典DataFrame
    :return: 情感引导词DataFrame
    """
    tqdm.pandas()  # 使用tqdm显示进度
    df["pos"] = df["word"].progress_apply(select_pos)  # 使用select_pos函数进行词性筛选
    df = df.dropna(subset=["pos"])  # 删除pos为空的行
    df["pos"] = df["pos"].apply(lambda x: ", ".join(x) if x else "")  # 将列表转换为字符串
    afw_df = df.sort_values(by=["emotion"], ascending=[True])  # 按序排列
    emo_counts = afw_df["emotion"].value_counts()  # 各情感类别引导词数量统计
    print(emo_counts)
    return afw_df


if __name__ == "__main__":
    csvpath = "../data/NRC-VAD-Lexicon.csv"
    df_7emotion = read_lexicon(csvpath)
    df_6emotion = filter_neutral(df_7emotion)
    afw_df = select_affective_words(df_6emotion)
    afw_df.to_csv("../data/adj_words.csv", index=False)