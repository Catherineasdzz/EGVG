# @File    : cal_func_vad.py
# @Description : 基于VAD情感模型的情感计算和情感引导函数

import re
import spacy
import random
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def read_lexicon():
    """
    读取及预处理vad词典
    :return: 包含词汇和vad值的DataFrame
    """
    df = pd.read_csv("../data/NRC-VAD-Lexicon.csv")
    df = df[~df["word"].isna()]  # 删除空值
    df["word"] = [x.lower() for x in df["word"]]  # 转换为小写
    df.dropna(inplace=True)  # 去除含有空值的行
    cols_vad = ["valence", "arousal", "dominance"]
    df[cols_vad] = df[cols_vad].astype(float)  # 将情感值转换为浮点数
    emo_df = pd.read_csv("../data/7EMO-VAD.csv")  # 情感类别映射
    emo_centers = emo_df[["valence", "arousal", "dominance"]].values  # 定义情感类别vad中心
    word_vad = df[["valence", "arousal", "dominance"]].values  # 获取情感值
    distances = np.linalg.norm(word_vad[:, np.newaxis, :] - emo_centers, axis=2)  # 计算每个样本到每个类别中心的距离
    predicted_labels = np.argmin(distances, axis=1)  # 获取每个样本所属的类别（距离最小的类别）
    df["emotion"] = emo_df["emotion"][predicted_labels].values  # 情感类别映射
    df.set_index("word", inplace=True)  # 将词汇设置为索引
    return df


def analyze_emotion(vad_df, raw_prompt):
    """
    对输入prompt进行情感分析
    :param vad_df: vad词典DataFrame
    :param raw_prompt: 输入prompt
    :return: 情感分布
    """
    prompt = [word.lower() for word in raw_prompt.split(",")]  # 将文本转换为小写，并按逗号分割
    prompt = [word for sublist in prompt for word in sublist.split(" ")]  # 按空格分割
    word_list = [w for w in prompt if w.isalpha()]  # 仅保留由字母组成的单词
    word_in_dict = [x for x in word_list if x in vad_df.index]  # 查找文本在词典中的单词
    filtered_df = vad_df[vad_df.index.isin(word_in_dict)]["emotion"]  # 获取对应词典信息
    # 情感计算
    emo_counts = Counter(filtered_df)
    emo_percentages = {emotion: count / len(filtered_df) for emotion, count in emo_counts.items()}  # 计算每个类别占比
    sorted_percentages = sorted(emo_percentages.items(), key=lambda x: x[1], reverse=True)  # 降序排列
    return sorted_percentages


def filter_prompt(vad_df, raw_prompt, emotion):
    """
    滤除输入prompt中具有非选择情感极性的部分
    :param vad_df: vad词典DataFrame
    :param raw_prompt: 输入prompt
    :param emotion: 选择情感类别
    :return: 过滤后的prompt
    """
    if emotion == "neutral":
        return raw_prompt
    else:
        # 情感倾向词过滤
        selected_tag = ["ADJ", "ADV"]  # 筛选词性标签
        model = spacy.load("en_core_web_lg")  # 加载模型
        doc = model(raw_prompt)  # 文本处理
        # 筛选词汇
        filtered_word = [token.text for token in doc if token.pos_ in selected_tag]
        for w in filtered_word:
            if vad_df.loc[w, "emotion"] != emotion:
                raw_prompt = raw_prompt.replace(w+" ", "")
        raw_prompt = re.sub(r'(?:^,|(?<=,))( |$)', '', raw_prompt)
        return raw_prompt


def get_afw_random(emotion):
    """
    随机获取指定情感类别的情感引导词
    :param emotion: 指定情感类别
    :return: 情感引导词
    """
    if emotion == "neutral":  # 选择中性不做情感引导
        return ""
    df = pd.read_csv("../data/affective_dictionary.csv")  # 情感引导词典
    # 选定情感类别的情感引导词
    selected_words = df[df["emotion"] == emotion]["affective_word"].tolist()
    afw_list = random.sample(selected_words, 3)  # 随机选择3个
    afw = ""
    for w in afw_list:
        afw = afw + ", " + w
    return afw


def get_afw_similarity(prompt, emotion):
    """
    根据文本主题相似度获取指定情感类别的情感引导词
    :param prompt: 输入prompt
    :param emotion: 指定情感类别
    :return: 情感引导词
    """
    if emotion == "neutral":  # 选择中性不做情感引导
        return ""
    df = pd.read_csv("../data/affective_dictionary.csv")  # 情感引导词典
    selected_data = df[df["emotion"] == emotion]  # 选定情感类别
    # 计算tf-idf向量
    vectorizer = TfidfVectorizer()
    related_vector = vectorizer.fit_transform(selected_data["related_prompt"].values.astype("U"))
    user_vector = vectorizer.transform([prompt])  # 计算输入prompt的向量
    similarity = cosine_similarity(user_vector, related_vector)  # 计算相似度
    # 获取相似度前3的情感引导词
    top_indices = similarity.argsort()[0][-3:][::-1]
    afw_list = selected_data.iloc[top_indices]["affective_word"].tolist()
    afw = ""
    for w in afw_list:
        afw = afw + ", " + w
    return afw


if __name__ == "__main__":
    prompt = "a forest"
    emotion = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
    for emo in emotion:
        afw = get_afw_similarity(prompt, emo)
        print(prompt+afw)
