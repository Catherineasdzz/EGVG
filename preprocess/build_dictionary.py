# @File    : build_dictionary.py
# @Description : 构建情感引导词典

import pandas as pd
from tqdm import tqdm

afw_df = pd.read_csv("../data/afw_manual.csv")  # 读取情感引导词
prompt_df = pd.read_csv("../data/diffusiondb.csv")  # 读取prompt数据
dict_df = pd.DataFrame(columns=["emotion", "affective_word", "related_prompt"])
for index, row in tqdm(afw_df.iterrows(), total=len(afw_df), desc="Processing Rows"):
    emotion = row["emotion"]
    affective_word = row["word"]
    prompt_df["prompt"] = prompt_df["prompt"].fillna("")  # 将缺失值填充为空字符串
    related_prompt = prompt_df[prompt_df["prompt"].str.contains(affective_word)]["prompt"].tolist()  # 获取包含引导词的prompt数据
    dict_df = pd.concat([dict_df, pd.DataFrame({
        "emotion": emotion,
        "affective_word": affective_word,
        "related_prompt": [related_prompt]
    })], ignore_index=True)
print(dict_df)
dict_df.to_csv("../data/affective_dictionary.csv", index=False)  # 写入文件
