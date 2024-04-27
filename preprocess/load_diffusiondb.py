# @File    : load_diffusiondb.py
# @Description : 下载DiffusionDB的prompt数据

import pandas as pd
from urllib.request import urlretrieve

# 下载DiffusionDB文本数据
table_url = f"https://huggingface.co/datasets/poloclub/diffusiondb/resolve/main/metadata.parquet"
urlretrieve(table_url, "../data/metadata.parquet")

# 写入csv文件
metadata_df = pd.read_parquet("../data/metadata.parquet")  # 读取数据
metadata_df = metadata_df.dropna(subset=["prompt"])  # 删除空值
metadata_df.drop_duplicates(subset=["prompt"], keep="first", inplace=True)  # 删除重复值
metadata_df["prompt"] = metadata_df["prompt"].replace("[^a-zA-Z0-9, ]", "", regex=True)  # 只保留数字和字母
metadata_df["prompt"].to_csv("../data/diffusiondb.csv", index=False)  # 写入文件