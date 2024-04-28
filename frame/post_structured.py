# @File    : post_structured.py
# @Description : 结构化后处理

import re
import spacy


def structured_text(text):
    """
    结构化后处理
    :param text: 输入文本
    :return: 结构化文本指令
    """
    nlp = spacy.load("en_core_web_lg")  # 加载模型
    text = text.lower().replace("noun", "")  # 预处理
    doc = nlp(text)  # 处理文本
    # 提取形容词-名词对
    adjective_noun_pairs = []
    for i, token in enumerate(doc):
        if token.pos_ == "NOUN":
            # 前向搜索
            j = i - 1
            if j >= 0 and doc[j].pos_ == "PUNCT":
                j -= 1
            if j >= 0 and doc[j].pos_ == "ADJ":
                adjective_noun_pairs.append(f"{doc[j].text} {token.text}")
            # 后向搜索
            k = i + 1
            if k < len(doc) and doc[k].pos_ == "PUNCT":
                k += 1
            if k < len(doc) and doc[k].pos_ == "ADJ":
                adjective_noun_pairs.append(f"{doc[k].text} {token.text}")
    drop_list = ["adjective", "noun", "context", "information",
                 "question", "answer", "query", "object",
                 "prompt", "pair", "sentence", "theme"]
    filtered_list = [item for item in adjective_noun_pairs if not any(drop_item in item for drop_item in drop_list)]
    unique_list = list(set(filtered_list))  # 去除重复的细节
    output = ", ".join(unique_list)  # 将结果输出为逗号分隔的形式
    output = re.sub(r'[^\w\s,]', '', output)  # 去除多余的符号
    output = re.sub(r'\b\w+\s,\s', '', output)
    return output


if __name__ == "__main__":
    text = input("input the text: ")
    result = structured_text(text)
    print(result)