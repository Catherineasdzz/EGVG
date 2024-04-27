# @File    : post_structured.py
# @Description : 结构化后处理

import spacy


def structured_text(text):
    """
    结构化后处理
    :param text: 输入文本
    :return: 结构化文本指令
    """
    nlp = spacy.load("en_core_web_sm")  # 加载模型
    doc = nlp(text)  # 处理文本
    # 提取形容词-名词对
    adjective_noun_pairs = []
    for i, token in enumerate(doc):
        if token.pos_ == "NOUN" and i >= 1 and doc[i-1].pos_ == "ADJ":
            adjective_noun_pairs.append(f"{doc[i-1].text} {token.text}")
    output = ", ".join(adjective_noun_pairs)  # 将结果输出为逗号分隔的形式
    return output


if __name__ == "__main__":
    text = input("input the text: ")
    result = structured_text(text)
    print(result)