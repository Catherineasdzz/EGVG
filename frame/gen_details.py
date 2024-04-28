# @File    : gen_details.py
# @Description : 构建细节扩写框架

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex


def expand_details(prompt, emotion):
    """
    为用户输入扩写类人情感-视觉细节
    :param prompt: 输入prompt
    :param emotion: 指定情感类别
    :return: 情感-视觉细节
    """
    model_name_or_path = "../model/llama-2-7b-chat-hf"  # LLM模型路径
    embed_name_or_path = "../model/bge-large-en"  # Embedding模型路径
    # LLM系统prompt
    system_prompt = """You are an AI assistant that answers questions in a friendly manner, \
        based on the given source documents and your own knowledge. Here are some rules you always follow:
        - Generate human readable output, avoid creating output with gibberish text.
        - Generate only the requested output, don't include any other language before or after the requested output.
        - Never say thank you, that you are happy to help, that you are an AI agent, etc. Just answer directly.
        - Generate professional language typically used in business documents in North America.
        - Never generate offensive or foul language.
        - Never generate emoji.
        """
    # prompt模板
    query_wrapper_prompt = PromptTemplate(
        "[INST]<<SYS>>\n" + system_prompt + "<</SYS>>\n\n{query_str}[/INST] "
    )
    # 加载LLM模型
    llm = HuggingFaceLLM(
        context_window=2048,
        max_new_tokens=1024,
        generate_kwargs={"temperature": 0.6, "do_sample": True},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name=model_name_or_path,
        model_name=model_name_or_path,
        device_map="auto",
    )
    # 加载Embedding模型
    embed_model = HuggingFaceEmbedding(model_name=embed_name_or_path)
    # 配置模型
    Settings.llm = llm
    Settings.embed_model = embed_model
    # 加载外部知识数据并向量化
    documents = SimpleDirectoryReader(f"../data/rag/{emotion}").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()
    # 检索背景及请求
    query_text = f"You will receive an emotion and an original prompt. \
                    Give some related visual objects with adjectives in only one paragraph. \
                    Only give the paragraph of visual objects. \
                    When you finish your response, you have to reformat your response into multiple adjective-noun pairs which \
                    separated by comma, like 'adjective noun, adjective noun, … ' \
                    emotion: {emotion}, original prompt: {prompt}"
    # 检索生成
    response = query_engine.query(query_text)
    return response


if __name__ == "__main__":
    emotion = input("select the emotion: ")
    raw_prompt = input("give the prompt: ")
    response = expand_details(raw_prompt, emotion)
    print(response)