import os
from typing import List, Tuple
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


def load_file_01(filepath):
    # TextLoader来自 langchain.document_loaders，支持多种文本格式的加载
    text_loader = TextLoader(filepath, autodetect_encoding=True)
    docs = text_loader.load()
    # 简单文本分割，按指定分隔符分割文本
    text_splitter = CharacterTextSplitter(
        separators='\n\n',
        chunk_size=500,
        chunk_overlap=50  # 块之间的字符重叠最大数量
    )
    results  = text_splitter.split_documents(docs)
    return results
# 可选参数还有
# strip_whitespace=True, # 是否去掉每个块的前后空白符
# length_function=len, # 计算文本长度的函数，默认为len


def load_file_02(filepath):
    # TextLoader来自 langchain.document_loaders，支持多种文本格式的加载
    text_loader = TextLoader(filepath, autodetect_encoding=True)
    docs = text_loader.load()
    # 最常用的分割器，适用于大多数RAG场景
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', ' ', ''], # 递归使用的分隔符列表，可以理解为会按顺序尝试每一个分隔符，如果用某个分隔符能切出大小合适的文本块，就用它
        chunk_size=500,
        chunk_overlap=50  # 块之间的字符重叠最大数量
    )
    results  = text_splitter.split_documents(docs)
    return results
# 可选参数还有
# length_function=len, # 计算文本长度的函数，默认为len
# is_separator_regex=False # 是否将分隔符视为正则表达式，默认为False



def load_file_03(filepath):
    # TextLoader来自 langchain.document_loaders，支持多种文本格式的加载
    text_loader = TextLoader(filepath, autodetect_encoding=True)
    docs = text_loader.load()
    # TokenTextSplitter 按照token数量来分割文本，适用于对token数量敏感的LLM
    text_splitter = TokenTextSplitter(
        encoding_name="gpt2", # 用于计算token数量的编码器，默认为"gpt2"
        chunk_size=500,
        chunk_overlap=50  # 块之间的字符重叠最大数量
    )
    results  = text_splitter.split_documents(docs)
    return results
# 可选参数还有
# allowed_special=set(), # 允许的特殊token集合，默认为空集
# disallowed_special=set(), # 不允许的特殊token集合，默认为空集
# encoding_name 还包括 "cl100k_base"（适用于GPT-4和GPT-3.5-turbo），"p50k_base"（适用于Llama模型）等，详情见 tiktoken 文档



# 此外还有 LanguageTextSplitter，支持按编程语言语法分割文本
# PythonCodeTextSplitter，支持按Python语法分割文本
# MarkdownTextSplitter，使用md特定的分隔符
# MarkdownHeaderTextSplitter，按md标题分割文本