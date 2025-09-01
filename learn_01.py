from transformers import AutoModelForCausalLM, AutoTokenizer
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
import os
import re
import torch
import argparse
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from typing import List, Tuple
import numpy as np
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA


class Qwen(LLM, ABC):   # 继承自 LLM 和 ABC，LLM是 LangChain 提供的基础语言模型类，ABC 是 Python 的抽象基类模块，后者可以让Qwen类定义抽象方法
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 3

    def __init__(self):  # 初始化方法
        super().__init__()

    @property  # 将方法转换为属性，实现只读属性，python没有严格的常量
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

    # 可以让 history_len 可改，但不能通过属性来改，只能通过方法来改
    def set_history_len(self, history_len: int = 10) -> None:
        self.history_len = history_len

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"max_token": self.max_token,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "history_len": self.history_len}


# 加载模型和分词器
model_name = "./Qwen3-8B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


# load_file() 定义
def load_file(filepath):
    # TextLoader来自 langchain.document_loaders，支持多种文本格式的加载
    loader = TextLoader(filepath, autodetect_encoding=True)
    # 使用自定义的中文文本分割器分割文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=[
            "\n\n",   # 段落分割
            "\n",     # 行分割  
            "。",     # 中文句号
            "！",     # 中文感叹号
            "？",     # 中文问号
            ".",      # 英文句号
            "!",      # 英文感叹号
            "?",      # 英文问号
            " ",      # 空格
            ""        # 字符级别分割
        ]
    )
    docs = loader.load_and_split(text_splitter)
    write_check_file(filepath, docs)
    return docs

# write_check_file() 定义
def write_check_file(filepath, docs):
    # 在原文件目录下创建tmp_files文件夹
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # 创建load_file.txt文件。用来写入加载并分割后的文档，达到记录和调试的效果。
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s, len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


# 这样处理后，每个元素就是一个完整的句子（含标点），便于后续中文文本处理。
class ChineseTextSplitter(CharacterTextSplitter):
    def __init__(self, pdf: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.pdf = pdf

    def split_text(self, text: str) -> List[str]:
        if self.pdf:
            # 将多个换行符替换为一个换行符
            text = re.sub(r"\n{3,}", "\n", text)
            # 将空白字符替换为一个空格
            text = re.sub('\s', ' ', text)
            # 将所有连续的两个换行符删除
            text = text.replace("\n\n", "")

        sent_sep_pattern = re.compile(
            r'([。；？！!?]["'"」』]{0,2}|(?<=[。；？！!?]["'"」』]{0,2})(?=[^a-zA-Z0-9_"\'《〈（\[】）〉》"'']))'
        )
        sent_list = []
        for ele in sent_sep_pattern.split(text):
            if sent_sep_pattern.match(ele) and sent_list:
                sent_list[-1] += ele
            elif ele:
                sent_list.append(ele)
        return sent_list
    





EMBEDDING_MODEL = 'bge-base-zh-v1.5'
# Embedding running device
EMBEDDING_DEVICE = "cuda"
embedding_model_dict = {
    "bge-base-zh-v1.5": "./bge-base-zh-v1.5",
}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL], model_kwargs={'device': EMBEDDING_DEVICE})




# 将1，2，3，5，6，8分割成[1,2,3],[5,6],[8]
def separate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists

# 继承自FAISS，后者是 LangChain 框架中用于向量检索的类，底层封装了 Facebook 的 FAISS 库，实现高效的向量相似度搜索。
class FAISSWrapper(FAISS):
    chunk_size = 250 # chunk的最大长度，超过该长度就不再合并相邻的chunk
    chunk_conent = True # 是否将相邻的chunk合并成一个文档返回
    score_threshold = 0 # 相似度阈值，低于该值的结果会被过滤掉

    # 重写父类的相似度搜索方法
    # embedding 是用户问题的输入向量，k 是返回的相似文档数量
    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        # self.index 是 FAISS 库中的索引对象，search 方法用于在索引中查找与给定向量最相似的 k 个向量
        # 返回的 scores 是相似度分数(越小越相似，因为返回的是距离值)，indices 是对应的向量索引
        # np.array([embedding], dtype=np.float32) 将输入的嵌入向量转换为 NumPy 数组，确保数据类型为 float32
        # 这里之所以是 [embedding]，是因为 FAISS 的 search 方法期望输入是一个二维数组，即使只查询一个向量，也需要将其包装在一个列表中
        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        # 最终返回的文档列表
        docs = []
        # 用来存储已经合并的chunk的索引，避免重复添加
        id_set = set()
        # store_len 是向量库中所有文档的数量
        store_len = len(self.index_to_docstore_id)
        # j表示当前处理的是第几个相似文档，i表示该文档在向量库中的索引，之所以是indices[0]，是因为上面search方法返回的indices是一个二维数组
        for j, i in enumerate(indices[0]):
            # This happens when not enough docs are returned.FAISS返回-1表示没有找到对应的向量
            # 或者者相似度分数低于设定的阈值，其实是距离值高于设定的阈值
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                continue
            # 根据向量索引找到对应的文档ID
            _id = self.index_to_docstore_id[i]
            # 根据文档ID从文档存储中检索出完整的文档内容
            doc = self.docstore.search(_id)
            # 如果不需要合并相邻的chunk，直接把当前文档添加到结果列表中
            if not self.chunk_conent:
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                doc.metadata["score"] = int(scores[0][j])
                docs.append(doc)
                continue
            # 如果需要合并相邻的chunk，先把当前文档的索引添加到id_set中
            id_set.add(i)
            docs_len = len(doc.page_content)
            for k in range(1, max(i, store_len - i)):
                break_flag = False
                # 同时向前和向后检查相邻的chunk
                for l in [i + k, i - k]:
                    if 0 <= l < len(self.index_to_docstore_id):
                        # 根据索引找到对应的文档ID和文档内容
                        _id0 = self.index_to_docstore_id[l]
                        doc0 = self.docstore.search(_id0)
                        # 如果合并后的长度超过设定的chunk_size，或者者者不是同一个文档，就停止合并
                        if docs_len + len(doc0.page_content) > self.chunk_size:
                            break_flag = True
                            break
                        # 确保是同一个文档
                        elif doc0.metadata["source"] == doc.metadata["source"]:
                            docs_len += len(doc0.page_content)
                            id_set.add(l)
                if break_flag:
                    break

            # if not self.chunk_conent:
            #     return docs
            # if len(id_set) == 0 and self.score_threshold > 0:
            #     return []
            # 将id_set转换为有序列表，方便后续处理
            id_list = sorted(list(id_set))
            # 将连续的索引分割成多个子列表，方便合并相邻的chunk
            id_lists = separate_list(id_list)
            for id_seq in id_lists:
                for id in id_seq:
                    if id == id_seq[0]:
                        _id = self.index_to_docstore_id[id]
                        doc = self.docstore.search(_id)
                    else:
                        _id0 = self.index_to_docstore_id[id]
                        doc0 = self.docstore.search(_id0)
                        # page_content 是 Document 类中的属性，表示文档的内容
                        doc.page_content += " " + doc0.page_content

                # 确保 doc 是 Document 类型的实例，防御性编程
                if not isinstance(doc, Document):
                    raise ValueError(f"Could not find document for id {_id}, got {doc}")
                # 如果被合并的文档位于K个相似文档中，就取它们的最低分数作为合并后文档的分数
                # .index(i) 是找到元素 i 在 indices[0] 中的下标，从而获取对应的分数
                doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
                doc.metadata["score"] = int(doc_score)
                docs.append((doc, doc_score))
            for doc in docs:
                print(doc[0].metadata['source'], doc[0].metadata['score'], len(doc[0].page_content))
            return docs



if __name__ == '__main__':
    # load docs (pdf file or txt file)
    filepath = './sword.md'
    # Embedding model name
    PROMPT_TEMPLATE = """Known information:
{context_str}

Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in Chinese. The question is {question}"""

    # 表示每次向量检索时，返回相似度最高的前3个文档片段（top-k 检索）。
    VECTOR_SEARCH_TOP_K = 3
    # LangChain 问答链的链类型参数，'stuff' 表示将所有检索到的文档内容“拼接”在一起后统一交给大模型生成答案。这是最常用、最简单的一种链式问答方式。
    CHAIN_TYPE = 'stuff'
    llm = Qwen()

    docs = load_file(filepath)

    # 根据传入的文档和嵌入模型，自动构建向量数据库实例
    docsearch = FAISSWrapper.from_documents(docs, embeddings)

    # PromptTemplate类来自 langchain.prompts.prompt，用于定义提示模板
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context_str", "question"]
    )

    # chain_type_kwargs 用于传递给问答链的额外参数，document_variable_name 指定了在提示模板中表示检索到的文档内容的变量名
    chain_type_kwargs = {"prompt": prompt, "document_variable_name": "context_str"}
    # RetrievalQA类来自 langchain.chains，封装了问答系统的逻辑
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=CHAIN_TYPE,
        retriever=docsearch.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K}),
        chain_type_kwargs=chain_type_kwargs
    )

    query = "请简要介绍下江吟风"
    print(qa.run(query))