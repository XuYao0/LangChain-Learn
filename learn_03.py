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
from langchain_core.output_parsers import StrOutputParser

model_name = "./Qwen3-8B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

EMBEDDING_MODEL = 'bge-base-zh-v1.5'
EMBEDDING_DEVICE = "cuda"
embedding_model_dict = {
    "bge-base-zh-v1.5": "./bge-base-zh-v1.5",
}
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_dict[EMBEDDING_MODEL], model_kwargs={'device': EMBEDDING_DEVICE})



class Qwen(LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history_len: int = 3

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "Qwen"

    @property
    def _history_len(self) -> int:
        return self.history_len

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
            max_new_tokens=4096
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



# load_file() 定义
def load_file(filepath):
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
    write_check_file(filepath, results)
    return results

def write_check_file(filepath, docs):
    folder_path = os.path.join(os.path.dirname(filepath), "tmp_files")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    fp = os.path.join(folder_path, 'load_file.txt')
    with open(fp, 'a+', encoding='utf-8') as fout:
        fout.write("filepath=%s, len=%s" % (filepath, len(docs)))
        fout.write('\n')
        for i in docs:
            fout.write(str(i))
            fout.write('\n')
        fout.close()


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


class FAISSWrapper(FAISS):
    chunk_size = 1000 
    chunk_conent = True 
    score_threshold = 0 

    def similarity_search_with_score_by_vector(
        self, embedding: List[float], k: int = 4
    ) -> List[Tuple[Document, float]]:
        
        print("测试，本函数被执行")
        if self.chunk_conent: 
            print("合并chunk")
        else: 
            print("不合并chunk")

        scores, indices = self.index.search(np.array([embedding], dtype=np.float32), k)
        docs = []
        id_set = set()
        store_len = len(self.index_to_docstore_id)
       
        for j, i in enumerate(indices[0]):
            if i == -1 or 0 < self.score_threshold < scores[0][j]:
                continue
            id_set.add(i)
            if not self.chunk_conent:
                continue

            _id = self.index_to_docstore_id[i]
            doc = self.docstore.search(_id)
            docs_len = len(doc.page_content)
            forward_stopped = False
            backward_stopped = False

            for k in range(1, max(i, store_len - i)):
                if forward_stopped and backward_stopped:
                    break

                if not forward_stopped and i + k < len(self.index_to_docstore_id):
                    _id0 = self.index_to_docstore_id[i + k]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        forward_stopped = True
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(i + k)
                    else:
                        forward_stopped = True   

                if not backward_stopped and i - k >= 0:
                    _id0 = self.index_to_docstore_id[i - k]
                    doc0 = self.docstore.search(_id0)
                    if docs_len + len(doc0.page_content) > self.chunk_size:
                        backward_stopped = True
                    elif doc0.metadata["source"] == doc.metadata["source"]:
                        docs_len += len(doc0.page_content)
                        id_set.add(i - k)

        id_list = sorted(list(id_set))
        id_lists = separate_list(id_list)
        print("id_lists=", id_lists)
        docs_len = 0
        for id_seq in id_lists:
            doc = None
            for id in id_seq:
                if id == id_seq[0]:
                    _id = self.index_to_docstore_id[id]
                    doc = self.docstore.search(_id)
                else:
                    _id0 = self.index_to_docstore_id[id]
                    doc0 = self.docstore.search(_id0)
                    doc.page_content += " " + doc0.page_content

            if not isinstance(doc, Document):
                raise ValueError(f"Could not find document for id {_id}, got {doc}")
            
            doc_score = min([scores[0][id] for id in [indices[0].tolist().index(i) for i in id_seq if i in indices[0]]])
            doc.metadata["score"] = int(doc_score)
            docs_len += len(doc.page_content)
            print("====================")
            print("docs_len=", docs_len)
            print(doc.page_content)
            docs.append((doc, doc_score))
        return docs



if __name__ == '__main__':
    # 构建所检索的向量库
    filepath = './sword.md'
    docs = load_file(filepath)
    docsearch = FAISSWrapper.from_documents(docs, embeddings)
    retriver = docsearch.as_retriever(search_kwargs={"k": 5})

    # 构建检索结果的格式化方式
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    # 构建提示词 
    PROMPT_TEMPLATE = """Known information:
{context_str}

Based on the above known information, respond to the user's question concisely and professionally. If an answer cannot be derived from it, say 'The question cannot be answered with the given information' or 'Not enough relevant information has been provided,' and do not include fabricated details in the answer. Please respond in Chinese. The question is {question}"""
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    # 模型实例
    llm = Qwen()

    # 构建LCEL链
    rag_chain = (
        {
            "context": retriver | format_docs,
            "question": llm
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    query = "涤罪僧会是韶华轻掷那个女人的父亲吗？"
    result = rag_chain.invoke(query)
    print(result)