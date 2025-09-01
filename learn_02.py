from langchain_community.document_loaders import CSVLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from utils import get_qwen_models

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 连接大模型
llm, chat, embed = get_qwen_models()

# 读取.csv文件
students_loader = CSVLoader(file_path="testfiles/students.csv")
docs = students_loader.load()

# 向量化入库
vectorstore = Chroma.from_documents(documents=docs, embedding=embed)

retriever = vectorstore.as_retriever(search_kwargs={"k":2})

prompt = ChatPromptTemplate.from_messages([
    ("human",""""You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:"""")
])

# RAG 链
rag_chain = (
    {"context": retriever | format_docs,
     "question":RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

rag_chain.invoke(input="谁的成绩最高？")