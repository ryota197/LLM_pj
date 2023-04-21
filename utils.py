import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

# .envファイルからAPIキーを読み込む
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# モデル定義
chatGPT = OpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

#定数
CHUNK_SIZE = 1000

# PDFファイルの読み込み関数
def load_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    # 文書の分割
    text_splitter = CharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    return texts

# Embeddings生成：ベクトルの集合データベースを返す
def create_embeddings(split_docs):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(split_docs, embeddings)

    return db

# ChatGPTへの入力処理
"""
    引数の説明
    file PDFファイルのパス
    query 指示
    chain_type
    k 参照するテキストチャンクの数
"""
def qa(file, query, chain_type, k):
    split_docs = load_pdf(file)
    vector_store = create_embeddings(split_docs)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa = RetrievalQA.from_chain_type(
        llm=chatGPT, chain_type=chain_type, retriever=retriever, return_source_documents=True
    )
    result = qa({"query": query})
    print(result['result'])
    return result