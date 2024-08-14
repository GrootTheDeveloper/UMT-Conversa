from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GPT4AllEmbeddings

# Khai bao bien
pdf_data_path = "data"
csv_data_path = "data/data.csv"
vector_db_path = "vectorstores/db_faiss"
def create_db_from_files():
    # Khai bao loader de quet toan bo thu muc dataa
    loader = DirectoryLoader(pdf_data_path, glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Embeding
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(vector_db_path)
    return db

def create_db_from_csv_files():
    # Tạo loader để tải dữ liệu từ file CSV
    loader = CSVLoader(file_path=csv_data_path,  encoding='utf-8')
    documents = loader.load()

    # Tách văn bản thành các đoạn nhỏ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Thiết lập và tạo embedding cho các đoạn văn bản
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.from_documents(chunks, embedding_model)

    # Lưu cơ sở dữ liệu vector
    db.save_local(vector_db_path)
    return db


create_db_from_csv_files()