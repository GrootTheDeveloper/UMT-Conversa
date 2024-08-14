from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from accelerate import Accelerator
import os
import torch

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Cấu hình
model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Đặt thiết bị GPU đúng
torch.cuda.set_device(0)  # Đảm bảo GPU RTX 3050 là thiết bị 0
#print("Using GPU:", torch.cuda.get_device_name(0))  # In tên GPU được sử dụng
#print("GPU Memory Allocated:", torch.cuda.memory_allocated(device=0) / 1e9, "GB")

# Khởi tạo Accelerator với cấu hình mặc định
accelerator = Accelerator()

# Load LLM
def load_llm(model_file):
    config = {
        'max_new_tokens': 500,
        'temperature': 0.01,
    }

    llm = CTransformers(model=model_file, model_type="llama", gpu_layers=15, device=0, config=config)
    llm = accelerator.prepare(llm)
    #print("Model Loaded to GPU")
    return llm

# Đọc từ VectorDB
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model)
    return db

db = read_vectors_db()
llm = load_llm(model_file)

# Tạo Prompt
#template = """system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời\n
#    {context}\nuser\n{question}\nassistant"""
#prompt = PromptTemplate(template=template, input_variables=["context", "question"])

prompt_template = """
Hãy sử dụng thông tin dưới đây để trả lời câu hỏi một cách chính xác nhất. Nếu không tìm thấy thông tin phù hợp trong bộ dữ liệu, xin hãy báo rằng "Không có thông tin".

Context:
{context}

Câu hỏi:
{question}

Cách trả lời:
- Chỉ trả lời câu hỏi về trường UMT, nếu không liên quan chỉ cần nói rằng "Tôi chỉ trả lời câu hỏi trong phạm vi trường UMT"
- Nếu bạn biết câu trả lời, hãy cung cấp thông tin cụ thể và chính xác.
- Nếu bạn không chắc chắn, hãy đề xuất các nguồn thông tin khác hoặc cách để tìm hiểu thêm.
- Nếu không có thông tin, hãy nói "Không có thông tin".

Trợ lý: """

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Tạo simple chain
def create_qa_chain(prompt, llm, db):
    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 1}, max_tokens_limit=1024),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return llm_chain

llm_chain = create_qa_chain(prompt, llm, db)

# Chạy cái chain
while(True):
    question = input("Nhập câu hỏi: ")
    response = llm_chain.invoke({"query": question})
    #print("Response:", response)
    print("Bot: ", response)

