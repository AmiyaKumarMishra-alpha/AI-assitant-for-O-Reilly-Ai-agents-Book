from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS 
import os

# Load the text file
BOOK_PATH = "./Book.txt"
VECTOR_DB_PATH = "./vector_db"
# enable autodetect to avoid UnicodeDecodeError on non-default encodings
# loader = TextLoader(BOOK_PATH, encoding="utf-8", autodetect_encoding=True)
file_path = "./Google guide.pdf"
loader = PyPDFLoader(file_path)
# try:
#     documents = loader.load()
# except Exception as e:
#     print(f"Failed to load {BOOK_PATH}: {e}")
#     raise

try:
    documents = loader.load()
except Exception as e:
    print(f"Failed to load {file_path}: {e}")
    raise

#chunking
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Number of chunks: {len(chunks)}")

#embeddings and vector store
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = FAISS.from_documents(chunks, embeddings)

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

vector_store.save_local(VECTOR_DB_PATH)

print("Vector store saved")