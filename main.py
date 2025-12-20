from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
import os

# Load the text file
BOOK_PATH = "./Book.txt"
VECTOR_DB_PATH = "./vector_db"
# enable autodetect to avoid UnicodeDecodeError on non-default encodings
loader = TextLoader(BOOK_PATH, encoding="utf-8", autodetect_encoding=True)
try:
    documents = loader.load()
except Exception as e:
    print(f"Failed to load {BOOK_PATH}: {e}")
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