import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

current_dir=os.getcwd()
transcript_path=os.path.join(current_dir,"documents","transcript.txt")
db_path=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_path,"chroma_db")

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

with open(transcript_path,"r",encoding="utf-8") as f:
    transcript=f.read()
chunks=splitter.split_text(transcript)  
print(f"Total chunks created: {len(chunks)}")


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# create vector store
vector_store=Chroma.from_texts(
    texts=chunks,
    embedding=embeddings,
    persist_directory=persistent_directory
)

print(f"Vector store created and persisted at {persistent_directory}")
print("Chunk generation and vector store creation completed.")