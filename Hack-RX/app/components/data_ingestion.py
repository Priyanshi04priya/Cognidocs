from langchain_community.vectorstores import Qdrant
from app.components.data_preproceesing import load_document
from langchain_community.embeddings import OpenAIEmbeddings 
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

QDRANT_URL=os.getenv("QDRANT_URL")
# QDRANT_COLLECTION_NAME=os.getenv("QDRANT_COLLECTION_NAME")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=os.getenv("OPENAI_API_KEY")
    )

def result(docs,QDRANT_COLLECTION_NAME):
    split_docs = text_splitter.split_documents(documents=docs)
    vectorstore = Qdrant.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION_NAME,
        force_recreate=True,
    )

