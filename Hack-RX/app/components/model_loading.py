from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import ParentDocumentRetriever

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    length_function=len,
    is_separator_regex=False,
)

parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    length_function=len,
    is_separator_regex=False,
)

store = InMemoryStore()

client = QdrantClient(":memory:")

client.create_collection(
    collection_name="demo_collection",
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="demo_collection",
    embedding=gemini_embeddings,
)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

docs=[]
retriever.add_documents(docs)

# len(list(store.yield_keys()))


# 

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor


llm=llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
compressor = LLMChainExtractor.from_llm(llm)

compression_retriever=ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

