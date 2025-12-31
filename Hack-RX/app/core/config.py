from dotenv import load_dotenv
import os
load_dotenv()

class Settings :
    # class to load env Variables
    OPENAI_API_KEY: str =os.getenv("OPENAI_API_KEY")
    QDRANT_URL: str=os.getenv("QDRANT_URL")

    QDRANT_COLLECTION_NAME : str=os.getenv("QDRANT_COLLECTION_NAME")
    
settings=Settings()
if not settings.OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY env not set")

if not settings.QDRANT_URL:
    raise ValueError("QDRANT_URL env not set")
    