import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import optparse

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store

required_settings = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
}

embedding_model = AzureOpenAIEmbeddings(
                    azure_endpoint=required_settings["AZURE_OPENAI_ENDPOINT"],
                    api_key=required_settings["AZURE_OPENAI_API_KEY"],
                    api_version=required_settings["AZURE_OPENAI_API_VERSION"],
                    azure_deployment=required_settings["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
                    chunk_size=16,
                    max_retries=5
            )

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}  
)

# Search for relevant documents

query = "In what year did Tesla begin production of the Roadster?"

## retriever = db.as_retriever(search_kwargs={"k": 5})

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,
        "score_threshold": 0.7  # Only return chunks with cosine similarity ≥ 0.3
    }
)

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


# Synthetic Questions: 

# 1. "What was NVIDIA's first graphics accelerator called?"
# 2. "Which company did NVIDIA acquire to enter the mobile processor market?"
# 3. "What was Microsoft's first hardware product release?"
# 4. "How much did Microsoft pay to acquire GitHub?"
# 5. "In what year did Tesla begin production of the Roadster?"
# 6. "Who succeeded Ze'ev Drori as CEO in October 2008?"
# 7. "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?"
# 8. "What was the original name of Microsoft before it became Microsoft?"