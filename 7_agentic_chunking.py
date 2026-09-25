from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

required_settings = {
        "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION"),
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        "AZURE_OPENAI_CHAT_DEPLOYMENT": os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
}


# Initilize the llm
llm = AzureChatOpenAI(
    azure_endpoint=required_settings["AZURE_OPENAI_ENDPOINT"],
    api_key=required_settings["AZURE_OPENAI_API_KEY"],
    api_version=required_settings["AZURE_OPENAI_API_VERSION"],
    azure_deployment=required_settings["AZURE_OPENAI_CHAT_DEPLOYMENT"]
)

# Tesla text to chunk
tesla_text = """Tesla's Q3 Results
Tesla reported record revenue of $25.2B in Q3 2024.
The company exceeded analyst expectations by 15%.
Revenue growth was driven by strong vehicle deliveries.

Model Y Performance  
The Model Y became the best-selling vehicle globally, with 350,000 units sold.
Customer satisfaction ratings reached an all-time high of 96%.
Model Y now represents 60% of Tesla's total vehicle sales.

Production Challenges
Supply chain issues caused a 12% increase in production costs.
Tesla is working to diversify its supplier base.
New manufacturing techniques are being implemented to reduce costs."""


#Create the prompt
prompt = f"""
You are a text chunking expert. Split this text into logical chunks.

Rules:
 - Each chunk should be around 200 characters or less
 - Split at natural topic boundaries
 - Keep realted information together
 - Put "<<<SPLIT>>>" between chunks

Text:
{tesla_text}

Return the text with <<<SPLIT>>> markers where you want to split:
"""

# Get AI response
print(" Asking AI to chunk the text...")
response = llm.invoke(prompt)
marked_text = response.content

#Split the text at the <<<SPLIT>>> markers

chunks = marked_text.split("<<<SPLIT>>>")

#Clean up the chunks (remove extra whitespace)
clean_chunks = []
for chunk in chunks:
    clean_chunk = chunk.strip()
    if clean_chunk:  # Only add non-empty chunks
        clean_chunks.append(clean_chunk)

#Show the results
print("Agentic Chunking Results:")
print("="*50)

for i, chunk in enumerate(clean_chunks, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()