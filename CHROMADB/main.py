import os
import chromadb
from dotenv import load_dotenv

load_dotenv(override=True)

chroma_key = os.getenv("CHROMA_API_KEY")
os.environ["CHROMA_API_KEY"] = chroma_key

client = chromadb.CloudClient(
    api_key=chroma_key,
    tenant='c083b317-a6b8-4ff7-9e45-97d73c87dbb1',
    database='helloworld'
)

collection = client.get_collection('cruciator18_rag_agentic_main')

print("Connection successful! Peeking at 1 document...")
print(collection.peek(1))    

results = collection.query(
    query_texts=["what is the basic structure of the retriever?"],
    n_results=1
)

print("\nResults of query:")
for i, query_results in enumerate(results["documents"][0]):
    print(f"\nResult {i}:")
    print(query_results)