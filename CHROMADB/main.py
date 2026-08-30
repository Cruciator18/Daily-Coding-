import chromadb
import uuid

client = chromadb.PersistentClient(path='./chroma_db')
collection  = client.create_collection('backend_rules')

with open('backend.txt', 'r', encoding ='utf-8') as f:
    backend_rules: list[str] = f.read().splitlines()
    
    
collection.add(
    ids = [str(uuid.uuid4()) for _ in backend_rules],
    documents = backend_rules,
    metadatas =[{"line": line} for line in range(len(backend_rules))]
    
)

print(collection.peek(5))    
    

results= collection.query(
    query_texts = ["How to to efficiently manage and reuse connections to the database server?",
                   "How to prevent denial-of-service via resource exhaustion?",
                   "How to optimize database queries for better performance?",
    ],
    n_results = 4
)

for i,query_results in enumerate(results["documents"]):
    print(f"\nQuery{i}")
    print('\n'.join(query_results))