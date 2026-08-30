import chromadb
import uuid

client = chromadb.Client()
collection  = client.create_collection('backend')

with open('backend.txt', 'r', encoding ='utf-8') as f:
    rules: list[str] = f.read().splitlines()
    
    
collection.add(
    ids = [str(uuid.uuid4()) for _ in rules],
    documents = rules,
    metadatas =[{"line": line} for line in range(len(rules))]
    
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