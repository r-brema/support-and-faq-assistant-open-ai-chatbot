import chromadb
import os
from dotenv import load_dotenv

from chromadb.utils import embedding_functions

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

default_ef = embedding_functions.DefaultEmbeddingFunction()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_api_key, model_name="text-embedding-3-small"
)
croma_client = chromadb.PersistentClient(path="./db/chroma_persist")

collection = croma_client.get_or_create_collection(
    "my_story",
    embedding_function=openai_ef,
    metadata={"hnsw:batch_size": 10000}
)

# Define text documents
documents = [
   
    {
        "id": "doc1",
        "text": "Microsoft is a technology company that develops software. It was founded by Bill Gates and Paul Allen in 1975.",
    },
    {
        "id": "doc2",
        "text": "The Turing Test, proposed by Alan Turing, is a measure of a machine's ability to exhibit intelligent behavior equivalent to, or indistinguishable from, that of a human.",
    },
]

for doc in documents:
    print("going to upsert for ids=", doc["id"])
    collection.add(ids=[doc["id"]], documents=[doc["text"]])
print('upsert done')
# define a query text
query_text = "find document related to Turing Test"

results = collection.query(
    query_texts=[query_text],
    n_results=3,
)

for idx, document in enumerate(results["documents"][0]):
    doc_id = results["ids"][0][idx]
    distance = results["distances"][0][idx]

    print(
        f" For the query: {query_text}, \n Found similar document: {document} (ID: {doc_id}, Distance: {distance})"
    )