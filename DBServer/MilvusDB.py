from pymilvus import MilvusClient
from EmbeddingModel import get_embedding
client = MilvusClient("milvus_demo.db") 

collection_name = "demo_collection"
if client.has_collection(collection_name=collection_name):
    client.drop_collection(collection_name=collection_name)
client.create_collection(
    collection_name=collection_name,
    dimension=768,  # The vectors we will use in this demo has 768 dimensions
)

docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]

vectors = [get_embedding(doc) for doc in docs]

data = [
    {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
    for i in range(len(vectors))
]

client.insert(
    collection_name=collection_name,
    data=data,
)


query_vectors = get_embedding("Who is Alan Turing?")
# If you don't have the embedding function you can use a fake vector to finish the demo:
# query_vectors = [ [ random.uniform(-1, 1) for _ in range(768) ] ]

res = client.search(
    collection_name="demo_collection",  # target collection
    data=[query_vectors],  # query vectors
    limit=1,  # number of returned entities
    output_fields=["text", "subject"],  # specifies fields to be returned
)


print(res)