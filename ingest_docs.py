import os
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

docs = []

for file in os.listdir("data/policies"):
    with open(f"data/policies/{file}", "r") as f:
        docs.append(f.read())

embeddings = []
for doc in docs:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    ).data[0].embedding
    embeddings.append(emb)

import numpy as np
index = faiss.IndexFlatL2(len(embeddings[0]))
index.add(np.array(embeddings).astype("float32"))

os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/index.faiss")

with open("vector_store/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("Documents ingested!")