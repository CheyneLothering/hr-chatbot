import os
import faiss
import pickle
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

VECTOR_PATH = "vector_store/index.faiss"
DOC_PATH = "vector_store/docs.pkl"

def load_vector_store():
    index = faiss.read_index(VECTOR_PATH)
    with open(DOC_PATH, "rb") as f:
        docs = pickle.load(f)
    return index, docs

def retrieve_context(query: str, k=3):
    index, docs = load_vector_store()

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    import numpy as np
    D, I = index.search(
        np.array([embedding]).astype("float32"), k
    )

    return "\n".join([docs[i] for i in I[0]])