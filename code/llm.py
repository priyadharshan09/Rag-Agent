import google.generativeai as genai
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Load the embedding model (same one used for FAISS index)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("D:/DS/Agreement/orchestro_contracts.index")

# Load chunk metadata
with open("D:/DS/Agreement/all_pdfs_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Configure Gemini with your API key
genai.configure(api_key="AIzaSyDN_jqSZiv0K79Paddz9-qTtdQlwiXx2AY")
gmodel = genai.GenerativeModel("gemini-1.5-flash")

def embed_query(query):
    """Embed the input query string."""
    query_vec = model.encode([query])
    return np.array(query_vec).astype('float32')

def search_index(query_vec, top_k=5):
    """Search FAISS index for nearest chunks."""
    distances, indices = index.search(query_vec, top_k)
    return distances[0], indices[0]

def generate_answer(query, retrieved_chunks):
    """Use Gemini to generate a structured answer."""
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a helpful assistant for contract analysis.
    Question: {query}

    Relevant contract excerpts:
    {context}

    Based on this, provide a clear and structured answer.
    """
    response = gmodel.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    query_vec = embed_query(user_query)
    distances, indices = search_index(query_vec)

    # Get top chunks
    retrieved_chunks = [chunks[i]["content"] for i in indices]

    # Generate answer using Gemini
    answer = generate_answer(user_query, retrieved_chunks)
    print("\nAnswer:", answer)
