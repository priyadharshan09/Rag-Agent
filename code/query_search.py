from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the same embedding model used for document chunks
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the FAISS index you created earlier
index = faiss.read_index("D:/DS/Agreement/orchestro_contracts.index")

# Load your chunk metadata (to map indexes to text)
import json
with open("D:/DS/Agreement/all_pdfs_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

def embed_query(query):
    """Embed the input query string."""
    query_vec = model.encode([query])
    return np.array(query_vec).astype('float32')

def search_index(query_vec, top_k=5):
    """Search FAISS index for nearest chunks."""
    distances, indices = index.search(query_vec, top_k)
    return distances[0], indices[0]

def get_results(indices):
    """Get text chunks corresponding to indices."""
    return [chunks[i]["content"] for i in indices]

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    query_vec = embed_query(user_query)
    distances, indices = search_index(query_vec)
    
    print(f"\nTop {len(indices)} relevant chunks:")
    for i, idx in enumerate(indices):
        print(f"\nChunk {i+1} (Distance: {distances[i]:.4f}):")
        print(get_results([idx])[0])
