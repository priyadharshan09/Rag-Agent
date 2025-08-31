import faiss
import numpy as np

# Load saved embeddings from disk
embeddings_np = np.load("D:/DS/Agreement/embeddings.npy")

# Create a FAISS index using L2 (Euclidean) distance
index = faiss.IndexFlatL2(embeddings_np.shape[1])

# Add embeddings to the index
index.add(embeddings_np)

# Save the index for later search queries
faiss.write_index(index, "D:/DS/Agreement/orchestro_contracts.index")

print("✅ FAISS index created and saved successfully.")
print(f"Total vectors indexed: {index.ntotal}")
