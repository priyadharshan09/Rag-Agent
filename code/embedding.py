# embedding_save.py

from sentence_transformers import SentenceTransformer
import json
import numpy as np

# Load the pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load chunked text data
json_path = "D:/DS/Agreement/all_pdfs_chunks.json"
with open(json_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Extract the text content from each chunk
texts = [chunk["content"] for chunk in chunks]

# Generate embeddings for all chunks
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings)

# Save embeddings to disk for later use
save_path = "D:/DS/Agreement/embeddings.npy"
np.save(save_path, embeddings_np)

print(f"✅ Embeddings saved to {save_path}")
print(f"Total chunks embedded: {len(texts)}")
