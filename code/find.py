import faiss
import numpy as np

# Load your saved embeddings from a file (for example, numpy .npy file) or re-run embedding here
embeddings_np = np.load("embeddings.npy")  # if you saved previously

index = faiss.IndexFlatL2(embeddings_np.shape[1])
index.add(embeddings_np)

faiss.write_index(index, "orchestro_contracts.index")
