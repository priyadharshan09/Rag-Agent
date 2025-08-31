from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

# Load the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load FAISS index
index = faiss.read_index("D:/DS/Agreement/orchestro_contracts.index")

# Load chunk metadata
with open("D:/DS/Agreement/all_pdfs_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Configure Gemini API key (replace with your key or environment variable)
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)
gmodel = genai.GenerativeModel("gemini-1.5-flash")

def embed_query(query):
    query_vec = embedding_model.encode([query])
    return np.array(query_vec).astype('float32')

def search_index(query_vec, top_k=5):
    distances, indices = index.search(query_vec, top_k)
    return distances[0], indices[0]

def generate_answer(query, retrieved_chunks):
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

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    query_vec = embed_query(question)
    distances, indices = search_index(query_vec, top_k=5)

    retrieved_chunks = [chunks[i]["content"] for i in indices]

    if not retrieved_chunks:
        return jsonify({"error": "No relevant data found"}), 404

    answer = generate_answer(question, retrieved_chunks)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
@app.route("/", methods=["GET"])
def home():
    return "API is running. Use POST /api/ask to get answers.", 200

