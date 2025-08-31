import fitz  # PyMuPDF
import json
import os

# -----------------------------
# Step 1: Folder with PDFs
# -----------------------------
pdf_folder = "D:\DS\Agreement\data"   # put all 10 PDFs inside this folder
output_file = "all_pdfs_chunks.json"

# -----------------------------
# Step 2: Function to split text
# -----------------------------
def split_into_chunks(text, chunk_size=500):
    """Split text into smaller parts (for RAG)."""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

# -----------------------------
# Step 3: Process all PDFs
# -----------------------------
chunked_data = []

for pdf_name in os.listdir(pdf_folder):
    if pdf_name.endswith(".pdf"):  # only take PDFs
        pdf_path = os.path.join(pdf_folder, pdf_name)
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                chunks = split_into_chunks(text)
                for idx, chunk in enumerate(chunks):
                    chunked_data.append({
                        "file": pdf_name,
                        "page": page_num,
                        "chunk_id": f"{pdf_name}_{page_num}_{idx+1}",
                        "content": chunk
                    })

# -----------------------------
# Step 4: Save everything
# -----------------------------
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunked_data, f, indent=4, ensure_ascii=False)

print(f"✅ Extracted {len(chunked_data)} chunks from {len(os.listdir(pdf_folder))} PDFs")
print(f"💾 Saved to {output_file}")
