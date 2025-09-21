import json
import numpy as np
import faiss
import os

# ========== CONFIG ==========
EMB_PATH = "embeddings_local.jsonl"   # your embeddings file
INDEX_PATH = "faiss_index.bin"        # output faiss index file
D = 1024  # embedding dimension for intfloat/multilingual-e5-large
# ============================

def build_faiss():
    print(f"📖 Loading embeddings from {EMB_PATH}...")
    ids, texts, embeddings = [], [], []

    with open(EMB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ids.append(obj["id"])
            texts.append(obj["text"])
            embeddings.append(np.array(obj["embedding"], dtype="float32"))

    embeddings = np.vstack(embeddings).astype("float32")
    print(f"✅ Loaded {len(embeddings)} embeddings with dim {embeddings.shape[1]}")

    # Create FAISS index
    print("🔄 Building FAISS index (L2, normalized)...")
    index = faiss.IndexFlatIP(D)  # cosine similarity if embeddings are normalized
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"💾 FAISS index saved to {INDEX_PATH}")

    # Save metadata (id + text) separately
    meta_path = INDEX_PATH + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "texts": texts}, f, ensure_ascii=False, indent=2)

    print(f"💾 Metadata saved to {meta_path}")
    print("🎉 FAISS index build complete!")

if __name__ == "__main__":
    build_faiss()
