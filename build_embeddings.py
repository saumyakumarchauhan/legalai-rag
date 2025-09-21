import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

# ========== CONFIG ==========
DATA_PATH = "merged_dataset.jsonl"
EMB_PATH = "embeddings_local.jsonl"
BATCH_SIZE = 50   # adjust based on GPU memory
MODEL_NAME = "intfloat/multilingual-e5-large"  # 1024-dim embeddings
# ============================

# Load model
print(f"🔄 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
    print("✅ Using GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ GPU not found, using CPU")

def main(resume_batch=0):
    print(f"📖 Reading dataset: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print(f"✅ Loaded {len(records)} records")

    # Check how many records already embedded
    processed = 0
    if os.path.exists(EMB_PATH):
        with open(EMB_PATH, "r", encoding="utf-8") as f_out:
            processed = sum(1 for _ in f_out)
        print(f"🔄 Resuming from record {processed} (batch {processed // BATCH_SIZE})")

    start_idx = max(processed, resume_batch * BATCH_SIZE)

    with open(EMB_PATH, "a", encoding="utf-8") as out_f:
        for i in tqdm(range(start_idx, len(records), BATCH_SIZE), desc="🔢 Embedding batches"):
            batch_records = records[i:i + BATCH_SIZE]
            texts, meta = [], []

            for j, r in enumerate(batch_records):
                text = r.get("output") or r.get("input") or r.get("text", "")
                if not text.strip():
                    continue
                text = text.strip()
                texts.append(text)
                meta.append({
                    "id": r.get("id", f"record_{i+j}"),
                    "text": text
                })

            if not texts:
                continue

            # Generate embeddings
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                batch_size=16,            # internal batch for GPU
                show_progress_bar=False,
                normalize_embeddings=True # optional: unit length
            )

            for m, emb in zip(meta, embeddings):
                out_f.write(json.dumps({
                    "id": m["id"],
                    "text": m["text"],
                    "embedding": emb.tolist()
                }, ensure_ascii=False) + "\n")

                snippet = m["text"][:60].replace("\n", " ")
                print(f"✅ Embedded ({m['id']}): {snippet}...")

    print(f"💾 Saved embeddings to {EMB_PATH}")
    print("🎉 All embeddings generated successfully!")

if __name__ == "__main__":
    # Example: pass resume_batch=127 to start from batch 127
    main(resume_batch=0)
