import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def query_faiss(query, index_path="data/index/faiss_index.index",
                meta_path="data/index/faiss_index_metadata.csv",
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                top_k=3):
    # 1) Завантаження індексу
    index = faiss.read_index(index_path)
    metadata = pd.read_csv(meta_path)
    model = SentenceTransformer(model_name)

    # 2) Ембединг запиту
    query_emb = model.encode([query], show_progress_bar=False)
    query_emb = np.array(query_emb, dtype="float32")

    # 3) Пошук
    distances, ids = index.search(query_emb, top_k)
    results = []
    for dist, idx_ in zip(distances[0], ids[0]):
        row = metadata.iloc[idx_]
        results.append({
            "score": float(dist),
            "chunk_id": int(row["chunk_id"]),
            "source_file": row["source_file"],
            "text_chunk": row["text_chunk"]
        })
    return results


if __name__ == "__main__":
    demo_query = "Який обов'язок інженера-програміста??"
    top_results = query_faiss(demo_query, top_k=3)
    for i, r in enumerate(top_results):
        print(f"Rank {i+1} | score={r['score']}")
        print(f"source_file: {r['source_file']}")
        print(r["text_chunk"][:200], "...")
        print("----")
