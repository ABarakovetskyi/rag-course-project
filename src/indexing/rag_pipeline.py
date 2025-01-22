import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder



class RAGPipeline:
    def __init__(self,
                 metadata_csv="data/index/faiss_index_metadata.csv",
                 faiss_index_path="data/index/faiss_index.index",
                 embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",  # або інший
                 bm25_enable=True,
                 re_ranker_enable=True):
        # Завантажимо метадані
        self.df = pd.read_csv(metadata_csv)
        self.model = SentenceTransformer(embedding_model)

        # FAISS
        self.index = faiss.read_index(faiss_index_path)

        # BM25 (створимо список tokenized_text для кожного chunk)
        self.bm25_enable = bm25_enable
        if bm25_enable:
            self.corpus = self.df["text_chunk"].tolist()
            tokenized_corpus = [c.split() for c in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

        # Cross-encoder (re-ranker)
        self.re_ranker_enable = re_ranker_enable
        if re_ranker_enable:
            # Завантажуємо крос-енкодер
            self.cross_encoder = CrossEncoder(cross_encoder_model)

    def hybrid_search(self, query, top_k=5):
        """
        Приклад двоетапного пошуку:
        1) BM25, беремо top_n (наприклад 50)
        2) Серед них робимо векторний пошук (або навпаки)
        """
        if not self.bm25_enable:
            # якщо BM25 вимкнено
            return self.vector_search(query, top_k=top_k)

        # 1) BM25 top_n
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        # відсортувати за спаданням
        indices_sorted = np.argsort(scores)[::-1]
        top_n = 50  # беремо 50 найрелевантніших з BM25
        top_n_indices = indices_sorted[:top_n]

        # 2) Тепер серед цих 50 робимо векторний пошук
        #    збираємо ці chunks у список
        sub_texts = [self.corpus[i] for i in top_n_indices]
        sub_ids = top_n_indices

        # ембедимо запит
        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb, dtype="float32")

        # ембедимо sub_texts
        chunk_embs = self.model.encode(sub_texts)
        chunk_embs = np.array(chunk_embs, dtype="float32")

        # знаходимо косинусну подібність (або L2)
        # косинусна подібність = (A dot B) / (||A||*||B||)
        # зробимо це вручну для всіх
        dot = np.sum(query_emb * chunk_embs, axis=1)
        norm_q = np.linalg.norm(query_emb)
        norm_c = np.linalg.norm(chunk_embs, axis=1)
        cos_sims = dot / (norm_q * norm_c)

        # відсортуємо
        cos_sims_sorted_idx = np.argsort(cos_sims)[::-1]
        final_indices = cos_sims_sorted_idx[:top_k]

        results = []
        for idx_ in final_indices:
            real_idx = sub_ids[idx_]
            chunk = self.df.iloc[real_idx]
            results.append({
                "score": float(cos_sims[idx_]),
                "chunk_id": int(chunk["chunk_id"]),
                "source_file": chunk["source_file"],
                "text_chunk": chunk["text_chunk"]
            })
        return results

    def vector_search(self, query, top_k=5):
        """
        Прямий пошук через FAISS
        """
        query_emb = self.model.encode([query], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype="float32")
        distances, ids = self.index.search(query_emb, top_k)
        results = []
        for dist, idx_ in zip(distances[0], ids[0]):
            row = self.df.iloc[idx_]
            results.append({
                "score": float(dist),
                "chunk_id": int(row["chunk_id"]),
                "source_file": row["source_file"],
                "text_chunk": row["text_chunk"]
            })
        return results

    def re_rank(self, query, candidates):
        """
        Перерозташувати (re-rank) candidates, використовуючи cross-encoder
        """
        if not self.re_ranker_enable or not candidates:
            return candidates

        # Формуємо пари (query, chunk_text)
        pairs = [(query, c["text_chunk"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)  # numpy array

        # з'єднуємо scores з candidates
        for i, sc in enumerate(scores):
            candidates[i]["re_rank_score"] = float(sc)

        # сортуємо за sc
        candidates.sort(key=lambda x: x["re_rank_score"], reverse=True)
        return candidates


if __name__ == "__main__":
    pipeline = RAGPipeline(
        bm25_enable=True,
        re_ranker_enable=True  # вмикаємо крос-енкодер
    )

    query = "Який обов’язок інженера-програміста?"
    # Запускаємо двоетапний пошук (BM25 -> вектор)
    candidates = pipeline.hybrid_search(query, top_k=10)

    # re-rank
    final_reranked = pipeline.re_rank(query, candidates)

    for i, res in enumerate(final_reranked[:3]):
        print(f"Rank {i + 1} | cross_enc_score={res.get('re_rank_score')}, vector_score={res['score']}")
        print(f"source_file: {res['source_file']}")
        print(res["text_chunk"][:200], "...")
        print("----")
