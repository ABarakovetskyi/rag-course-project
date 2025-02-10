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
                 embedding_model="intfloat/multilingual-e5-base",
                 cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 bm25_enable=True,
                 re_ranker_enable=True,
                 use_cosine=True):
        """
        - metadata_csv, faiss_index_path: шляхи до CSV і FAISS-індексу.
        - embedding_model: модель SentenceTransformer.
        - cross_encoder_model: модель для re-rank (пошуковий cross-енкодер).
        - bm25_enable, re_ranker_enable: вмикають BM25 і re-rank.
        - use_cosine: якщо True, припускаємо, що FAISS-індекс створений із cosine similarity
          (IndexFlatIP + normalize_L2). Якщо False, то IndexFlatL2.
        """

        # 1) Завантажимо метадані
        self.df = pd.read_csv(metadata_csv)
        # 2) Завантажимо SentenceTransformer
        self.model = SentenceTransformer(embedding_model)
        # 3) FAISS index
        self.index = faiss.read_index(faiss_index_path)

        self.use_cosine = use_cosine

        # 4) BM25
        self.bm25_enable = bm25_enable
        if bm25_enable:
            self.corpus = self.df["text_chunk"].tolist()
            tokenized_corpus = [c.split() for c in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)

        # 5) Cross-encoder (re-ranker)
        self.re_ranker_enable = re_ranker_enable
        if re_ranker_enable:
            self.cross_encoder = CrossEncoder(cross_encoder_model)

    def hybrid_search(self, query, top_k=5):
        """
        Двоетапний пошук:
          1) BM25 => беремо top_n (50)
          2) Векторний пошук серед цих top_n
        """
        if not self.bm25_enable:
            return self.vector_search(query, top_k=top_k)

        # 1) BM25 -> топ-50
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        indices_sorted = np.argsort(scores)[::-1]
        top_n = 50
        top_n_indices = indices_sorted[:top_n]

        # 2) Векторний пошук серед цих 50
        sub_texts = [self.corpus[i] for i in top_n_indices]
        sub_ids = top_n_indices

        # embed query
        formatted_query = f"query: {query}"
        query_emb = self.model.encode([formatted_query])
        query_emb = np.array(query_emb, dtype="float32")
        query_emb /= np.linalg.norm(query_emb)

        formatted_sub_texts = [f"passage: {text}" for text in sub_texts]
        chunk_embs = self.model.encode(formatted_sub_texts)
        chunk_embs = np.array(chunk_embs, dtype="float32")

        dot = np.sum(query_emb * chunk_embs, axis=1)
        norm_c = np.linalg.norm(chunk_embs, axis=1)
        cos_sims = dot / norm_c

        cos_sims_sorted_idx = np.argsort(cos_sims)[::-1]
        final_indices = cos_sims_sorted_idx[:top_k]

        results = []
        for idx_ in final_indices:
            real_idx = sub_ids[idx_]
            row = self.df.iloc[real_idx]
            results.append({
                "score": float(cos_sims[idx_]),
                "chunk_id": int(row["chunk_id"]),
                "source_file": row["source_file"],
                "text_chunk": row["text_chunk"]
            })
        for idx_, res in enumerate(results):
            print(f"Фрагмент {idx_ + 1}: Score = {res['score']:.4f}, Source = {res['source_file']}")
        return results

    def vector_search(self, query, top_k=5):
        """
        Прямий пошук у FAISS-індексі
        """
        formatted_query = f"query: {query}"
        query_emb = self.model.encode([formatted_query], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype="float32")
        query_emb /= np.linalg.norm(query_emb)

        distances, ids = self.index.search(query_emb, top_k)

        results = []
        for dist, idx_ in zip(distances[0], ids[0]):
            row = self.df.iloc[idx_]
            score = -dist if self.use_cosine else -dist

            results.append({
                "score": float(score),
                "chunk_id": int(row["chunk_id"]),
                "source_file": row["source_file"],
                "text_chunk": row["text_chunk"]
            })

        results = [r for r in results if r["score"] > 0.2]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def re_rank(self, query, candidates):
        """
        Перерозташувати (re-rank) candidates, використовуючи cross-encoder
        """
        if not self.re_ranker_enable or not candidates:
            return candidates

        pairs = [(query, c["text_chunk"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for i, sc in enumerate(scores):
            candidates[i]["re_rank_score"] = float(sc)

        candidates.sort(key=lambda x: x["re_rank_score"], reverse=True)
        return candidates


if __name__ == "__main__":
    pipeline = RAGPipeline(
        metadata_csv="data/index/faiss_index_metadata.csv",
        faiss_index_path="data/index/faiss_index.index",
        embedding_model="intfloat/multilingual-e5-base",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        bm25_enable=True,
        re_ranker_enable=True,
        use_cosine=True
    )

    query = "Який обов’язок начальника сектору СЕД та УП?"
    candidates = pipeline.hybrid_search(query, top_k=8)
    final_reranked = pipeline.re_rank(query, candidates)

    for i, res in enumerate(final_reranked[:3]):
        print(f"Rank {i + 1} | cross_enc_score={res.get('re_rank_score')}, vector_score={res['score']}")
        print(f"source_file: {res['source_file']}")
        print(res["text_chunk"][:200], "...")
        print("----")
