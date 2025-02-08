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
                 embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
                 cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 bm25_enable=True,
                 re_ranker_enable=True,
                 use_cosine=True):
        """
        - metadata_csv, faiss_index_path: шляхи до CSV і FAISS-індексу.
        - embedding_model: модель SentenceTransformer, що має збігатися з тією,
          якою ви створювали FAISS-індекс.
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
        # Якщо use_cosine=True, тоді векторний пошук у FAISS видає
        # "distance = 1 - similarity" чи "distance = -dot_product"?
        # Для IndexFlatIP: FAISS повертає результати від "найменшого" значення
        # (яке насправді = -найбільший dot product), такий фреймворк FAISS.

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
            # Якщо вимкнено BM25
            return self.vector_search(query, top_k=top_k)

        # 1) BM25 -> топ-50
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        # сортуємо за спаданням
        indices_sorted = np.argsort(scores)[::-1]
        top_n = 50
        top_n_indices = indices_sorted[:top_n]

        # 2) Векторний пошук серед цих 50
        sub_texts = [self.corpus[i] for i in top_n_indices]
        sub_ids = top_n_indices

        # embed query
        query_emb = self.model.encode([query])
        query_emb = np.array(query_emb, dtype="float32")
        chunk_embs = self.model.encode(sub_texts)
        chunk_embs = np.array(chunk_embs, dtype="float32")

        # косинусна подібність
        dot = np.sum(query_emb * chunk_embs, axis=1)
        norm_q = np.linalg.norm(query_emb)
        norm_c = np.linalg.norm(chunk_embs, axis=1)
        cos_sims = dot / (norm_q * norm_c)

        # сортуємо за cos_sims спаданням
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
        return results

    def vector_search(self, query, top_k=5):
        """
        Прямий пошук у FAISS-індексі
        Якщо use_cosine=True => index = IP => distance = -similarity
        Якщо use_cosine=False => index = L2 => distance = евклідова відстань
        У кожному разі FAISS повертає "найкращі" (мінімальні dist) першими.
        """
        # embed запит
        query_emb = self.model.encode([query], show_progress_bar=False)
        query_emb = np.array(query_emb, dtype="float32")

        # FAISS search
        distances, ids = self.index.search(query_emb, top_k)
        # distances.shape = (1, top_k), ids.shape = (1, top_k)

        results = []
        for dist, idx_ in zip(distances[0], ids[0]):
            row = self.df.iloc[idx_]
            # Якщо use_cosine=True (IndexFlatIP), dist = "distance" = negative dot product
            # Ми можемо перетворити на "score" = -dist, щоб "більший" score = краща схожість
            if self.use_cosine:
                score = -dist  # оскільки IP = dot_product * (-1) => the smaller dist => the bigger dot => => score = -dist
            else:
                # L2 => менша dist => краща. Тож score = -dist, щоб більше = краще
                score = -dist

            results.append({
                "score": float(score),
                "chunk_id": int(row["chunk_id"]),
                "source_file": row["source_file"],
                "text_chunk": row["text_chunk"]
            })

        # Вони вже прийшли в порядку від найменшого dist,
        # тож від найбільшого score. Але надійніше ще раз відсортувати.
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def re_rank(self, query, candidates):
        """
        Перерозташувати (re-rank) candidates, використовуючи cross-encoder
        """
        if not self.re_ranker_enable or not candidates:
            return candidates

        pairs = [(query, c["text_chunk"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)  # numpy array

        for i, sc in enumerate(scores):
            candidates[i]["re_rank_score"] = float(sc)

        candidates.sort(key=lambda x: x["re_rank_score"], reverse=True)
        return candidates


if __name__ == "__main__":
    pipeline = RAGPipeline(
        metadata_csv="data/index/faiss_index_metadata.csv",
        faiss_index_path="data/index/faiss_index.index",
        embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
        bm25_enable=True,
        re_ranker_enable=True,
        use_cosine=True  # з урахуванням, що в create_embeddings.py теж було use_cosine=True
    )

    query = "Який обов’язок начальника сектору СЕД та УП?"
    candidates = pipeline.hybrid_search(query, top_k=10)
    final_reranked = pipeline.re_rank(query, candidates)

    for i, res in enumerate(final_reranked[:3]):
        print(f"Rank {i + 1} | cross_enc_score={res.get('re_rank_score')}, vector_score={res['score']}")
        print(f"source_file: {res['source_file']}")
        print(res["text_chunk"][:200], "...")
        print("----")
