from src.rag_pipeline.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    pipeline = RAGPipeline(
        bm25_enable=True,  # Вмикаємо BM25
        re_ranker_enable=True  # Вмикаємо cross-encoder
    )

    query = "Який обов’язок інженера-програміста?"

    print("\n🔍 Виконуємо двоетапний пошук (BM25 -> векторний)")
    candidates = pipeline.hybrid_search(query, top_k=10)

    print("\n📌 Re-ranking через cross-encoder")
    final_reranked = pipeline.re_rank(query, candidates)

    print("\n📜 Топ-3 результати:")
    for i, res in enumerate(final_reranked[:3]):
        print(f"🔹 Rank {i + 1} | cross_enc_score={res.get('re_rank_score')}, vector_score={res['score']}")
        print(f"📂 source_file: {res['source_file']}")
        print(f"📝 {res['text_chunk'][:200]}...")
        print("----")
