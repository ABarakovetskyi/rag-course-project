import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def create_faiss_index(
    input_path="data/processed/rag_chunks.json",
    input_format="json",  # 'json' або 'csv'
    model_name="intfloat/multilingual-e5-base",
    output_path="data/index/faiss_index",
    use_cosine=True
):
    """
    1) Зчитує DataFrame з JSON або CSV (пріоритетно JSON).
    2) Якщо є колонка 'chunk', перейменовуємо на 'text_chunk'.
    3) Переконуємось, що є колонки 'text_chunk' і 'source_file'. За потреби створюємо 'chunk_id'.
    4) Обчислюємо ембеддинги за допомогою SentenceTransformer(model_name).
    5) Якщо use_cosine=True, нормалізуємо вектори (faiss.normalize_L2) і використовуємо IndexFlatIP (inner product).
       Якщо use_cosine=False, використовуємо IndexFlatL2 (Евклідова відстань).
    6) Зберігаємо індекс у файл (faiss_index.index) та метадані у CSV (faiss_index_metadata.csv).

    Приклад виклику:
      python create_embeddings.py
    """

    # Видаляємо старі індекси та метадані
    if os.path.exists(f"{output_path}.index"):
        os.remove(f"{output_path}.index")
        print(f"Видалено старий індекс: {output_path}.index")
    if os.path.exists(f"{output_path}_metadata.csv"):
        os.remove(f"{output_path}_metadata.csv")
        print(f"Видалено старі метадані: {output_path}_metadata.csv")

    # 1) Пріоритетно читаємо JSON, інакше CSV
    if os.path.exists(input_path) and input_format == "json":
        df = pd.read_json(input_path, orient='records')
    elif input_format == "csv":
        df = pd.read_csv(input_path)
    else:
        raise FileNotFoundError(f"Файл {input_path} не знайдено або формат не підтримується.")

    # Якщо "chunk" - це текст, перейменовуємо на "text_chunk"
    if "chunk" in df.columns and "text_chunk" not in df.columns:
        df.rename(columns={"chunk": "text_chunk"}, inplace=True)

    # Перевіряємо наявність потрібних полів
    if "text_chunk" not in df.columns:
        raise ValueError("Не знайдено 'text_chunk' у вхідних даних. Перевірте JSON/CSV.")
    if "source_file" not in df.columns:
        raise ValueError("Не знайдено 'source_file' у вхідних даних.")
    if "chunk_id" not in df.columns:
        df["chunk_id"] = range(len(df))

    # 2) Завантажуємо модель
    model = SentenceTransformer(model_name)
    print(f"Використовується модель ембеддингів: {model_name}")

    # 3) Обчислюємо ембеддинги
    texts = [f"passage: {text}" for text in df["text_chunk"].tolist()]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # 4) Якщо треба косинусна схожість - нормалізуємо вектори і використовуємо Inner Product
    if use_cosine:
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        metric_mode = "cosine (IP + normalized)"
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        metric_mode = "L2"

    # Додаємо ембеддинги до індексу
    index.add(embeddings)
    print(f"FAISS index size: {index.ntotal} (dimension={embeddings.shape[1]}, metric={metric_mode})")

    # 5) Зберігаємо індекс
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, f"{output_path}.index")

    # 6) Зберігаємо метадані
    meta_df = df[["chunk_id", "source_file", "text_chunk"]].copy()
    meta_csv = f"{output_path}_metadata.csv"
    meta_df.to_csv(meta_csv, index=False, encoding='utf-8-sig')

    print(f"Індекс і метадані збережено у:\n  {output_path}.index\n  {meta_csv}")


if __name__ == "__main__":
    create_faiss_index(
        input_path="data/processed/rag_chunks.json",
        input_format="json",
        model_name="intfloat/multilingual-e5-base",
        output_path="data/index/faiss_index",
        use_cosine=True
    )
