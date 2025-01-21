import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

def create_faiss_index(csv_path="data/processed/rag_chunks.csv",
                       model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                       output_path="data/index/faiss_index"):
    # 1) Завантаження даних
    df = pd.read_csv(csv_path)

    # 2) Ініціалізація моделі
    model = SentenceTransformer(model_name)

    # 3) Отримуємо ембединги
    texts = df["text_chunk"].tolist()
    embeddings = model.encode(texts, show_progress_bar=True)

    # Конвертуємо в np.float32 для FAISS
    embeddings = np.array(embeddings, dtype="float32")

    # 4) Створення FAISS-індексу
    dimension = embeddings.shape[1]  # розмір вектора
    index = faiss.IndexFlatL2(dimension)  # або інший тип індексу
    index.add(embeddings)

    print(f"FAISS index size: {index.ntotal}")

    # 5) Збереження індексу та метаданих
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    faiss.write_index(index, f"{output_path}.index")

    # Також збережемо DataFrame (або частину) з метаданими
    df["embedding_id"] = range(len(df))  # співставлення
    df.to_csv(f"{output_path}_metadata.csv", index=False)

    print("Index і метадані збережено.")

if __name__ == "__main__":
    create_faiss_index()
