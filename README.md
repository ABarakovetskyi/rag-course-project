# RAG-Course-Project

Цей проєкт демонструє **Retrieval-Augmented Generation (RAG)**‑підхід для роботи з українськомовними документами.  
Поточна реалізація використовує:

- **Sentence Transformers** (модель `paraphrase-multilingual-MiniLM-L12-v2`) для ембедингів  
- **FAISS** для векторного пошуку  
- **LangChain** і **transformers** (за потреби)  

---

## **Інструкції з використання**

### **1. Клонування репозиторію та встановлення залежностей**
```bash
git clone https://github.com/username/rag-course-project.git
cd rag-course-project

 ### **2. Створення та активація віртуального середовища (venv)**

python -m venv new_venv
# Windows:
.\new_venv\Scripts\activate
# macOS/Linux:
source new_venv/bin/activate

### **3. Встановлення залежностей**

pip install --upgrade pip
pip install -r requirements.txt

### **4. Попередня обробка документів**

1.) Помістіть усі ваші PDF/DOCX у папку data/raw_docs.
2.) Запустіть скрипт для створення шматків (chunks):

python src/data_preparation/parse_documents.py

Скрипт розіб’є PDF/DOCX на шматки та збереже їх у data/processed/rag_chunks.csv (також у форматі .json).
Опціонально: можна ввімкнути/вимкнути знеособлення ПІБ.

### **5. Створення ембедингів та FAISS-індексу**

1.) Запустіть: python src/indexing/create_embeddings.py
Цей скрипт:

- Зчитає rag_chunks.csv.
- Обчислить ембединги (модель paraphrase-multilingual-MiniLM-L12-v2).
- Збереже FAISS-індекс у data/index/faiss_index.index.
- Збереже метадані у data/index/faiss_index_metadata.csv.

Очікуваний вивід:

FAISS index size: ...
Індекс збережено.

### **6. Пошук у FAISS (query_index.py)**


Файл query_index.py містить функцію query_faiss(query, index_path, top_k=3), яка:

1.)Читає індекс faiss_index.index.
2.) Обчислює ембединг запиту.
3.) Виконує пошук у FAISS і повертає найбільш схожі chunks.

Для тестування: python src/indexing/query_index.py

Приклад виводу:
Rank 1 | score=5.25377
source_file: Інженера 2 категорії групиHR.pdf
...
Rank 2 ...
...

### **7. Наступні кроки**

Для інтеграції з LLM (GPT-4, Llama та ін.) можна створити RAG-пайплайн:

Користувач вводить запит.
Знаходимо найближчі chunks у FAISS.
Формуємо prompt + chunks.
Генеруємо відповідь (через OpenAI API чи локальну модель).
Це можна реалізувати у файлі rag_pipeline.py.
