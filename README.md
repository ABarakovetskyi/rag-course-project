RAG-Course-Project
Цей проєкт демонструє Retrieval-Augmented Generation (RAG)-підхід для роботи з документами (DOCX/PDF) переважно українською мовою. Мета – швидко знаходити відповідні фрагменти з локальної бази (через BM25 + векторний пошук) і доповнювати відповідь генеративною моделлю (GPT‑2 або іншою LLM).

Основні компоненти
Парсинг документів:

parse_documents.py: зчитує DOCX/PDF, розбиває на частини (chunks) приблизно по 2000 символів, уникає розриву слів на межах.
Результат: rag_chunks.csv / rag_chunks.json у папці data/processed/.
Створення ембеддингів і FAISS-індексу:

create_embeddings.py: бере rag_chunks.csv/rag_chunks.json, обчислює ембеддинги (SentenceTransformer, за замовчуванням distiluse-base-multilingual-cased-v2), створює FAISS-індекс (IndexFlatIP або IndexFlatL2).
Параметр use_cosine=True вмикає нормалізацію векторів і IndexFlatIP (фактично косинусна схожість).
Вихід: faiss_index.index та faiss_index_metadata.csv.
RAG-пайплайн:

rag_pipeline.py:
Завантажує метадані й FAISS-індекс.
Містить hybrid_search (BM25 → top_n, потім векторний пошук), vector_search (прямий пошук у FAISS) і re-rank (CrossEncoder).
Повертає релевантні chunks, які можна використати для генерації відповіді.
Демо‑додаток:

app.py: Gradio-вебінтерфейс, що приймає запит, викликає rag_pipeline (hybrid_search + re-rank), бере top-3 chunks → формує prompt для GPT‑2, обрізає prompt, щоб не перевищувати 1024 токенів, і генерує фінальну відповідь.
Виводить «Відповідь від LLM» + «Список джерел».
Інструкції з використання
1. Клонування репозиторію
bash
Копіювати
Редагувати
git clone https://github.com/username/rag-course-project.git
cd rag-course-project
2. Створення та активація venv
bash
Копіювати
Редагувати
python -m venv new_venv
# Windows:
.\new_venv\Scripts\activate
# macOS/Linux:
source new_venv/bin/activate
3. Встановлення залежностей
bash
Копіювати
Редагувати
pip install --upgrade pip
pip install -r requirements.txt
4. Підготовка документів
Покладіть PDF/DOCX у папку data/raw_docs.
Запустіть parse_documents.py, щоб зчитати й нарізати chunks:
bash
Копіювати
Редагувати
python src/data_preparation/parse_documents.py
Результати:

rag_chunks.csv / rag_chunks.json у data/processed/ (із полями chunk, source_file).
5. Створення ембеддингів та FAISS-індексу
Запустіть:

bash
Копіювати
Редагувати
python src/indexing/create_embeddings.py
Зчитає (за замовчуванням) rag_chunks.json (чи .csv)
Використає модель distiluse-base-multilingual-cased-v2 (можна змінити в коді).
Створить faiss_index.index + faiss_index_metadata.csv у data/index.
6. Тестовий запуск RAG-пайплайну (необов’язково)
bash
Копіювати
Редагувати
python src/rag_pipeline/rag_pipeline.py
Може завантажити індекс, виконати hybrid_search(query), роздрукувати результати.
Перевірте, чи знаходить потрібні chunks.
7. Запуск веб‑додатку Gradio
bash
Копіювати
Редагувати
python app.py
Відкрийте http://127.0.0.1:7860.
Введіть запит (наприклад, «Які обов’язки начальника сектору СЕД та УП?»).
Система знайде chunks → GPT‑2 згенерує відповідь → відобразить у інтерфейсі.
8. Налаштування моделей
В create_embeddings.py можна змінити модель (model_name) і use_cosine.
У rag_pipeline.py відповідно в конструкторі передати use_cosine=True/False і embedding_model=....
У app.py можна змінити генеративну модель із gpt2 на щось більше (наприклад, Bloom), або ж підключити API OpenAI.
Додаткові відомості
Якщо модель GPT‑2 «галюцинує» або дає не надто якісні відповіді – розгляньте більшу LLM (GPT‑3.5 через API, Bloom, T5).
Якщо у вас великі дані (більше 10–100 тис. chunks), можна переходити до індексів Faiss типу IVF чи HNSW.
rank-bm25 використовується для текстового пошуку (BM25) і CrossEncoder – для re-rank.
