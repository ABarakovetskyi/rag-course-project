from dotenv import load_dotenv
import os

import gradio as gr
import openai
import traceback

from src.rag_pipeline.rag_pipeline import RAGPipeline
# Завантаження змінних середовища з .env файлу
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Ім'я моделі для OpenAI API
MODEL_NAME = "gpt-3.5-turbo"

# Ініціалізуємо RAG-пайплайн
rag = RAGPipeline(
    metadata_csv="data/index/faiss_index_metadata.csv",
    faiss_index_path="data/index/faiss_index.index",
    embedding_model="intfloat/multilingual-e5-base",  # Додаємо нову модель ембеддингу
    bm25_enable=True,  # Увімкнено BM25 пошук для покращення результатів
    re_ranker_enable=True,  # Увімкнено повторне ранжування результатів
    use_cosine=True  # Використовується косинусна схожість для метрики
)


def build_instruct_prompt(fragments, user_query: str) -> str:
    """
    Формує prompt у форматі '### Instruction: ... ### Response:'
    Додаємо інструкцію про "строге цитування" chunks.
    """
    fragments_text = "\n".join(
        f"- {frag['text_chunk']}" for frag in fragments
    )
    return f"""Ти відповідаєш ВИКЛЮЧНО на основі наведених фрагментів, БЕЗ додавання вигаданих фактів.
Якщо в тексті нижче відсутня потрібна інформація, напиши \"Я не знаю\".
**Використовуй прямі цитати** (дослівно) з фрагментів. Не додавай власні припущення чи вигадану інформацію.

Ось фрагменти:
{fragments_text}

Запит: {user_query}
"""


def generate_openai(prompt: str) -> str:
    """
    Викликає OpenAI API для моделі GPT-3.5 Turbo.
    """
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Ти корисний помічник."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def get_response(query: str):
    """
    Обробляє запит користувача, виконує пошук, повторне ранжування та генерує відповідь.
    """
    if not query.strip():
        return ("Будь ласка, введіть запит", "")

    try:
        # Пошук з використанням гібридного методу з top_k=20
        candidates = rag.hybrid_search(query, top_k=20)
        if not candidates:
            return "Нічого не знайдено для вашого запиту", ""

        # Повторне ранжування кандидатів
        candidates = rag.re_rank(query, candidates)
        top_candidates = candidates[:8]

        # Фільтрація результатів за порогом схожості
        filtered = [c for c in top_candidates if c["score"] > 0.3]
        if len(filtered) < 3:
            filtered = top_candidates

        # Вибір топ-5 фрагментів для формування prompt
        top_fragments = filtered[:5]

        # Формування інструкційного prompt
        prompt = build_instruct_prompt(top_fragments, query)
        print("==== PROMPT ====")
        print(prompt)
        print("================")

        # Виклик моделі GPT-3.5 Turbo
        answer = generate_openai(prompt)

        # Формування списку джерел
        sources_md = "### Список джерел та оцінки схожості\n"
        for frag in top_fragments:
            sources_md += f"- [{frag['source_file']}](raw_docs/{frag['source_file']}) (Схожість: {frag['score']:.2f})\n"

        return answer, sources_md

    except Exception as e:
        traceback.print_exc()
        return f"Сталася помилка: {e}", ""


# Налаштування Gradio інтерфейсу
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(label="Введіть запит"),
    outputs=[
        gr.Textbox(label="Відповідь від GPT-3.5 Turbo"),
        gr.Markdown(label="Список джерел")
    ],
    title="RAG Pipeline Demo (GPT-3.5 Turbo)",
    description="Збільшили top_k для re-ranker, посилили prompt (строге цитування)."
)

# Запуск додатку
if __name__ == "__main__":
    iface.launch()
