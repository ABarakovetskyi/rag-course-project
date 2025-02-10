import gradio as gr
import subprocess
import traceback

from src.rag_pipeline.rag_pipeline import RAGPipeline

MODEL_NAME = "mistral:7b-instruct"

# Ініціалізуємо RAG-пайплайн
rag = RAGPipeline(
    metadata_csv="data/index/faiss_index_metadata.csv",
    faiss_index_path="data/index/faiss_index.index",
    embedding_model="intfloat/multilingual-e5-base",  # Додаємо нову модель
    bm25_enable=True,
    re_ranker_enable=True,
    use_cosine=True
)

def build_instruct_prompt(fragments, user_query: str) -> str:
    """
    Формує prompt у форматі '### Instruction: ... ### Response:'
    Додаємо інструкцію про "строге цитування" chunks.
    """
    # Об'єднаємо текст фрагментів
    fragments_text = "\n".join(
        f"- {frag['text_chunk']}" for frag in fragments
    )
    return f"""### Instruction:
Ти відповідаєш ВИКЛЮЧНО на основі наведених фрагментів, БЕЗ додавання вигаданих фактів.
Якщо в тексті нижче відсутня потрібна інформація, напиши "Я не знаю".
**Використовуй прямі цитати** (дослівно) з фрагментів. Не додавай власні припущення чи вигадану інформацію.

Ось фрагменти:
{fragments_text}

Запит: {user_query}

### Response:
"""

def generate_ollama(prompt: str) -> str:
    """
    Викликає ollama з командою "run" і потрібною моделлю (mistral:7b-instruct).
    """
    try:
        cmd = [
            "ollama",
            "run",
            MODEL_NAME,
            # Можна додати "--nopersist", якщо Ollama підтримує
        ]
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if result.returncode != 0:
            return f"Ollama error: {result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error calling ollama: {e}"

def get_response(query: str):
    if not query.strip():
        return ("Будь ласка, введіть запит", "")

    try:
        # 1) Hybrid search з top_k=12 (замість 8), щоб зібрати більше кандидатів
        candidates = rag.hybrid_search(query, top_k=20)
        if not candidates:
            return "Нічого не знайдено для вашого запиту", ""

        # 2) Re-rank
        candidates = rag.re_rank(query, candidates)
        # Після re-rank беремо, скажімо, top-8
        # (бо хотіли збільшити re-rank)
        # Потім ще відфільтруємо
        top_candidates = candidates[:8]

        # 3) Фільтруємо за score > 0.2 (або 0.15, 0.25 — налаштуйте)
        filtered = [c for c in top_candidates if c["score"] > 0.3]
        if len(filtered) < 3:
            # якщо надто агресивна фільтрація — fallback
            filtered = top_candidates

        # 4) Залишимо в prompt top-5 фрагментів
        top_fragments = filtered[:5]

        # 5) Формуємо "строге" інструкції
        prompt = build_instruct_prompt(top_fragments, query)
        print("==== PROMPT ====")
        print(prompt)
        print("================")

        answer = generate_ollama(prompt)

        # Формуємо список джерел
        sources_md = "### Список джерел та оцінки схожості\n"
        for frag in top_fragments:
            sources_md += f"- [{frag['source_file']}](raw_docs/{frag['source_file']}) (Схожість: {frag['score']:.2f})\n"

        return answer, sources_md

    except Exception as e:
        traceback.print_exc()
        return f"Сталася помилка: {e}", ""

# Gradio UI
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(label="Введіть запит"),
    outputs=[
        gr.Textbox(label="Відповідь від Ollama (Mistral)"),
        gr.Markdown(label="Список джерел")
    ],
    title="RAG Pipeline Demo (Ollama, Mistral Instruct)",
    description="Збільшили top_k для re-ranker, посилили prompt (строге цитування)."
)

if __name__ == "__main__":
    iface.launch()
