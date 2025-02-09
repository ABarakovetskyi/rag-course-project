import gradio as gr
import subprocess
import traceback

from src.rag_pipeline.rag_pipeline import RAGPipeline

MODEL_NAME = "mistral:7b-instruct"  # назва моделі в ollama
OLLAMA_TEMPERATURE = "0.2"  # знижує "творчість"

rag = RAGPipeline(
    metadata_csv="data/index/faiss_index_metadata.csv",
    faiss_index_path="data/index/faiss_index.index",
    bm25_enable=True,
    re_ranker_enable=True,
    use_cosine=True
)


def generate_ollama(prompt: str) -> str:
    """
    Виклик ollama CLI. Повертає відповідь.
    """
    try:
        cmd = [
            "ollama",
            "run",
            MODEL_NAME
        ]
        result = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8")

        if result.returncode != 0:
            return f"Ollama error: {result.stderr}"
        return result.stdout.strip()
    except Exception as e:
        return f"Error calling ollama: {e}"


def get_response(query: str):
    if not query.strip():
        return ("Будь ласка, введіть запит", "")

    try:
        # 1) Hybrid search з top_k=8, щоб отримати більше кандидатів
        # (далі re-rank та фільтруємо)
        candidates = rag.hybrid_search(query, top_k=8)
        if not candidates:
            return "Нічого не знайдено для вашого запиту", ""

        # 2) Re-rank
        candidates = rag.re_rank(query, candidates)

        # 3) Опційна фільтрація за схожістю (score), щоб відкинути низькорелевантні chunks
        #   Напр.: беремо лише ті, в яких score > 0.2
        filtered = [c for c in candidates if c["score"] > 0.2]

        # Якщо після фільтра лишилося занадто мало, fallback:
        if len(filtered) < 3:
            # Якщо надто агресивна фільтрація, візьмемо хоч щось
            filtered = candidates

        # 4) Візьмемо top-5 замість 3
        top_fragments = filtered[:5]

        # 5) Формуємо розширену інструкцію:
        prompt = """Ти відповідаєш лише на основі наведених фрагментів.
Не додавай жодної вигаданої інформації. Відповідай коротко і чітко.
Якщо немає інформації, напиши "Я не знаю".
Ось фрагменти:\n\n"""
        for frag in top_fragments:
            prompt += f"- {frag['text_chunk']}\n"

        prompt += f"\nЗапит: {query}\nВідповідь: "

        # Для діагностики
        print("================ PROMPT ===============")
        print(prompt)
        print("========================================")

        # Викликаємо ollama
        answer = generate_ollama(prompt)

        # Формуємо список джерел
        sources_md = "### Список джерел\n"
        for frag in top_fragments:
            sources_md += f"- [{frag['source_file']}](raw_docs/{frag['source_file']})\n"

        return answer, sources_md

    except Exception as e:
        traceback.print_exc()
        return f"Сталася помилка: {e}", ""


iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(label="Введіть запит"),
    outputs=[
        gr.Textbox(label="Відповідь від Ollama LLM"),
        gr.Markdown(label="Список джерел")
    ],
    title="RAG Pipeline Demo (Ollama)",
    description="Demonstration of RAG with local LLM via Ollama. Prompt engineering, top_k=5, filtering low-score fragments."
)

if __name__ == "__main__":
    iface.launch()
