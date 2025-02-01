import gradio as gr
from src.rag_pipeline.rag_pipeline import RAGPipeline
from transformers import pipeline as hf_pipeline

# Ініціалізуємо RAG-пайплайн (за потреби налаштуйте параметри)
rag = RAGPipeline(
    metadata_csv="data/index/faiss_index_metadata.csv",
    faiss_index_path="data/index/faiss_index.index",
    bm25_enable=True,
    re_ranker_enable=True
)

# Ініціалізуємо генератор тексту з безкоштовної моделі GPT-2
generator = hf_pipeline("text-generation", model="gpt2")


def get_response(query: str) -> str:
    """
    Функція обробки запиту:
      1. За допомогою RAG-пайплайну отримуємо релевантні фрагменти.
      2. Формуємо prompt із включенням отриманих фрагментів та початкового запиту.
      3. Викликаємо GPT-2 для генерації відповіді.
      4. Повертаємо згенеровану відповідь разом із джерелами (source_file).
    """
    # Отримуємо кандидати (наприклад, 10 фрагментів) та робимо re-ranking
    candidates = rag.hybrid_search(query, top_k=10)
    candidates = rag.re_rank(query, candidates)

    # Вибираємо top-3 фрагменти для включення у prompt
    top_fragments = candidates[:3]

    # Формуємо prompt: спочатку перелік фрагментів, потім запит
    prompt = "Використовуючи наступні фрагменти:\n"
    for frag in top_fragments:
        prompt += f"- {frag['text_chunk']}\n"
    prompt += f"\nЗапит: {query}\nВідповідь: "

    # Викликаємо LLM (GPT-2) для генерації відповіді
    output = generator(prompt, max_length=250, do_sample=True, top_p=0.9, num_return_sequences=1)
    answer = output[0]['generated_text']

    # Формуємо список джерел (за бажанням)
    sources = "\n".join([f"Файл: {frag['source_file']}" for frag in top_fragments])

    full_response = answer + "\n\nДжерела:\n" + sources
    return full_response


# Налаштовуємо Gradio інтерфейс
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(label="Введіть запит", placeholder="Наприклад: Який обов’язок інженера-програміста?"),
    outputs=gr.Textbox(label="Відповідь від LLM"),
    title="RAG Pipeline Demo",
    description="Мінімальний веб‑додаток для отримання відповіді від LLM з використанням RAG‑логіки"
)

if __name__ == "__main__":
    iface.launch()
