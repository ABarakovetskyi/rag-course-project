import gradio as gr
from src.rag_pipeline.rag_pipeline import RAGPipeline
from transformers import pipeline as hf_pipeline, GPT2Tokenizer

# Ініціалізуємо RAG-пайплайн (налаштуйте параметри за потреби)
rag = RAGPipeline(
    metadata_csv="data/index/faiss_index_metadata.csv",
    faiss_index_path="data/index/faiss_index.index",
    bm25_enable=True,
    re_ranker_enable=True
)

# Ініціалізуємо генератор тексту з безкоштовної моделі GPT-2
generator = hf_pipeline("text-generation", model="gpt2")

# Завантажуємо токенізатор GPT-2 для обрізання prompt-а
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_model_length = tokenizer.model_max_length  # Наприклад, 1024 для GPT-2

def truncate_prompt(prompt: str) -> str:
    """
    Обрізає prompt до максимально допустимої кількості токенів для моделі.
    """
    tokens = tokenizer.encode(prompt, truncation=True, max_length=max_model_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def get_response(query: str) -> str:
    """
    Обробка запиту:
      1. Отримання релевантних фрагментів за допомогою RAG-пайплайну.
      2. Формування prompt із включенням отриманих фрагментів та початкового запиту.
      3. Виклик GPT-2 для генерації відповіді.
      4. Повернення згенерованої відповіді разом із джерелами.
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

    # Обрізаємо prompt, щоб він не перевищував максимально допустиму довжину
    prompt = truncate_prompt(prompt)

    # Викликаємо LLM (GPT-2) для генерації відповіді
    output = generator(prompt, max_length=250, do_sample=True, top_p=0.9, num_return_sequences=1)
    answer = output[0]['generated_text']

    # Формуємо список джерел (source_file)
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
