import gradio as gr
from transformers import pipeline as hf_pipeline, GPT2Tokenizer
import traceback

# Імпорт нашого RAGPipeline
# Припустимо, ви зберегли його як src.rag_pipeline.rag_pipeline, де вже додано логіку use_cosine
from src.rag_pipeline.rag_pipeline import RAGPipeline

# Ініціалізація (налаштуйте параметри відповідно до вашої конфігурації)
rag = RAGPipeline(
    metadata_csv="data/index/faiss_index_metadata.csv",
    faiss_index_path="data/index/faiss_index.index",
    embedding_model="sentence-transformers/distiluse-base-multilingual-cased-v2",
    cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    bm25_enable=True,
    re_ranker_enable=True,
    use_cosine=True  # якщо у create_embeddings.py ви також вмикали use_cosine
)

# Ініціалізуємо текстовий генератор із GPT-2 (або іншу модель, якщо бажано)
generator = hf_pipeline("text-generation", model="gpt2")

# Токенізатор GPT-2, щоби контролювати довжину промпту
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
max_model_length = tokenizer.model_max_length  # 1024 для GPT-2

def get_response(query: str):
    """
    1) Використовуємо RAG-пайплайн для пошуку (hybrid_search + re_rank).
    2) Беремо топ-3 фрагменти => формуємо промпт.
    3) Обрізаємо промпт, щоб уникнути 'index out of range' в GPT-2.
    4) Генеруємо відповідь GPT-2.
    5) Формуємо список джерел (Markdown).
    """
    if not query.strip():
        return ("Будь ласка, введіть запит", "")

    try:
        print(f"[DEBUG] Отримано запит: {query}")

        # 1) Hybrid search
        candidates = rag.hybrid_search(query, top_k=10)
        if not candidates:
            return "Нічого не знайдено для вашого запиту", ""

        # 2) Re-rank (cross-encoder)
        candidates = rag.re_rank(query, candidates)
        top_fragments = candidates[:3]
        print(f"[DEBUG] Знайдено {len(candidates)} кандидатів. Беремо top-3 фрагменти.")

        # Формуємо промпт
        prompt = "Використовуючи наступні фрагменти:\n"
        for frag in top_fragments:
            prompt += f"- {frag['text_chunk']}\n"
        prompt += f"\nЗапит: {query}\nВідповідь: "

        # 3) Обрізаємо промпт, щоб не перевищувати 1024 токени (GPT-2 контекст)
        desired_gen_tokens = 80
        max_allowed = max_model_length - desired_gen_tokens

        tokens_prompt = tokenizer.encode(prompt)
        if len(tokens_prompt) > max_allowed:
            tokens_prompt = tokens_prompt[:max_allowed]
        truncated_prompt = tokenizer.decode(tokens_prompt)

        # 4) Генеруємо відповідь
        output = generator(
            truncated_prompt,
            max_new_tokens=desired_gen_tokens,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )
        answer = output[0]['generated_text']

        # 5) Формуємо список джерел
        sources_md = "### Список джерел\n"
        for frag in top_fragments:
            src_file = frag["source_file"]
            # Припускаємо, що файли лежать у raw_docs, і в UI хочемо дати лінк
            sources_md += f"- [{src_file}](raw_docs/{src_file})\n"

        return answer, sources_md

    except Exception as e:
        print("=== EXCEPTION CAUGHT ===")
        traceback.print_exc()
        return f"Сталася помилка: {e}", ""

# Налаштуємо Gradio інтерфейс
iface = gr.Interface(
    fn=get_response,
    inputs=gr.Textbox(label="Введіть запит"),
    outputs=[
        gr.Textbox(label="Відповідь від LLM"),
        gr.Markdown(label="Список джерел")
    ],
    title="RAG Pipeline Demo (Truncate Prompt)",
    description="Demonstration of RAG + GPT-2, with prompt-truncation to avoid exceeding the 1024 token limit of GPT-2."
)

if __name__ == "__main__":
    iface.launch()
