import os
import re
import glob
import json
import pypdf
import docx2txt
import pandas as pd

# LangChain TextSplitters
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

# Для підрахунку токенів
# pip install tiktoken
import tiktoken

# Приклад: використання енкодера для GPT-3.5
ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    """Повертає приблизну кількість токенів тексту з урахуванням GPT-3.5."""
    tokens = ENCODER.encode(text)
    return len(tokens)

def read_pdf(file_path):
    """
    Зчитує текст із PDF-сторінок і об'єднує в один рядок.
    """
    text_pages = []
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    return "\n".join(text_pages)

def read_docx(file_path):
    """
    Зчитує текст із DOCX-файлу.
    """
    return docx2txt.process(file_path) or ""

def clean_text(text: str) -> str:
    """
    1. Прибирає зайві порожні рядки.
    2. Видаляє колонтитули, 'технічні' блоки, підписи, сертифікати тощо.
       Налаштуйте регулярки під свої потреби.
    """
    # Приклад: Видалити повторювані переводи рядків (2+ підряд) замінюючи на 1-2
    text = re.sub(r'\n\s*\n+', '\n\n', text)

    # Приклад: Видалити фрагменти з "Затверджую Голова Правління..."
    # (?i) - регістронезалежний пошук
    text = re.sub(r'(?i)затверджую голова правління.*?\n', '', text)

    # Приклад: Видалити рядки з "Сертифікат:" + будь-який текст до кінця рядка
    text = re.sub(r'(?i)сертифікат:.*', '', text)

    # Приклад: Видалити підписи "документ підписаний ..."
    text = re.sub(r'(?i)документ підписаний.*', '', text)

    # Приклад: Видалити часові мітки формату "17:54"
    text = re.sub(r'\b\d{1,2}:\d{2}\b', '', text)

    # Приклад: Видалити зайві пробіли
    text = re.sub(r'[ \t]+', ' ', text)

    # Обрізаємо пробіли/переноси з початку та кінця
    text = text.strip()
    return text

def choose_text_splitter(splitter_type="recursive", chunk_size=1000, chunk_overlap=200):
    """
    Повертає потрібний TextSplitter із LangChain залежно від splitter_type:
      - "recursive" -> RecursiveCharacterTextSplitter
      - "character" -> CharacterTextSplitter
    """
    if splitter_type == "character":
        return CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:  # "recursive" за замовчуванням
        return RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )



def parse_and_chunk_documents(
    raw_docs_path="data/raw_docs",
    output_folder="data/processed",
    splitter_type="recursive",
    chunk_size=1000,
    chunk_overlap=200
):
    """
    1. Збирає всі файли PDF/DOCX у папці raw_docs_path.
    2. Зчитує та очищує текст.
    3. Розбиває на чанки одним із TextSplitter (Recursive/Character/Spacy).
    4. Підраховує токени кожного чанка.
    5. Зберігає результати в CSV та JSON.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Обираємо потрібний TextSplitter
    text_splitter = choose_text_splitter(
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_records = []
    # Збираємо список PDF та DOCX файлів
    files = glob.glob(os.path.join(raw_docs_path, "*.pdf")) + \
            glob.glob(os.path.join(raw_docs_path, "*.docx"))

    for file_path in files:
        print(f"Processing: {file_path}")
        filename = os.path.basename(file_path)

        # 1) Зчитування
        if file_path.endswith(".pdf"):
            text_data = read_pdf(file_path)
        else:
            text_data = read_docx(file_path)

        # 2) Очищення
        text_data = clean_text(text_data)
        if not text_data.strip():
            # Якщо після очищення текст порожній, пропускаємо
            continue

        # 3) Розбиття на чанки
        chunks = text_splitter.split_text(text_data)

        # 4) Формуємо записи
        for idx, chunk in enumerate(chunks):
            chunk_text = chunk.strip()
            token_count = count_tokens(chunk_text)
            record = {
                "source_file": filename,
                "chunk_id": idx,
                "text_chunk": chunk_text,
                "token_count": token_count
            }
            all_records.append(record)

    # Якщо взагалі немає даних, уникаємо помилки при створенні DataFrame
    if not all_records:
        print("No text data found after parsing and cleaning.")
        return

    # 5) Збереження у CSV / JSON
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_folder, "rag_chunks.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')

    json_path = os.path.join(output_folder, "rag_chunks.json")
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Total chunks generated: {len(all_records)}")

if __name__ == "__main__":
    # Приклад виклику з параметрами
    parse_and_chunk_documents(
        raw_docs_path="data/raw_docs",
        output_folder="data/processed",
        splitter_type="recursive",  # або "character", "spacy"
        chunk_size=1000,
        chunk_overlap=200
    )
