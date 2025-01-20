import os
import re
import glob
import pandas as pd
import json

# Для PDF
import pypdf
# Для DOCX
import docx2txt


def anonymize_text(text):
    """
    Примітивне знеособлення ПІБ українською мовою.
    Шукаємо:
      1) [А-ЯҐЄІЇ][а-яґєії'’]+ \s+ [А-ЯҐЄІЇ]\.[А-ЯҐЄІЇ]\.
         (наприклад, "Бараковецький А.В.")
      2) [А-ЯҐЄІЇ][а-яґєії'’]+ \s+ [А-ЯҐЄІЇ]{2,}
         (наприклад, "Андрій БАРАКОВЕЦЬКИЙ")

    Замінюємо усе знайдене на "ПІБ_ЗНЕОСОБЛЕНО".
    """

    pattern = re.compile(
        r"\b[А-ЯҐЄІЇ][а-яґєії'’]+\s+[А-ЯҐЄІЇ]\.[А-ЯҐЄІЇ]\."
        r"|\b[А-ЯҐЄІЇ][а-яґєії'’]+\s+[А-ЯҐЄІЇ]{2,}\b",
        flags=re.U
    )
    # Замінюємо знайдене на "ПІБ_ЗНЕОСОБЛЕНО"
    anonymized = pattern.sub("ПІБ_ЗНЕОСОБЛЕНО", text)
    return anonymized


def read_pdf(file_path):
    """
    Зчитує PDF і повертає текст одним рядком.
    """
    text_pages = []
    with open(file_path, 'rb') as f:
        reader = pypdf.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text_pages.append(page_text)
    full_text = "\n".join(text_pages)
    return full_text


def read_docx(file_path):
    """
    Зчитує DOCX і повертає текст одним рядком.
    """
    text = docx2txt.process(file_path)
    return text if text else ""


def split_text_by_paragraphs(text):
    """
    Розбиває текст за абзацами (два або більше переносів).
    Повертає список абзаців (рядків) без порожніх.
    """
    # Наприклад, два або більше \n
    paragraphs = re.split(r'\n\s*\n+', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    return paragraphs


def split_paragraph_into_subchunks(paragraph, max_words=300):
    """
    Якщо абзац > max_words,
    1) розбити його на речення за (.?!;),
    2) об'єднати речення у блоки, де <= max_words слів.
    """
    sentences = re.split(r'(?<=[\.?!;])\s+', paragraph)

    chunks = []
    current_chunk = []
    current_count = 0

    for sentence in sentences:
        words = re.split(r'\s+', sentence.strip())
        if not words or all(w == '' for w in words):
            continue

        if current_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = words
            current_count = len(words)
        else:
            current_chunk.extend(words)
            current_count += len(words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_text(text, max_words=300):
    """
    1) Розбиває текст на абзаци,
    2) Якщо абзац > max_words, розбиває на підшматки (речення) до max_words слів,
    3) Повертає список chunks.
    """
    paragraphs = split_text_by_paragraphs(text)
    all_chunks = []

    for paragraph in paragraphs:
        word_count = len(re.split(r'\s+', paragraph))
        if word_count <= max_words:
            all_chunks.append(paragraph)
        else:
            subchunks = split_paragraph_into_subchunks(paragraph, max_words=max_words)
            all_chunks.extend(subchunks)
    return all_chunks


def parse_and_chunk_documents(raw_docs_path="data/raw_docs",
                              output_folder="data/processed",
                              max_words=300,
                              anonymize=False):
    """
    Основна функція:
      1) Знаходить PDF/DOCX файли у raw_docs_path
      2) Читає та (за потреби) знеособлює text
      3) Розбиває його на chunks
      4) Зберігає у CSV та JSON (для подальшого RAG)

    :param anonymize: Якщо True, викликаємо anonymize_text(text_data).
    """
    os.makedirs(output_folder, exist_ok=True)

    pdf_files = glob.glob(os.path.join(raw_docs_path, "*.pdf"))
    docx_files = glob.glob(os.path.join(raw_docs_path, "*.docx"))

    all_records = []

    for pdf_file in pdf_files:
        print(f"Reading PDF: {pdf_file}")
        text_data = read_pdf(pdf_file)

        # Знеособлення, якщо треба
        if anonymize:
            text_data = anonymize_text(text_data)

        # Розбиваємо на chunks
        chunks = process_text(text_data, max_words=max_words)

        for idx, chunk in enumerate(chunks):
            all_records.append({
                "source_file": os.path.basename(pdf_file),
                "chunk_id": idx,
                "text_chunk": chunk
            })

    for docx_file in docx_files:
        print(f"Reading DOCX: {docx_file}")
        text_data = read_docx(docx_file)

        if anonymize:
            text_data = anonymize_text(text_data)

        chunks = process_text(text_data, max_words=max_words)

        for idx, chunk in enumerate(chunks):
            all_records.append({
                "source_file": os.path.basename(docx_file),
                "chunk_id": idx,
                "text_chunk": chunk
            })

    # Зберігаємо в CSV
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(output_folder, "rag_chunks.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"Chunks saved to CSV: {csv_path}")

    # Додатково збережемо в JSON
    json_path = os.path.join(output_folder, "rag_chunks.json")
    df.to_json(json_path, orient='records', force_ascii=False, indent=2)
    print(f"Chunks also saved to JSON: {json_path}")


if __name__ == "__main__":
    # Приклад виклику:
    parse_and_chunk_documents(
        raw_docs_path="data/raw_docs",
        output_folder="data/processed",
        max_words=300,
        anonymize=True  # якщо треба вимкнути, поставити False
    )
