import os
import csv
import json

try:
    import docx
except ImportError:
    print("Встановіть бібліотеку python-docx: pip install python-docx")

try:
    import PyPDF2
except ImportError:
    print("Встановіть бібліотеку PyPDF2: pip install PyPDF2")

SECTION_HEADERS = [
    "ЗАГАЛЬНІ ПОЛОЖЕННЯ",
    "ЗАВДАННЯ ТА ОБОВ’ЯЗКИ",
    "ПРАВА",
    "ВІДПОВІДАЛЬНІСТЬ",
    "ВЗАЄМОВІДНОСИНИ",
    "ПОВИНЕН ЗНАТИ",
    "КВАЛІФІКАЦІЙНІ ВИМОГИ"
]


def read_docx_with_tables(file_path: str) -> str:
    """
    Зчитує текст і таблиці з .docx файлу у порядку їх розташування.
    """
    doc = docx.Document(file_path)
    full_text = []

    for element in doc.element.body:
        if element.tag.endswith('p'):  # Якщо абзац
            paragraph = docx.text.paragraph.Paragraph(element, doc)
            full_text.append(paragraph.text.strip())
        elif element.tag.endswith('tbl'):  # Якщо таблиця
            table = docx.table.Table(element, doc)
            table_text = []
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                table_text.append(" | ".join(row_text))
            full_text.append("\n".join(table_text))  # Додаємо таблицю як текст

    return "\n".join(full_text)



def read_pdf(file_path: str) -> str:
    """
    Зчитує текст із PDF-файлу за допомогою PyPDF2.
    Якщо PDF – це скан, може знадобитися PDF + OCR (pdfplumber + pytesseract).
    """
    text = ""
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_into_sections(text: str) -> list:
    """
    Розділяє текст на секції на основі заголовків із SECTION_HEADERS.
    """
    import re

    # Об'єднуємо заголовки в єдиний регулярний вираз (ігноруємо регістр)
    header_pattern = re.compile(rf"({'|'.join(SECTION_HEADERS)})", re.IGNORECASE)

    sections = []
    current_section = ""
    lines = text.split("\n")

    for line in lines:
        if header_pattern.match(line.strip()):
            if current_section:
                sections.append(current_section.strip())
            current_section = line.strip() + "\n"
        else:
            current_section += line + "\n"

    if current_section:
        sections.append(current_section.strip())

    return sections


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Розбиває текст на фрагменти, включаючи таблиці у відповідні розділи.
    """
    chunks = []
    sections = split_into_sections(text)

    for section in sections:
        words = section.split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start = max(end - overlap, end)

    return chunks



def process_file(file_path: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """
    Обробляє .docx або .pdf файли, додаючи підтримку таблиць для .docx.
    """
    base, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".docx":
        text_data = read_docx_with_tables(file_path)  # Використовуємо нову функцію для .docx
    elif ext == ".pdf":
        text_data = read_pdf(file_path)
    else:
        return []

    return chunk_text(text_data, chunk_size=chunk_size, overlap=overlap)



def save_chunks_to_csv(chunks_info: list, csv_path: str):
    """
    Зберігає список словників (із ключами 'chunk' та 'source_file') у CSV (UTF-8 з BOM),
    щоб відкривати без проблем у Excel.
    """
    with open(csv_path, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chunk', 'source_file'])
        writer.writeheader()
        for row in chunks_info:
            writer.writerow(row)


def save_chunks_to_json(chunks_info: list, json_path: str):
    """
    Зберігає список словників (із ключами 'chunk' та 'source_file') у JSON (UTF-8).
    """
    with open(json_path, mode='w', encoding='utf-8') as f:
        json.dump(chunks_info, f, ensure_ascii=False, indent=2)


def convert_csv_cp1251_to_utf8(src_path: str, dst_path: str):
    """
    Конвертація CSV з Windows-1251 (cp1251) у UTF-8.
    Якщо файл справді збережений у cp1251, тоді буде ОК.
    """
    with open(src_path, 'r', encoding='cp1251', errors='replace') as src_file:
        content = src_file.read()

    with open(dst_path, 'w', encoding='utf-8', newline='') as dst_file:
        dst_file.write(content)

    print(f"Файл {src_path} сконвертовано з cp1251 у {dst_path} (UTF-8).")


if __name__ == "__main__":
    raw_docs_folder = os.path.join("data", "raw_docs")
    processed_folder = os.path.join("data", "processed")
    os.makedirs(processed_folder, exist_ok=True)

    csv_out = os.path.join(processed_folder, "rag_chunks.csv")
    json_out = os.path.join(processed_folder, "rag_chunks.json")

    all_chunks = []

    for filename in os.listdir(raw_docs_folder):
        if not filename.lower().endswith((".docx", ".pdf")):
            print(f"Пропускаємо непідтримуваний формат файлу: {filename}")
            continue

        file_path = os.path.join(raw_docs_folder, filename)
        print(f"Обробляємо файл: {file_path}")

        chunks = process_file(file_path, chunk_size=500, overlap=100)
        for ch in chunks:
            all_chunks.append({
                "chunk": ch,
                "source_file": filename
            })

    save_chunks_to_csv(all_chunks, csv_out)
    save_chunks_to_json(all_chunks, json_out)

    print(f"Готово! Збережено {len(all_chunks)} chunks до:\n  {csv_out}\n  {json_out}")
