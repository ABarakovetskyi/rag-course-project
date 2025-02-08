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

def read_docx(file_path: str) -> str:
    """
    Зчитує текст із .docx файлу за допомогою python-docx
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
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

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list:
    """
    Розбиває рядок `text` на список шматків (chunk-ів) з урахуванням:
      - макс. довжини `chunk_size`
      - перекриття `overlap`
      - розриву chunk-ів по пробілах/переносах (щоб не різати слова всередині).

    Якщо трапляється дуже довге слово, chunk все одно може взяти його цілим.
    """
    chunks = []
    text_length = len(text)
    start = 0

    while start < text_length:
        # Межа chunk-а "спочатку"
        end = min(start + chunk_size, text_length)

        if end < text_length:
            # Перевіряємо, чи не закінчуємо chunk всередині слова
            # Якщо символ text[end] - не пробіл і не перевідрядка, рухаємося назад.
            temp_end = end
            while temp_end > start and not text[temp_end].isspace():
                temp_end -= 1

            # Якщо temp_end == start, це означає, що слово довше за chunk_size,
            # тому відкотитися до пробілу неможливо - chunk візьме весь [start:end].
            if temp_end > start:
                end = temp_end

        chunk = text[start:end].rstrip()  # strip вправо, щоб уникнути зайвих пробілів

        # Додаємо, тільки якщо chunk непорожній
        if chunk:
            chunks.append(chunk)

        # Якщо досягнули кінця тексту, завершуємо
        if end >= text_length:
            break

        # Обчислюємо новий start з урахуванням overlap
        # (якщо end == start + chunk_size, наступний початок = end - overlap)
        start = max(end - overlap, end)  # щоб уникнути start < end

        if start >= text_length:
            break

    return chunks


def process_file(file_path: str,
                 chunk_size: int = 2000,
                 overlap: int = 200) -> list:
    """
    Залежно від розширення файлу (.docx або .pdf),
    зчитує текст і нарізає його на chunk-и.
    Повертає список рядків (кожен chunk).
    """
    base, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".docx":
        text_data = read_docx(file_path)
    elif ext == ".pdf":
        text_data = read_pdf(file_path)
    else:
        # Пропускаємо інші формати
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

    # Обробляємо всі docx/pdf у папці raw_docs
    for filename in os.listdir(raw_docs_folder):
        if not filename.lower().endswith((".docx", ".pdf")):
            print(f"Пропускаємо непідтримуваний формат файлу: {filename}")
            continue

        file_path = os.path.join(raw_docs_folder, filename)
        print(f"Обробляємо файл: {file_path}")

        chunks = process_file(file_path, chunk_size=2000, overlap=200)
        for ch in chunks:
            all_chunks.append({
                "chunk": ch,
                "source_file": filename
            })

    # Зберігаємо результати у CSV та JSON
    save_chunks_to_csv(all_chunks, csv_out)
    save_chunks_to_json(all_chunks, json_out)

    print(f"Готово! Збережено {len(all_chunks)} chunks до:\n  {csv_out}\n  {json_out}")
