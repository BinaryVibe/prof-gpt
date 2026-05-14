import fitz
import re
import os

def clean_academic_text(raw_text: str) -> str:
    text = re.sub(r'\n+', ' ', raw_text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        return ""

    try:
        doc = fitz.open(pdf_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            full_text.append(page.get_text("text"))
        
        raw_text = " ".join(full_text)
        return clean_academic_text(raw_text)
    except Exception:
        return ""

if __name__ == "__main__":
    test_file = "backend/data/sample.pdf"
    extracted = extract_text_from_pdf(test_file)
    if extracted:
        print(extracted[:500])