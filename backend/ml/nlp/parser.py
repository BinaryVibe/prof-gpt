import fitz  # PyMuPDF
import re
import os

def clean_academic_text(raw_text: str) -> str:
    """
    Cleans messy PDF text by removing excessive whitespace and broken lines.
    """
    # Replace multiple newlines with a single space
    text = re.sub(r'\n+', ' ', raw_text)
    # Remove multiple spaces
    text = re.sub(r'\s{2,}', ' ', text)
    # Remove special characters that might break the ML model later
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
    return text.strip()

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF, extracts text from all pages, and returns a single clean string.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Error: Could not find the file at {pdf_path}")

    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        full_text = []

        # Loop through every page and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            full_text.append(text)

        # Combine all pages into one giant string
        raw_text = " ".join(full_text)
        
        # Clean it up before sending it to the ML models
        cleaned_text = clean_academic_text(raw_text)
        
        return cleaned_text

    except Exception as e:
        print(f"Failed to process {pdf_path}: {e}")
        return ""

# --- TESTING BLOCK ---
# This only runs if you execute this specific file directly
if __name__ == "__main__":
    # Put a test PDF (like a syllabus) inside the backend/ml/nlp/ folder
    test_file = "backend/ml/nlp/sample_syllabus.pdf"
    
    print(f"Attempting to extract text from {test_file}...\n")
    extracted = extract_text_from_pdf(test_file)
    
    if extracted:
        print("✅ Extraction Successful! Here is a preview of the first 500 characters:\n")
        print("-" * 50)
        print(extracted[:500] + "...")
        print("-" * 50)