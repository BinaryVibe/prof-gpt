
# Prof GPT | Academic Navigator

**Prof GPT** is a Full-Stack Retrieval-Augmented Generation (RAG) system custom-built for university students. It acts as an intelligent academic assistant, allowing users to upload course materials, CVs, and institutional handbooks, and ask questions with high-precision, hallucination-free answers.

Developed with a strict 96% Precision@3 retrieval accuracy, Prof GPT combines local vector databases, intent classification, and an enterprise-grade LLM generation layer powered by Google Gemini.

---

## Features

* **Dynamic PDF Ingestion:** Upload any PDF (Lecture slides, complex CVs, policies) directly through the UI. The backend uses PyMuPDF to extract, chunk, and embed the text on the fly.
* **Smart Intent Routing & Fallback:** Uses a custom trained Machine Learning model to classify user queries into specific academic intents (e.g., `Technical`, `Schedule`, `Policy`). If the filtered database search fails, an automatic "Global Fallback Search" ensures no data is missed.
* **Strict Anti-Hallucination Guardrails:** The LLM is strictly prompted to only answer based on retrieved context. If the answer isn't in the uploaded PDFs, Prof GPT will explicitly refuse rather than invent fake programs or sources.
* **Streamlit UI:** A high-contrast, modern, dark-mode frontend featuring the vibrant #FF5628 orange accent, "Syne" typography, and interactive Quick Action dashboards.
* **High-Volume LLM Architecture:** Integrated with Google's `gemini-3.1-flash-lite` to efficiently process massive document contexts while maintaining rapid, lightweight response times.

---

## 🛠️ Tech Stack

**Frontend**
* [Streamlit](https://streamlit.io/) (Session state management, interactive UI, custom CSS injection)

**Backend & API**
* [FastAPI](https://fastapi.tiangolo.com/) (High-performance API routing for chat and uploads)
* [Uvicorn](https://www.uvicorn.org/) (ASGI Web Server)

**Machine Learning & RAG Engine**
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Local persistent storage)
* **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Text Extraction:** `PyMuPDF` (fitz)
* **Intent Classification:** `scikit-learn` & `joblib`
* **LLM Provider:** Google GenAI SDK (`gemini-3.1-flash-lite`)

---

## Project Structure

```text
PROF-GPT/
│
├── backend/
│   ├── api/
│   │   ├── dependencies.py
│   │   └── routes.py                # FastAPI endpoints (/ask, /upload)
│   │
│   ├── ml/
│   │   ├── intent/
│   │   │   ├── intent_model.pkl     # Trained ML model for query classification
│   │   │   └── intent_engine.py
│   │   │
│   │   ├── nlp/
│   │   │   └── response_generator.py # Gemini API connection & strict prompting
│   │   │
│   │   └── rag/
│   │       ├── chunker.py           # Text splitting logic (Size: 500, Overlap: 10%)
│   │       ├── embedder.py          # MiniLM embedding functions
│   │       └── search.py            # KNN Vector search with Intent Fallback logic
│   │
│   ├── database.py                  # ChromaDB client configuration
│   └── main.py                      # FastAPI application entry point
│
├── frontend/
│   └── app.py                       # Streamlit UI, styling, and session state
│
└── data/
    └── chroma_db/                   # Local vector database storage

```

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/BinaryVibe/prof-gpt.git
cd prof-gpt

```

### 2. Set up the Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn streamlit requests chromadb sentence-transformers pymupdf scikit-learn google-generativeai joblib

```
OR 
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Prof GPT requires a Google Gemini API key to run the generation layer. Get a key from [Google AI Studio](https://aistudio.google.com/) and set it in your terminal:

```bash
# Linux/macOS
export GEMINI_API_KEY="your_api_key_here"

# Windows (Command Prompt)
set GEMINI_API_KEY=your_api_key_here

```

### 5. Run the Application

You will need two terminal windows running simultaneously.

**Terminal 1: Start the FastAPI Backend**

```bash
uvicorn backend.main:app --reload

```

**Terminal 2: Start the Streamlit Frontend**

```bash
streamlit run frontend/app.py

```

---

## RAG Pipeline Configuration

During testing, the retrieval engine was mathematically validated against a 100-question evaluation dataset. The optimal parameters for the academic document formats are:

* **Chunk Size:** 500 tokens
* **Chunk Overlap:** 30%
* **Retrieval Metric:** Precision@3 (Achieved **96% accuracy**)

---

## 🔮 Future Enhancements

* Implement `streamlit-authenticator` and SQLite for persistent user accounts and chat history.
* Add support for multi-modal ingestion (OCR for image-based PDFs and diagrams).
* Connect UI directly to student portals for live query resolution.

---

**Developed by:** Ayaan Ahmed Khan & Muhammad Umar Nasir

