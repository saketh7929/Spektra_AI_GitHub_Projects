# AI Study Buddy – Intelligent Notes Assistant

A Streamlit app for studying from PDF/TXT notes. It splits notes into chunks and retrieves relevant passages with TF-IDF. Optional OpenAI integration can answer questions using retrieved context, summarize notes, and generate practice quizzes.

## Features
- Upload PDF or TXT study notes
- Local TF-IDF passage retrieval for questions
- AI question answering grounded in provided notes (optional)
- AI summaries and quiz generation (optional)
- No API key needed for local passage retrieval

## Tech stack
Python, Streamlit, pypdf, scikit-learn, OpenAI API (optional)

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Enable AI features
Set `OPENAI_API_KEY` in your environment. Optional `OPENAI_MODEL` defaults to `gpt-4.1-mini`. API usage may incur charges. Never commit secrets.

## How retrieval works
The app uses TF-IDF vectorization and cosine similarity to rank note chunks against the question. When AI is configured, the retrieved passages are sent as context and the assistant is instructed to prioritize the notes and acknowledge missing information.

## Limitations
- TF-IDF is a lightweight keyword-based retrieval baseline, not a full vector database.
- PDF extraction works best with selectable text; scanned documents need OCR.
- AI responses can still be imperfect; verify important study material.
