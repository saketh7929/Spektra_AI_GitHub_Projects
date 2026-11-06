# AI-Powered Resume Analyzer

A Streamlit application that compares a PDF resume with a job description. It reports a transparent baseline skill-match score, matched skills, and possible missing keywords. Optional OpenAI-powered suggestions can be enabled with an API key.

## Features
- PDF resume text extraction
- Job-description keyword comparison
- Match score and matched/missing skill lists
- Optional AI-generated, honest resume improvement suggestions
- Local baseline analysis without an API key

## Tech stack
Python, Streamlit, pypdf, OpenAI API (optional)

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

## Enable optional AI suggestions
Set `OPENAI_API_KEY` in your environment before starting the app. You may also set `OPENAI_MODEL` (defaults to `gpt-4.1-mini`). API usage may incur charges. Never commit API keys.

## Limitations
- The baseline score is a simple skill-keyword heuristic, not a hiring prediction.
- The skill dictionary is small and should be expanded for specific industries.
- Scanned PDFs without selectable text are not OCR-processed.
- Add skills to a resume only when they accurately represent your abilities.
