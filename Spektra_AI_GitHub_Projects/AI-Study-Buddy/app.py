import os
import io
import re
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="wide")
st.title("📚 AI Study Buddy – Intelligent Notes Assistant")
st.caption("Upload study notes, retrieve relevant passages, summarize, and optionally ask an AI model to explain them.")


def read_upload(file):
    if file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file.getvalue().decode("utf-8", errors="ignore")


def split_chunks(text, chunk_words=180, overlap=35):
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_words - overlap, 1)
    for start in range(0, len(words), step):
        part = words[start:start + chunk_words]
        if part:
            chunks.append(" ".join(part))
        if start + chunk_words >= len(words):
            break
    return chunks


def retrieve(query, chunks, top_k=4):
    if not chunks or not query.strip():
        return []
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [query])
    similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = similarities.argsort()[::-1][:top_k]
    return [(chunks[i], float(similarities[i])) for i in ranked if similarities[i] > 0]


def optional_ai(prompt, context):
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    client = OpenAI(api_key=key)
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {"role": "system", "content": "You are a study assistant. Use the supplied notes as the primary source. If the notes do not contain the answer, clearly say so. Explain in student-friendly language."},
            {"role": "user", "content": f"Notes context:\n{context[:14000]}\n\nTask: {prompt}"}
        ]
    )
    return response.output_text


with st.sidebar:
    st.header("Workflow")
    st.write("Upload a PDF or TXT file, then ask a question or generate a summary/quiz.")
    st.info("Document retrieval uses TF-IDF locally. Optional AI responses require OPENAI_API_KEY.")

file = st.file_uploader("Upload your notes (PDF or TXT)", type=["pdf", "txt"])
if file:
    try:
        raw_text = read_upload(file).strip()
        if not raw_text:
            st.error("No text was extracted. Try a text-based PDF or TXT file.")
        else:
            chunks = split_chunks(raw_text)
            st.success(f"Loaded notes: {len(raw_text.split())} words in {len(chunks)} searchable chunks.")
            tab1, tab2, tab3 = st.tabs(["Ask a question", "Summarize", "Create practice quiz"])
            with tab1:
                question = st.text_input("Your question", placeholder="Explain the main idea of these notes...")
                if st.button("Find answer", type="primary", key="answer"):
                    hits = retrieve(question, chunks)
                    if not hits:
                        st.warning("No relevant passage was found. Try different keywords.")
                    else:
                        context = "\n\n".join(text for text, score in hits)
                        answer = optional_ai(question, context)
                        if answer:
                            st.markdown("#### AI answer")
                            st.write(answer)
                        else:
                            st.markdown("#### Relevant passages (local retrieval)")
                            st.caption("Configure OPENAI_API_KEY for an AI-generated explanation; these excerpts are retrieved from your file.")
                            for i, (passage, score) in enumerate(hits, 1):
                                with st.expander(f"Passage {i} · relevance {score:.2f}"):
                                    st.write(passage)
            with tab2:
                if st.button("Generate summary", type="primary", key="summary"):
                    summary = optional_ai("Summarize these notes into clear bullet points and list key terms.", raw_text)
                    if summary:
                        st.write(summary)
                    else:
                        # Simple offline fallback: show opening chunks as source-based overview.
                        st.info("AI summary needs OPENAI_API_KEY. Here is a short excerpt-based overview instead.")
                        st.write(" ".join(raw_text.split()[:220]) + (" …" if len(raw_text.split()) > 220 else ""))
            with tab3:
                n_questions = st.slider("Number of questions", 3, 10, 5)
                if st.button("Create quiz", type="primary", key="quiz"):
                    quiz = optional_ai(f"Create {n_questions} practice questions from these notes. Include an answer key at the end. Avoid inventing facts not supported by the notes.", raw_text)
                    if quiz:
                        st.write(quiz)
                    else:
                        st.info("AI quiz generation needs OPENAI_API_KEY. Here are study prompts based on retrieved note chunks.")
                        for i, chunk in enumerate(chunks[:n_questions], 1):
                            st.markdown(f"**{i}.** What are the key concepts in this passage?\n\n<details><summary>Study passage</summary>{chunk}</details>")
    except Exception as exc:
        st.error(f"Could not process the uploaded file: {exc}")
else:
    st.markdown("### Try it with a class note PDF or text file")
