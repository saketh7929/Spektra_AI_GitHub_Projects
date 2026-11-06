import os
import re
import io
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Resume Analyzer")
st.caption("Compare a resume with a job description and identify relevant skills and keywords.")

DEFAULT_SKILLS = {
    "Programming": ["python", "java", "javascript", "typescript", "c++", "sql"],
    "Web & APIs": ["html", "css", "react", "node.js", "rest api", "json", "http"],
    "AI & Data": ["machine learning", "artificial intelligence", "nlp", "pandas", "numpy", "llm", "prompt engineering"],
    "Engineering": ["git", "github", "debugging", "testing", "agile", "oop", "data structures", "database", "docker"]
}


def extract_pdf_text(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages).strip()


def tokenize(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", text.lower()))


def find_skills(text):
    lower = text.lower()
    found = []
    for group, skills in DEFAULT_SKILLS.items():
        for skill in skills:
            if skill in lower:
                found.append((group, skill))
    return found


def keyword_match(resume, job):
    # Skill phrase matching gives a transparent baseline score without requiring an API key.
    job_skills = [skill for skills in DEFAULT_SKILLS.values() for skill in skills if skill in job.lower()]
    job_terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", job.lower()))
    resume_terms = tokenize(resume)
    matched_terms = sorted(job_terms & resume_terms)
    missing_skills = sorted(skill for skill in job_skills if skill not in resume.lower())
    matched_skills = sorted(skill for skill in job_skills if skill in resume.lower())
    score = round(100 * len(matched_skills) / max(len(job_skills), 1)) if job_skills else round(100 * len(matched_terms) / max(len(job_terms), 1))
    return min(score, 100), matched_skills, missing_skills, matched_terms


with st.sidebar:
    st.header("How it works")
    st.write("1. Upload your PDF resume.\n2. Paste the job description.\n3. Review the transparent keyword analysis.\n4. Optionally request AI-generated suggestions.")
    st.info("The baseline analysis runs locally. AI suggestions require an OpenAI API key set as OPENAI_API_KEY.")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
with col2:
    job_description = st.text_area("Paste job description", height=240, placeholder="Paste the target role's responsibilities and requirements...")

if st.button("Analyze Resume", type="primary", use_container_width=True):
    if not resume_file or not job_description.strip():
        st.warning("Please upload a PDF resume and paste a job description.")
    else:
        try:
            resume_text = extract_pdf_text(resume_file)
            if not resume_text:
                st.error("No selectable text found. Try a text-based PDF instead of a scanned image PDF.")
            else:
                score, matched, missing, terms = keyword_match(resume_text, job_description)
                st.subheader("Analysis results")
                st.metric("Keyword skill match", f"{score}%")
                st.progress(score / 100)
                a, b = st.columns(2)
                with a:
                    st.markdown("#### Matching skills")
                    st.write(matched if matched else "No skills from the built-in skill list matched.")
                with b:
                    st.markdown("#### Skills to consider adding (only if true)")
                    st.write(missing if missing else "No missing skills from the built-in skill list.")
                st.markdown("#### Shared keywords")
                st.write(terms[:80] if terms else "No shared keywords found.")

                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    if st.button("Generate AI suggestions"):
                        try:
                            from openai import OpenAI
                            client = OpenAI(api_key=api_key)
                            response = client.responses.create(
                                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                                input=[{"role": "system", "content": "You are a career assistant. Give concise, honest resume improvement suggestions. Never recommend claiming skills or experience the applicant does not have."},
                                       {"role": "user", "content": f"Resume text:\n{resume_text[:12000]}\n\nJob description:\n{job_description[:8000]}\n\nProvide 5 tailored suggestions and a short ATS keyword list."}]
                            )
                            st.markdown("#### AI suggestions")
                            st.write(response.output_text)
                        except Exception as exc:
                            st.error(f"AI suggestion request failed: {exc}")
                else:
                    st.caption("AI suggestions are hidden because OPENAI_API_KEY is not configured. Baseline analysis above is available without it.")
        except Exception as exc:
            st.error(f"Could not read the PDF: {exc}")
