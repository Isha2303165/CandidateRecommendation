
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from PyPDF2 import PdfReader
import docx
from PIL import Image
import pytesseract
import io
import numpy as np
import pandas as pd

st.set_page_config(page_title="Candidate Recommendation Engine", layout="wide")


EMBED_MODEL_NAME = "all-MiniLM-L6-v2"           # small, fast, good for semantic similarity
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"  # faster summarizer
TOP_K = 10  # default number of top candidates to show


@st.cache_resource(show_spinner=False)
def load_models():
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    try:
        summarizer = pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)  # CPU by default
    except Exception as e:
        summarizer = None
        st.warning(f"Summarizer load failed: {e}. Summaries will be disabled.")
    return embed_model, summarizer

embed_model, summarizer = load_models()


def extract_text_from_pdf(file) -> str:
    try:
        reader = PdfReader(file)
        texts = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                texts.append(txt)
        return "\n".join(texts)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_docx(file) -> str:
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs if p.text])
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_image(file) -> str:
    try:
        image = Image.open(file).convert("RGB")
    
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error doing OCR on image: {e}")
        return ""

def extract_text_from_txt(file) -> str:
    try:
        raw = file.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return ""

def extract_text(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif name.endswith(".docx") or name.endswith(".doc"):
        return extract_text_from_docx(uploaded_file)
    elif any(name.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
        return extract_text_from_image(uploaded_file)
    elif name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        # try fallback to reading bytes as text
        return extract_text_from_txt(uploaded_file)

# -------------------------
# Embedding & similarity helpers
# -------------------------
@st.cache_data(show_spinner=False)
def embed_texts(texts: list):
    # returns numpy array (n, d)
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-12
    return embeddings / norms

def compute_similarities(job_emb: np.ndarray, resume_embs: np.ndarray) -> np.ndarray:
   
    return np.dot(resume_embs, job_emb)


def summarize_text(text: str, max_length=120, min_length=30):
    if summarizer is None:
        return "Summary unavailable (summarizer failed to load)."
    # reduce size for extremely long texts (pipeline can fail on extremely long inputs)
    if len(text) > 4000:
        text = text[:4000]
    try:
        out = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        return out[0]["summary_text"]
    except Exception as e:
        return f"Summary error: {e}"

st.title("🔍 Candidate Recommendation Engine — Local (No OpenAI)")
top_k = st.number_input("Show top K candidates", value=TOP_K, min_value=1, max_value=50, step=1)
show_full_text = st.checkbox("Show full extracted resume text (in results)", value=False)
show_summaries = st.checkbox("Generate AI summaries (local model)", value=True)


col1, col2 = st.columns([2, 1])

with col1:
    uploaded_files = st.file_uploader(
        "Upload Candidate Resumes (multiple accepted):",
        type=["pdf", "docx", "doc", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    job_description = st.text_area("📝 Enter the Job Description", height=240)

    recommend_btn = st.button("🚀 Recommend Candidates")


if recommend_btn:
    if (not uploaded_files or len(uploaded_files) == 0) and (not st.session_state.get("pasted_resumes")):
        st.warning("Please upload at least one resume file.")
    elif not job_description or job_description.strip() == "":
        st.warning("Please enter a job description.")
    else:
        # Read all resumes
        names = []
        texts = []
        for f in uploaded_files:
            with st.spinner(f"Extracting {f.name}..."):
                txt = extract_text(f)
            if not txt or txt.strip() == "":
                st.warning(f"No text extracted from {f.name}. Skipping.")
                continue
            names.append(f.name)
            texts.append(txt)

        if len(texts) == 0:
            st.error("No readable resumes found.")
        else:
            # show progress and encode
            with st.spinner("Embedding job description and resumes..."):
                job_emb = embed_model.encode(job_description, convert_to_numpy=True, show_progress_bar=False)
                job_emb = job_emb / (np.linalg.norm(job_emb) + 1e-12)
                resume_embs = embed_texts(texts)  # normalized array (n, d)

            # compute similarity scores
            sims = compute_similarities(job_emb, resume_embs)  # shape (n,)
            # build results
            results = []
            for i, (n, t, s) in enumerate(zip(names, texts, sims)):
                results.append({"name": n, "score": float(s), "text": t})

            # sort descending
            results = sorted(results, key=lambda x: x["score"], reverse=True)
            top_results = results[:top_k]

            # Display table
            df = pd.DataFrame([{"Candidate": r["name"], "Similarity": round(r["score"], 4)} for r in top_results])
            st.subheader(f"Top {len(top_results)} candidates")
            st.table(df)

            # CSV download
            csv = pd.DataFrame([{"Candidate": r["name"], "Similarity": r["score"], "ResumeText": r["text"]} for r in results]).to_csv(index=False)
            st.download_button("Download full ranked CSV", csv, file_name="ranked_candidates.csv", mime="text/csv")

            # Show each candidate
            for idx, r in enumerate(top_results, start=1):
                st.markdown("---")
                st.subheader(f"#{idx} — {r['name']}")
                st.write(f"**Similarity score:** {r['score']:.4f}")
                if show_full_text:
                    st.expander("Extracted resume text", expanded=False).write(r["text"])
                # short preview
                preview = r["text"][:2000]
                st.write("**Resume preview:**")
                st.write(preview + ("..." if len(r["text"]) > len(preview) else ""))

                # Summarize if enabled
                if show_summaries:
                    with st.spinner("Generating summary..."):
                        summ = summarize_text(r["text"], max_length=120, min_length=30)
                    st.markdown("**Why this candidate might be a good fit (summary):**")
                    st.write(summ)


