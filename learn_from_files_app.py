"""
Learn-From-My-Files AI (starter)

Idea: the bot starts EMPTY. It only knows what you upload.
Technique: RAG (Retrieval-Augmented Generation)
  1. Upload files -> text is split into small chunks
  2. When you ask a question -> the most relevant chunks are found
  3. Claude answers ONLY from those chunks

Setup:
  pip install streamlit anthropic scikit-learn pypdf
  export ANTHROPIC_API_KEY="your-key"      (Windows: set ANTHROPIC_API_KEY=your-key)
  streamlit run learn_from_files_app.py
"""

import streamlit as st
from anthropic import Anthropic
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL = "claude-sonnet-5-5"
CHUNK_SIZE = 800      # characters per chunk
TOP_K = 5             # chunks sent to Claude per question

client = Anthropic()  # reads ANTHROPIC_API_KEY from environment

st.set_page_config(page_title="Learn From My Files", page_icon="📚")
st.title("📚 Learn From My Files")
st.caption("Ye AI shuru me kuch nahi jaanta. Jo files upload karoge, bas wahi seekhega.")

# ---------- memory of the app (kept per browser session) ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []      # list of (filename, text)
if "history" not in st.session_state:
    st.session_state.history = []     # chat messages
if "learned_files" not in st.session_state:
    st.session_state.learned_files = set()


def read_file(f):
    name = f.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(f)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return f.read().decode("utf-8", errors="ignore")   # txt, md, csv, code, etc.


def split_text(text):
    text = " ".join(text.split())
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


def find_relevant(question):
    texts = [c[1] for c in st.session_state.chunks]
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best = scores.argsort()[::-1][:TOP_K]
    return [st.session_state.chunks[i] for i in best if scores[i] > 0]


# ---------- sidebar: teach the AI ----------
with st.sidebar:
    st.header("Files upload karo")
    files = st.file_uploader(
        "PDF / TXT / MD", type=["pdf", "txt", "md", "csv"], accept_multiple_files=True
    )
    if st.button("🧠 Seekho (Learn)") and files:
        for f in files:
            if f.name in st.session_state.learned_files:
                continue
            for chunk in split_text(read_file(f)):
                st.session_state.chunks.append((f.name, chunk))
            st.session_state.learned_files.add(f.name)
        st.success(f"{len(st.session_state.learned_files)} file(s) seekh li!")

    st.write("**Seekhi hui files:**")
    for n in sorted(st.session_state.learned_files):
        st.write("•", n)
    if st.button("🗑️ Sab bhool jao (Reset)"):
        st.session_state.chunks = []
        st.session_state.history = []
        st.session_state.learned_files = set()
        st.rerun()

# ---------- chat ----------
for m in st.session_state.history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

question = st.chat_input("Apni files se kuch bhi pucho...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if not st.session_state.chunks:
            answer = "Mujhe abhi kuch nahi aata. Pehle sidebar se files upload karke 'Seekho' dabao."
            st.write(answer)
        else:
            context = "\n\n".join(
                f"[Source: {n}]\n{t}" for n, t in find_relevant(question)
            ) or "(no relevant text found)"

            system = (
                "You are a study assistant that knows NOTHING except the context "
                "given below from the user's uploaded files. Answer only from that "
                "context. If the answer is not there, say you haven't learned it yet. "
                "Mention the source file name. Reply in the same language the user writes in."
            )
            prompt = f"Context from uploaded files:\n{context}\n\nQuestion: {question}"

            placeholder = st.empty()
            answer = ""
            with client.messages.stream(
                model=MODEL,
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    answer += text
                    placeholder.write(answer)

    st.session_state.history.append({"role": "assistant", "content": answer})
