import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time

st.set_page_config(
    page_title="Universal Web-Based AI System",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
.step-done   { color: #22c55e; font-weight: 500; }
.step-active { color: #f59e0b; font-weight: 500; }
.step-wait   { color: #6b7280; }
.answer-box  {
    background: #f0f9ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 16px 20px;
    margin-top: 12px;
    font-size: 15px;
    line-height: 1.7;
}
.metric-row  { display: flex; gap: 12px; margin: 12px 0; }
.metric-card {
    flex: 1;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    text-align: center;
}
.metric-val  { font-size: 22px; font-weight: 700; color: #1e293b; }
.metric-lbl  { font-size: 12px; color: #64748b; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key in ["index", "chunks", "embed_model", "llm_model", "llm_tok", "ready", "stats"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "ready" not in st.session_state:
    st.session_state.ready = False


# ── Model loaders (cached) ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_llm():
    tok   = T5Tokenizer.from_pretrained("google/flan-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    return model, tok


# ── Scraping ───────────────────────────────────────────────────────────────────
def scrape(url: str):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()
    texts = []
    for tag in soup.find_all(["h1","h2","h3","h4","p","li","td","th","blockquote","article"]):
        t = tag.get_text(separator=" ", strip=True)
        if t:
            texts.append(t)
    return texts, soup.title.string if soup.title else url


# ── Cleaning ───────────────────────────────────────────────────────────────────
STOP_PHRASES = {"cookie","privacy policy","terms of service","login","sign up",
                "subscribe","newsletter","all rights reserved","click here",
                "read more","learn more","contact us","follow us"}

def clean(texts):
    cleaned, seen = [], set()
    for t in texts:
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"[^\x00-\x7F]+", " ", t)
        if len(t) < 20:
            continue
        low = t.lower()
        if any(ph in low for ph in STOP_PHRASES):
            continue
        key = re.sub(r"\W+", "", low)[:80]
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)
    return cleaned


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk(texts, max_words=120, overlap=20):
    chunks = []
    for t in texts:
        words = t.split()
        if len(words) <= max_words:
            chunks.append(t)
        else:
            for i in range(0, len(words), max_words - overlap):
                chunk_words = words[i : i + max_words]
                if len(chunk_words) >= 15:
                    chunks.append(" ".join(chunk_words))
    return chunks


# ── Embedding + FAISS ──────────────────────────────────────────────────────────
def build_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


# ── Retrieval ──────────────────────────────────────────────────────────────────
def retrieve(query, index, chunks, model, top_k=5):
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    results = [(chunks[i], float(scores[0][j])) for j, i in enumerate(ids[0]) if i < len(chunks)]
    return results


# ── Answer generation ──────────────────────────────────────────────────────────
def generate_answer(query, context_chunks, model, tokenizer, max_new_tokens=256):
    context = " ".join([c for c, _ in context_chunks[:4]])
    context = context[:1800]
    prompt = (
        f"Answer the following question using only the context provided.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🧠 Universal Web-Based AI System")
st.caption("Enter any website URL → the system scrapes, cleans, indexes, and lets you query it with AI.")

st.divider()

# ── Section 1: URL input ───────────────────────────────────────────────────────
st.subheader("Step 1 — Process a website")

col_url, col_btn = st.columns([4, 1])
with col_url:
    url = st.text_input("Website URL", placeholder="https://example.com", label_visibility="collapsed")
with col_btn:
    process_btn = st.button("Process", use_container_width=True, type="primary")

# ── Pipeline progress placeholder ─────────────────────────────────────────────
progress_placeholder = st.empty()

STEPS = [
    ("🔍", "Scraping website…",          "Scraping complete"),
    ("🧹", "Cleaning data…",             "Data cleaned"),
    ("✂️", "Chunking text…",             "Text chunked"),
    ("🔢", "Generating embeddings…",     "Embeddings generated"),
    ("🗄️", "Building vector database…", "Vector DB created"),
    ("🤖", "Loading LLM…",               "Model ready"),
]

def render_steps(done_up_to: int, active: int):
    lines = []
    for i, (icon, active_label, done_label) in enumerate(STEPS):
        if i < done_up_to:
            lines.append(f'<p class="step-done">✅ {done_label}</p>')
        elif i == active:
            lines.append(f'<p class="step-active">⏳ {icon} {active_label}</p>')
        else:
            lines.append(f'<p class="step-wait">○ {done_label}</p>')
    return "\n".join(lines)

if process_btn and url:
    st.session_state.ready = False
    try:
        with st.spinner(""):

            # Step 0 — Scrape
            progress_placeholder.markdown(render_steps(-1, 0), unsafe_allow_html=True)
            raw_texts, page_title = scrape(url)
            time.sleep(0.3)

            # Step 1 — Clean
            progress_placeholder.markdown(render_steps(1, 1), unsafe_allow_html=True)
            cleaned = clean(raw_texts)
            time.sleep(0.2)

            # Step 2 — Chunk
            progress_placeholder.markdown(render_steps(2, 2), unsafe_allow_html=True)
            chunks = chunk(cleaned)
            time.sleep(0.2)

            if not chunks:
                st.error("No usable text found on this page. Try a different URL.")
                st.stop()

            # Step 3 — Embeddings
            progress_placeholder.markdown(render_steps(3, 3), unsafe_allow_html=True)
            embed_model = load_embed_model()
            index, _ = build_index(chunks, embed_model)

            # Step 4 — FAISS done (covered by build_index above)
            progress_placeholder.markdown(render_steps(4, 4), unsafe_allow_html=True)
            time.sleep(0.2)

            # Step 5 — LLM
            progress_placeholder.markdown(render_steps(5, 5), unsafe_allow_html=True)
            llm_model, llm_tok = load_llm()

            # All done
            progress_placeholder.markdown(render_steps(6, -1), unsafe_allow_html=True)

            # Store in session
            st.session_state.index       = index
            st.session_state.chunks      = chunks
            st.session_state.embed_model = embed_model
            st.session_state.llm_model   = llm_model
            st.session_state.llm_tok     = llm_tok
            st.session_state.ready       = True
            st.session_state.stats       = {
                "url":       url,
                "title":     page_title,
                "raw":       len(raw_texts),
                "cleaned":   len(cleaned),
                "chunks":    len(chunks),
            }

    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch the URL: {e}")
    except Exception as e:
        st.error(f"Pipeline error: {e}")

elif process_btn and not url:
    st.warning("Please enter a URL first.")


# ── Stats cards ────────────────────────────────────────────────────────────────
if st.session_state.ready and st.session_state.stats:
    s = st.session_state.stats
    st.success(f"✅ **{s['title']}** is ready to query!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw text blocks", s["raw"])
    c2.metric("Cleaned blocks",  s["cleaned"])
    c3.metric("Index chunks",    s["chunks"])

st.divider()

# ── Section 2: Query ───────────────────────────────────────────────────────────
st.subheader("Step 2 — Ask a question")

if not st.session_state.ready:
    st.info("Process a website above before querying.")
else:
    query = st.text_input("Your question", placeholder="What is this website about?")
    ask_btn = st.button("Get Answer", type="primary")

    if ask_btn and query:
        with st.spinner("Searching and generating answer…"):
            results = retrieve(
                query,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.embed_model,
            )
            answer = generate_answer(
                query,
                results,
                st.session_state.llm_model,
                st.session_state.llm_tok,
            )

        st.markdown("#### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

        with st.expander("View retrieved context chunks"):
            for i, (chunk_text, score) in enumerate(results, 1):
                st.markdown(f"**Chunk {i}** (similarity: `{score:.3f}`)")
                st.caption(chunk_text)
                st.divider()

    elif ask_btn and not query:
        st.warning("Please enter a question.")

st.divider()
st.caption("Built with Streamlit · FAISS · SentenceTransformers · Flan-T5 · BeautifulSoup")
