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
.answer-box {
    background: #1a1a2e;
    border-left: 4px solid #7c6ef7;
    border-radius: 8px;
    padding: 20px 24px;
    margin-top: 12px;
    font-size: 15px;
    line-height: 1.9;
    color: #e8e6f0;
}
.step-card {
    background: #16213e;
    border: 1px solid #2a2a4a;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 14px;
}
.step-done   { color: #22c55e; }
.step-active { color: #f59e0b; }
.step-wait   { color: #4a4a6a; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
for key in ["index", "chunks", "embed_model", "llm_model", "llm_tok", "ready", "stats"]:
    if key not in st.session_state:
        st.session_state[key] = None
if "ready" not in st.session_state:
    st.session_state.ready = False


# ── Cached model loaders ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_llm():
    tok   = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return model, tok


# ── IMPROVED SCRAPING ──────────────────────────────────────────────────────────
def scrape(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    r = requests.get(url, headers=headers, timeout=25)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # Remove all non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "noscript", "form", "iframe", "button", "input",
                     "aside", "meta", "link"]):
        tag.decompose()

    texts = []

    # ── Strategy 1: grab structured semantic blocks ──
    # Join list items under their parent so "We supply and support: item1 item2"
    # becomes one coherent chunk instead of orphaned fragments
    for ul in soup.find_all(["ul", "ol"]):
        # find nearest preceding heading or paragraph as context
        prev = ul.find_previous(["h1","h2","h3","h4","p"])
        context = prev.get_text(" ", strip=True) if prev else ""
        items = [li.get_text(" ", strip=True) for li in ul.find_all("li") if li.get_text(strip=True)]
        if items:
            joined = (context + " " if context else "") + " | ".join(items)
            texts.append(joined)

    # ── Strategy 2: grab headings + next sibling paragraphs together ──
    for heading in soup.find_all(["h1","h2","h3","h4"]):
        heading_text = heading.get_text(" ", strip=True)
        siblings = []
        for sib in heading.find_next_siblings():
            if sib.name in ["h1","h2","h3","h4"]:
                break
            t = sib.get_text(" ", strip=True)
            if t:
                siblings.append(t)
        if heading_text and siblings:
            block = heading_text + ". " + " ".join(siblings[:4])
            texts.append(block)
        elif heading_text:
            texts.append(heading_text)

    # ── Strategy 3: plain paragraphs ──
    for p in soup.find_all("p"):
        t = p.get_text(" ", strip=True)
        if t:
            texts.append(t)

    # ── Strategy 4: divs / sections with meaningful text ──
    for div in soup.find_all(["div", "section", "article", "span"]):
        # Only direct text, not all descendants (avoids duplication)
        direct = " ".join(
            child.strip()
            for child in div.strings
            if isinstance(child, str) and child.strip()
        )
        if len(direct) > 60:
            texts.append(direct)

    page_title = soup.title.string.strip() if soup.title and soup.title.string else url
    return texts, page_title


# ── EMBEDDING + FAISS ──────────────────────────────────────────────────────────
def build_index(chunks, model):
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ── RETRIEVAL ──────────────────────────────────────────────────────────────────
def retrieve(query, index, chunks, model, top_k=6):
    q_emb = model.encode([query], show_progress_bar=False)
    q_emb = np.array(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    results = [
        (chunks[i], float(scores[0][j]))
        for j, i in enumerate(ids[0])
        if i < len(chunks)
    ]
    return results


# ── IMPROVED ANSWER GENERATION ─────────────────────────────────────────────────
def generate_answer(query, context_chunks, model, tokenizer):
    # Build rich context from top chunks
    context_parts = []
    total_words = 0
    for chunk_text, score in context_chunks:
        words = chunk_text.split()
        if total_words + len(words) > 400:
            break
        context_parts.append(chunk_text)
        total_words += len(words)

    context = " ".join(context_parts)

    # Clear, detailed prompt that forces a full paragraph answer
    prompt = (
        f"You are a helpful assistant. Based on the information below, "
        f"write a detailed paragraph answering the question.\n\n"
        f"Information: {context}\n\n"
        f"Question: {query}\n\n"
        f"Write a complete, detailed paragraph answer (at least 3 sentences):\n"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=700
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        min_new_tokens=60,       # Force at least 60 tokens — prevents empty answers
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.5,      # Encourages longer answers
        temperature=0.7,
        do_sample=False,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # If model still returns something too short, fall back to a summary from context
    if len(answer.strip()) < 40:
        answer = _fallback_answer(query, context_parts)

    return answer


def _fallback_answer(query, context_parts):
    """
    Simple extractive fallback: stitch the most relevant sentences together
    into a readable paragraph when the LLM output is too short.
    """
    all_sentences = []
    for part in context_parts:
        sentences = re.split(r'(?<=[.!?])\s+', part)
        all_sentences.extend([s.strip() for s in sentences if len(s.strip()) > 30])

    if not all_sentences:
        return "I could not find a clear answer to your question in the website content."

    # Return up to 5 sentences as a coherent paragraph
    selected = all_sentences[:5]
    return " ".join(selected)


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════

st.title("🧠 Universal Web-Based AI System")
st.caption("Enter any website URL → the system scrapes, cleans, indexes, and lets you query it with AI.")
st.divider()

# ── STEP 1: URL input ──────────────────────────────────────────────────────────
st.subheader("Step 1 — Process a website")

col_url, col_btn = st.columns([4, 1])
with col_url:
    url = st.text_input("Website URL", placeholder="https://example.com",
                        label_visibility="collapsed")
with col_btn:
    process_btn = st.button("Process", use_container_width=True, type="primary")

progress_box = st.empty()

STEPS = [
    ("Scraping website…",          "Scraping complete"),
    ("Cleaning & filtering data…", "Data cleaned"),
    ("Chunking into segments…",    "Text chunked"),
    ("Generating embeddings…",     "Embeddings generated"),
    ("Building FAISS vector DB…",  "Vector DB ready"),
    ("Loading LLM model…",         "Model ready — ask your question!"),
]

def render_steps(done: int, active: int):
    html = ""
    for i, (active_lbl, done_lbl) in enumerate(STEPS):
        if i < done:
            html += f'<div class="step-card step-done">✅ {done_lbl}</div>'
        elif i == active:
            html += f'<div class="step-card step-active">⏳ {active_lbl}</div>'
        else:
            html += f'<div class="step-card step-wait">○ {done_lbl}</div>'
    return html


if process_btn and url:
    st.session_state.ready = False
    try:
        progress_box.markdown(render_steps(-1, 0), unsafe_allow_html=True)

        # Step 0 — Scrape
        raw_texts, page_title = scrape(url)
        progress_box.markdown(render_steps(1, 1), unsafe_allow_html=True)

        # Step 1 — Clean
        cleaned = clean(raw_texts)
        if not cleaned:
            st.error("No usable text found. The site may block scrapers or have no readable content.")
            st.stop()
        progress_box.markdown(render_steps(2, 2), unsafe_allow_html=True)

        # Step 2 — Chunk
        chunks = chunk(cleaned)
        if not chunks:
            st.error("Could not create text chunks. Try a different URL.")
            st.stop()
        progress_box.markdown(render_steps(3, 3), unsafe_allow_html=True)

        # Step 3 — Embed
        embed_model = load_embed_model()
        index = build_index(chunks, embed_model)
        progress_box.markdown(render_steps(4, 4), unsafe_allow_html=True)
        time.sleep(0.3)

        # Step 4 — FAISS done (covered above)
        progress_box.markdown(render_steps(5, 5), unsafe_allow_html=True)

        # Step 5 — Load LLM
        llm_model, llm_tok = load_llm()
        progress_box.markdown(render_steps(6, -1), unsafe_allow_html=True)

        # Save to session
        st.session_state.index       = index
        st.session_state.chunks      = chunks
        st.session_state.embed_model = embed_model
        st.session_state.llm_model   = llm_model
        st.session_state.llm_tok     = llm_tok
        st.session_state.ready       = True
        st.session_state.stats = {
            "title":   page_title,
            "raw":     len(raw_texts),
            "cleaned": len(cleaned),
            "chunks":  len(chunks),
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the URL: {e}")
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        raise e

elif process_btn and not url:
    st.warning("Please enter a URL first.")


# ── Stats ──────────────────────────────────────────────────────────────────────
if st.session_state.ready and st.session_state.stats:
    s = st.session_state.stats
    st.success(f"✅ **{s['title']}** is ready to query!")
    c1, c2, c3 = st.columns(3)
    c1.metric("Raw text blocks",  s["raw"])
    c2.metric("Cleaned blocks",   s["cleaned"])
    c3.metric("Index chunks",     s["chunks"])

st.divider()

# ── STEP 2: Query ──────────────────────────────────────────────────────────────
st.subheader("Step 2 — Ask a question")

if not st.session_state.ready:
    st.info("Process a website above first.")
else:
    query   = st.text_input("Your question",
                            placeholder="What services do you provide?")
    ask_btn = st.button("Get Answer", type="primary")

    if ask_btn and query:
        with st.spinner("Searching knowledge base and generating answer…"):
            results = retrieve(
                query,
                st.session_state.index,
                st.session_state.chunks,
                st.session_state.embed_model,
                top_k=6,
            )
            answer = generate_answer(
                query,
                results,
                st.session_state.llm_model,
                st.session_state.llm_tok,
            )

        st.markdown("#### Answer")
        st.markdown(f'<div class="answer-box">{answer}</div>',
                    unsafe_allow_html=True)

        # Show source context — collapsed by default, clean display
        with st.expander("View source context used for this answer"):
            for i, (chunk_text, score) in enumerate(results, 1):
                st.markdown(
                    f"**Source {i}** &nbsp; similarity score: `{score:.3f}`",
                    unsafe_allow_html=True
                )
                st.write(chunk_text)
                st.divider()

    elif ask_btn and not query:
        st.warning("Please type a question first.")

st.divider()
st.caption("Built with Streamlit · FAISS · SentenceTransformers · Flan-T5-Large · BeautifulSoup")
