# Universal Web-Based AI System

A fully cloud-deployable RAG (Retrieval-Augmented Generation) pipeline built with Streamlit.

## What it does

1. Takes any website URL as input
2. Scrapes all text content (headings, paragraphs, lists)
3. Cleans and deduplicates the text
4. Chunks text into optimal segments
5. Generates semantic embeddings (SentenceTransformers)
6. Stores in a FAISS vector index
7. Accepts user queries and returns AI-generated answers (Flan-T5)

## Files

```
your-repo/
├── app.py                   # Main Streamlit application
├── requirements.txt         # All dependencies
└── .streamlit/
    └── config.toml          # Theme and server config
```

## Deploy on Streamlit Cloud (Free)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository and branch (`main`)
5. Set **Main file path** to `app.py`
6. Click **"Deploy"**

Your app will be live at:
`https://YOUR_USERNAME-YOUR_REPO-app-XXXXX.streamlit.app`

Share this link with your mentor — they can open it in any browser, no installation needed.

## Run locally (optional)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to use

1. Paste any website URL into the input field
2. Click **"Process"** — watch the pipeline run step by step:
   - Scraping complete
   - Data cleaned
   - Text chunked
   - Embeddings generated
   - Vector DB created
   - Model ready
3. Type a question about the website content
4. Click **"Get Answer"**

## Tech stack

| Component        | Library                          |
|-----------------|----------------------------------|
| UI              | Streamlit                        |
| Web scraping    | requests + BeautifulSoup         |
| Embeddings      | SentenceTransformers (MiniLM-L6) |
| Vector DB       | FAISS (CPU)                      |
| LLM             | Flan-T5-Base (HuggingFace)       |
| Hosting         | Streamlit Cloud (free)           |

## Notes

- First run downloads models (~500MB) — subsequent runs use cache
- Streamlit Cloud provides ~1GB RAM — sufficient for this stack
- No API keys or paid services required
- Works on any public website
