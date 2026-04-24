# NewsLens 

An AI-powered news research tool built with LangChain, FAISS, and Streamlit. Paste up to 5 article URLs, build a local RAG index, and ask natural-language questions with source citations.

## Architecture

```
URLs → UnstructuredURLLoader → RecursiveCharacterTextSplitter
     → OpenAIEmbeddings → FAISS (persisted to disk)
     → RetrievalQAWithSourcesChain (GPT-4o-mini) → Answer + Sources
```

## Setup

**1. Clone & install dependencies**
```bash
git clone <your-repo>
cd news-research
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**3. Run**
```bash
streamlit run main.py
```

## Usage

1. Paste 1–5 article URLs into the sidebar
2. Click ** Process URLs** — this loads, chunks, embeds, and indexes the content
3. Type a question in the main input
4. Get an answer with source citations

The FAISS index is persisted to `faiss_index/` so you don't need to re-process URLs between sessions.

## Key Improvements Over v1

| Area | Before | After |
|------|--------|-------|
| Dependencies | `langchain==0.0.284` (2023) | `langchain>=0.3.0` (current) |
| LLM | `OpenAI` (legacy) | `ChatOpenAI` (gpt-4o-mini) |
| Persistence | `pickle` (insecure) | `FAISS.save_local()` |
| Error handling | None | Try/except on all I/O |
| URL validation | None | http/https check |
| Session state | None | Full Streamlit session |
| History | None | Per-session Q&A history |
| Config | Hardcoded | Constants + `.env` |
| Cross-platform | Windows-only deps | Clean, portable deps |

## Project Structure

```
news-research/
├── main.py              # Application entrypoint
├── requirements.txt     # Pinned, cross-platform deps
├── .env.example         # Environment variable template
├── .gitignore           # Excludes secrets + generated files
├── README.md            # This file
└── faiss_index/         # Generated at runtime — gitignored
```

## Requirements

- Python 3.10+
- OpenAI API key
