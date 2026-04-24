import os
import streamlit as st
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FAISS_INDEX_PATH = Path("faiss_index")
MAX_URLS = 5
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NewsLens — AI Research Tool",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0f;
    color: #e8e8f0;
  }

  .stApp { background-color: #0a0a0f; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
  }

  h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800;
    letter-spacing: -0.02em;
  }

  /* Inputs */
  .stTextInput > div > div > input {
    background: #13131f !important;
    border: 1px solid #2a2a3e !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
  }
  .stTextInput > div > div > input:focus {
    border-color: #6c63ff !important;
    box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #6c63ff, #a855f7) !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(108,99,255,0.4) !important;
  }

  /* Answer card */
  .answer-card {
    background: #13131f;
    border: 1px solid #2a2a3e;
    border-left: 3px solid #6c63ff;
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    font-family: 'DM Mono', monospace;
    line-height: 1.7;
  }

  /* Source tag */
  .source-tag {
    display: inline-block;
    background: #1a1a2e;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.75rem;
    color: #a78bfa;
    font-family: 'DM Mono', monospace;
  }

  /* Status messages */
  .status-msg {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #6c63ff;
    padding: 8px 0;
  }

  /* Hero */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #e8e8f0 0%, #a78bfa 50%, #6c63ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 8px;
  }

  .hero-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    color: #4a4a6a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  /* Divider */
  hr { border-color: #1e1e2e !important; }

  /* Metric cards */
  .metric-row {
    display: flex;
    gap: 12px;
    margin: 16px 0;
  }
  .metric-card {
    background: #13131f;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    text-align: center;
  }
  .metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #a78bfa;
  }
  .metric-label {
    font-size: 0.7rem;
    color: #4a4a6a;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_urls(urls: list[str]) -> list[str]:
    """Return only non-empty, http(s) URLs."""
    valid = []
    for u in urls:
        u = u.strip()
        if u and u.startswith(("http://", "https://")):
            valid.append(u)
    return valid


def build_vector_store(urls: list[str], status) -> FAISS | None:
    """Load URLs → split → embed → persist FAISS index. Returns store or None."""
    try:
        status.markdown('<p class="status-msg">⬇ Loading articles…</p>', unsafe_allow_html=True)
        loader = UnstructuredURLLoader(urls=urls)
        docs = loader.load()
        if not docs:
            st.error("No content could be loaded from the provided URLs.")
            return None

        status.markdown('<p class="status-msg">✂ Splitting text into chunks…</p>', unsafe_allow_html=True)
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", ", "],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)
        logger.info("Split into %d chunks", len(chunks))

        status.markdown('<p class="status-msg">🧠 Generating embeddings…</p>', unsafe_allow_html=True)
        embeddings = OpenAIEmbeddings()
        store = FAISS.from_documents(chunks, embeddings)

        status.markdown('<p class="status-msg">💾 Saving index to disk…</p>', unsafe_allow_html=True)
        store.save_local(str(FAISS_INDEX_PATH))

        # Store metadata in session
        st.session_state["doc_count"] = len(docs)
        st.session_state["chunk_count"] = len(chunks)
        st.session_state["urls_indexed"] = urls

        return store

    except Exception as e:
        logger.exception("Failed to build vector store")
        st.error(f"Error processing URLs: {e}")
        return None


def load_vector_store() -> FAISS | None:
    """Load persisted FAISS index from disk."""
    try:
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(
            str(FAISS_INDEX_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.warning("Could not load vector store: %s", e)
        return None


def query_store(store: FAISS, question: str) -> dict | None:
    """Run RAG chain and return result dict."""
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=800)
        retriever = store.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in docs])
        sources = "\n".join(set([
            d.metadata.get("source", "")
            for d in docs
            if d.metadata.get("source")
        ]))

        from langchain_core.messages import HumanMessage, SystemMessage
        messages = [
            SystemMessage(content=f"Answer the question based only on the context below.\n\nContext:\n{context}"),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        return {"answer": response.content, "sources": sources}

    except Exception as e:
        logger.exception("Query failed")
        st.error(f"Query failed: {e}")
        return None


# ── Session state init ────────────────────────────────────────────────────────
for key, default in {
    "doc_count": 0,
    "chunk_count": 0,
    "urls_indexed": [],
    "history": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 NewsLens")
    st.markdown('<p class="hero-sub">URL Ingestion Panel</p>', unsafe_allow_html=True)
    st.markdown("---")

    urls = []
    for i in range(MAX_URLS):
        url = st.text_input(f"URL {i + 1}", placeholder="https://...", key=f"url_{i}")
        urls.append(url)

    process_clicked = st.button("⚡ Process URLs", use_container_width=True)

    if st.session_state["urls_indexed"]:
        st.markdown("---")
        st.markdown("### Index Stats")
        st.markdown(f"""
        <div class="metric-card" style="margin:4px 0">
          <div class="metric-val">{st.session_state['doc_count']}</div>
          <div class="metric-label">Documents</div>
        </div>
        <div class="metric-card" style="margin:4px 0">
          <div class="metric-val">{st.session_state['chunk_count']}</div>
          <div class="metric-label">Chunks</div>
        </div>
        <div class="metric-card" style="margin:4px 0">
          <div class="metric-val">{len(st.session_state['urls_indexed'])}</div>
          <div class="metric-label">URLs Indexed</div>
        </div>
        """, unsafe_allow_html=True)

    if not os.getenv("OPENAI_API_KEY"):
        st.markdown("---")
        st.warning("⚠ OPENAI_API_KEY not found in environment. Add it to your .env file.")


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">NewsLens</div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">AI-powered news research · RAG pipeline · GPT-4o-mini</p>', unsafe_allow_html=True)
st.markdown("---")

status_placeholder = st.empty()

# Process URLs
if process_clicked:
    valid_urls = validate_urls(urls)
    if not valid_urls:
        st.error("Please enter at least one valid URL (must start with http:// or https://)")
    else:
        with st.spinner(""):
            store = build_vector_store(valid_urls, status_placeholder)
        if store:
            status_placeholder.success(f"✅ Indexed {len(valid_urls)} URL(s) successfully — {st.session_state['chunk_count']} chunks ready.")
            st.session_state["store"] = store

# Query section
st.markdown("### Ask a Question")
query = st.text_input("", placeholder="What are analysts saying about Fed interest rate cuts?", label_visibility="collapsed")

if query:
    # Load from session or disk
    store = st.session_state.get("store") or load_vector_store()

    if not store:
        st.warning("No index found. Please process some URLs first.")
    else:
        with st.spinner("Thinking…"):
            result = query_store(store, query)

        if result:
            st.session_state["history"].append({"q": query, "r": result})
            st.markdown("### Answer")
            st.markdown(f'<div class="answer-card">{result.get("answer", "No answer returned.")}</div>', unsafe_allow_html=True)

            sources = result.get("sources", "").strip()
            if sources:
                st.markdown("**Sources**")
                for src in sources.split("\n"):
                    if src.strip():
                        st.markdown(f'<span class="source-tag">🔗 {src.strip()}</span>', unsafe_allow_html=True)

# Q&A History
if st.session_state["history"]:
    st.markdown("---")
    with st.expander(f"📜 Session History ({len(st.session_state['history'])} questions)"):
        for i, item in enumerate(reversed(st.session_state["history"]), 1):
            st.markdown(f"**Q{i}:** {item['q']}")
            st.markdown(f"<small>{item['r'].get('answer','')[:200]}…</small>", unsafe_allow_html=True)
            st.markdown("---")