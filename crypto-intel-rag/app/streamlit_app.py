"""
Crypto Market Intelligence RAG — Streamlit Application
A multi-page AI assistant grounded in real crypto data using Endee vector DB.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="CryptoInsight RAG",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# Custom CSS — dark premium theme
# ─────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1527 50%, #0a1020 100%);
        min-height: 100vh;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #0a1628 100%);
        border-right: 1px solid #1e3a5f;
    }

    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a237e 0%, #0d47a1 40%, #006064 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 100, 200, 0.3);
        border: 1px solid rgba(100, 200, 255, 0.15);
    }
    .header-banner h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
        letter-spacing: -0.5px;
    }
    .header-banner p {
        color: rgba(200, 220, 255, 0.85);
        font-size: 1rem;
        margin: 0;
    }

    /* Status pills */
    .status-ok {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(0, 200, 100, 0.15);
        color: #00e676; border: 1px solid #00e676;
        border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; font-weight: 600;
    }
    .status-err {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255, 50, 50, 0.12);
        color: #ff5252; border: 1px solid #ff5252;
        border-radius: 20px; padding: 4px 14px; font-size: 0.8rem; font-weight: 600;
    }

    /* Answer card */
    .answer-card {
        background: rgba(13, 27, 42, 0.9);
        border: 1px solid rgba(100, 180, 255, 0.25);
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 100, 200, 0.15);
    }
    .answer-card h3 {
        color: #64b5f6;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 0 0.8rem 0;
    }
    .answer-text {
        color: #e8eaf6;
        font-size: 1rem;
        line-height: 1.7;
    }

    /* Source citation card */
    .source-card {
        background: rgba(26, 35, 126, 0.35);
        border: 1px solid rgba(100, 160, 255, 0.2);
        border-left: 3px solid #42a5f5;
        border-radius: 10px;
        padding: 0.75rem 1.2rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-left-color: #00e5ff;
        background: rgba(26, 35, 126, 0.5);
    }
    .source-title { color: #90caf9; font-weight: 600; font-size: 0.9rem; }
    .source-meta  { color: #78909c; font-size: 0.78rem; margin-top: 2px; }

    /* Chunk card */
    .chunk-card {
        background: rgba(10, 20, 40, 0.6);
        border: 1px solid rgba(60, 100, 160, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
        color: #b0bec5;
    }
    .chunk-card .sim-badge {
        display: inline-block;
        background: rgba(0, 150, 255, 0.2);
        color: #29b6f6;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-right: 8px;
    }

    /* Metric card */
    .metric-card {
        background: rgba(13, 27, 42, 0.8);
        border: 1px solid rgba(100, 180, 255, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 80, 180, 0.15);
    }
    .metric-value { color: #64ffda; font-size: 2rem; font-weight: 700; }
    .metric-label { color: #78909c; font-size: 0.8rem; margin-top: 4px; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1565c0, #0277bd) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.6rem 1.6rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1976d2, #0288d1) !important;
        box-shadow: 0 4px 16px rgba(0, 150, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Input */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: rgba(13, 27, 42, 0.9) !important;
        border: 1px solid rgba(100, 180, 255, 0.3) !important;
        border-radius: 10px !important;
        color: #e8eaf6 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #78909c !important;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #64b5f6 !important;
        border-bottom: 2px solid #64b5f6 !important;
    }

    /* Divider */
    hr { border-color: rgba(100, 180, 255, 0.1); }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def check_endee_status() -> bool:
    from vector_store.endee_client import health_check
    return health_check()


def check_api_key() -> bool:
    # Re-read directly from .env every time so changes are picked up without restart
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    key = os.getenv("GEMINI_API_KEY", "")
    return bool(key and key != "your_gemini_api_key_here")


def render_status_bar():
    endee_ok = check_endee_status()
    api_ok = check_api_key()

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if endee_ok:
            st.markdown('<span class="status-ok">⬤ Endee Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">⬤ Endee Offline</span>', unsafe_allow_html=True)
    with col2:
        if api_ok:
            st.markdown('<span class="status-ok">⬤ Gemini Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-err">⬤ No API Key</span>', unsafe_allow_html=True)
    with col3:
        if not endee_ok:
            st.caption("🐳 Start Endee: `docker run -p 8080:8080 endeeio/endee-server:latest`")
        if not api_ok:
            st.caption("🔑 Add `GEMINI_API_KEY=...` to your `.env` file")


# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0;">
            <div style="font-size:2.5rem;">🔮</div>
            <div style="color:#64b5f6; font-size:1.1rem; font-weight:700; margin-top:0.3rem;">CryptoInsight</div>
            <div style="color:#78909c; font-size:0.75rem;">RAG Intelligence System</div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🔍 Ask a Question", "📥 Ingest Documents", "📊 Evaluation"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="color:#546e7a; font-size:0.75rem; padding: 0.5rem;">
        <b style="color:#78909c;">How it works:</b><br><br>
        1️⃣ Ingest crypto news<br>
        2️⃣ Embeddings stored in Endee<br>
        3️⃣ Ask questions → RAG retrieves context<br>
        4️⃣ Gemini generates grounded answers<br>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    top_k = st.slider("Top-K results", min_value=1, max_value=10, value=5)
    st.caption("Number of chunks retrieved per query")


# ─────────────────────────────────────────
# Page 1: Ask a Question
# ─────────────────────────────────────────

if page == "🔍 Ask a Question":
    st.markdown(
        """
        <div class="header-banner">
            <h1>🔮 CryptoInsight RAG</h1>
            <p>Ask anything about crypto markets, DeFi, regulations, and whitepapers — grounded in real data</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_status_bar()
    st.markdown("<br>", unsafe_allow_html=True)

    # Suggested questions
    st.markdown("**💡 Suggested Questions:**")
    suggestions = [
        "What is happening with Bitcoin price?",
        "Explain the Ethereum Petra upgrade",
        "What is the state of DeFi TVL?",
        "Tell me about US crypto regulations",
        "What are the top altcoin gainers?",
    ]
    cols = st.columns(len(suggestions))
    query_prefill = ""
    for i, col in enumerate(cols):
        with col:
            if st.button(suggestions[i], key=f"sug_{i}", use_container_width=True):
                query_prefill = suggestions[i]

    st.markdown("<br>", unsafe_allow_html=True)

    # Query input
    query = st.text_input(
        "Your Question",
        value=query_prefill,
        placeholder="e.g. What is driving the Bitcoin rally in 2026?",
        key="query_input",
    )

    ask_col, clear_col = st.columns([1, 5])
    with ask_col:
        ask_clicked = st.button("Ask ✨", type="primary", use_container_width=True)

    if ask_clicked and query.strip():
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            from rag.generator import ask as rag_ask
            result = rag_ask(query.strip(), top_k=top_k)

        st.markdown("<br>", unsafe_allow_html=True)

        # Answer
        st.markdown(
            f"""
            <div class="answer-card">
                <h3>💬 Answer</h3>
                <div class="answer-text">{result['answer'].replace(chr(10), '<br>')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Model info
        model_used = result.get("model", "unknown")
        st.caption(f"Generated by: `{model_used}` · Retrieved: `{len(result.get('chunks', []))}` chunks")

        # Sources
        sources = result.get("sources", [])
        if sources:
            st.markdown("---")
            st.markdown("**📚 Sources:**")
            for s in sources:
                parts = s.split(" — ", 1)
                title = parts[1] if len(parts) > 1 else s
                meta = parts[0] if len(parts) > 1 else ""
                st.markdown(
                    f"""
                    <div class="source-card">
                        <div class="source-title">📄 {title}</div>
                        <div class="source-meta">{meta}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Retrieved chunks
        chunks = result.get("chunks", [])
        if chunks:
            with st.expander(f"🔬 View Retrieved Chunks ({len(chunks)})", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    sim = chunk.get("similarity", 0.0)
                    st.markdown(
                        f"""
                        <div class="chunk-card">
                            <span class="sim-badge">{sim:.3f}</span>
                            <b>{chunk.get('source', '?')}</b> · {chunk.get('date', '')}<br>
                            <span style="color:#90a4ae; font-size:0.8rem;">{chunk.get('text', '')[:300]}...</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    elif ask_clicked and not query.strip():
        st.warning("Please enter a question.")


# ─────────────────────────────────────────
# Page 2: Ingest Documents
# ─────────────────────────────────────────

elif page == "📥 Ingest Documents":
    st.markdown(
        """
        <div class="header-banner">
            <h1>📥 Document Ingestion</h1>
            <p>Load crypto news articles into Endee vector database</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_status_bar()
    st.markdown("<br>", unsafe_allow_html=True)

    # Show dataset info
    data_path = PROJECT_ROOT / "data" / "raw" / "crypto_news.json"
    if data_path.exists():
        import json
        with open(data_path) as f:
            articles = json.load(f)

        st.markdown("**📰 Available Dataset:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{len(articles)}</div>
                    <div class="metric-label">Articles</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            categories = list({a.get("category", "?") for a in articles})
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{len(categories)}</div>
                    <div class="metric-label">Categories</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col3:
            total_words = sum(len(a.get("content", "").split()) for a in articles)
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{total_words:,}</div>
                    <div class="metric-label">Total Words</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Article list
        with st.expander("📋 View Articles", expanded=False):
            for a in articles:
                st.markdown(
                    f"""
                    <div class="chunk-card">
                        <b style="color:#90caf9;">{a.get('title', 'No title')}</b><br>
                        <span style="color:#78909c; font-size:0.8rem;">
                            📰 {a.get('source')} · 📅 {a.get('date')} · 🏷️ {a.get('category')}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### ⚡ Run Ingestion Pipeline")
    st.markdown(
        """
        <div style="color:#78909c; font-size:0.9rem; margin-bottom:1rem;">
        This will: <br>
        1. Create the Endee index (if needed)<br>
        2. Chunk articles with overlapping windows<br>
        3. Embed with <code>all-MiniLM-L6-v2</code><br>
        4. Upsert vectors into Endee
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🚀 Run Ingestion", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def ui_progress(step, total, msg):
            progress_bar.progress(step / total)
            status_text.markdown(f"**Step {step}/{total}:** {msg}")
            time.sleep(0.2)

        from ingestion.ingest_news import run_ingestion
        result = run_ingestion(progress_callback=ui_progress)

        progress_bar.progress(1.0)

        if result["status"] == "success":
            st.success(
                f"✅ Ingestion complete! "
                f"{result['articles']} articles → {result['chunks']} chunks stored in Endee."
            )
            st.balloons()
        else:
            st.error(f"❌ Ingestion failed:\n\n{result.get('error', 'Unknown error')}")


# ─────────────────────────────────────────
# Page 3: Evaluation
# ─────────────────────────────────────────

elif page == "📊 Evaluation":
    st.markdown(
        """
        <div class="header-banner">
            <h1>📊 Evaluation Dashboard</h1>
            <p>Measure retrieval quality: Precision@K, relevance scores, and more</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_status_bar()
    st.markdown("<br>", unsafe_allow_html=True)

    # Metrics explanation
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        | Metric | Description |
        |--------|-------------|
        | **Precision@K** | Fraction of top-K results that are relevant (ground truth) |
        | **Relevance Score** | Keyword overlap between query and retrieved chunks |
        | **Similarity** | Cosine similarity score from Endee (0–1) |
        """)

    top_k_eval = st.slider("K for Precision@K", 1, 10, 5, key="eval_k")

    if st.button("▶️ Run Evaluation Suite", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def eval_progress(step, total, msg):
            progress_bar.progress(step / total)
            status_text.markdown(f"**Testing {step}/{total}:** {msg}")

        from evaluation.metrics import run_evaluation
        report = run_evaluation(top_k=top_k_eval, progress_callback=eval_progress)

        progress_bar.progress(1.0)
        status_text.empty()

        st.markdown("---")
        st.markdown("### 📈 Aggregate Results")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{report['avg_precision_at_k']:.2%}</div>
                    <div class="metric-label">Avg Precision@{top_k_eval}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mc2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{report['avg_relevance_score']:.2%}</div>
                    <div class="metric-label">Avg Relevance Score</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with mc3:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{report['successful']}/{report['total_queries']}</div>
                    <div class="metric-label">Queries Successful</div>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔬 Per-Query Results")

        import pandas as pd
        df = pd.DataFrame(report["results"])
        df = df[["query", "precision_at_k", "relevance_score", "top_source", "top_similarity", "status"]]
        df.columns = ["Query", "Precision@K", "Relevance", "Top Source", "Similarity", "Status"]
        df["Precision@K"] = df["Precision@K"].apply(lambda x: f"{x:.2%}")
        df["Relevance"] = df["Relevance"].apply(lambda x: f"{x:.2%}")
        df["Similarity"] = df["Similarity"].apply(lambda x: f"{x:.3f}")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Visual chart
        try:
            import plotly.express as px
            raw_df = pd.DataFrame(report["results"])
            fig = px.bar(
                raw_df,
                x="query",
                y=["precision_at_k", "relevance_score"],
                barmode="group",
                title="Precision@K vs Relevance Score per Query",
                color_discrete_map={"precision_at_k": "#42a5f5", "relevance_score": "#26c6da"},
                labels={"value": "Score", "variable": "Metric"},
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="rgba(10,20,40,0.5)",
                paper_bgcolor="rgba(10,20,40,0.0)",
                font_color="#b0bec5",
                xaxis_tickangle=-20,
                legend_title_text="",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig.update_xaxes(ticktext=[q[:30] + "..." for q in raw_df["query"]], tickvals=raw_df["query"])
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            pass
