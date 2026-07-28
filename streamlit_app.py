import streamlit as st

from rag_graph import run_query

st.set_page_config(
    page_title="Agentic AI eBook Chatbot",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #1b2140 0%, #0e1120 45%, #0a0c16 100%);
        color: #e7e9f5;
    }

    .kv-hero {
        padding: 1.6rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(120deg, rgba(124,92,255,0.22), rgba(56,189,248,0.12));
        border: 1px solid rgba(148,133,255,0.35);
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px rgba(76,60,180,0.15);
    }
    .kv-hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.75rem;
        margin: 0 0 0.35rem 0;
        background: linear-gradient(90deg, #b6a4ff, #7dd3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kv-hero p {
        margin: 0;
        color: #a9adcf;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    .kv-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        margin-right: 0.4rem;
        margin-top: 0.6rem;
    }
    .kv-pill.high   { background: rgba(52,211,153,0.18); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
    .kv-pill.medium { background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
    .kv-pill.low    { background: rgba(248,113,113,0.18); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 0.4rem 0.2rem;
        margin-bottom: 0.6rem;
    }

    .kv-chunk {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #7c5cff;
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        color: #c7cae6;
    }
    .kv-chunk .kv-chunk-meta {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 0.78rem;
        color: #8f93c2;
        margin-bottom: 0.3rem;
    }

    section[data-testid="stSidebar"] {
        background: #0c0e1c;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 14px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("### 📘 About")
    st.markdown(
        "This bot answers **only** from Konverge AI's *Agentic AI for "
        "Executives* eBook. It retrieves the most relevant passages, "
        "grades them for actual relevance, and generates an answer "
        "strictly from what's left -- or tells you it isn't in the book."
    )
    st.markdown("---")
    st.markdown("### Confidence key")
    st.markdown(
        "<span class='kv-pill high'>HIGH</span> strong match to the book<br>"
        "<span class='kv-pill medium'>MEDIUM</span> partial / indirect match<br>"
        "<span class='kv-pill low'>LOW</span> weak or no match",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    """
    <div class="kv-hero">
        <h1>📘 Agentic AI eBook Chatbot</h1>
        <p>Ask questions about Konverge AI's <i>Agentic AI</i> eBook.
        Every answer is grounded strictly in the retrieved document text --
        if it's not in the book, the bot says so instead of guessing.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def confidence_badge(confidence: float) -> str:
    if confidence >= 0.45:
        css_class, label = "high", "High confidence"
    elif confidence >= 0.22:
        css_class, label = "medium", "Medium confidence"
    else:
        css_class, label = "low", "Low confidence"
    return f"<span class='kv-pill {css_class}'>{label}</span> <code>{confidence:.2f}</code>"


if "messages" not in st.session_state:
    st.session_state.messages = []  

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "📘"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("result"):
            result = msg["result"]
            st.markdown(confidence_badge(result["confidence"]), unsafe_allow_html=True)
            chunks = result["context_chunks"]
            with st.expander(f"Retrieved chunks ({len(chunks)})"):
                if not chunks:
                    st.caption("No chunks were retrieved.")
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(
                        f"""<div class="kv-chunk">
                        <div class="kv-chunk-meta">Chunk {i} · Page {chunk['page']} · score {chunk['score']:.3f}</div>
                        {chunk['text']}
                        </div>""",
                        unsafe_allow_html=True,
                    )

query = st.chat_input("Ask a question about the eBook...")

if query and query.strip():
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📘"):
        with st.spinner("Retrieving and generating..."):
            result = run_query(query)
        st.markdown(result["answer"])
        st.markdown(confidence_badge(result["confidence"]), unsafe_allow_html=True)
        chunks = result["context_chunks"]
        with st.expander(f"Retrieved chunks ({len(chunks)})"):
            if not chunks:
                st.caption("No chunks were retrieved.")
            for i, chunk in enumerate(chunks, start=1):
                st.markdown(
                    f"""<div class="kv-chunk">
                    <div class="kv-chunk-meta">Chunk {i} · Page {chunk['page']} · score {chunk['score']:.3f}</div>
                    {chunk['text']}
                    </div>""",
                    unsafe_allow_html=True,
                )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )