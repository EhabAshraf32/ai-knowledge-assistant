"""
AI Knowledge Assistant — Streamlit App
Pipeline loads automatically on startup.
User picks chunk_size and chunk_overlap from the sidebar.
"""

import torch
import streamlit as st

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg,#667eea,#764ba2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.chat-user {
    background:#f3f4f6; border-radius:12px 12px 4px 12px;
    padding:10px 14px; margin:6px 0; max-width:80%;
    margin-left:auto; font-size:.95rem;
}
.chat-bot {
    background:#eef2ff; border-radius:12px 12px 12px 4px;
    padding:10px 14px; margin:6px 0; max-width:80%;
    font-size:.95rem; border-left:4px solid #667eea;
}
.src-tag {
    display:inline-block; background:#dcfce7; color:#15803d;
    border-radius:6px; padding:2px 8px; font-size:.75rem; margin:2px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 AI Knowledge Assistant</p>', unsafe_allow_html=True)
st.caption("RAG over 30 Wikipedia AI documents · exact notebook pipeline")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Chunking Settings")

    chunk_size = st.slider(
        "Chunk Size (tokens)",
        min_value=100, max_value=800, value=300, step=50,
        help="Number of tokens per chunk (notebook default: 300)"
    )
    chunk_overlap = st.slider(
        "Chunk Overlap (tokens)",
        min_value=0, max_value=200, value=50, step=10,
        help="Overlap between consecutive chunks (notebook default: 50)"
    )

    apply = st.button("✅ Apply & Rebuild", type="primary", use_container_width=True)

    st.markdown("---")
    hf_token = st.text_input(
        "HuggingFace Token",
        type="password",
        placeholder="hf_xxxxxxxxxx",
    )
    chroma_dir = st.text_input("Chroma directory", value="./chroma_ai_kb")

    st.markdown("---")
    st.subheader("📚 Topics (30 docs)")
    topics = [
        "Artificial Intelligence","Machine Learning","Deep Learning",
        "Generative Artificial Intelligence","Large Language Model",
        "Natural Language Processing","Computer Vision",
        "Reinforcement Learning","Retrieval-Augmented Generation",
        "MLOps","Data Science","Neural Network",
        "Transformer (deep learning architecture)",
        "Prompt Engineering","Agentic AI",
    ]
    for t in topics:
        st.markdown(f"- {t}")

# ── Session state ─────────────────────────────────────────────────────────────
for key in ("pipeline_ready","vector_db","stuff_chain","memory",
            "chat_history","built_chunk","built_overlap"):
    if key not in st.session_state:
        st.session_state[key] = None if key not in (
            "pipeline_ready","chat_history"
        ) else (False if key == "pipeline_ready" else [])

# ── Build pipeline ────────────────────────────────────────────────────────────
def build_pipeline(token, persist_dir, c_size, c_overlap):
    import wikipedia
    import nltk
    from pathlib import Path
    from langchain_community.document_loaders import WikipediaLoader
    from langchain_text_splitters import NLTKTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipe
    from langchain_chroma import Chroma
    from langchain_classic.chains.question_answering import load_qa_chain
    from langchain_core.prompts import PromptTemplate
    from langchain_classic.memory import ConversationBufferMemory

    wikipedia.set_user_agent("MyLangChainBot/1.0 (user@example.com)")

    # 1. Load docs
    with st.spinner("📥 Loading 30 Wikipedia documents …"):
        docs = []
        for topic in topics:
            loader = WikipediaLoader(query=topic, load_max_docs=2,
                                     doc_content_chars_max=100_000)
            docs.extend(loader.load())

    documents = [d.page_content for d in docs]
    metadatas = [{"file_name": Path(d.metadata.get("source","Unknown")).stem}
                 for d in docs]

    # 2. Split
    with st.spinner(f"✂️ Splitting — chunk_size={c_size}, overlap={c_overlap} …"):
        nltk.download("punkt",     quiet=True)
        nltk.download("punkt_tab", quiet=True)
        splitter_tok = AutoTokenizer.from_pretrained(
            "stabilityai/stablelm-tuned-alpha-3b", token=token)
        splitter = NLTKTextSplitter.from_huggingface_tokenizer(
            splitter_tok, chunk_size=c_size, chunk_overlap=c_overlap)
        chunks = splitter.create_documents(documents, metadatas=metadatas)

    # 3. Embeddings
    with st.spinner("🔢 Building embeddings …"):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Chroma
    with st.spinner("🗄️ Creating Chroma vector DB …"):
        vector_db = Chroma.from_documents(
            chunks, embeddings, persist_directory=persist_dir)

    # 5. LLM
    with st.spinner("🧠 Loading Qwen/Qwen2.5-1.5B-Instruct …"):
        llm_tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct", token=token)
        llm_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        pipe = hf_pipe(
            "text-generation",
            model=llm_model, tokenizer=llm_tok,
            max_new_tokens=500, temperature=0.9,
            return_full_text=False,
            pad_token_id=llm_tok.eos_token_id,
        )
        llm = HuggingFacePipeline(pipeline=pipe)

    # 6. Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=False)

    # 7. Prompt + chain
    template = """
You are an AI Knowledge Assistant.
Use the conversation history and provided context to answer the user's question.
If the answer cannot be found in the context, respond exactly:
NO ANSWER IS AVAILABLE

====================
Conversation History:
{chat_history}

====================
Context:
{context}

====================
Question:
{question}

====================
Answer:
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["chat_history","context","question"],
        verbose=True,
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return vector_db, chain, memory


def ask_question(question):
    vdb    = st.session_state.vector_db
    chain  = st.session_state.stuff_chain
    mem    = st.session_state.memory

    similar_docs = vdb.similarity_search(question, k=5)
    chat_history = mem.buffer

    result = chain.invoke({
        "input_documents": similar_docs,
        "question": question,
        "chat_history": chat_history,
    })
    answer = result["output_text"]

    mem.save_context({"input": question}, {"output": answer})

    sources = sorted({
        doc.metadata.get("file_name","Unknown") for doc in similar_docs
    })
    return answer, sources


# ── Auto-build on first run OR when user clicks Apply ────────────────────────
needs_build = (
    not st.session_state.pipeline_ready
    or apply
    or st.session_state.built_chunk   != chunk_size
    or st.session_state.built_overlap != chunk_overlap
)

if needs_build:
    if not hf_token:
        st.warning("👈 Enter your HuggingFace token in the sidebar to get started.")
        st.stop()
    try:
        vdb, chain, mem = build_pipeline(
            hf_token, chroma_dir, chunk_size, chunk_overlap)
        st.session_state.vector_db      = vdb
        st.session_state.stuff_chain    = chain
        st.session_state.memory         = mem
        st.session_state.pipeline_ready = True
        st.session_state.built_chunk    = chunk_size
        st.session_state.built_overlap  = chunk_overlap
        st.session_state.chat_history   = []   # reset chat on rebuild
        st.success(f"✅ Ready — chunk_size={chunk_size}, overlap={chunk_overlap}")
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# ── Chat UI ───────────────────────────────────────────────────────────────────
st.markdown("---")

for role, text, sources in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="chat-user">👤 {text}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">🤖 {text}</div>',
                    unsafe_allow_html=True)
        if sources:
            tags = "".join(f'<span class="src-tag">📄 {s}</span>' for s in sources)
            st.markdown(f"<small>Sources: {tags}</small>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input(
        "Ask anything about AI …",
        label_visibility="collapsed",
        placeholder="What is a Transformer model?",
    )
    sent = st.form_submit_button("Send ➤", type="primary")

if sent and user_input.strip():
    st.session_state.chat_history.append(("user", user_input, []))
    with st.spinner("Thinking …"):
        try:
            answer, sources = ask_question(user_input)
        except Exception as e:
            answer, sources = f"⚠️ {e}", []
    st.session_state.chat_history.append(("assistant", answer, sources))
    st.rerun()

if st.session_state.chat_history:
    if st.button("🗑️ Clear chat"):
        from langchain_classic.memory import ConversationBufferMemory
        st.session_state.chat_history = []
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=False)
        st.rerun()

st.markdown("---")
st.caption("LangChain · Chroma · HuggingFace · Streamlit — exact notebook pipeline")
