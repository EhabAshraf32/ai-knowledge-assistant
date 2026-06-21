# 🧠 AI Knowledge Assistant — RAG-Powered Conversational AI

A Retrieval-Augmented Generation (RAG) chatbot that answers questions grounded in a custom knowledge base spanning 15 core AI/ML domains, with full source attribution and conversational memory.

Built with **LangChain**, **Hugging Face Transformers**, **ChromaDB**, and **Qwen2.5-1.5B-Instruct**.

---

## Overview

AI Knowledge Assistant is an end-to-end RAG application that ingests documents (PDFs and Wikipedia articles), indexes them in a vector database, and uses a large language model to answer user questions strictly from that retrieved context — citing its sources and explicitly declining to answer when the knowledge base doesn't contain the information.

The project was built to explore the full RAG stack hands-on: document loading, chunking strategy, embeddings, vector search, prompt engineering, conversational memory, and grounded generation.

---

## Features

- **Multi-source ingestion** — loads documents from local PDFs (`PyPDFDirectoryLoader`) and live Wikipedia articles (`WikipediaLoader`) across 15 AI/ML topics (Machine Learning, Deep Learning, NLP, Computer Vision, RAG, MLOps, Transformers, Prompt Engineering, Agentic AI, and more)
- **Semantic chunking** — splits documents using `NLTKTextSplitter` driven by a Hugging Face tokenizer, balancing chunk size and overlap for retrieval quality
- **Dense vector retrieval** — embeds chunks with `sentence-transformers/all-MiniLM-L6-v2` and stores them in a persisted **ChromaDB** vector store
- **Grounded generation** — answers are generated only from retrieved context via a custom prompt template; the assistant explicitly responds `NO ANSWER IS AVAILABLE` rather than hallucinating
- **Conversational memory** — maintains chat history across turns using LangChain's `ConversationBufferMemory`
- **Source attribution** — every answer returns the originating document(s) so users can verify the response
- **Interactive CLI** — simple terminal loop for asking questions in real time

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Data Sources       │
                    │  PDFs + Wikipedia    │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  NLTK Text Splitter   │
                    │ (HF tokenizer-based)  │
                    │  chunk_size=300       │
                    │  overlap=50           │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Embedding Model      │
                    │ all-MiniLM-L6-v2      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   ChromaDB            │
                    │  (persisted vector    │
                    │   store)              │
                    └──────────┬───────────┘
                               │  similarity_search (k=5)
                    ┌──────────▼───────────┐
        User Query →│  Stuff QA Chain       │
                    │  + Prompt Template     │
                    │  + Chat History        │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │   LLM                 │
                    │  Qwen2.5-1.5B-Instruct │
                    │  (HF pipeline)         │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Answer + Sources      │
                    └───────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain (`langchain`, `langchain-community`, `langchain-classic`) |
| LLM | Qwen2.5-1.5B-Instruct (Hugging Face `transformers`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (`langchain-huggingface`) |
| Vector Store | ChromaDB (`langchain-chroma`) |
| Text Splitting | NLTK + Hugging Face Tokenizer |
| Document Loaders | `PyPDFDirectoryLoader`, `WikipediaLoader` |
| Memory | `ConversationBufferMemory` |
| Language | Python 3.12 |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-knowledge-assistant.git
cd ai-knowledge-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the NLTK tokenizer data
python -m nltk.downloader punkt
```

### Environment Variables

This project authenticates with the Hugging Face Hub. **Never hard-code your token in the notebook or source files.** Create a `.env` file in the project root:

```bash
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Then load it in code with `python-dotenv` instead of pasting the token directly:

```python
from dotenv import load_dotenv
load_dotenv()
```

> ⚠️ If you ever committed a real token to version control, treat it as compromised — revoke it immediately from your [Hugging Face token settings](https://huggingface.co/settings/tokens) and generate a new one.

---

## Usage Guide

1. **Add your documents** (optional) — place PDFs in `data/pdfs/` to be indexed alongside the Wikipedia sources.
2. **Run the notebook or script** to build the vector store:
   ```bash
   jupyter notebook notebooks/ai_knowledge_assistant.ipynb
   ```
3. **Ask questions** via the interactive CLI loop:
   ```text
   what do you want to ask about: What is Retrieval-Augmented Generation?

   Assistant:
   Retrieval-Augmented Generation (RAG) is an approach that combines a
   retrieval system with a generative language model, allowing the model
   to ground its responses in external, up-to-date information rather
   than relying solely on parameters learned during training...

   Sources:
   - Retrieval-Augmented_Generation
   ```
4. Type `exit`, `quit`, or `bye` to end the session.

---

## Example Questions

- "What is the difference between machine learning and deep learning?"
- "Explain how a transformer architecture works."
- "What is prompt engineering and why does it matter for LLMs?"
- "How does Retrieval-Augmented Generation reduce hallucinations?"
- "What is MLOps?"

## Example Output

```text
what do you want to ask about: What is machine learning?

Assistant:
Machine learning is a field of study in artificial intelligence concerned
with the development and study of statistical algorithms that can learn
from data and generalize to unseen data. It involves creating models that
can make predictions or decisions without explicit programming. Advances
in deep learning have particularly improved performance in this area.

Sources:
- Artificial_intelligence
- Machine_Learning
```

---

## Screenshots

> Add terminal screenshots or a short demo GIF of the assistant answering questions here once available — this section significantly boosts a project's credibility for recruiters skimming GitHub.

---

## Future Improvements

- [ ] Swap the CLI for a web UI (Streamlit or Gradio) for easier demoing
- [ ] Add streaming token-by-token responses
- [ ] Replace `ConversationBufferMemory` with summarized/windowed memory to control context length on long sessions
- [ ] Add evaluation harness (retrieval precision/recall, answer faithfulness) using a framework like RAGAS
- [ ] Containerize with Docker for reproducible deployment
- [ ] Add unit tests for the ingestion, chunking, and retrieval pipeline
- [ ] Support hybrid search (keyword + semantic) for improved retrieval on exact-match queries
- [ ] Deploy as a hosted API (FastAPI) with a simple authentication layer

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**Ehab Ashraf Mohamed**
AI & Machine Learning Engineer

- 📧 ehab44221@gmail.com
- 💼 [LinkedIn](https://www.linkedin.com/public-profile/settings/?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_self_edit_contact_info%3Bv6i0SdM2SzmhBZg%2FDfsqMg%3D%3D)
- 🐙 [GitHub](https://github.com/EhabAshraf32)
- 📊 [Kaggle](https://kaggle.com/ehabashraf)

If you found this project useful, consider giving it a ⭐ on GitHub!
