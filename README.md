# RAG with Cross-Encoder Reranking

A Retrieval-Augmented Generation (RAG) app that answers questions about a text document (`paracetamol.txt`). It retrieves candidate chunks with FAISS, reranks them with a cross-encoder, and generates the final answer from only the top-ranked chunks.

**Live demo:** <PASTE YOUR STREAMLIT APP URL HERE>

## How it works

1. **Load and split:** the document is split into 500-character chunks with 100 overlap.
2. **Embed and index:** chunks are embedded with `sentence-transformers/all-MiniLM-L6-v2` and stored in a FAISS index.
3. **Retrieve:** the 20 chunks closest to the question are fetched.
4. **Rerank:** `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each (question, chunk) pair, and the top 5 are kept.
5. **Generate:** an LLM answers using only those 5 chunks as context.

Why rerank? Vector search is fast but approximate. A cross-encoder reads the question and chunk together, so it judges relevance more accurately, and it is only run on the 20 candidates to keep it cheap.

## Tech stack

- Python, Streamlit
- LangChain (community, huggingface, text-splitters, groq)
- FAISS (`faiss-cpu`)
- Sentence-Transformers (embeddings and cross-encoder)
- Groq API (`llama-3.3-70b-versatile`) for generation

## Run locally

1. Clone the repo and install the packages in `requirements.txt`.
2. Create `.streamlit/secrets.toml` containing:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
   You can get a free key at [console.groq.com](https://console.groq.com).
3. Start the app with `streamlit run app.py`.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app with the full RAG and reranking pipeline |
| `paracetamol.txt` | Source document the app answers from |
| `requirements.txt` | Python dependencies |

## Notes

- The original version of this project ran the LLM locally with Ollama (`qwen2.5:7b`). It was switched to Groq so the app can run on Streamlit Community Cloud, which cannot host Ollama.
- Never commit your API key. Keep it in `.streamlit/secrets.toml` locally and in the app's Secrets settings on Streamlit Cloud.
