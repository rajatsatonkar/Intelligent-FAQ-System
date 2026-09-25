# Intelligent FAQ System

A semantic-search FAQ bot: given a natural-language question, it retrieves the most relevant Q&A pair from a knowledge base by embedding similarity, then generates a grounded answer with an LLM — a small, early RAG pipeline (predates and complements the larger [RAG over SEC 10-K Filings](https://github.com/rajatsatonkar/agentic_rag_sec_filings) project).

## How it works

1. **Ingest** — `faqs.csv` (question/answer pairs) is loaded and embedded with `sentence-transformers/all-MiniLM-L6-v2`, then indexed in a local **FAISS** vector store.
2. **Retrieve** — an incoming question is embedded and matched against the FAISS index for the closest stored Q&A pair.
3. **Generate** — the retrieved context is passed to `google/flan-t5-large` (via the Hugging Face Hub) through a LangChain `RetrievalQA` chain, with a prompt that instructs the model to answer only from the retrieved context and say "I don't know" otherwise.
4. **Serve** — a Streamlit UI (`main.py`) lets a user build the knowledge base and ask questions interactively.

## Repository layout

- `langchain_helper.py` — vector store creation (`create_vector_db`) and the retrieval-QA chain (`get_qa_chain`).
- `main.py` — Streamlit app.
- `faqs.csv` — sample FAQ dataset (bootcamp-support Q&A pairs, used here as demo data).
- `index.faiss` / `index.pkl` — the pre-built FAISS index (regenerate any time with `create_vector_db()`).

## Running it

```bash
pip install -r requirements.txt
```

Create a `.env` file with a Hugging Face access token:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

Then launch the app:
```bash
streamlit run main.py
```
Click **Create Knowledgebase** once to build the FAISS index, then ask questions.

## Tech stack

Python, LangChain, FAISS, Hugging Face (Sentence Transformers, Flan-T5), Streamlit

## Author

Rajat Satonkar — rajatsatonkar@gmail.com
