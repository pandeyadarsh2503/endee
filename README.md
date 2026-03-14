# Crypto Market Intelligence RAG 🔮

> An AI assistant grounded in real crypto data — powered by **Endee** vector database, **Google Gemini 2.5 Flash**, and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![Endee](https://img.shields.io/badge/VectorDB-Endee-green)
![Gemini](https://img.shields.io/badge/LLM-Gemini_2.5_Flash-orange)

---

## 📖 Problem Statement

In the fast-paced cryptocurrency market, traders and analysts are overwhelmed by the sheer volume of news, reports, and data. Traditional keyword searches are often inaccurate, and raw Large Language Models (LLMs) frequently "hallucinate" wrong financial data.

**The Solution:** This project implements an advanced **Retrieval-Augmented Generation (RAG)** pipeline.
1. It ingests thousands of crypto news articles.
2. It embeds them using local HuggingFace sentence-transformers.
3. It stores them in **Endee**, a highly-efficient vector database.
4. When a user asks a question, it retrieves only the most factually relevant articles using semantic vector search.
5. It feeds those articles to **Google Gemini 2.5 Flash**, forcing the AI to generate a precise answer grounded *strictly* in the provided text, complete with source citations.

---

## 🏗 Architecture

```
User Query
    ↓
Query Embedding (sentence-transformers/all-MiniLM-L6-v2, runs locally)
    ↓
Endee Vector Search (cosine similarity, FLOAT32 precision via Endee Python SDK)
    ↓
Top-K Relevant Chunks
    ↓
Gemini 2.5 Flash (Grounded Prompt via google-genai SDK)
    ↓
Final Answer + Inline Source Citations
```

---

## ⚡ Quick Start Guide (How to Run)

Follow these exact steps to run the complete pipeline on your local machine.

### 1. Prerequisites
- Python 3.10 or 3.11
- Docker (required to run the Endee Vector Database)
- A free Google Gemini API key ([Get one here](https://aistudio.google.com/))

### 2. Start the Endee Vector Database
Endee is a lightning-fast vector database that runs in a Docker container. Start it up:

```bash
docker run -d -p 8080:8080 --name endee-server endeeio/endee-server:latest
```

Verify it's running by checking its health endpoint:
```bash
curl http://localhost:8080/api/v1/health
```

### 3. Setup the Python Environment
Clone this repository and navigate to the project directory:

```bash
git clone <your-repo-link>
cd crypto-intel-rag

# (Optional but recommended) Create a virtual environment
python -m venv venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Install all required Python packages
pip install -r requirements.txt
```

### 4. Configure Your Environment Variables
Open the `.env` file in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 5. Start the Application
Run the Streamlit application:

```bash
streamlit run app/streamlit_app.py
```

### 6. 📥 Ingest Data (Crucial First Step!)
When the Web UI opens at `http://localhost:8501`:
1. Navigate to the **"📥 Ingest Documents"** tab on the left sidebar.
2. Click **"Run Ingestion"**.
3. Wait for the pipeline to chunk, embed, and upsert the crypto articles into your Endee database. (This usually takes < 1 minute for 500 articles).
4. Once completed, navigate back to the **"💬 Ask a Question"** tab and start talking to your data!

*(If you don't ingest data first, the database will be empty and the AI will reply "Not found in knowledge base".)*

---

## 🐳 Docker Compose (Alternative Full Stack)

If you prefer to run both the Endee database AND the Streamlit app together using Docker Compose:

```bash
# From the root of the project
docker-compose up --build -d
```
The app will be available at `http://localhost:8501`.

---

## 📂 Project Structure

```
crypto-intel-rag/
│
├── data/
│   ├── raw/               ← Sample crypto news JSON articles
│   └── processed/         ← Auto-generated chunks
│
├── ingestion/
│   └── ingest_news.py     ← Chunking + embedding pipeline
│
├── embeddings/
│   └── embedder.py        ← Local sentence-transformers (all-MiniLM-L6-v2)
│
├── vector_store/
│   └── endee_client.py    ← Endee Python SDK wrapper for create_index, upsert, and query
│
├── rag/
│   ├── retriever.py       ← Fetches relevant context from Endee
│   ├── prompt_template.py ← Enforces grounding and citations
│   └── generator.py       ← Connects to Gemini 2.5 Flash
│
├── evaluation/
│   └── metrics.py         ← Calculates Precision@K and retrieval relevance
│
├── app/
│   └── streamlit_app.py   ← Streamlit UI frontend
│
├── docker-compose.yml     ← Orchestrates the Endee Server + Streamlit App
├── requirements.txt       ← Python dependencies
├── .env                   ← Configuration and API keys
└── README.md              ← You are here
```

---

## 🔥 Key Technical Features

| Feature | Implementation |
|---------|---------|
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim). Runs 100% locally and free. |
| **Vector DB** | **Endee**, utilizing `Precision.FLOAT32` for high-accuracy cosine similarity. |
| **LLM Backend** | Google Gemini `gemini-2.5-flash` using the new `google-genai` SDK. |
| **Data Upsertion** | Msgpack binary batch serialization via the official Endee Python SDK. |
| **Strict Grounding** | System prompts force the AI to refuse answering if context is missing. |
| **Transparency** | UI provides the exact source title, chunk, and similarity score. |

---

## 🛠 Advanced: Running Components Manually

You can test individual pipeline modules from the command line:

```bash
# Test Endee connection and index creation
python vector_store/endee_client.py

# Ingest specifically 500 articles
python ingestion/ingest_news.py --max 500

# Run evaluation suite
python evaluation/metrics.py
```
