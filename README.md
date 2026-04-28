# ⚡ LOKI: RAG Terminal

A high-tech, terminal-themed **Retrieval-Augmented Generation (RAG)** system. Built for speed, privacy, and aesthetic dominance. 

Upload your documents (PDF, TXT, Markdown), and LOKI will parse, chunk, and embed them into a local vector vault, allowing you to query your data in natural language with cited, real-time responses.

```text
╔══════════════════════════════════════════════════╗
║          LOKI  DOCUMENT Q&A TERMINAL             ║
║     Powered by Gemma 2 · ChromaDB · RAG          ║
╚══════════════════════════════════════════════════╝
```

---

## 📸 Preview

<div align="center">
  <img src="docs/screenshots/onboarding.jpeg" width="45%" alt="LOKI OS Onboarding" />
  <img src="docs/screenshots/logo.jpeg" width="45%" alt="LOKI Logo Glitch" />
  <br/>
  <img src="docs/screenshots/terminal_empty.jpeg" width="45%" alt="Terminal UI" />
  <img src="docs/screenshots/terminal_active.jpeg" width="45%" alt="RAG in Action" />
</div>

---

## ⚡ Deployment & Hybrid Mode

LOKI is engineered to be **hosting-agnostic**. It detects its environment automatically:

*   **Local Mode**: Uses **Ollama** (LLM) and **ChromaDB** (Vector Store). Perfect for privacy and offline use.
*   **Hosted Mode (Render)**: Switches to **Groq** (LLM) and **Pinecone** (Vector Store). If `HF_TOKEN` is present it uses the **Hugging Face Inference API** for embeddings; otherwise it falls back to the local `sentence-transformers` model.

---

## ✨ Features

- **Terminal UI** — dark CRT aesthetic, scanlines, typewriter streaming, blinking cursor.
- **RAG Pipeline** — Automatic parsing, chunking, and vector embedding.
- **Natural Language Processing** — Built-in typo correction and conversational memory.
- **Privacy First** — Runs locally via Ollama with zero external API costs.
- **Hybrid Deployment** — Seamlessly switch between Local (ChromaDB) and Cloud (Pinecone).
- **No Frameworks** — Vanilla HTML/CSS/JS frontend; no npm, no build steps.

---

## 🛠️ The Tech Stack

| Layer            | Local                        | Hosted                  |
| ---------------- | ---------------------------- | ----------------------- |
| **Brain (LLM)**  | Ollama (Gemma 2 / any model) | Groq (Llama 3.1)        |
| **Vector Store** | ChromaDB (Embedded)          | Pinecone (Serverless)   |
| **Embeddings**   | `all-MiniLM-L6-v2`           | `all-MiniLM-L6-v2`      |
| **Backend**      | Flask 3 + SSE Streaming      | Flask 3 + SSE Streaming |
| **Frontend**     | Vanilla JS / CRT CSS         | Vanilla JS / CRT CSS    |

---

## 🚀 Quick Start (Local)

### 1. Requirements
*   **Ollama**: [Download here](https://ollama.com/)
*   **Model**: `ollama pull gemma:2b`

### 2. Setup
```bash
# Clone the repo
git clone https://github.com/Yashpreetg24/LOKI-RAG
cd LOKI-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launch
```bash
cp .env.example .env
python run.py
```
👉 Access at **http://localhost:5001**

---

## ⌨️ Terminal Commands

| Command             | Action                                           |
| ------------------- | ------------------------------------------------ |
| `<any text>`        | Ask a question about your documents              |
| `/upload`           | Open file picker to ingest a new PDF, TXT, or MD |
| `/docs`             | List all documents currently in the vector vault |
| `/summarize <file>` | Generate an AI summary of a specific document    |
| `/delete <file>`    | Wipe a file and its vectors from the system      |
| `/status`           | Check LLM backend and system vitals              |
| `/clear`            | Clear the terminal display                       |
| `/help`             | List all available protocols                     |

---

## ⚙️ Environment Variables

| Variable           | Default      | Description                             |
| ------------------ | ------------ | --------------------------------------- |
| `RENDER`           | `0`          | Set to `1` to force Cloud Mode          |
| `OLLAMA_MODEL`     | `gemma:2b`   | The model LOKI will use locally         |
| `GROQ_API_KEY`     | *(optional)* | API key for Groq (Cloud Mode)           |
| `PINECONE_API_KEY` | *(optional)* | API key for Pinecone (Cloud Mode)       |
| `HF_TOKEN`         | *(optional)* | Enables Hugging Face-hosted embeddings in Cloud Mode |
| `FLASK_PORT`       | `5001`       | Local server port                       |

---

## 📂 Project Structure

```text
LOKI-RAG/
├── app/
│   ├── ingestion/     # Parsing, Chunking & Embedding logic
│   ├── rag/           # LLM chains, Query rewriting & Prompts
│   ├── models/        # Data structures & History
│   ├── static/        # CRT Terminal UI (HTML/CSS/JS)
│   └── routes.py      # Flask API Endpoints
├── docs/screenshots/  # Visual assets for README
├── run.py             # System entry point
├── requirements.txt   # Core dependencies
└── .env               # System configuration
```

---

## ☁️ Cloud Deployment (Render)

LOKI includes a `render.yaml` for one-click deployment.

1.  Create a **Pinecone** index (`dimension: 384`).
2.  Connect this repo to **Render**.
3.  Add `GROQ_API_KEY` and `PINECONE_API_KEY` to your Environment Secrets. `HF_TOKEN` is optional if you want hosted embeddings instead of the local fallback.
4.  Deploy! LOKI will automatically switch to cloud mode.

---

## 👤 Author
**Yashpreet Gupta**  
*"Burdened with glorious purpose."*

---
LICENSE: MIT
