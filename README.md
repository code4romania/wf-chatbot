# Dopomoha Q&A System

Dopomoha Q&A System — a full-stack pipeline for building, querying, and interacting with question & Answer corpora. The system is designed to support automated corpus generation, hybrid retrieval with reranking, and multiple user interfaces including a Smart FAQ and Chat UI.

---

## 📁 Documentation Overview

This monorepo includes documentation for three key components:

### 1. 📚 [Corpus Generation Pipeline](docs/CorpusGenerationDocs.md)

Automates the creation of a Q&A corpus from web content using the Google Gemini model.

- **Stage 1**: Scrape pages using URL lists and HTML boundaries.
- **Stage 2**: Generate relevant questions based on the scraped content.
- **Stage 3**: Use Gemini to generate answers to those questions.

📖 **[Read more →](docs/CorpusGenerationDocs.md)**

---

### 2. 🔍 [Retrieval Pipeline](docs/RetrievalDocs.md)

Retrieves the most relevant Q&A pairs using a hybrid search engine:

- Combines **dense**, **sparse**, and **ColBERT** representations.
- Reranks results using a **cross-encoder**.
- Supports both **FastAPI web server** and **CLI mode**.

📖 **[Read more →](docs/RetrievalDocs.md)**

---

### 3. 🖥️ [Frontend Interface](docs/FrontendDocs.md)

Two user-friendly ways to interact with the system:

- **Smart FAQ**: Query multiple answers with review flow.
- **Chat Interface**: Conversational mode with top-1 answer responses.

📖 **[Read more →](docs/FrontendDocs.md)**

---

## Quick Start with Docker Compose

You can run the entire system using Docker Compose from the project root (where `docker-compose.yml` is located):

```bash
docker-compose up --build