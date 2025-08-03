# Retrieval Pipeline Documentation



This documentation describes how to use the PromptMatcher-based retrieval system under the `/backend` directory. It supports hybrid semantic + lexical search with reranking using a cross-encoder, and it can be run either via a FastAPI server or directly from the command line.

- [Setup](#setup)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

---

## Setup 

### 📁 Step 1: Change Directory

All scripts and config files live in the `/backend` directory:

```bash
cd backend
```

### 📦 Step 2: Install Requirements

Install dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

## ⚙️ Step 3: Configure Retrieval Parameters

The system reads from `RetrievalParams.json`:

```json
{
  "questions_path": "dopomoha_questions_pro",
  "answers_path": "dopomoha_questions_pro_answers",
  "base_path": "data_whole_page",
  "concat_qa": true
}
```

- `questions_path`: Folder containing question JSON files.  
- `answers_path`: Folder containing corresponding answer files.  
- `base_path`: Base folder under `CorpusGeneration/corpus/` where the above folders live.  
- `concat_qa`: If true, the system concatenates question + answer before embedding.  

**Example expected directory layout:**

```
CorpusGeneration/corpus/data_whole_page/
├── dopomoha_questions_pro/
│   └── en/
│       └── some_file.json
└── dopomoha_questions_pro_answers/
    └── en/
        └── some_file.json
```

## Usage
### 🚀 Option 1: Run the FastAPI Server

Run the web API:

```bash
uvicorn main:app --reload
```


**Endpoints**

#### `POST /query`

**Request:**

```json
{
  "query": "How do I apply for asylum?",
  "top_k": 3,
  "session_id": "session123",
  "use_concat_matcher": true
}
```

**Response:**

```json
{
  "session_id": "session123",
  "results": [
    {
      "matched_prompt": "How do I apply for asylum in Romania?",
      "response": "Go to the immigration office and fill out the asylum form.",
      "instruction": "Provide steps to apply for asylum.",
      "score": 0.92,
      "question_id": 101,
      "answer_id": 202
    }
  ],
  "query_id": 1
}
```

#### `POST /review`

Used to submit feedback on answers.

### 🖥️ Option 2: Run CLI Interface

You can run `PromptMatcher.py` directly and interact with it:

```bash
python PromptMatcher.py
```

You will enter an interactive REPL:

```
Ask something (or 'quit'): What documents do I need to apply?
```

It returns the top reranked results along with debug scores:

```
--- Final Reranked Top 3 result(s) ---

Match 1: (Rerank Score: 12.3456)
  Q-ID 123 » What documents are required to apply for refugee status?
  A-ID 456 » You need a passport, proof of residence, and ID photos.
  Instruction: List necessary documents for application.
  (Debug: Initial Hybrid=0.8473, Dense=0.9123, Sparse=0.7510, Colbert=0.7832)
```

#### What Happens Internally?

- Loads all Q&A pairs from the configured question/answer folders.  
- Encodes them into:
  - Dense vectors (via BGE-M3)
  - Sparse lexical weights
  - ColBERT token-level vectors  
- Caches all vectors to disk.  
- For each query:
  - Encodes the input prompt.
  - Computes cosine and lexical similarity.
  - Fuses scores via weighted hybrid formula.
  - Selects top results.
  - Applies cross-encoder reranking.
  - Returns final best matches.

#### 📂 Cache Files

Cached embeddings are stored under:

```
CorpusGeneration/corpus/<base_path>/_embed_cache/
  └── BAAI_bge_m3/
      └── en/
          └── qna/
              ├── corpus_df.pkl
              ├── corpus_dense_vec.npy
              ├── corpus_lexical_weights.pkl
              └── corpus_colbert_vecs.pkl
```

These are loaded automatically if they exist.

#### Scoring & Weights

You can adjust scoring weights during query:

```python
matcher.query(
    user_prompt="Where to find housing?",
    top_k=3,
    sparse_weight=0.3,
    colbert_weight=0.3,
    retrieval_top_n=10,
    rerank_top_n=3
)
```

- Dense weight is automatically computed: `1.0 - sparse_weight - colbert_weight`.  
- All similarity scores are min-max normalized before hybrid fusion.

#### 🗃️ Database Integration

When using the FastAPI server, all queries and reviews are stored in a SQL database using SQLAlchemy:

- `UserQuery`: Logs the input query and returned answers.  
- `UserReview`: Stores feedback from the `/review` endpoint.  

## Troubleshooting

- **No Q&A loaded?**  
  Check if `RetrievalParams.json` points to the correct folders.  
  Make sure the JSON files have valid structure.  


