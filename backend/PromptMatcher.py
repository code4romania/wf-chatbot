"""
PromptMatcher using INT8-quantized multilingual-E5-base
"""

import os, json
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


INT8_E5_MODEL = "intfloat/multilingual-e5-base"
MAX_LENGTH    = 512                                             # truncate longer passages


class PromptMatcher:
    """
    Build an ANN-ready matrix of *question+answer* embeddings using a CPU-optimized
    INT8 multilingual-E5 model.  Incoming queries are prefixed with 'query:'.
    """

    def __init__(
        self,
        base_data_path: str,
        language: str = "en",
        model_name: str = INT8_E5_MODEL,
        device: str = "cpu",                # change to "cuda" later if you add a GPU
    ):
        self.base_data_path = base_data_path
        self.language       = language.lower()
        self.model_name     = model_name
        self.device         = device

        # ── load INT8 Sentence-Transformer checkpoint ────────────────────────────
        print(f"Loading model {self.model_name} on {self.device} …")
        self.model = SentenceTransformer(self.model_name, device=self.device)  # quantized weights

        # data holders
        self.df             : pd.DataFrame          | None = None
        self.display_qs     : List[str]                     = []  # for printing results
        self.display_as     : List[str]                     = []
        self.embedding_txts : List[str]                     = []  # 'passage: Q A'
        self.prompt_vectors : np.ndarray           | None = None
        self.q_ids          : List[int]                     = []
        self.a_ids          : List[int]                     = []

        self._process_data()

    # --------------------------------------------------------------------- #

    def _process_data(self):
        """Load Q&A json files → DataFrame → embeddings."""
        q_dir = os.path.join(self.base_data_path, "dopomoha_questions", self.language)
        a_dir = os.path.join(self.base_data_path, "dopomoha_answers",   self.language)
        if not (os.path.isdir(q_dir) and os.path.isdir(a_dir)):
            raise FileNotFoundError("Q&A directories not found. Check base_data_path.")

        # ---- read questions ----
        questions: Dict[int, str] = {}
        for f in filter(lambda x: x.endswith(".json"), os.listdir(q_dir)):
            with open(os.path.join(q_dir, f), encoding="utf-8") as fh:
                for q in json.load(fh).get("questions", []):
                    if "question_id" in q and "question" in q:
                        questions[q["question_id"]] = q["question"]

        # ---- read answers ----
        answers: Dict[int, Dict[str, Union[str, int]]] = {}
        for f in filter(lambda x: x.endswith(".json"), os.listdir(a_dir)):
            with open(os.path.join(a_dir, f), encoding="utf-8") as fh:
                for a in json.load(fh).get("answers", []):
                    if {"question_id", "answer_id", "answer"} <= a.keys():
                        answers[a["question_id"]] = {
                            "answer":    a["bResponse"],
                            "answer_id": a["answer_id"],
                        }

        # ---- link Q ↔ A and build embedding strings ----
        rows = []
        for q_id, q_txt in questions.items():
            if q_id in answers:
                a_txt      = answers[q_id]["answer"]
                passage    = f"passage: {q_txt.strip()} {a_txt.strip()}"
                rows.append({
                    "Prompt":      q_txt,
                    "Response":    a_txt,
                    "question_id": q_id,
                    "answer_id":   answers[q_id]["answer_id"],
                    "embed_txt":   passage,
                })

        if not rows:
            raise ValueError("No linked questions/answers found.")

        self.df             = pd.DataFrame(rows)
        self.display_qs     = self.df["Prompt"].tolist()
        self.display_as     = self.df["Response"].tolist()
        self.embedding_txts = self.df["embed_txt"].tolist()
        self.q_ids          = self.df["question_id"].tolist()
        self.a_ids          = self.df["answer_id"].tolist()

        # ---- embed everything (batch, cpu-friendly) ----
        print("Encoding corpus …")
        self.prompt_vectors = self.model.encode(
            self.embedding_txts,
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        print(f"Ready – indexed {len(self.prompt_vectors)} Q+A passages.\n")

    # --------------------------------------------------------------------- #

    def query(
        self,
        user_prompt: str,
        metric: str = "cosine",
        top_k: int = 1,
    ) -> Union[Dict[str, Union[str, float, int]], List[Dict[str, Union[str, float, int]]]]:
        """Return best -K matches for the user query."""
        if self.prompt_vectors is None:
            raise RuntimeError("Corpus not encoded.")

        # ----- embed the incoming query -----
        query_vec = self.model.encode(
            [f"query: {user_prompt.strip()}"],
            normalize_embeddings=True
        )

        if metric == "cosine":
            scores = cosine_similarity(query_vec, self.prompt_vectors)[0]
            idxs   = np.argsort(scores)[::-1]
        elif metric == "euclidean":
            scores = euclidean_distances(query_vec, self.prompt_vectors)[0]
            idxs   = np.argsort(scores)   # smaller distance = better
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        results = []
        for rank in range(min(top_k, len(idxs))):
            i = idxs[rank]
            results.append({
                "matched_prompt": self.display_qs[i],
                "response":       self.display_as[i],
                "score":          float(scores[i]),
                "metric":         metric,
                "question_id":    self.q_ids[i],
                "answer_id":      self.a_ids[i],
            })
        return results[0] if top_k == 1 else results


# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    actual_data_path = "./WebScrape/data_whole_page"       # <-- adjust
    matcher = PromptMatcher(base_data_path=actual_data_path, language="en")

    while True:
        q = input("\nAsk something (or 'quit'): ").strip()
        if q.lower() == "quit":
            break
        top_k = 3
        hits  = matcher.query(q, metric="cosine", top_k=top_k)
        print(f"\n--- Top {top_k} result(s) ---")
        for rnk, hit in enumerate(hits, 1) if isinstance(hits, list) else [(1, hits)]:
            print(f"\nMatch {rnk}:")
            print(f"  Q-ID {hit['question_id']}  » {hit['matched_prompt']}")
            print(f"  A-ID {hit['answer_id']}  » {hit['response']}")
            print(f"  Score: {hit['score']:.4f}")