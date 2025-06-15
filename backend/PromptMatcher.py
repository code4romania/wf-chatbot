"""
PromptMatcher – flexible Q&A embedding search
------------------------------------------------
* Adds pluggable embedding models (INT8‑E5, all‑MiniLM‑L6, BGE‑base) – or any HF id via --custom-model.
* Lets you toggle whether the answer text is concatenated to the question when creating passage embeddings.
* Keeps the simple brute‑force similarity search that suits small/medium corpora.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# -----------------------------------------------------------------------------
# Configurable model registry – extend as you like
# -----------------------------------------------------------------------------
MODEL_OPTIONS: Dict[str, str] = {
    "e5-int8": "intfloat/multilingual-e5-base",           # default (quantised)
    "allmini6": "sentence-transformers/all-MiniLM-L6-v2", # small & fast (English‑only)
    "bge-base": "BAAI/bge-base-en-v1.5",
    "e5-large": "intfloat/multilingual-e5-large"
}

MAX_LENGTH = 512  # currently unused but kept for future truncation logic
logger = logging.getLogger("promptmatcher")
logger.setLevel(logging.INFO)


class PromptMatcher:
    """Build an embedding matrix for Q&A pairs and perform brute‑force search."""

    def __init__(
        self,
        base_data_path: Union[str, Path],
        language: str = "en",
        model_choice: str = "e5-int8",
        custom_model_name: Optional[str] = None,
        device: str = "cpu",
        concat_q_and_a: bool = True,
    ):
        self.base_data_path = Path(base_data_path)
        self.language = language.lower()
        self.concat_q_and_a = concat_q_and_a

        # ----- resolve model name -----
        if custom_model_name:
            self.model_name = custom_model_name
        else:
            if model_choice not in MODEL_OPTIONS:
                raise ValueError(
                    f"Unknown model_choice '{model_choice}'. Valid keys: {', '.join(MODEL_OPTIONS)}"
                )
            self.model_name = MODEL_OPTIONS[model_choice]

        self.device = device

        logger.info(f"Loading model {self.model_name} on {self.device} …")
        self.model = SentenceTransformer(self.model_name, device=self.device)

        # Holders
        self.df: Optional[pd.DataFrame] = None
        self.prompt_vectors: Optional[np.ndarray] = None

        self._process_data()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _process_data(self):
        """Load question/answer JSON → DataFrame → embeddings."""

        q_dir = self.base_data_path / "dopomoha_questions" / self.language
        a_dir = self.base_data_path / "dopomoha_answers" / self.language
        if not (q_dir.is_dir() and a_dir.is_dir()):
            raise FileNotFoundError(
                f"Expected directories {q_dir} and {a_dir}. Check --base-data-path."
            )

        # ---------- ingest questions ----------
        questions: Dict[int, str] = {}
        for fp in q_dir.glob("*.json"):
            with open(fp, encoding="utf-8", errors="replace") as fh:
                payload = json.load(fh)
            for q in payload.get("questions", []):
                if {"question_id", "question"} <= q.keys():
                    questions[q["question_id"]] = q["question"]

        # ---------- ingest answers ----------
        answers: Dict[int, Dict[str, Union[str, int]]] = {}
        for fp in a_dir.glob("*.json"):
            with open(fp, encoding="utf-8", errors="replace") as fh:
                payload = json.load(fh)
            for a in payload.get("answers", []):
                if {"question_id", "answer_id", "answer"} <= a.keys():
                    answers[a["question_id"]] = {
                        "answer": a["answer"],  # fixed field name
                        "answer_id": a["answer_id"],
                    }

        # ---------- link Q ↔ A ----------
        rows: List[Dict[str, Union[str, int]]] = []
        for q_id, q_txt in questions.items():
            if q_id in answers:
                a_txt = answers[q_id]["answer"]
                passage_parts = [q_txt.strip()]
                if self.concat_q_and_a:
                    passage_parts.append(a_txt.strip())
                passage = f"passage: {' '.join(passage_parts)}"

                rows.append(
                    {
                        "Prompt": q_txt,
                        "Response": a_txt,
                        "question_id": q_id,
                        "answer_id": answers[q_id]["answer_id"],
                        "embed_txt": passage,
                    }
                )

        if not rows:
            raise ValueError("No linked questions/answers found.")

        self.df = pd.DataFrame(rows)

        # ---------- encode embeddings ----------
        logger.info("Encoding corpus …")
        self.prompt_vectors = self.model.encode(
            self.df["embed_txt"].tolist(),
            batch_size=64,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32")  # FAISS‑friendly precision
        logger.info(f"Ready – indexed {len(self.prompt_vectors)} passages.\n")

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def query(
        self,
        user_prompt: str,
        metric: str = "cosine",
        top_k: int = 1,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """Return top‑K matches for the user prompt."""

        if self.prompt_vectors is None:
            raise RuntimeError("Corpus not encoded.")

        # ----- embed user prompt -----
        query_vec = self.model.encode([f"query: {user_prompt.strip()}"], normalize_embeddings=True)

        # ----- similarity scores -----
        if metric.lower() == "cosine":
            scores = cosine_similarity(query_vec, self.prompt_vectors)[0]
            best_idx = np.argsort(scores)[::-1]
        elif metric.lower() in {"euclidean", "l2"}:
            scores = euclidean_distances(query_vec, self.prompt_vectors)[0]
            best_idx = np.argsort(scores)  # smaller distance = better
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        # ----- assemble results -----
        results: List[Dict[str, Union[str, float, int]]] = []
        for rank in range(min(top_k, len(best_idx))):
            i = best_idx[rank]
            results.append(
                {
                    "matched_prompt": self.df.iloc[i]["Prompt"],
                    "response": self.df.iloc[i]["Response"],
                    "score": float(scores[i]),
                    "metric": metric.lower(),
                    "question_id": int(self.df.iloc[i]["question_id"]),
                    "answer_id": int(self.df.iloc[i]["answer_id"]),
                }
            )
        return results

    # helper for REPL – keeps original behaviour but now exposes args


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Interactive semantic Q&A matcher (brute‑force)")
    parser.add_argument("--base-data-path", type=str, default="./WebScrape/data_whole_page", help="Root folder holding dopomoha_questions/ and dopomoha_answers/")
    parser.add_argument("--language", type=str, default="en", help="Language subfolder to load (e.g. 'en', 'uk')")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-choice", choices=MODEL_OPTIONS.keys(), default="e5-int8", help="Predefined model key")
    model_group.add_argument("--custom-model", type=str, help="Custom HuggingFace model id (overrides --model-choice)")

    parser.add_argument("--no-answer", action="store_true", help="Embed question only (skip answer text)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="Similarity metric")
    parser.add_argument("--top-k", type=int, default=3, help="Number of hits to display")

    args = parser.parse_args()

    matcher = PromptMatcher(
        base_data_path=args.base_data_path,
        language=args.language,
        model_choice=args.model_choice,
        custom_model_name=args.custom_model,
        concat_q_and_a=not args.no_answer,
    )

    try:
        while True:
            q = input("\nAsk something (or 'quit'): ").strip()
            if q.lower() == "quit":
                break
            hits = matcher.query(q, metric=args.metric, top_k=args.top_k)
            print(f"\n--- Top {len(hits)} result(s) ---")
            for rnk, hit in enumerate(hits, 1):
                print(f"\nMatch {rnk}:")
                print(f"  Q-ID {hit['question_id']} » {hit['matched_prompt']}")
                print(f"  A-ID {hit['answer_id']} » {hit['response']}")
                print(f"  Score: {hit['score']:.4f}")
    except KeyboardInterrupt:
        print("\nExiting. Bye!")
