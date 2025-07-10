import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
import asyncio
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

MAX_LENGTH = 512
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
        city_names: Optional[List[str]] = None,
    ):
        self.base_data_path = Path(base_data_path)
        self.language = language.lower()
        self.concat_q_and_a = concat_q_and_a
        self.city_names = [city.lower() for city in city_names] if city_names else []

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

        # Holders for ALL pre-encoded Q&A data
        self.full_df: Optional[pd.DataFrame] = None
        self.full_vectors: Optional[np.ndarray] = None

        # These will hold the subset of the corpus relevant to the current query
        self._current_df_subset: Optional[pd.DataFrame] = None
        self._current_vectors_subset: Optional[np.ndarray] = None
        
        self._process_all_data_and_embed() # Process and embed ALL data upfront

    # ------------------------------------------------------------------
    # Data preparation - Load all raw data and embed upfront
    # ------------------------------------------------------------------
    def _process_all_data_and_embed(self):
        """
        Loads all general and city-specific Q&A data, prepares the 'embed_txt'
        and encodes all passages upfront into self.full_df and self.full_vectors.
        """
        q_dir = self.base_data_path / "dopomoha_no_yes_no" / self.language
        a_dir = self.base_data_path / "dopomoha_batch_pointing" / self.language

        if not (q_dir.is_dir() and a_dir.is_dir()):
            raise FileNotFoundError(
                f"Expected directories {q_dir} and {a_dir}. Check --base-data-path."
            )

        all_linked_qa_rows: List[Dict[str, Union[str, int, None]]] = []

        # Iterate over all JSON files in the questions directory to find Q&A pairs
        for q_fp in q_dir.glob("*.json"):
            file_name_stem = q_fp.stem
            
            # Determine if this file belongs to a specific city or is general
            city_name_for_file = None
            for city in self.city_names:
                if file_name_stem == f"dopomoha-{city}":
                    city_name_for_file = city.lower() # Ensure lowercase for consistency
                    break
            
            # Load questions from this file
            questions_in_file: Dict[int, Dict[str, Union[str, None]]] = {}
            with open(q_fp, encoding="utf-8", errors="replace") as fh:
                q_payload = json.load(fh)
            for q_entry in q_payload.get("questions", []):
                if {"question_id", "question"} <= q_entry.keys():
                    questions_in_file[q_entry["question_id"]] = {
                        "question": q_entry["question"],
                        "city": city_name_for_file
                    }

            # Load answers from the corresponding answer file (assuming same filename)
            answers_in_file: Dict[int, Dict[str, Union[str, int, None]]] = {}
            a_fp = a_dir / q_fp.name
            if a_fp.is_file():
                with open(a_fp, encoding="utf-8", errors="replace") as fh:
                    a_payload = json.load(fh)
                for a_entry in a_payload.get("answers", []):
                    if {"question_id", "answer_id", "answer", "instruction"} <= a_entry.keys():
                        answers_in_file[a_entry["question_id"]] = {
                            "answer": a_entry["answer"],
                            "instruction": a_entry["instruction"],
                            "answer_id": a_entry["answer_id"],
                            "city": city_name_for_file
                        }
            
            # Link Q&A pairs from this specific file and add to the main list
            for q_id, q_data in questions_in_file.items():
                if q_id in answers_in_file:
                    a_data = answers_in_file[q_id]

                    # Optional: consistency check for city association (should match based on filename)
                    if q_data["city"] != a_data["city"]:
                         logger.warning(f"City mismatch in loaded data for Q/A ID {q_id} from file {q_fp.name}: Q city={q_data['city']}, A city={a_data['city']}. Using Q's city.")
                    
                    # Prepare the row for the full DataFrame
                    row_data = {
                        "Prompt": q_data["question"],
                        "Response": a_data["answer"],
                        "Instruction": a_data["instruction"],
                        "city": q_data["city"], # Use question's city for the row
                        "question_id": q_id,
                        "answer_id": a_data["answer_id"],
                    }
                    all_linked_qa_rows.append(row_data)

        if not all_linked_qa_rows:
            raise ValueError("No linked questions/answers found across all loaded files. Cannot build corpus.")

        self.full_df = pd.DataFrame(all_linked_qa_rows)
        
        # Prepare 'embed_txt' column for encoding
        embed_texts = []
        for _, row in self.full_df.iterrows():
            passage_parts = [row["Prompt"].strip()]
            if self.concat_q_and_a:
                passage_parts.append(row["Response"].strip()) # Only answer is concatenated with question
            embed_texts.append(f"passage: {' '.join(passage_parts)}")

        self.full_df['embed_txt'] = embed_texts

        logger.info(f"Encoding entire corpus of {len(self.full_df)} passages…")
        self.full_vectors = self.model.encode(
            self.full_df["embed_txt"].tolist(),
            batch_size=64, # Adjust batch size based on memory/performance
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype("float32") # FAISS-friendly precision
        logger.info(f"Full corpus with {len(self.full_vectors)} embeddings ready.")


    # NEW: Helper method to detect city in user query
    def _detect_city_in_query(self, query_text: str) -> Optional[str]:
        query_lower = query_text.lower()
        for city in self.city_names:
            # Simple substring check. Could be improved with more robust NLP if needed
            if city in query_lower:
                return city
        return None

    def _select_corpus_subset(self, detected_city: Optional[str]):
        """
        Selects the relevant subset of the pre-encoded full corpus using a boolean mask.
        Sets self._current_df_subset and self._current_vectors_subset.
        """
        if self.full_df is None or self.full_vectors is None:
            raise RuntimeError("Full corpus not initialized. Call _process_all_data_and_embed first.")

        # Initialize mask: True for general data (where 'city' column is None)
        mask = self.full_df['city'].isna()
        
        if detected_city:
            logger.info(f"Including data for city '{detected_city}' in corpus subset.")
            # OR the mask with conditions for the detected city
            mask = mask | (self.full_df['city'] == detected_city)
        else:
            logger.info("No specific city detected. Corpus subset will contain general data only.")

        # Apply the boolean mask directly
        self._current_df_subset = self.full_df[mask].copy()
        self._current_vectors_subset = self.full_vectors[mask] # NumPy directly accepts boolean arrays for indexing
        
        if self._current_df_subset.empty:
            logger.warning("No data in the selected corpus subset. This might indicate an issue with your data or city detection.")
            self._current_vectors_subset = np.array([]) # Ensure it's an empty array if df is empty
        
        logger.info(f"Corpus subset built with {len(self._current_df_subset)} passages for query context.")


    async def query(
            self,
            user_prompt: str,
            metric: str = "cosine",
            top_k: int = 1,
        ) -> List[Dict[str, Union[str, float, int, None]]]:
            """Return top‑K matches for the user prompt, using pre-encoded, dynamically selected corpus subset."""

            detected_city = self._detect_city_in_query(user_prompt)
            
            # Select the relevant subset of the pre-encoded corpus for this query
            self._select_corpus_subset(detected_city)

            if self._current_vectors_subset is None or self._current_df_subset is None or self._current_vectors_subset.size == 0:
                logger.warning("Current corpus subset is empty. Cannot perform search.")
                return []

            # --- Define a synchronous helper function to encapsulate the blocking logic ---
            def _sync_query_logic():
                # The _current_df_subset and _current_vectors_subset are already set by _select_corpus_subset
                current_df_for_search = self._current_df_subset
                current_vectors_for_search = self._current_vectors_subset

                # ----- embed user prompt -----
                query_vec = self.model.encode([f"query: {user_prompt.strip()}"], normalize_embeddings=True)

                # ----- similarity scores -----
                if metric.lower() == "cosine":
                    scores = cosine_similarity(query_vec, current_vectors_for_search)[0]
                    best_idx_in_current = np.argsort(scores)[::-1] # descending for cosine
                elif metric.lower() in {"euclidean", "l2"}:
                    scores = euclidean_distances(query_vec, current_vectors_for_search)[0]
                    best_idx_in_current = np.argsort(scores)  # ascending for euclidean (smaller is better)
                else:
                    raise ValueError("metric must be 'cosine' or 'euclidean'")

                # ----- assemble results -----
                results: List[Dict[str, Union[str, float, int, None]]] = []
                for rank in range(min(top_k, len(best_idx_in_current))):
                    i_current = best_idx_in_current[rank]
                    
                    # Access the row directly from the _current_df_subset
                    matched_row = current_df_for_search.iloc[i_current]

                    results.append(
                        {
                            "matched_prompt": matched_row["Prompt"],
                            "response": matched_row["Response"],
                            "instruction": matched_row["Instruction"],
                            "city": matched_row["city"],
                            "score": float(scores[i_current]),
                            "metric": metric.lower(),
                            "question_id": int(matched_row["question_id"]),
                            "answer_id": int(matched_row["answer_id"]),
                        }
                    )
                return results

            # --- Run the synchronous logic in a separate thread ---
            return await asyncio.to_thread(_sync_query_logic)

# helper for REPL – keeps original behaviour but now exposes args
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Interactive semantic Q&A matcher (brute‑force)")
    parser.add_argument("--base-data-path", type=str, default="./WebScrape/data_whole_page", help="Root folder holding dopomoha_questions/ and dopomoha_answers/")
    parser.add_argument("--language", type=str, default="en", help="Language subfolder to load (e.g. 'en', 'uk')")

    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-choice", choices=MODEL_OPTIONS.keys(), default="e5-large", help="Predefined model key")
    model_group.add_argument("--custom-model", type=str, help="Custom HuggingFace model id (overrides --model-choice)")

    parser.add_argument("--no-answer", action="store_true", help="Embed question only (skip answer text)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine", help="Similarity metric")
    parser.add_argument("--top-k", type=int, default=3, help="Number of hits to display")

    args = parser.parse_args()

    # Define the list of cities from your screenshot
    CITY_LIST = [
        "braila", "brasov", "cluj-napoca", "constanta", "galati",
        "iasi", "oradea", "sibiu", "suceava", "timisoara"
    ]

    matcher = PromptMatcher(
        base_data_path=args.base_data_path,
        language=args.language,
        model_choice=args.model_choice,
        custom_model_name=args.custom_model,
        concat_q_and_a= True, # Always concatenate Q+A for embedding
        city_names=CITY_LIST, # Pass the list of cities to the matcher
    )

    async def main_repl():
        try:
            while True:
                q = input("\nAsk something (e.g., 'Kindergarten in Braila?' or 'quit'): ").strip()
                if q.lower() == "quit":
                    break
                hits = await matcher.query(q, metric=args.metric, top_k=args.top_k)
                print(f"\n--- Top {len(hits)} result(s) ---")
                for rnk, hit in enumerate(hits, 1):
                    print(f"\nMatch {rnk}:")
                    print(f"  Q-ID {hit['question_id']} » {hit['matched_prompt']}")
                    print(f"  A-ID {hit['answer_id']} » {hit['response']}")
                    print(f"  Instruction: {hit['instruction']}")
                    print(f"  City: {hit['city'] if hit['city'] else 'General'}")
                    print(f"  Score: {hit['score']:.4f}")
        except KeyboardInterrupt:
            print("\nExiting. Bye!")

    asyncio.run(main_repl())