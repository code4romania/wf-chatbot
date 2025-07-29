import os
import json
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional, Any

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BGE_M3_MODEL = "BAAI/bge-m3"
MAX_LENGTH = 8192
logger = logging.getLogger("promptmatcher")
logger.setLevel(logging.INFO)

# These paths can be configured if your directory structure is different
questions_path = "dopomoha_questions_pro"
answers_path = "dopomoha_questions_pro_answers"


class PromptMatcher:
    """
    Builds dense, sparse, and Colbert embedding matrices for a Q&A corpus and performs
    a hybrid search by combining scores from all representations using BGE-M3
    via FlagEmbedding.
    """

    def __init__(
        self,
        base_data_path: Union[str, Path],
        language: str = "en",
        device: str = "cpu",
        concat_q_and_a: bool = False,
        city_names=None, # Retained for compatibility with your existing __init__ call
    ):
        self.base_data_path = Path(base_data_path)
        self.language = language.lower()
        self.concat_q_and_a = concat_q_and_a
        self.model_name = BGE_M3_MODEL
        self.device = device

        logger.info(f"Loading model {self.model_name} on {self.device}…")
        self.model = BGEM3FlagModel(self.model_name, device=self.device, use_fp16=True)

        # Holders for ALL pre-encoded Q&A data
        self.full_df: Optional[pd.DataFrame] = None
        self.full_dense_vectors: Optional[np.ndarray] = None
        # full_sparse_vectors holds List[Dict[int, float]] (token_id: weight)
        self.full_sparse_vectors: Optional[List[Dict[int, float]]] = None
        # IMPORTANT CHANGE: full_colbert_vectors will now hold List[np.ndarray]
        self.full_colbert_vectors: Optional[List[np.ndarray]] = None

        self._process_all_data_and_embed() # Process and embed ALL data upfront

    # ------------------------------------------------------------------
    # Data preparation - Load all raw data and embed upfront
    # ------------------------------------------------------------------
    def _safe_model_name(self) -> str:
        """Return a filesystem-safe version of the model name."""
        return self.model_name.replace('/', '_').replace('-', '_')

    def _cache_dir(self) -> Path:
        """Return the full cache directory for the current config."""
        qna_str = "qna" if self.concat_q_and_a else "q"
        return self.base_data_path / "_embed_cache" / self._safe_model_name() / self.language / qna_str

    def _process_all_data_and_embed(self):
        """
        Loads all Q&A data, prepares the 'embed_txt', and encodes all passages
        upfront into dense, sparse, and Colbert vectors. Uses cache if possible.
        """
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_df_path = cache_dir / "corpus_df.pkl"
        cache_dense_vec_path = cache_dir / "corpus_dense_vec.npy"
        cache_sparse_vec_path = cache_dir / "corpus_lexical_weights.pkl"
        # IMPORTANT CHANGE: Cache Colbert vectors as a pickle of List[np.ndarray]
        cache_colbert_vec_path = cache_dir / "corpus_colbert_vecs.pkl"

        # Try loading from cache
        if all([
            cache_df_path.exists(),
            cache_dense_vec_path.exists(),
            cache_sparse_vec_path.exists(),
            cache_colbert_vec_path.exists()
        ]):
            logger.info(f"Loading embeddings from cache: {cache_dir}")
            self.full_df = pd.read_pickle(cache_df_path)
            self.full_dense_vectors = np.load(cache_dense_vec_path)
            with open(cache_sparse_vec_path, "rb") as f:
                self.full_sparse_vectors = pickle.load(f)
            # Load Colbert vectors as a list of numpy arrays
            with open(cache_colbert_vec_path, "rb") as f:
                self.full_colbert_vectors = pickle.load(f)
            return

        # Data directories
        q_dir = self.base_data_path / questions_path / self.language
        a_dir = self.base_data_path / answers_path / self.language

        if not (q_dir.is_dir() and a_dir.is_dir()):
            raise FileNotFoundError(f"Expected directories {q_dir} and {a_dir}. Check path.")

        all_linked_qa_rows = []
        for q_fp in q_dir.glob("*.json"):
            with open(q_fp, encoding="utf-8", errors="replace") as fh:
                q_payload = json.load(fh)
            questions_in_file = {
                q_entry["question_id"]: q_entry["question"]
                for q_entry in q_payload.get("questions", [])
                if {"question_id", "question"} <= q_entry.keys()
            }

            a_fp = a_dir / q_fp.name
            if a_fp.is_file():
                with open(a_fp, encoding="utf-8", errors="replace") as fh:
                    a_payload = json.load(fh)
                for a_entry in a_payload.get("answers", []):
                    q_id = a_entry.get("question_id")
                    if q_id in questions_in_file:
                        row_data = {
                            "Prompt": questions_in_file[q_id],
                            "Response": a_entry["answer"],
                            "Instruction": a_entry["instruction"],
                            "question_id": q_id,
                            "answer_id": a_entry["answer_id"],
                        }
                        all_linked_qa_rows.append(row_data)

        if not all_linked_qa_rows:
            raise ValueError("No linked Q&A found. Cannot build corpus.")

        self.full_df = pd.DataFrame(all_linked_qa_rows)

        # Prepare 'embed_txt' column for encoding
        embed_texts = []
        passage_prefix = "represent the passage for retrieval: "
        for _, row in self.full_df.iterrows():
            passage_parts = [row["Prompt"].strip()]
            if self.concat_q_and_a:
                passage_parts.append(row["Response"].strip())
            embed_texts.append(f"{passage_prefix}{' '.join(passage_parts)}")

        self.full_df['embed_txt'] = embed_texts

        logger.info(f"Encoding entire corpus of {len(self.full_df)} passages…")
        embeddings = self.model.encode(
            self.full_df["embed_txt"].tolist(),
            batch_size=32,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True, # This will now return 'colbert_vecs' as List[np.ndarray]
        )
        
        # --- Handling output from BGEM3FlagModel ---
        if isinstance(embeddings, dict):
            if 'dense_vecs' in embeddings and 'lexical_weights' in embeddings and 'colbert_vecs' in embeddings:
                self.full_dense_vectors = embeddings['dense_vecs'].astype("float32")
                self.full_sparse_vectors = embeddings['lexical_weights'] # Store lexical weights directly
                # Store colbert_vecs as a list of numpy arrays, no .astype on the list itself
                self.full_colbert_vectors = [vec.astype("float32") for vec in embeddings['colbert_vecs']]
            else:
                raise ValueError(
                    "Embeddings dictionary from BGEM3FlagModel does not contain 'dense_vecs', 'lexical_weights', or 'colbert_vecs' keys. "
                    f"Keys found: {embeddings.keys()}. This is unexpected."
                )
        else:
            raise TypeError(
                f"Unexpected type for embeddings from BGEM3FlagModel: {type(embeddings)}. Expected dict."
                f"Full embeddings object: {embeddings}"
            )

        logger.info(f"Full corpus with {len(self.full_df)} dense/sparse/Colbert embeddings ready.")

        # Save to cache
        logger.info(f"Saving corpus and vectors to cache: {cache_dir}")
        self.full_df.to_pickle(cache_df_path)
        np.save(cache_dense_vec_path, self.full_dense_vectors)
        with open(cache_sparse_vec_path, "wb") as f:
            pickle.dump(self.full_sparse_vectors, f) # Save lexical weights (list of dicts)
        # Save Colbert vectors (list of numpy arrays)
        with open(cache_colbert_vec_path, "wb") as f:
            pickle.dump(self.full_colbert_vectors, f)

    def _compute_sparse_scores(self, query_lexical_weights: Dict[int, float], corpus_lexical_weights_list: List[Dict[int, float]]) -> np.ndarray:
        """
        Computes lexical matching scores using BGEM3FlagModel's method.
        """
        scores = []
        for doc_lexical_weights in corpus_lexical_weights_list:
            score = self.model.compute_lexical_matching_score(query_lexical_weights, doc_lexical_weights)
            scores.append(score)
        return np.array(scores)
    
    def _compute_colbert_scores(self, query_colbert_vec: np.ndarray, corpus_colbert_vecs_list: List[np.ndarray]) -> np.ndarray:
        """
        Computes Colbert matching scores by performing pairwise comparisons
        using self.model.colbert_score(q_reps, p_reps).
        """
        scores = []
        # query_colbert_vec is (num_tokens, dim) for the single query
        
        # Iterate through each passage's colbert vectors in the corpus list
        for passage_colbert_vec in corpus_colbert_vecs_list:
            # colbert_score expects (num_tokens, dim) for both q_reps and p_reps
            # Wrap query_colbert_vec and passage_colbert_vec in a list to match expected batching if necessary,
            # but based on the example, it takes single (num_tokens, dim) arrays.
            # However, the doc states `q_reps (np.ndarray)` implying it could take (num_queries, num_tokens, dim)
            # Let's try passing them as-is first, or adjust if an error occurs.
            # The most direct interpretation of `colbert_score(output_1['colbert_vecs'][0], output_2['colbert_vecs'][0])`
            # implies it takes single (num_tokens, dim) arrays.
            
            # Since query_colbert_vec is already (num_tokens, dim) after extracting [0] in query method,
            # and each passage_colbert_vec is (num_tokens, dim), we can pass them directly.
            
            # The result from colbert_score is a torch.Tensor, so convert to numpy
            score_tensor = self.model.colbert_score(query_colbert_vec, passage_colbert_vec)
            scores.append(score_tensor.cpu().numpy().item()) # .item() gets the scalar value from a 0-dim tensor
            
        return np.array(scores)
    
    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Applies min-max normalization to a set of scores."""
        min_s, max_s = scores.min(), scores.max()
        if max_s == min_s:
            return np.zeros_like(scores, dtype=np.float32)
        return (scores - min_s) / (max_s - min_s)
    
    def _print_top_k_results(self, dense_scores: np.ndarray, sparse_scores: np.ndarray, colbert_scores: np.ndarray, hybrid_scores: np.ndarray, top_k: int):
        """
        Prints the top K results for dense, sparse, Colbert, and hybrid scores,
        including the score and the matched prompt.
        This method assumes self.full_df is available and populated.
        """
        if self.full_df is None:
            logger.error("full_df is not available for printing results.")
            return

        logger.info("\n--- Top K Results by Score Type ---")

        # Dense Scores
        logger.info(f"\nTop {top_k} results by DENSE Score:")
        top_k_dense_indices = np.argsort(dense_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_k_dense_indices):
            matched_prompt = self.full_df.iloc[idx]["Prompt"]
            logger.info(f"  {rank+1}. Score: {dense_scores[idx]:.4f} - Prompt: {matched_prompt[:100]}{'...' if len(matched_prompt) > 100 else ''}")

        # Sparse Scores
        logger.info(f"\nTop {top_k} results by SPARSE Score:")
        top_k_sparse_indices = np.argsort(sparse_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_k_sparse_indices):
            matched_prompt = self.full_df.iloc[idx]["Prompt"]
            logger.info(f"  {rank+1}. Score: {sparse_scores[idx]:.4f} - Prompt: {matched_prompt[:100]}{'...' if len(matched_prompt) > 100 else ''}")

        # Colbert Scores
        logger.info(f"\nTop {top_k} results by COLBERT Score:")
        top_k_colbert_indices = np.argsort(colbert_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_k_colbert_indices):
            matched_prompt = self.full_df.iloc[idx]["Prompt"]
            logger.info(f"  {rank+1}. Score: {colbert_scores[idx]:.4f} - Prompt: {matched_prompt[:100]}{'...' if len(matched_prompt) > 100 else ''}")

        # Hybrid Scores
        logger.info(f"\nTop {top_k} results by HYBRID Score:")
        top_k_hybrid_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        for rank, idx in enumerate(top_k_hybrid_indices):
            matched_prompt = self.full_df.iloc[idx]["Prompt"]
            logger.info(f"  {rank+1}. Score: {hybrid_scores[idx]:.4f} - Prompt: {matched_prompt[:100]}{'...' if len(matched_prompt) > 100 else ''}")

    def query(
        self,
        user_prompt: str,
        top_k: int = 1,
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.3,
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Return top-K matches for a prompt using a weighted hybrid of dense, sparse, and Colbert search.
        """
        if self.full_df is None or self.full_dense_vectors is None or \
           self.full_sparse_vectors is None or self.full_colbert_vectors is None:
            logger.error("Corpus is not initialized. Cannot perform search.")
            return []
        
        if not (0.0 <= sparse_weight <= 1.0) or not (0.0 <= colbert_weight <= 1.0):
            raise ValueError("sparse_weight and colbert_weight must be between 0.0 and 1.0")
        
        if sparse_weight + colbert_weight > 1.0:
            raise ValueError("Sum of sparse_weight and colbert_weight cannot exceed 1.0")

        # ----- Embed user prompt (dense and sparse) -----
        query_prefix = "retrieve the most relevant Q&A for the query: "
        query_embeddings = self.model.encode(
            [f"{query_prefix}{user_prompt.strip()}"],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True # Request Colbert vectors for query
        )

        query_dense_vec = None
        query_lexical_weights = None
        query_colbert_vec = None # New: query Colbert vector

        if isinstance(query_embeddings, dict):
            if 'dense_vecs' in query_embeddings and 'lexical_weights' in query_embeddings and 'colbert_vecs' in query_embeddings:
                query_dense_vec = query_embeddings['dense_vecs'] # (1, dim)
                query_lexical_weights = query_embeddings['lexical_weights'][0] # Dict[int, float]
                # IMPORTANT: Extract the single query's colbert vec, which is (num_tokens, dim)
                query_colbert_vec = query_embeddings['colbert_vecs'][0].astype("float32")
            else:
                raise ValueError(
                    "Query embeddings dictionary from BGEM3FlagModel does not contain 'dense_vecs', 'lexical_weights', or 'colbert_vecs' keys. "
                    f"Keys found: {query_embeddings.keys()}. This is unexpected."
                )
        else:
            raise TypeError(
                f"Unexpected type for query_embeddings from BGEM3FlagModel: {type(query_embeddings)}. Expected dict."
            )

        if query_lexical_weights is None and sparse_weight > 0:
            logger.warning("Lexical weights not available for query, but sparse_weight > 0. Setting sparse_weight to 0.")
            sparse_weight = 0.0
        
        if query_colbert_vec is None and colbert_weight > 0:
            logger.warning("Colbert vectors not available for query, but colbert_weight > 0. Setting colbert_weight to 0.")
            colbert_weight = 0.0

        # ----- 1. Dense Similarity (Cosine) -----
        dense_scores = cosine_similarity(query_dense_vec, self.full_dense_vectors)[0]

        # ----- 2. Sparse (Lexical) Similarity -----
        sparse_scores = np.zeros_like(dense_scores) # Initialize with zeros
        if query_lexical_weights is not None:
            sparse_scores = self._compute_sparse_scores(query_lexical_weights, self.full_sparse_vectors)

        # ----- 3. Colbert Similarity -----
        colbert_scores = np.zeros_like(dense_scores)
        if query_colbert_vec is not None:
            # Pass the single query's colbert vector and the list of corpus colbert vectors
            colbert_scores = self._compute_colbert_scores(query_colbert_vec, self.full_colbert_vectors)

        # ----- 4. Normalize and Combine Scores -----
        norm_dense = self._normalize_scores(dense_scores)
        norm_sparse = self._normalize_scores(sparse_scores)
        norm_colbert = self._normalize_scores(colbert_scores)

        # Calculate dense_weight based on remaining weight
        dense_weight = 1.0 - sparse_weight - colbert_weight
        if dense_weight < 0: # Should not happen if previous check passes, but as a safeguard
             dense_weight = 0 
        
        hybrid_scores = ((dense_weight) * norm_dense) + \
                        (sparse_weight * norm_sparse) + \
                        (colbert_weight * norm_colbert)
        
        self._print_top_k_results(dense_scores, sparse_scores, colbert_scores, hybrid_scores, top_k)
        
        # ----- 5. Get Top-K Results -----
        best_indices = np.argsort(hybrid_scores)[::-1]

        # ----- Assemble results -----
        results: List[Dict[str, Union[str, float, int]]] = []
        for rank in range(min(top_k, len(best_indices))):
            idx = best_indices[rank]
            matched_row = self.full_df.iloc[idx]
            results.append({
                "matched_prompt": matched_row["Prompt"],
                "response": matched_row["Response"],
                "instruction": matched_row["Instruction"],
                "score": float(hybrid_scores[idx]),
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "colbert_score": float(colbert_scores[idx]), # New: Colbert score in results
                "question_id": int(matched_row["question_id"]),
                "answer_id": int(matched_row["answer_id"]),
            })
        return results

# Helper for REPL – keeps original behaviour but simplified
if __name__ == "__main__":
    
    # Initialize the matcher
    # Make sure './WebScrape/data_whole_page' exists and contains your Q&A data
    # in 'dopomoha_general_pro/en' and 'dopomoha_general_pro_answers/en'
    matcher = PromptMatcher(
        base_data_path="./WebScrape/data_whole_page", # Adjust this path if needed
        language="en",
        concat_q_and_a=False, # Set to True to embed Q+A pairs together
    )

    def main_repl():
        """A simple Read-Eval-Print-Loop to test the matcher."""
        try:
            while True:
                q = input("\nAsk something (or 'quit'): ").strip()
                if q.lower() == "quit":
                    break
                
                # Perform the hybrid query
                # Ensure sparse_weight + colbert_weight <= 1.0
                hits = matcher.query(q, top_k=3, sparse_weight=0.3, colbert_weight=0.3)
                
                print(f"\n--- Top {len(hits)} result(s) ---")
                for rnk, hit in enumerate(hits, 1):
                    print(f"\nMatch {rnk}: (Hybrid Score: {hit['score']:.4f})")
                    print(f"  Q-ID {hit['question_id']} » {hit['matched_prompt']}")
                    print(f"  A-ID {hit['answer_id']} » {hit['response']}")
                    print(f"  Instruction: {hit['instruction']}")
                    print(f"  (Debug: Dense={hit['dense_score']:.4f}, Sparse={hit['sparse_score']:.4f}, Colbert={hit['colbert_score']:.4f})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye! 👋")

    main_repl()