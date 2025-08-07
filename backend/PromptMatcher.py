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
import torch # <--- ADD THIS IMPORT
from transformers import AutoTokenizer, AutoModelForSequenceClassification # For cross-encoder

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BGE_M3_MODEL = "BAAI/bge-m3"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3" # Or "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 8192 # Consider removing or commenting on its non-use for BGE-M3 internal truncation
logger = logging.getLogger("promptmatcher")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(message)s')

with open("RetrievalParams.json", 'r') as f:
    params = json.load(f)
    
questions_path = params["questions_path"]
answers_path = params["answers_path"]
base_path= params["base_path"]
concat_qa= params["concat_qa"]



class PromptMatcher:
    """
    Builds dense, sparse, and Colbert embedding matrices for a Q&A corpus and performs
    a hybrid search by combining scores from all representations using BGE-M3
    via FlagEmbedding. Includes a reranking step with a cross-encoder.
    """

    def __init__(
        self,
        language: str = "en",
        device: str = "cpu"
    ):
        self.base_data_path = Path("CorpusGeneration/corpus/",base_path )
        self.language = language.lower()
        self.concat_q_and_a = concat_qa
        self.model_name = BGE_M3_MODEL
        self.cross_encoder_model_name = CROSS_ENCODER_MODEL
        self.device = device
        self.torch_device = torch.device(device) # Store as torch.device for consistency

        logger.info(f"Loading retriever model {self.model_name} on {self.device}…")
        # Ensure use_fp16=True is compatible with your device and desired precision
        self.model = BGEM3FlagModel(self.model_name, device=self.device, use_fp16=True)

        logger.info(f"Loading cross-encoder model {self.cross_encoder_model_name} on {self.device}…")
        self.cross_encoder_tokenizer = AutoTokenizer.from_pretrained(self.cross_encoder_model_name)
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(self.cross_encoder_model_name).to(self.torch_device).eval()

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
        logger.info(f"About to encode {len(embed_texts)} passages…")

        embeddings = self.model.encode(
            self.full_df["embed_txt"].tolist(),
            batch_size=32,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
        )
        
        # --- Handling output from BGEM3FlagModel ---
        if isinstance(embeddings, dict):
            if 'dense_vecs' in embeddings and 'lexical_weights' in embeddings and 'colbert_vecs' in embeddings:
                self.full_dense_vectors = embeddings['dense_vecs']
                self.full_sparse_vectors = embeddings['lexical_weights']
                self.full_colbert_vectors = embeddings['colbert_vecs']
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

        # Ensure all vectors are float32 for consistency in numpy operations, especially if use_fp16=True was set
        if self.full_dense_vectors.dtype != np.float32:
            self.full_dense_vectors = self.full_dense_vectors.astype(np.float32)
        self.full_colbert_vectors = [vec.astype(np.float32) if vec.dtype != np.float32 else vec for vec in self.full_colbert_vectors]


        logger.info(f"Full corpus with {len(self.full_df)} dense/sparse/Colbert embeddings ready.")

        # Save to cache
        logger.info(f"Saving corpus and vectors to cache: {cache_dir}")
        self.full_df.to_pickle(cache_df_path)
        np.save(cache_dense_vec_path, self.full_dense_vectors)
        with open(cache_sparse_vec_path, "wb") as f:
            pickle.dump(self.full_sparse_vectors, f)
        with open(cache_colbert_vec_path, "wb") as f:
            pickle.dump(self.full_colbert_vectors, f)


    # Reverted to original concept for _pad_and_stack_colbert_vecs, but it's not used by _compute_colbert_scores anymore
    # The batching for colbert_score will be handled explicitly within _compute_colbert_scores
    # to address the FlagEmbedding library's expectation.
    def _pad_and_stack_colbert_vecs(self, colbert_vecs_list: List[np.ndarray]) -> torch.Tensor:
        """
        Helper function to pad and stack a list of variable-length Colbert vectors
        into a single batched torch tensor. (Used for other potential batching, not
        directly for the current self.model.colbert_score as it seems to expect
        individual vectors).
        """
        if not colbert_vecs_list:
            return torch.empty(0, 0, 0, device=self.torch_device)

        max_tokens = max(vec.shape[0] for vec in colbert_vecs_list)
        # colbert_dim = colbert_vecs_list[0].shape[1] # Not strictly needed

        padded_vecs = []
        for vec in colbert_vecs_list:
            padding_needed = max_tokens - vec.shape[0]
            if padding_needed > 0:
                padded_vec = np.pad(vec, ((0, padding_needed), (0, 0)), mode='constant', constant_values=0)
            else:
                padded_vec = vec
            padded_vecs.append(padded_vec)
        
        return torch.from_numpy(np.stack(padded_vecs, axis=0)).to(self.torch_device)


    def _compute_sparse_scores(self, query_lexical_weights: Dict[int, float], corpus_lexical_weights_list: List[Dict[int, float]]) -> np.ndarray:
        """
        Computes lexical matching scores using BGEM3FlagModel's method.
        """
        scores = []
        for doc_lexical_weights in corpus_lexical_weights_list:
            score = self.model.compute_lexical_matching_score(query_lexical_weights, doc_lexical_weights)
            scores.append(score)
        return np.array(scores, dtype=np.float32)
    
    def _compute_colbert_scores(self, query_colbert_vec: np.ndarray, corpus_colbert_vecs_list: List[np.ndarray]) -> np.ndarray:
        """
        Computes Colbert matching scores by iterating and scoring each corpus vector
        individually against the query vector, as FlagEmbedding's colbert_score
        method expects individual (N_tokens, D) inputs, not (Batch, N_tokens, D).
        """
        if not corpus_colbert_vecs_list:
            return np.array([], dtype=np.float32)

        # Ensure query_colbert_vec has a "batch" dimension of 1 for model.colbert_score,
        # but keep it as NumPy array for now.
        # It needs to be (1, N_q, D) if colbert_score internally expects a batch of 1 query.
        # Or, if it expects (N_q, D) it should be just query_colbert_vec.
        # Let's assume it expects (1, N_q, D) as it often does for similar models.
        # UPDATE: FlagEmbedding's m3.py's colbert_score method actually expects (num_tokens, dim) for both q_reps and p_reps,
        # and it internally adds a batch dimension if needed. So, we should pass the raw 2D query_colbert_vec.
        
        scores = []
        # Iterate through each passage's Colbert vector
        for passage_colbert_vec in corpus_colbert_vecs_list:
            # Pass individual (N_q, D) and (N_p, D) numpy arrays
            score_tensor = self.model.colbert_score(query_colbert_vec, passage_colbert_vec)
            scores.append(score_tensor.item()) # .item() to get scalar from 0-dim tensor

        return np.array(scores, dtype=np.float32) # Convert list of scores to numpy array
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


    def _rerank_results(self, user_query: str, retrieval_results: List[Dict[str, Any]], rerank_top_n: int) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieval results using a cross-encoder model.
        """
        if not retrieval_results:
            return []

        logger.info(f"Reranking top {len(retrieval_results)} results with cross-encoder (top {rerank_top_n} desired)...")

        # Prepare pairs for cross-encoder
        # Each pair is [query, passage]
        rerank_pairs = []
        for res in retrieval_results:
            # You can combine Prompt and Response for the passage if concat_q_and_a is True for main embedding
            # or just use the Prompt, depending on what you want the cross-encoder to evaluate.
            # For simplicity, let's use the 'Prompt' (question) as the passage text for reranking.
            # If you concatenated Q+A for initial retrieval, using Q+A here might be more consistent.
            passage_text = res["matched_prompt"]
            if self.concat_q_and_a: # Or decide if you *always* want Q+A for reranker
                passage_text = f"{res['matched_prompt']} {res['response']}" # Reranker should see all relevant text
            rerank_pairs.append([user_query, passage_text])

        # Batch tokenization and inference
        inputs = self.cross_encoder_tokenizer(
            rerank_pairs, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.torch_device)

        with torch.no_grad():
            scores = self.cross_encoder_model(**inputs).logits.squeeze().cpu().numpy()

        # If only one result, scores might be a scalar. Convert to array.
        if scores.ndim == 0:
            scores = np.array([scores.item()])

        # Associate scores back with results
        for i, score in enumerate(scores):
            retrieval_results[i]["rerank_score"] = float(score)

        # Sort by rerank_score in descending order
        reranked_results = sorted(retrieval_results, key=lambda x: x["rerank_score"], reverse=True)

        return reranked_results[:rerank_top_n]


    def query(
        self,
        user_prompt: str,
        top_k: int = 1, # Number of final results after reranking
        sparse_weight: float = 0.3,
        colbert_weight: float = 0.3,
        retrieval_top_n: int = 10, # Number of hybrid hits to consider for reranking
        rerank_top_n: int = 3 # Number of final results after reranking
    ) -> List[Dict[str, Union[str, float, int]]]:
        """
        Return top-K matches for a prompt using a weighted hybrid of dense, sparse, and Colbert search,
        followed by a cross-encoder reranking step.
        """
        if self.full_df is None or self.full_dense_vectors is None or \
           self.full_sparse_vectors is None or self.full_colbert_vectors is None:
            logger.error("Corpus is not initialized. Cannot perform search.")
            return []
        
        if not (0.0 <= sparse_weight <= 1.0) or not (0.0 <= colbert_weight <= 1.0):
            raise ValueError("sparse_weight and colbert_weight must be between 0.0 and 1.0")
        
        if sparse_weight + colbert_weight > 1.0:
            raise ValueError("Sum of sparse_weight and colbert_weight cannot exceed 1.0")

        # Ensure top_k is respected for rerank_top_n if it's smaller
        rerank_top_n = min(rerank_top_n, top_k)
        if rerank_top_n < top_k:
            logger.warning(f"rerank_top_n ({rerank_top_n}) is less than top_k ({top_k}). Setting top_k to rerank_top_n.")
            top_k = rerank_top_n


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
                query_colbert_vec = query_embeddings['colbert_vecs'][0]
            else:
                raise ValueError(
                    "Query embeddings dictionary from BGEM3FlagModel does not contain 'dense_vecs', 'lexical_weights', or 'colbert_vecs' keys. "
                    f"Keys found: {query_embeddings.keys()}. This is unexpected."
                )
        else:
            raise TypeError(
                f"Unexpected type for query_embeddings from BGEM3FlagModel: {type(query_embeddings)}. Expected dict."
            )

        # Ensure query vectors are float32 for consistency in similarity calculations if corpus is float32
        if query_dense_vec.dtype != np.float32:
            query_dense_vec = query_dense_vec.astype(np.float32)
        if query_colbert_vec is not None and query_colbert_vec.dtype != np.float32: # colbert_vec can be None
            query_colbert_vec = query_colbert_vec.astype(np.float32)

        if query_lexical_weights is None and sparse_weight > 0:
            logger.warning("Lexical weights not available for query, but sparse_weight > 0. Setting sparse_weight to 0.")
            sparse_weight = 0.0
        
        if query_colbert_vec is None and colbert_weight > 0:
            logger.warning("Colbert vectors not available for query, but colbert_weight > 0. Setting colbert_weight to 0.")
            colbert_weight = 0.0

        # ----- 1. Dense Similarity (Cosine) -----
        dense_scores = cosine_similarity(query_dense_vec, self.full_dense_vectors)[0]

        # ----- 2. Sparse (Lexical) Similarity -----
        sparse_scores = np.zeros_like(dense_scores, dtype=np.float32) # Initialize with zeros, specify dtype
        if query_lexical_weights is not None:
            sparse_scores = self._compute_sparse_scores(query_lexical_weights, self.full_sparse_vectors)

        # ----- 3. Colbert Similarity -----
        colbert_scores = np.zeros_like(dense_scores, dtype=np.float32)
        if query_colbert_vec is not None and self.full_colbert_vectors: # Check if corpus colbert vecs exist too
            colbert_scores = self._compute_colbert_scores(query_colbert_vec, self.full_colbert_vectors)

        # ----- 4. Normalize and Combine Scores (Initial Hybrid Retrieval) -----
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
        
        self._print_top_k_results(dense_scores, sparse_scores, colbert_scores, hybrid_scores, retrieval_top_n) # Print for retrieval_top_n
        
        # ----- 5. Get Top-N Retrieval Results for Reranking -----
        # Get more results than final top_k, to allow reranker to improve quality
        best_retrieval_indices = np.argsort(hybrid_scores)[::-1][:retrieval_top_n]

        retrieval_results: List[Dict[str, Union[str, float, int]]] = []
        for idx in best_retrieval_indices:
            matched_row = self.full_df.iloc[idx]
            retrieval_results.append({
                "matched_prompt": matched_row["Prompt"],
                "response": matched_row["Response"],
                "instruction": matched_row["Instruction"],
                "score": float(hybrid_scores[idx]), # Hybrid score from initial retrieval
                "dense_score": float(dense_scores[idx]),
                "sparse_score": float(sparse_scores[idx]),
                "colbert_score": float(colbert_scores[idx]),
                "question_id": int(matched_row["question_id"]),
                "answer_id": int(matched_row["answer_id"]),
            })

        # ----- 6. Rerank the Retrieval Results -----
        reranked_final_results = self._rerank_results(user_prompt, retrieval_results, rerank_top_n)

        # Log final reranked results (optional, for debugging)
        if reranked_final_results:
            logger.info(f"\n--- Final Reranked Top {len(reranked_final_results)} Results ---")
            for rank, hit in enumerate(reranked_final_results, 1):
                logger.info(f"  {rank}. Rerank Score: {hit['rerank_score']:.4f} - Prompt: {hit['matched_prompt'][:100]}{'...' if len(hit['matched_prompt']) > 100 else ''}")
                logger.info(f"     (Initial Hybrid Score: {hit['score']:.4f})")

        return reranked_final_results


# Helper for REPL – keeps original behaviour but simplified
if __name__ == "__main__":
    
    # Initialize the matcher
    # Make sure './WebScrape/data_whole_page' exists and contains your Q&A data
    # in 'dopomoha_general_pro/en' and 'dopomoha_general_pro_answers/en'
    matcher = PromptMatcher()

    def main_repl():
        """A simple Read-Eval-Print-Loop to test the matcher."""
        try:
            while True:
                q = input("\nAsk something (or 'quit'): ").strip()
                if q.lower() == "quit":
                    break
                
                # Perform the hybrid query with reranking
                # retrieval_top_n: how many hybrid hits to pass to reranker (e.g., 50)
                # rerank_top_n: how many final results to return after reranking (e.g., 5)
                hits = matcher.query(q, top_k=5, sparse_weight=0.3, colbert_weight=0.3, retrieval_top_n=10, rerank_top_n=3)
                
                print(f"\n--- Final Reranked Top {len(hits)} result(s) ---")
                for rnk, hit in enumerate(hits, 1):
                    print(f"\nMatch {rnk}: (Rerank Score: {hit['rerank_score']:.4f})")
                    print(f"  Q-ID {hit['question_id']} » {hit['matched_prompt']}")
                    print(f"  A-ID {hit['answer_id']} » {hit['response']}")
                    print(f"  Instruction: {hit['instruction']}")
                    print(f"  (Debug: Initial Hybrid={hit['score']:.4f}, Dense={hit['dense_score']:.4f}, Sparse={hit['sparse_score']:.4f}, Colbert={hit['colbert_score']:.4f})")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Bye! 👋")

    main_repl()