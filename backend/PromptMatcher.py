import pandas as pd
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Dict, Union

class PromptMatcher:
    def __init__(self, base_data_path: str, language: str = "en", model_name: str = "all-MiniLM-L6-v2"):
        self.base_data_path = base_data_path
        self.language = language
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.prompts: List[str] = []
        self.responses: List[str] = []
        self.question_ids: List[int] = []
        self.answer_ids: List[int] = []
        self.prompt_embeddings: np.ndarray = None
        self.process_data()

    def process_data(self):
        questions_dir = os.path.join(self.base_data_path, "dopomoha_questions", self.language)
        answers_dir = os.path.join(self.base_data_path, "dopomoha_answers", self.language)

        if not os.path.isdir(questions_dir):
            raise FileNotFoundError(f"Questions directory not found: {questions_dir}")
        if not os.path.isdir(answers_dir):
            raise FileNotFoundError(f"Answers directory not found: {answers_dir}")

        all_questions: Dict[int, str] = {}
        all_answers_with_ids: Dict[int, Dict[str, Union[str, int]]] = {}

        for filename in os.listdir(questions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(questions_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "questions" in data:
                        for q_entry in data["questions"]:
                            if "question_id" in q_entry and "question" in q_entry:
                                all_questions[q_entry["question_id"]] = q_entry["question"]

        for filename in os.listdir(answers_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(answers_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "answers" in data:
                        for a_entry in data["answers"]:
                            if "question_id" in a_entry and "answer_id" in a_entry and "answer" in a_entry:
                                all_answers_with_ids[a_entry["question_id"]] = {
                                    "answer": a_entry["bResponse"],
                                    "answer_id": a_entry["answer_id"]
                                }

        linked_data = []
        for q_id, question_text in all_questions.items():
            if q_id in all_answers_with_ids:
                linked_data.append({
                    "Prompt": question_text,
                    "Response": all_answers_with_ids[q_id]["answer"],
                    "question_id": q_id,
                    "answer_id": all_answers_with_ids[q_id]["answer_id"]
                })

        if not linked_data:
            print("Warning: No linked questions and answers found. Please check data files.")
            return

        self.df = pd.DataFrame(linked_data)
        self.prompts = self.df["Prompt"].tolist()
        self.responses = self.df["Response"].tolist()
        self.question_ids = self.df["question_id"].tolist()
        self.answer_ids = self.df["answer_id"].tolist()

        self.prompt_embeddings = self.model.encode(self.prompts, normalize_embeddings=True)
        print(f"Processed {len(self.prompts)} prompt-response pairs for language '{self.language}'.")

    def query(self, user_prompt: str, metric: str = "cosine", top_k: int = 1) -> Union[Dict[str, Union[str, float, int]], List[Dict[str, Union[str, float, int]]]]:
        if self.prompt_embeddings is None or not self.prompts:
            raise ValueError("Data has not been processed or no valid prompts were found.")

        user_embedding = self.model.encode([user_prompt], normalize_embeddings=True)

        if metric == "cosine":
            similarities = cosine_similarity(user_embedding, self.prompt_embeddings)[0]
            sorted_indices = np.argsort(similarities)[::-1]
            scores = similarities
        elif metric == "euclidean":
            distances = euclidean_distances(user_embedding, self.prompt_embeddings)[0]
            sorted_indices = np.argsort(distances)
            scores = distances
        else:
            raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

        results = []
        for i in range(min(top_k, len(sorted_indices))):
            index = sorted_indices[i]
            results.append({
                "matched_prompt": self.prompts[index],
                "response": self.responses[index],
                "score": scores[index],
                "metric": metric,
                "question_id": self.question_ids[index],
                "answer_id": self.answer_ids[index]
            })

        return results[0] if top_k == 1 else results

if __name__ == "__main__":
    # IMPORTANT: Set this to the actual path where your 'dopomoha_questions' and 'dopomoha_answers' folders reside
    # For example: actual_data_path = "/home/s4402146/Documents/CommitGlobal/WebScrape/data"
    actual_data_path = "~/WebScrape/data_whole_page"

    try:
        matcher = PromptMatcher(base_data_path=actual_data_path, language="en")

        while True:
            user_query = input("\nAsk something (or type 'quit'): ")
            if user_query.lower() == "quit":
                break

            # Set top_k to a value greater than 1 to get multiple matches
            # For example, top_k=3 will return the 3 best matches
            top_k_value = 1
            results = matcher.query(user_query, metric="cosine", top_k=top_k_value)

            print(f"\n--- Top {top_k_value} Matches ---")
            if isinstance(results, list): # This will be true if top_k > 1
                if results:
                    for i, result in enumerate(results):
                        print(f"\nMatch {i+1}:")
                        print(f"  Matched Prompt (ID: {result['question_id']}): {result['matched_prompt']}")
                        print(f"  Response (ID: {result['answer_id']}): {result['response']}")
                        print(f"  Score: {result['score']:.4f}")
                        print("-" * 30) # Separator for readability
                else:
                    print("No suitable matches found.")
            else: # This case handles when top_k was explicitly set to 1
                print(f"Matched Prompt (ID: {results['question_id']}): {results['matched_prompt']}")
                print(f"Response (ID: {results['answer_id']}): {results['response']}")
                print(f"Score: {results['score']:.4f}")

    except FileNotFoundError as e:
        print(f"\nError: {e}. Please ensure 'base_data_path' is set correctly and the directories exist.")
    except ValueError as e:
        print(f"\nError processing data or during query: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")