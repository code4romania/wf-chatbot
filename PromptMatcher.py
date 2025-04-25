import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class PromptMatcher:
    def __init__(self, csv_path, model_name="all-MiniLM-L6-v2"):
        self.csv_path = csv_path
        self.model = SentenceTransformer(model_name)
        self.df = None
        self.prompts = []
        self.responses = []
        self.prompt_embeddings = None
        self.process_data()

    def process_data(self):
        """Load the CSV, preprocess, and compute prompt embeddings."""
        self.df = pd.read_csv(self.csv_path)
        self.df = self.df.dropna(subset=["Prompt", "Response"])
        self.prompts = self.df["Prompt"].tolist()
        self.responses = self.df["Response"].tolist()
        self.prompt_embeddings = self.model.encode(self.prompts, normalize_embeddings=True)

    def query(self, user_prompt, metric="cosine"):
        """Find the best matching response for a given prompt."""
        if self.prompt_embeddings is None:
            raise ValueError("Data has not been processed. Call process_data() first.")

        user_embedding = self.model.encode([user_prompt], normalize_embeddings=True)

        if metric == "cosine":
            similarities = cosine_similarity(user_embedding, self.prompt_embeddings)[0]
            best_index = np.argmax(similarities)
            score = similarities[best_index]
        elif metric == "euclidean":
            distances = euclidean_distances(user_embedding, self.prompt_embeddings)[0]
            best_index = np.argmin(distances)
            score = distances[best_index]
        else:
            raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")

        return {
            "matched_prompt": self.prompts[best_index],
            "response": self.responses[best_index],
            "score": score,
            "metric": metric
        }

# Example usage
if __name__ == "__main__":
    matcher = PromptMatcher("data.csv")

    while True:
        query = input("\nAsk something (or type 'quit'): ")
        if query.lower() == "quit":
            break
        result = matcher.query(query, metric="cosine")
        print(f"\nMatched Prompt: {result['matched_prompt']}\nScore: {result['score']:.4f}\nResponse: {result['response']}")
