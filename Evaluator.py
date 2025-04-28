import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==== Load your datasets ====

# Test set of paraphrased prompts
test_data = pd.read_csv('your_paraphrased_test_dataset.csv')  # <-- adjust filename if needed
# Your "ground truth" dataset
original_data = pd.read_csv('your_original_dataset.csv')  # <-- adjust filename if needed

# ==== Setup your Prompt Matcher (cosine-based) ====

vectorizer = TfidfVectorizer()

# Vectorize the original prompts
original_prompts = original_data['Prompt'].fillna('').tolist()
X_original = vectorizer.fit_transform(original_prompts)

# ==== Setup ChatterBot ====

chatbot = ChatBot(
    'ComparisonBot',
    read_only=True,
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'No good match found',
            'maximum_similarity_threshold': 0.90
        }
    ]
)

# Train it with your original prompt-response pairs
trainer = ListTrainer(chatbot)
trainer.train(list(original_data[['Prompt', 'Response']].dropna().to_numpy().flatten()))

# ==== Compare accuracies ====

promptmatcher_correct = 0
chatterbot_correct = 0
total = 0

for idx, row in test_data.iterrows():
    paraphrased_prompt = row['generated_question']
    original_prompt = row['original_prompt']

    if pd.isna(paraphrased_prompt) or pd.isna(original_prompt):
        continue  # skip invalid rows

    # ---- Prompt Matcher (Cosine) prediction ----
    X_test = vectorizer.transform([paraphrased_prompt])
    cosine_scores = cosine_similarity(X_test, X_original)
    best_idx = cosine_scores.argmax()
    predicted_prompt = original_prompts[best_idx]

    if predicted_prompt.strip().lower() == original_prompt.strip().lower():
        promptmatcher_correct += 1

    # ---- ChatterBot prediction ----
    bot_response = chatbot.get_response(paraphrased_prompt)
    # Find the closest matching original prompt that led to that answer
    matched_prompt = None
    for i, (prompt, response) in enumerate(zip(original_data['Prompt'], original_data['Response'])):
        if str(bot_response) == str(response):
            matched_prompt = prompt
            break

    if matched_prompt and matched_prompt.strip().lower() == original_prompt.strip().lower():
        chatterbot_correct += 1

    total += 1

# ==== Results ====

print(f"Total test samples: {total}")
print(f"PromptMatcher Accuracy: {promptmatcher_correct / total:.2%}")
print(f"ChatterBot Accuracy: {chatterbot_correct / total:.2%}")
