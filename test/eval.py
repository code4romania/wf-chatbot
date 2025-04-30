import os
import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from PromptMatcher import PromptMatcher

# === 1. Reset ChatterBot DB ===
if os.path.exists("mybot.sqlite3"):
    os.remove("mybot.sqlite3")

chatbot = ChatBot(
    'MyCustomBot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    database_uri='sqlite:///mybot.sqlite3',
    logic_adapters=['chatterbot.logic.BestMatch']
)

trainer = ListTrainer(chatbot)

# === 2. Train ChatterBot from scratch ===
training_data = pd.read_csv("data.csv")
training_data = training_data[training_data["Status"].str.lower() == "trained"]
training_data = training_data.dropna(subset=["Prompt", "Response"])

print(f"Training ChatterBot on {len(training_data)} prompt-response pairs...")

for _, row in training_data.iterrows():
    trainer.train([row["Prompt"].strip(), row["Response"].strip()])

# === 3. Load paraphrased test prompts ===
test_df = pd.read_csv("good_questions.csv")
test_df.columns = ["generated_question", "original_prompt", "original_response"]
test_df = test_df.dropna(subset=["generated_question", "original_prompt"])

# Ground truth from main dataset
ground_truth = dict(zip(training_data["Prompt"], training_data["Response"]))

# === 4. Evaluate both systems ===
matcher = PromptMatcher("data.csv")



cb_correct = 0
pm_correct = 0
total = 0

for _, row in test_df.iterrows():
    paraphrase = row["generated_question"].strip()
    original = row["original_prompt"].strip()

    expected = ground_truth.get(original, "").strip()
    if not expected:
        continue  # Skip if we don't have a matching training pair

    # PromptMatcher prediction
    pm_result = matcher.query(paraphrase)
    pm_pred = pm_result["response"].strip()
    if pm_pred == expected:
        pm_correct += 1

    # ChatterBot prediction
    cb_pred = chatbot.get_response(paraphrase).text.strip()
    if cb_pred == expected:
        cb_correct += 1

    total += 1

# === 5. Results ===
print("\n=== Evaluation Results ===")
print(f"Total test cases:        {total}")
print(f"PromptMatcher accuracy:  {pm_correct / total:.2%}")
print(f"ChatterBot accuracy:     {cb_correct / total:.2%}")
