import pandas as pd
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from PromptMatcher import PromptMatcher  # <-- ADJUST if needed

# ====== Load datasets ======

test_data = pd.read_csv('your_paraphrased_test_dataset.csv')  # generated_question, original_prompt
original_data = pd.read_csv('your_original_dataset.csv')      # Prompt, Response

# ====== Setup PromptMatcher ======

matcher = PromptMatcher('your_original_dataset.csv', model_name="all-MiniLM-L6-v2")

# ====== Setup ChatterBot ======

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

trainer = ListTrainer(chatbot)
trainer.train(list(original_data[['Prompt', 'Response']].dropna().to_numpy().flatten()))

# ====== Evaluation ======

promptmatcher_correct = 0
chatterbot_correct = 0
total = 0

detailed_results = []

for idx, row in test_data.iterrows():
    paraphrased_prompt = row['generated_question']
    true_original_prompt = row['original_prompt']

    if pd.isna(paraphrased_prompt) or pd.isna(true_original_prompt):
        continue

    # ----- PromptMatcher prediction -----
    match_result = matcher.query(paraphrased_prompt, metric="cosine")
    matched_prompt_pm = match_result['matched_prompt']

    is_pm_correct = matched_prompt_pm.strip().lower() == true_original_prompt.strip().lower()
    if is_pm_correct:
        promptmatcher_correct += 1

    # ----- ChatterBot prediction -----
    bot_response = chatbot.get_response(paraphrased_prompt)
    matched_prompt_cb = None

    for i, (prompt, response) in enumerate(zip(original_data['Prompt'], original_data['Response'])):
        if str(bot_response) == str(response):
            matched_prompt_cb = prompt
            break

    is_cb_correct = matched_prompt_cb and (matched_prompt_cb.strip().lower() == true_original_prompt.strip().lower())
    if is_cb_correct:
        chatterbot_correct += 1

    total += 1

    detailed_results.append({
        'paraphrased_prompt': paraphrased_prompt,
        'true_original_prompt': true_original_prompt,
        'promptmatcher_matched_prompt': matched_prompt_pm,
        'promptmatcher_correct': is_pm_correct,
        'chatterbot_matched_prompt': matched_prompt_cb,
        'chatterbot_correct': is_cb_correct,
    })

# ====== Print Results ======

print(f"\n=== Evaluation Results ===")
print(f"Total samples tested: {total}")
print(f"PromptMatcher Accuracy: {promptmatcher_correct / total:.2%}")
print(f"ChatterBot Accuracy: {chatterbot_correct / total:.2%}")

# ====== Save detailed results ======

pd.DataFrame(detailed_results).to_csv('matching_detailed_results.csv', index=False)
print("\nDetailed results saved to matching_detailed_results.csv ✅")
