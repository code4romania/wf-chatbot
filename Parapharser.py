import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

model_name = "ramsrigouthamg/t5_paraphraser"

tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the original CSV
df = pd.read_csv("data.csv")
df = df.dropna(subset=["Prompt", "Response"])

# Storage for paraphrased dataset
paraphrased_data = []

def generate_paraphrases(prompt, n_variations=10):
    """Generates n paraphrases for a given prompt locally."""
    paraphrases = []

    input_text = f"paraphrase: {prompt} </s>"

    encoding = tokenizer(
        [input_text] * n_variations,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_length=256,
            num_beams=10,
            num_return_sequences=n_variations,
            temperature=1.5,
            top_k=120,
            top_p=0.95,
            early_stopping=True
        )

    paraphrases = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for output in outputs]

    return paraphrases

# Go through each prompt
for idx, row in tqdm(df.iterrows(), total=len(df)):
    original_prompt = row["Prompt"]
    response = row["Response"]

    try:
        paraphrases = generate_paraphrases(original_prompt, n_variations=10)
    except Exception as e:
        print(f"Error paraphrasing prompt '{original_prompt}': {e}")
        paraphrases = []

    for paraphrased_prompt in paraphrases:
        paraphrased_data.append({
            "paraphrased_prompt": paraphrased_prompt,
            "original_prompt": original_prompt,
            "response": response
        })

# Create and save the new test dataset
test_df = pd.DataFrame(paraphrased_data)
test_df.to_csv("test_dataset_paraphrased_local.csv", index=False)

print("\n✅ Local paraphrased dataset created: test_dataset_paraphrased_local.csv")
