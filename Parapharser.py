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
    paraphrases = []

    input_text = f"paraphrase: {prompt} </s>"

    encoding = tokenizer(
        [input_text],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **encoding,
            max_length=256,
            num_return_sequences=n_variations,
            do_sample=True,
            temperature=1.2,
            top_k=50,
            top_p=0.92
        )

    paraphrases = [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                   for output in outputs]

    # Deduplicate locally just in case
    paraphrases = list(dict.fromkeys(paraphrases))

    return paraphrases[:n_variations]

# Create and save the new test dataset
test_df = pd.DataFrame(paraphrased_data)
test_df.to_csv("test_dataset_paraphrased_local.csv", index=False)

print("\n✅ Local paraphrased dataset created: test_dataset_paraphrased_local.csv")
