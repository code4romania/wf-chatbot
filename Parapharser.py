import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ====== SETTINGS ======
csv_input_path = "data.csv"         # ← Your input file
csv_output_path = "human_like_questions_mistral_batched.csv"
n_questions_per_prompt = 5                # ← Number of human-like questions per original prompt
batch_size = 8                             # ← Number of prompts to process together
max_new_tokens = 512

# ====== Load Mistral Model ======
model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading model '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====== Helper function ======
def format_instruction(prompt, n_variations):
    return (
        f"Based on the following topic, create {n_variations} casual, natural questions. "
        f"Stay in the same language as the topic. Ask questions like a curious real person would.\n\n"
        f"Topic: {prompt}\n\n"
        f"List the questions as numbered lines."
    )

def parse_generated_output(output_text):
    """Split model output into list of human-like questions."""
    questions = []
    lines = output_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line and any(c.isalpha() for c in line):
            if "." in line:
                line = line.split(".", 1)[-1].strip()
            questions.append(line)
    return questions

# ====== Load CSV ======
print(f"Loading prompts from {csv_input_path}...")
df = pd.read_csv(csv_input_path)
df = df.dropna(subset=["Prompt", "Response"])

generated_data = []

# ====== Prepare all instructions ======
instructions = [format_instruction(prompt, n_questions_per_prompt) for prompt in df["Prompt"].tolist()]
responses = df["Response"].tolist()
original_prompts = df["Prompt"].tolist()

# ====== Batch Processing ======
print("Generating human-like questions with Mistral in batches...")

for batch_start in tqdm(range(0, len(instructions), batch_size)):
    batch_end = batch_start + batch_size
    batch_instructions = instructions[batch_start:batch_end]
    batch_original_prompts = original_prompts[batch_start:batch_end]
    batch_responses = responses[batch_start:batch_end]

    # Tokenize the batch
    inputs = tokenizer(
        batch_instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            num_return_sequences=1,
            early_stopping=True
        )

    # Decode all outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Parse and save each set of questions
    for decoded_output, original_prompt, response in zip(decoded_outputs, batch_original_prompts, batch_responses):
        questions = parse_generated_output(decoded_output)
        questions = list(dict.fromkeys(questions))  # Deduplicate

        for question in questions[:n_questions_per_prompt]:
            generated_data.append({
                "generated_question": question,
                "original_prompt": original_prompt,
                "response": response
            })

# ====== Save Output ======
test_df = pd.DataFrame(generated_data)
test_df.to_csv(csv_output_path, index=False)

print(f"\n✅ Done! Human-like questions saved to {csv_output_path}")
