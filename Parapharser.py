import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# ====== SETTINGS ======
csv_input_path = "data.csv"         # ← Your input CSV file
csv_output_path = "test_deepseek.csv"
n_questions_per_prompt = 5                # ← Number of human-like questions per original prompt
batch_size = 4                             # ← Adjust based on your GPU memory
max_new_tokens = 256

# ====== Load DeepSeek Model ======
model_name = "deepseek-ai/deepseek-llm-7b-chat"

print(f"Loading model '{model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # Required for batching

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
).eval()

# No need to move manually with device_map="auto"

# ====== Helper functions ======

def format_instruction(prompt, n_variations):
    """Format strict instruction to force clean output."""
    return (
        f"<|system|>\n"
        f"You are a helpful assistant.\n"
        f"Your task:\n"
        f"- Given a topic, generate exactly {n_variations} casual, natural questions.\n"
        f"- Stay in the SAME LANGUAGE as the topic.\n"
        f"- Output ONLY the {n_variations} questions as a numbered list.\n"
        f"- DO NOT repeat the topic, do not explain anything, do not add any extra text.\n"
        f"- Each line must start with a number and a dot (e.g., '1.').\n\n"
        f"<|user|>\n"
        f"Topic: {prompt}\n\n"
        f"<|assistant|>\n"
        f"1. "
    )

def parse_generated_output(output_text):
    """Extract clean numbered questions."""
    questions = []
    lines = output_text.strip().split("\n")
    for line in lines:
        match = re.match(r"^\d+\.\s*(.+)", line.strip())
        if match:
            question = match.group(1).strip()
            if question and any(c.isalpha() for c in question):
                questions.append(question)
    return questions

# ====== Load CSV ======
print(f"Loading prompts from {csv_input_path}...")
df = pd.read_csv(csv_input_path)
df = df.dropna(subset=["Prompt", "Response"])

generated_data = []

# ====== Prepare instructions ======
instructions = [format_instruction(prompt, n_questions_per_prompt) for prompt in df["Prompt"].tolist()]
responses = df["Response"].tolist()
original_prompts = df["Prompt"].tolist()

# ====== Batch Processing ======
print("Generating human-like questions with DeepSeek in batches...")

for batch_start in tqdm(range(0, len(instructions), batch_size)):
    batch_end = batch_start + batch_size
    batch_instructions = instructions[batch_start:batch_end]
    batch_original_prompts = original_prompts[batch_start:batch_end]
    batch_responses = responses[batch_start:batch_end]

    # Tokenize batch
    inputs = tokenizer(
        batch_instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

    # Generate outputs
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            num_return_sequences=1,
            early_stopping=True
        )

    # Decode all outputs
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Parse and save questions
    for decoded_output, original_prompt, response in zip(decoded_outputs, batch_original_prompts, batch_responses):
        questions = parse_generated_output(decoded_output)
        questions = list(dict.fromkeys(questions))  # Deduplicate

        if not questions:
            print(f"⚠️ No valid questions generated for: {original_prompt}")
            continue

        for question in questions[:n_questions_per_prompt]:
            generated_data.append({
                "generated_question": question,
                "original_prompt": original_prompt,
                "response": response
            })

# ====== Save Output ======
print(f"Saving generated dataset to {csv_output_path}...")
test_df = pd.DataFrame(generated_data)
test_df.to_csv(csv_output_path, index=False)

print(f"\n✅ Done! Human-like questions saved to {csv_output_path}")
