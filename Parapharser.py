import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm
import re
import csv
import os

# ====== SETTINGS ======
csv_input_path = "data.csv"                   # Your input CSV
good_output_path = "good_questions.csv"             # Output for good generations
bad_output_path = "bad_questions.csv"               # Output for faulty generations
n_questions_per_prompt = 5
batch_size = 4                                       # Safe now with quantized model
max_new_tokens = 256                                # Generation length

# ====== Load DeepSeek Model (4-bit Quantized) ======
model_name = "deepseek-ai/deepseek-llm-7b-chat"

print(f"Loading model '{model_name}' (4bit quantized)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",           # Best near-lossless quantization
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
).eval()

model.config.use_cache = False  # Important: reduce VRAM use

# ====== Helper Functions ======

def format_instruction(prompt, n_variations):
    """Format system/user/assistant instruction for DeepSeek."""
    return (
        f"<|system|>\n"
        f"You are a helpful assistant.\n"
        f"Your task:\n"
        f"- Given a topic, generate exactly {n_variations} casual, natural questions.\n"
        f"- Stay in the SAME LANGUAGE as the topic.\n"
        f"- Output ONLY the {n_variations} questions as a numbered list.\n"
        f"- DO NOT repeat the topic, do not explain anything, do not add extra text.\n"
        f"- Each line must start with a number and a dot (e.g., '1.')\n\n"
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

def save_rows_to_csv(file_path, rows, header):
    """Append rows to a CSV file."""
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode="a", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)

# ====== Load Prompts ======
print(f"Loading prompts from {csv_input_path}...")
df = pd.read_csv(csv_input_path)
df = df.dropna(subset=["Prompt", "Response"])

instructions = [format_instruction(prompt, n_questions_per_prompt) for prompt in df["Prompt"].tolist()]
responses = df["Response"].tolist()
original_prompts = df["Prompt"].tolist()

# ====== Prepare Output Files ======
good_header = ["generated_question", "original_prompt", "response"]
bad_header = ["original_prompt", "model_raw_output"]

open(good_output_path, 'w').close()
open(bad_output_path, 'w').close()

print("Generating human-like questions with DeepSeek 4bit...")

# ====== Main Loop ======
for batch_start in tqdm(range(0, len(instructions), batch_size)):
    batch_end = batch_start + batch_size
    batch_instructions = instructions[batch_start:batch_end]
    batch_original_prompts = original_prompts[batch_start:batch_end]
    batch_responses = responses[batch_start:batch_end]

    inputs = tokenizer(
        batch_instructions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    ).to(model.device)

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

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    good_rows = []
    bad_rows = []

    for decoded_output, original_prompt, response in zip(decoded_outputs, batch_original_prompts, batch_responses):
        questions = parse_generated_output(decoded_output)
        questions = list(dict.fromkeys(questions))  # Deduplicate

        if not questions:
            print(f"⚠️ No valid questions generated for: {original_prompt}")
            bad_rows.append({
                "original_prompt": original_prompt,
                "model_raw_output": decoded_output
            })
            continue

        for question in questions[:n_questions_per_prompt]:
            good_rows.append({
                "generated_question": question,
                "original_prompt": original_prompt,
                "response": response
            })

    if good_rows:
        save_rows_to_csv(good_output_path, good_rows, good_header)
    if bad_rows:
        save_rows_to_csv(bad_output_path, bad_rows, bad_header)

print("\n✅ Generation complete. Outputs saved to:")
print(f" - {good_output_path} (good generations)")
print(f" - {bad_output_path} (faulty generations)")
