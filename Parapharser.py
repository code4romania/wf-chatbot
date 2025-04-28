import pandas as pd
import torch
import random
import csv
import os
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ====== SETTINGS ======
csv_input_path = "data.csv"             # Input CSV
good_output_path = "good_questions.csv"       # Save good generations here
bad_output_path = "bad_questions.csv"         # Save bad generations here
n_questions_per_prompt = 5                    # Fixed 5
batch_size = 4
max_new_tokens = 256

# ====== LOAD DEEPSEEK MODEL (4-BIT) ======
model_name = "deepseek-ai/deepseek-llm-7b-chat"

print(f"Loading model '{model_name}' (4bit quantized)...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
).eval()

model.config.use_cache = False  # Save VRAM during generation

# ====== HELPER FUNCTIONS ======

def format_instruction(prompt):
    """Create a dynamic instruction with occasional keyword avoidance."""
    avoid_keywords = random.random() < 0.5  # 30% chance to encourage not copying
    instruction = (
        f"<|system|>\n"
        f"You are a helpful assistant.\n"
        f"Your task:\n"
        f"- Given a topic, generate exactly {n_questions_per_prompt} casual, CONCISE, natural questions.\n"
        f"- Stay in the SAME LANGUAGE as the topic.\n"
        f"- Output ONLY the {n_questions_per_prompt} questions as a numbered list.\n"
        f"- Each question must be short and natural.\n"
        f"- Each line must start with a number and a dot (e.g., '1.')\n"
    )
    if avoid_keywords:
        instruction += (
            "- Try to vary wording creatively and avoid copying exact words from the topic if possible.\n"
        )

    instruction += (
        f"\n<|user|>\n"
        f"Topic: {prompt}\n"
        f"\n<|assistant|>\n"
        f"1. "
    )
    return instruction

def parse_generated_output(output_text):
    """Smarter parsing to recover questions even if numbering is broken."""
    questions = []
    lines = output_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\d+\.\s*(.+)", line)
        if match:
            question = match.group(1).strip()
        else:
            question = line

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

# ====== LOAD DATA ======
print(f"Loading prompts from {csv_input_path}...")
df = pd.read_csv(csv_input_path)
df = df.dropna(subset=["Prompt", "Response"])

prompts = df["Prompt"].tolist()
responses = df["Response"].tolist()

instructions = [format_instruction(prompt) for prompt in prompts]

# ====== PREPARE OUTPUT ======
good_header = ["generated_question", "original_prompt", "response"]
bad_header = ["original_prompt", "model_raw_output"]

open(good_output_path, 'w').close()
open(bad_output_path, 'w').close()

print("Generating questions with DeepSeek...")

# ====== MAIN GENERATION LOOP ======
for batch_start in tqdm(range(0, len(instructions), batch_size)):
    batch_end = batch_start + batch_size
    batch_instructions = instructions[batch_start:batch_end]
    batch_prompts = prompts[batch_start:batch_end]
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

    for decoded_output, original_prompt, response in zip(decoded_outputs, batch_prompts, batch_responses):
        questions = parse_generated_output(decoded_output)
        questions = list(dict.fromkeys(questions))  # Remove duplicates

        if not questions or len(questions) < 3:
            print(f"⚠️ Few or no questions for: {original_prompt}")
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

print("\n✅ All done! Good generations saved to:")
print(f" - {good_output_path}")
print(f"Bad generations saved to:")
print(f" - {bad_output_path}")
