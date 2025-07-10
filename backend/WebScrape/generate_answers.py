import json
import os
import sys

#from DeepSeek import DeepSeek
from Gemini import GeminiChat

chat = GeminiChat()

language_map = {
    'en': 'English',
}

QUESTIONS_ROOT = "./data_whole_page/dopomoha_no_yes_no/"
SOURCE_FOLDER = "./data_whole_page/dopomoha_stripped/"
OUTPUT_ROOT = "./data_whole_page/dopomoha_batch_pointing/"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

import re

def extract_json_list(text):
    # Try to extract a list in JSON
    match = re.search(r'\[\s*{[\s\S]*?}\s*\]', text)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except Exception as e:
            print(f"Regex extracted JSON is still invalid: {e}")
            return None
    return None


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == '__main__':
    for code, language in language_map.items():
        questions_folder = os.path.join(QUESTIONS_ROOT, str(code))
        answers_folder = os.path.join(OUTPUT_ROOT, str(code))
        os.makedirs(answers_folder, exist_ok=True)
        aid = 1
        for fname in os.listdir(questions_folder):
            print(fname)
            if not fname.endswith('.json'):
                continue
            page_name, _ = os.path.splitext(fname)
            out_path = os.path.join(answers_folder, f"{page_name}.json")
            if os.path.exists(out_path):
                print(f"Skipping {page_name} as answers already exist in {out_path}")
                continue

            questions_data = load_json(os.path.join(questions_folder, fname))['questions']
            website = f"https://dopomoha.ro/en/{page_name}"
            content_data = load_json(os.path.join(SOURCE_FOLDER, f"{page_name}.json"))

            # For this version, we assume one content per page, take the first (or concatenate if needed)
            page_content = "\n\n".join([e['summary'] for e in content_data if e.get('summary')])
            qa_list = [
                {
                    'question': q['question'],
                    'question_id': q['question_id'],
                    'content_block_id': q['content_block_id'],
                }
                for q in questions_data
            ]
            if not qa_list or not page_content.strip():
                continue

            # --- Prompt Building ---
            batch_prompt = (
                f"Given the following content from website {website}:\n"
                f"\"\"\"\n{page_content}\n\"\"\"\n\n"
                f"For each question, provide a concise answer in {language} (1-2 sentences) using ONLY the content provided. "
                "Following each answer, include a 'find_instruction' that guides the user to the relevant part of the website. "
                "The 'find_instruction' must be in the format: "
                "\"Go to website <https URL> and read the section under <relevant section heading>.\" "
                "If no specific section heading is obvious, use \"Go to website <https URL> and read the page carefully.\" instead. "
                "Do NOT mention 'content' or 'summary' in your answers or instructions. "
                "Ensure the website URL provided in 'find_instruction' is always accurate and complete, e.g., 'https://dopomoha.ro/en/page_name'.\n"
                "Return a Python list of dicts, each with keys: 'question', 'answer', 'find_instruction'.\n\n"
                "Example:\n"
                "[\n"
                "  {\n"
                "    \"question\": \"What documents do I need to register?\",\n"
                "    \"answer\": \"You need an ID card and proof of address.\",\n"
                "    \"find_instruction\": \"Go to website https://dopomoha.ro/en/registration and read the section under 'Required Documents'.\"\n"
                "  },\n"
                "  {\n"
                "    \"question\": \"How can I update my contact details?\",\n"
                "    \"answer\": \"You can update your contact details by filling in the online form.\",\n"
                "    \"find_instruction\": \"Go to website https://dopomoha.ro/en/profile and read the page carefully.\"\n"
                "  }\n"
                "]\n\n"
                "Questions:\n"
            )
            for idx, qa in enumerate(qa_list, 1):
                batch_prompt += f"{idx}. {qa['question']}\n"
            batch_prompt += f"\nWebsite URL: {website}\n"

            print(f"Sending {len(qa_list)} questions for page {page_name}...")

            # --- Model Call & JSON Fallback ---
            resp, _ = chat.send(batch_prompt)
            try:
                answers_batch = json.loads(resp)
                assert isinstance(answers_batch, list)
            except Exception as e:
                print(f"Primary json.loads() failed for {page_name}: {e}\nTrying regex fallback...")
                answers_batch = extract_json_list(resp)
                if answers_batch is None:
                    print(f"Regex fallback failed. Skipping {page_name}.\nModel output:\n{resp}")
                    continue

            answers_out = []
            aid = 1
            for idx, qa in enumerate(qa_list):
                abatch = answers_batch[idx] if idx < len(answers_batch) else {}
                answer_text = abatch.get('answer', '').strip()
                find_instruction = abatch.get('find_instruction', '').strip() # This remains the same as it's the instruction

                answers_out.append({
                    'answer_id': aid,
                    'question_id': qa['question_id'],
                    'content_block_id': qa['content_block_id'],
                    'answer': answer_text,
                    'instruction': find_instruction, # Changed from bResponse to instruction
                    'cSubject': '',
                    'dLanguage': language,
                    'eVerified Translation': 'No',
                    'fStatus': 'Scraped'
                })
                aid += 1

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'answers': answers_out}, f, ensure_ascii=False, indent=2)
            print(f"Wrote answers for {page_name} to {answers_folder}/{page_name}.json")