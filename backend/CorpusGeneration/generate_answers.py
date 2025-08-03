import json
import os
import sys

#from DeepSeek import DeepSeek
from Gemini import GeminiChat

chat = GeminiChat()

# --- Load parameters from JSON ---
try:
    with open('GenerationParams.json', 'r', encoding='utf-8') as f:
        params = json.load(f)
except FileNotFoundError:
    print("Error: GenerationParams.json not found.")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON from GenerationParams.json: {e}")
    sys.exit(1)

OUTPUT_DIR = params['OUTPUT_DIR']
corpus_path = f"corpus/{OUTPUT_DIR}"

QUESTIONS_ROOT = os.path.join(corpus_path, params['AnswerGenerationParams']['questions_directory'])
SOURCE_FOLDER = corpus_path
OUTPUT_ROOT = os.path.join(corpus_path, params['AnswerGenerationParams']['answers_directory'])

language_map = params['QuestionGenerationParams']['languages']
ANSWER_PROMPT_TEMPLATE = params['AnswerGenerationParams']['answer_prompt_template']

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


def answer_generation():
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
            
            print("aaaaaaaaa",os.path.join(questions_folder, fname))
            questions_data = load_json(os.path.join(questions_folder, fname))['questions']
            website_base_url = f"https://dopomoha.ro/{code}/{page_name}" # Using 'code' for language in URL
            content_data = load_json(os.path.join(SOURCE_FOLDER, f"{page_name}.json"))

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
            questions_list_str = "\n".join([f"{idx}. {qa['question']}" for idx, qa in enumerate(qa_list, 1)])
            batch_prompt = ANSWER_PROMPT_TEMPLATE.format(
                page_content=page_content,
                website_base_url=website_base_url,
                language=language,
                questions_list=questions_list_str
            )

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
                # Safely get values from the model's response, providing defaults
                abatch = answers_batch[idx] if idx < len(answers_batch) else {}
                answer_text = abatch.get('answer', '').strip()
                website_url = abatch.get('website', website_base_url).strip() # Use website_base_url as fallback
                read_instruction = abatch.get('read_instruction', 'read the page carefully.').strip()

                # --- Merge and Parse in Python ---
                find_instruction = f"Go to website {website_url} and {read_instruction}"


                answers_out.append({
                    'answer_id': aid,
                    'question_id': qa['question_id'],
                    'content_block_id': qa['content_block_id'],
                    'question': qa['question'],
                    'answer': answer_text,
                    'website': website_url,
                    'instruction': find_instruction, # The final combined instruction
                    'read_instruction': read_instruction,
                    'cSubject': '',
                    'dLanguage': language,
                    'eVerified Translation': 'No',
                    'fStatus': 'Scraped'
                })
                aid += 1

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'answers': answers_out}, f, ensure_ascii=False, indent=2)
            print(f"Wrote answers for {page_name} to {answers_folder}/{page_name}.json")
            
if __name__ == '__main__':
    answer_generation()