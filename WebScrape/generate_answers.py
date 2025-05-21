
# generate_answers.py
import json
import os
import sys
import re
from DeepSeek import DeepSeek

chat = DeepSeek()

# Languages to process
language_map = {
    'en': 'English',
}

QUESTIONS_ROOT = "questions_dopomoha"
SOURCE_FOLDER = "dopomoha"
OUTPUT_ROOT = "answers_dopomoha"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


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
            if not fname.endswith('.json'):
                continue
            page_name, _ = os.path.splitext(fname)
            questions_data = load_json(os.path.join(questions_folder, fname))['questions']
            content_data = load_json(os.path.join(SOURCE_FOLDER, f"{page_name}.json"))
            answers_out = []
            for q in questions_data:
                entry = next((e for e in content_data if e['id'] == q['content_block_id']), None)
                if not entry:
                    continue
                ans_prompt = (
                    f"Answer in {language} using only the summary: '{entry['summary']}'.\n"
                    f"Question: {q}\n"
                    "Give a concise answer of the question while providing details. "
                    "Don't just give a yes/no answer, give a full answer from the summary. "
                    "Don't mention that the response is coming from the summary. "
                    "Don't say anything like according to the summary. Return only the answer text."
                )
                resp, _ = chat.send(ans_prompt)
                ans_text = resp.strip()
                answers_out.append({
                    'answer_id': aid,
                    'question_id': q['question_id'],
                    'content_block_id': q['content_block_id'],
                    'answer': ans_text,
                    'bResponse': ans_text + f" For more details, visit https://dopomoha.ro/en/{page_name}",
                    'cSubject': entry['heading'],
                    'dLanguage': language,
                    'eVerified Translation': 'No',
                    'fStatus': 'Scraped'
                })
                aid += 1
            out_path = os.path.join(answers_folder, f"{page_name}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump({'answers': answers_out}, f, ensure_ascii=False, indent=2)
            print(f"Wrote answers for {page_name} to {answers_folder}/{page_name}.json")
