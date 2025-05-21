# generate_questions.py
import json
import os
import sys
import re
from DeepSeek import DeepSeek

chat = DeepSeek()

# Languages to process
language_map = {
    'en': 'English',
    # add more codes if needed, e.g. 'ro': 'Romanian'
}

INPUT_FOLDER = "dopomoha"
OUTPUT_ROOT = "questions_dopomoha"
qid = 1

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def load_scraped(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_questions_for_language(scraped, code, language, page_name):
    questions_out = []
    
    for entry in scraped:
        base_prompt = (
        f"Given the title: '{entry['heading']}' and the summary: '{entry['summary']}', "
        f"generate 5 short, distinct questions in {language} that someone curious about '{entry['heading']}' might naturally ask "
        "BEFORE reading any details! "
        "IMPORTANT: Each question MUST be fully understandable on its own, as if seen in isolation. It should not require knowledge of the title or summary to grasp what the question is about. "
        "For example, if the title is 'New Community Gardening Program Launch', a BAD question is 'What are the benefits?' (Benefits of what?). "
        "Other BAD Examples that LACK CONTEXT: "
        "  - 'What is the purpose of these measures?' (Context needed: 'these measures' is vague). "
        "  - 'Is participation open to both adults and children?' (Context needed: 'Participation in what activity/program?'). "
        "  - 'How can I apply?' (Context needed: 'Apply for what?'). "
        "A GOOD question is 'What are the benefits of the new community gardening program?' or 'How can one join the new community gardening program?'. "
        "If the question is about a specific program, event, or concept mentioned in the title, try to include that specific program, event, or concept in the question itself. "
        "Do not include the words “title”, 'these', or “summary” in your questions. "
        "Return only a Python list literal of exactly 5 items.")
        history = []
        questions = []
        # Attempt up to 4 times to get a valid list of 5
        for _ in range(4):
            resp1, _ = chat.send(base_prompt)
            history += [('user', base_prompt), ('assistant', resp1)]
            match = re.search(r"(\[.*\])", resp1, re.S)
            if match:
                try:
                    lst = eval(match.group(1))
                    if isinstance(lst, list) and len(lst) == 5:
                        questions = lst
                        break
                except:
                    pass
            # corrective prompt
            correction = (
                "Response couldn't be parsed as a python list with 5 values. "
                "Make sure you return a valid Python list literal with 5 items, e.g. ['a','b','c','d','e']."
            )
            history.append(('user', correction))
            convo = ''.join([f"{('User' if r=='user' else 'Bot')}: {msg}\n" for r, msg in history])
            resp2, _ = chat.send(convo)
            history.append(('assistant', resp2))
            match2 = re.search(r"(\[.*\])", resp2, re.S)
            if match2:
                try:
                    lst2 = eval(match2.group(1))
                    if isinstance(lst2, list) and len(lst2) == 5:
                        questions = lst2
                        break
                except:
                    pass
        if not questions:
            continue
        # collect questions
        global qid
        for q in questions:
            questions_out.append({
                'question_id': qid,
                'content_block_id': entry['id'],
                'question': q
            })
            
            qid += 1
    # write output per language and page
    output_dir = os.path.join(OUTPUT_ROOT, str(code))
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{page_name}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'questions': questions_out}, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith('.json'):
            continue
        page_name, _ = os.path.splitext(filename)
        scraped = load_scraped(os.path.join(INPUT_FOLDER, filename))
        for code, lang in language_map.items():
            generate_questions_for_language(scraped, code, lang, page_name)
            print(f"Wrote questions for {page_name} to {OUTPUT_ROOT}_{code}/{page_name}.json")
