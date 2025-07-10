import json
import os
import sys
import re
from datetime import datetime
#from DeepSeek import DeepSeek
from Gemini import GeminiChat

chat = GeminiChat()

# Languages to process
language_map = {
    'en': 'English',
    # add more codes if needed, e.g. 'ro': 'Romanian'
}

INPUT_FOLDER = "./data_whole_page/dopomoha_stripped"
OUTPUT_ROOT = "./data_whole_page/dopomoha_no_yes_no"
QID_FILE = os.path.join(OUTPUT_ROOT, "next_qid.txt") # File to store next_qid
FAILURES_FILE = os.path.join(OUTPUT_ROOT, "notes", "fails.json") # Path for failures log

# Ensure output directories exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_ROOT, "notes"), exist_ok=True)

# Initialize qid
def load_qid():
    if os.path.exists(QID_FILE):
        with open(QID_FILE, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 1
    return 1

def save_qid(qid_value):
    with open(QID_FILE, 'w') as f:
        f.write(str(qid_value))

qid = load_qid()

def load_scraped(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_failures():
    if os.path.exists(FAILURES_FILE):
        with open(FAILURES_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_failures(failures):
    with open(FAILURES_FILE, 'w', encoding='utf-8') as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

def generate_questions_for_language(scraped, code, language, page_name, verbose=False):
    questions_out = []
    n_questions = 5
    max_questions= 20
    failures = load_failures() # Load existing failures

    for entry in scraped:
        base_prompt = (
            f"Given content: '{entry['summary']}', "
            f"generate {n_questions} to {max_questions} concise, distinct, open-ended questions in {language} "
            "that someone curious might naturally ask BEFORE reading any details. "
            "DO NOT generate any question that can be answered with 'yes' or 'no'. "
            "IMPORTANT: Do NOT start any question with 'Is', 'Are', 'Was', 'Were', 'Does', 'Do', 'Did', 'Can', 'Could', 'Will', 'Would', 'Should', or 'Has'. "
            "Use open-ended question words such as 'What', 'How', 'Why', 'Which', 'Who', or 'In what way'. "
            "Each question MUST be fully understandable on its own and should not require knowledge of the content to grasp what the question is about. "
            "Pretend each question will be seen in isolation, as if in an FAQ. "
            "BAD Examples that LACK CONTEXT or are yes/no: "
            "  - 'Is participation open to both adults and children?' (Yes/no and context missing: participation in what?). "
            "  - 'Can I apply?' (Yes/no and context missing: apply for what?). "
            "  - 'What is the purpose of these measures?' ('these measures' is vague). "
            "GOOD Examples: "
            "  - 'What are the benefits of the new community gardening program?' "
            "  - 'How can someone join the new community gardening program?' "
            "If the question is about a specific program, event, or concept, include that specific name or concept in the question. "
            "Do not use the words 'this', 'these', 'summary', or 'content' in your questions. "
            f"SUPER IMPORTANT: Return only a Python list literal."
        )

        history = []
        questions = []
        
        # Attempt up to 4 times to get a valid list of questions
        for i in range(4):
            resp1, _ = chat.send(base_prompt)
            history += [('user', base_prompt), ('assistant', resp1)]
            match = re.search(r"(\[.*\])", resp1, re.S)
            if match:
                try:
                    lst = eval(match.group(1))
                    # MODIFICATION: Relaxed check here
                    if isinstance(lst, list) and len(lst) >= 1: # Check if it's a list and has at least one item
                        questions = lst
                        break
                    elif verbose:
                        print(f"Attempt {i+1} for {page_name}, content block {entry['id']}: Parsed list is empty or not a list.")
                except Exception as e:
                    if verbose:
                        print(f"Attempt {i+1} for {page_name}, content block {entry['id']}: Evaluation failed - {e}")
            if verbose:
                print(f"Attempt {i+1} failed for {page_name}, content block {entry['id']}: {resp1}")
            
            # Corrective prompt
            correction = (
                f"Response couldn't be parsed as a python list or was empty. "
                f"Make sure you return a valid Python list literal with at least one item, e.g. ['a','b','c'....]."
            )
            history.append(('user', correction))
            convo = ''.join([f"{('User' if r=='user' else 'Bot')}: {msg}\n" for r, msg in history])
            resp2, _ = chat.send(convo)
            history.append(('assistant', resp2))
            match2 = re.search(r"(\[.*\])", resp2, re.S)
            if match2:
                try:
                    lst2 = eval(match2.group(1))
                    # MODIFICATION: Relaxed check here
                    if isinstance(lst2, list) and len(lst2) >= 1: # Check if it's a list and has at least one item
                        questions = lst2
                        break
                    elif verbose:
                        print(f"Correction attempt {i+1} for {page_name}, content block {entry['id']}: Parsed list is empty or not a list.")
                except Exception as e:
                    if verbose:
                        print(f"Correction attempt {i+1} for {page_name}, content block {entry['id']}: Evaluation failed - {e}")
            if verbose:
                print(f"Correction attempt {i+1} failed for {page_name}, content block {entry['id']}: {resp2}")

        if not questions:
            # Log the failure
            failures.append({
                'page_name': page_name,
                'content_block_id': entry['id'],
                'last_model_response': resp2 if 'resp2' in locals() else resp1 # Store the last response for debugging
            })
            if verbose:
                print(f"FAILURE: Could not generate questions for {page_name}, content block {entry['id']} in {language}.")
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
            save_qid(qid) # Save qid after each question to ensure progress is saved

    # write output per language and page
    if questions_out: # Only write if there are successfully generated questions
        output_dir = os.path.join(OUTPUT_ROOT, str(code))
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{page_name}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'questions': questions_out}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(questions_out)} questions for {page_name} to {out_path}")
    else:
        print(f"No questions generated for {page_name} in {language}. See fails.json for details if errors occurred.")

    save_failures(failures) # Save updated failures list

if __name__ == '__main__':
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith('.json'):
            continue
        page_name, _ = os.path.splitext(filename)
        scraped = load_scraped(os.path.join(INPUT_FOLDER, filename))
        for code, lang in language_map.items():
            generate_questions_for_language(scraped, code, lang, page_name, verbose=True) # Set verbose to True for detailed console output
            print(f"Finished processing {page_name} for {lang}.")