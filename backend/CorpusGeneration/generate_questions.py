import json
import os
import sys
import re
from datetime import datetime
#from DeepSeek import DeepSeek
from Gemini import GeminiChat

chat = GeminiChat()

def load_generation_params(filename):
    """Loads generation parameters from the JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Parameter file '{filename}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filename}': {e}")
        sys.exit(1)

def load_qid(path):
    if os.path.exists(path):
        with open(path, 'r') as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 1
    return 1

def save_qid(qid_value, path):
    with open(path, 'w') as f:
        f.write(str(qid_value))

def load_scraped(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_failures(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_failures(failures, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

def generate_questions():
    """
    Main function to orchestrate question generation from scraped content.
    Loads all parameters from GenerationParams.json.
    """
    # Load parameters from the JSON file
    PARAMS_FILE = "GenerationParams.json"
    params = load_generation_params(PARAMS_FILE)
    
    # Get relevant parameters
    top_level_output_dir = params.get('OUTPUT_DIR')
    q_params = params.get('QuestionGenerationParams', {})
    
    question_directory_name = q_params.get('question_directory')
    prompt_template = q_params.get('question_prompt')
    n_questions = q_params.get('n_questions', 3)
    max_questions = q_params.get('max_questions', 5)
    languages = q_params.get('languages', {})
    
    if not top_level_output_dir or not prompt_template or not languages:
        print("Error: Missing required parameters in GenerationParams.json.")
        sys.exit(1)

    # Define paths based on parameters
    INPUT_FOLDER = os.path.join('corpus', top_level_output_dir)
    if question_directory_name:
        OUTPUT_ROOT = os.path.join('corpus', top_level_output_dir, question_directory_name)
    else:
        OUTPUT_ROOT = os.path.join('corpus', top_level_output_dir, 'questions')
    
    QID_FILE = os.path.join(OUTPUT_ROOT, "next_qid.txt")
    FAILURES_FILE = os.path.join(OUTPUT_ROOT, "notes", "fails.json")

    # Ensure output directories exist
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "notes"), exist_ok=True)
    
    qid = load_qid(QID_FILE)
    failures = load_failures(FAILURES_FILE)
    
    for filename in os.listdir(INPUT_FOLDER):
        if not filename.endswith('.json'):
            continue
        
        page_name, _ = os.path.splitext(filename)
        try:
            scraped_content = load_scraped(os.path.join(INPUT_FOLDER, filename))
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        for code, lang in languages.items():
            questions_out = []
            print(f"\nProcessing {page_name} for {lang}...")
            
            for entry in scraped_content:
                base_prompt = prompt_template.format(
                    summary=entry['summary'],
                    n_questions=n_questions,
                    max_questions=max_questions,
                    language=lang
                )

                questions = []
                # Attempt up to 4 times to get a valid list of questions
                for i in range(4):
                    resp, _ = chat.send(base_prompt)
                    match = re.search(r"(\[.*\])", resp, re.S)
                    if match:
                        try:
                            lst = eval(match.group(1))
                            if isinstance(lst, list) and len(lst) >= 1:
                                questions = lst
                                break
                        except Exception:
                            pass
                    
                    # If parsing failed, retry with a corrective prompt
                    correction = "The response was not a valid Python list. Please provide a valid list of questions."
                    resp, _ = chat.send(correction)
                    match = re.search(r"(\[.*\])", resp, re.S)
                    if match:
                         try:
                            lst = eval(match.group(1))
                            if isinstance(lst, list) and len(lst) >= 1:
                                questions = lst
                                break
                         except Exception:
                            pass

                if not questions:
                    failures.append({
                        'page_name': page_name,
                        'content_block_id': entry['id'],
                        'last_model_response': resp
                    })
                    print(f"FAILURE: Could not generate questions for {page_name} in {lang}.")
                    continue

                for q in questions:
                    questions_out.append({
                        'question_id': qid,
                        'content_block_id': entry['id'],
                        'question': q
                    })
                    qid += 1
            
            if questions_out:
                output_lang_dir = os.path.join(OUTPUT_ROOT, code)
                os.makedirs(output_lang_dir, exist_ok=True)
                out_path = os.path.join(output_lang_dir, f"{page_name}.json")
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump({'questions': questions_out}, f, ensure_ascii=False, indent=2)
                print(f"Wrote {len(questions_out)} questions for {page_name} to {out_path}")
            
    save_qid(qid, QID_FILE)
    save_failures(failures, FAILURES_FILE)
    print("\nQuestion generation complete.")

if __name__ == '__main__':
    generate_questions()