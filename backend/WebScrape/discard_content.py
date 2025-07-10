import os
import json
import logging

# Set up logging
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(message)s')

# Your key phrases
START_PHRASE = (
    "Access the last two solutions designed to support refugees: "
    "Law made simple helping citizens better understand their rights and obligations & PTSD Help - your free assistant for managing Post Traumatic Stress Disorder (PTSD)."
)
END_PHRASE = "If you’d like to report any mistakes, inconsistencies or omissions please let us know"

# Directory with JSON files
INPUT_DIR = "data_whole_page/dopomoha_parsed"
OUTPUT_DIR = "data_whole_page/dopomoha_stripped"  # can be same as input

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def process_summary(text, file_name, entry_id):
    start_idx = text.find(START_PHRASE)
    if start_idx == -1:
        logging.info(f"START PHRASE NOT FOUND in {file_name} id={entry_id}")
        return None
    start_idx += len(START_PHRASE)
    end_idx = text.find(END_PHRASE, start_idx)
    if end_idx == -1:
        logging.info(f"END PHRASE NOT FOUND in {file_name} id={entry_id}")
        return None
    # Extract the wanted content
    return text[start_idx:end_idx].strip()

for file_name in os.listdir(INPUT_DIR):
    if not file_name.endswith('.json'):
        continue
    input_path = os.path.join(INPUT_DIR, file_name)
    output_path = os.path.join(OUTPUT_DIR, file_name)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logging.info(f"FAILED TO LOAD {file_name}: {e}")
        continue

    changed = False
    for entry in data:
        summary = entry.get('summary', '')
        new_summary = process_summary(summary, file_name, entry.get('id', '?'))
        if new_summary is not None:
            entry['summary'] = new_summary
            changed = True
        else:
            # Optionally blank out summary if not matched:
            # entry['summary'] = ''
            pass

    # Only write out if any change was made
    if changed:
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.info(f"FAILED TO WRITE {output_path}: {e}")

print("Processing complete. Check processing.log for details.")
