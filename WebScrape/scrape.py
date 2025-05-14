import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import json
import os
from dotenv import load_dotenv
import re
from DeepSeek import DeepSeek

load_dotenv()
#api_key = os.getenv("OPENAI_API_KEY")

# instantiate chat session
chat = DeepSeek()


def save_extracted_to_json(entries, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(entries)} entries to {output_path}")


def extract_h1_and_paragraphs(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    entries = []
    content = soup.find_all(['h1', 'p'])
    i = 0
    while i < len(content):
        tag = content[i]
        if tag.name == 'h1':
            heading_text = tag.get_text(strip=True)
            summary_parts = []
            i += 1
            while i < len(content) and content[i].name == 'p':
                summary_parts.append(content[i].get_text(strip=True))
                i += 1
            if summary_parts:
                summary = ' '.join(summary_parts)
                entries.append({"heading": heading_text, "summary": summary, "url": url})
        else:
            i += 1
    return entries


def build_training_data_from_scraped(scraped: list[dict], language: str) -> list[dict]:
    training = []
    for entry in scraped:
        # Generate questions prompt
        q_prompt = (
            f"Generate 5 short, distinct questions a user might ask under the heading: '{entry['heading']}'. "
            "Return only a Python list."
        )
        try:
            questions_text = chat.send(q_prompt)
            # Use regex to extract Python list literal
            match = re.search(r"\[.*\]", questions_text, re.S)
            if match:
                questions = eval(match.group(0))
            else:
                print(f"Could not parse questions list. Raw response: {questions_text}")
                questions = []
        except Exception as e:
            print(f"Error generating questions: {e}")
            continue

        for question in questions:
            a_prompt = f"Answer the question '{question}' using the information here: '{entry['summary']}'"
            try:
                answer_text = chat.send(a_prompt)
            except Exception as e:
                print(f"Error generating answer: {e}")
                answer_text = ""

            full_answer = answer_text + f" For more details, visit {entry['url']}"
            training.append({
                "aPrompt": question,
                "bResponse": full_answer,
                "cSubject": "",
                "dLanguage": language,
                "eVerified Translation": "No",
                "fStatus": "Scraped"
            })
    return training

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scrape for training and generate Q&A")
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("format", help="Output format (unused)")
    args = parser.parse_args()

    # Step 1: Scrape and save raw JSON
    scraped = extract_h1_and_paragraphs(args.url)
    save_extracted_to_json(scraped, "scrape_education.json")

    # Determine language from URL path
    parsed = urlparse(args.url)
    lang_code = parsed.path.strip('/').split('/')[0]
    language_map = {'ro': 'Romanian', 'ru': 'Russian', 'uk': 'Ukrainian', 'en': 'English'}
    language = language_map.get(lang_code, 'Unrecognized')

    # Step 2: Load scraped data back for LLM processing
    with open("scrape_education.json", 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)

    # Generate training entries
    training_entries = build_training_data_from_scraped(scraped_data, language)

    # Save final conversation JSON
    with open("scrape_data.json", "w", encoding='utf-8') as f:
        json.dump({"conversation": training_entries}, f, ensure_ascii=False, indent=2)

    print("Done writing scrape_data.json.")