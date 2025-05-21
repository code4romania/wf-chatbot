import requests
from bs4 import BeautifulSoup
import json
import os

# List all page name segments here
PAGE_NAMES = [
    'education',
    'legal-status',
    'human-trafficking',
    'sanatate',
    'transportation'
    'call-center',
    'about',
    'romanian-eng',
    'job-for-refugees-from-ukraine-in-romania',
    'legal-help',
    'support-for-drug-users',
    'psychological-support-for-ukrainian-refugees',
    'integration'
]

BASE_URL = 'https://dopomoha.ro/en'
OUTPUT_DIR = 'dopomoha'


def extract_h1_and_paragraphs(url):
    """
    Fetch the page at `url`, extract each <h1> and its following <p> siblings,
    and return a list of dicts with 'heading' and 'summary'.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    entries = []
    elements = soup.find_all(['h1', 'p'])
    i = 0
    while i < len(elements):
        el = elements[i]
        if el.name == 'h1':
            heading = el.get_text(strip=True)
            i += 1
            summary_parts = []
            while i < len(elements) and elements[i].name == 'p':
                summary_parts.append(elements[i].get_text(strip=True))
                i += 1
            if summary_parts:
                entries.append({'heading': heading, 'summary': ' '.join(summary_parts)})
        else:
            i += 1
    return entries


def save_to_json(entries, path):
    """
    Save `entries` to JSON file at `path`, creating directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(entries)} entries to {path}")
    except IOError as e:
        print(f"Error writing {path}: {e}")


def main():
    """
    Iterate over PAGE_NAMES, build each URL, scrape, assign unique IDs across all entries,
    and save to dopomoha/<name>.json.
    """
    id_counter = 1
    for name in PAGE_NAMES:
        url = f"{BASE_URL}/{name}"
        print(f"Processing {url}")
        entries = extract_h1_and_paragraphs(url)
        if not entries:
            print(f"No entries found for {name}, skipping.")
            continue
        # Assign unique integer ID to each entry
        for entry in entries:
            entry['id'] = id_counter
            id_counter += 1
        # Save to JSON file
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        save_to_json(entries, output_path)


if __name__ == '__main__':
    main()
