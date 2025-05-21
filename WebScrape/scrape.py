# scrape.py
"""
Script: scrape.py
Extracts h1 headings and paragraphs from a URL and saves them to JSON.
Usage: python scrape.py <url> [output_json]
"""
import requests
from bs4 import BeautifulSoup
import json
import sys


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
                entries.append({
                    "heading": heading_text,
                    "summary": summary
                })
        else:
            i += 1
    return entries


def save_to_json(entries, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(entries)} entries to {path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python scrape.py <url> [output_json]")
        sys.exit(1)
    url = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else 'scrape_education.json'
    data = extract_h1_and_paragraphs(url)
    save_to_json(data, output)
