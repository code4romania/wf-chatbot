import requests
from bs4 import BeautifulSoup
import json
import os

# List all page name segments here
PAGE_NAMES = [
    'education',
    'legal-status', # This page will use the custom parser
    'human-trafficking',
    'sanatate',
    'transportation', # Fix: added comma
    'call-center',
    'about',
    'romanian-eng',
    'job-for-refugees-from-ukraine-in-romania',
    'legal-help',
    'support-for-drug-users',
    'psychological-support-for-ukrainian-refugees',
    'integration'
]
PAGE_NAMES = [
    'legal-status', # This page will use the custom parser
]

BASE_URL = 'https://dopomoha.ro/en'
OUTPUT_DIR = 'dopomoha_custom_parse'

# --- Custom structure map for specific pages ---
# Each entry specifies a 'selector' for the base element (e.g., 'body > main')
# and a list of 'rules' for its direct children.
# 'type': '+' means the element is a content block.
# 'type': '-' means the element is ignored.
# 'type': '*' means its direct children (of 'child_tag') are content blocks.
# 'index': refers to the 0-based index of the direct child element within the base element.
PAGE_SPECIFIC_STRUCTURES = {
    'legal-status': {
        'selector': 'body > main > div',
        'rules': [
            {'tag': 'div', 'index': 0, 'type': '-'},
            {'tag': 'div', 'index': 1, 'type': '+'},
            {'tag': 'div', 'index': 2, 'type': '+'},
            {'tag': 'div', 'index': 3, 'type': '*', 'child_tag': 'div'},
            {'tag': 'div', 'index': 4, 'type': '+'},
        ]
    },
    # Add other pages here if they need custom parsing in the future
    # 'some-other-page': {
    #     'selector': 'body > div.main-content',
    #     'rules': [
    #         {'tag': 'p', 'index': 0, 'type': '+'},
    #         {'tag': 'section', 'index': 1, 'type': '*', 'child_tag': 'h2'},
    #     ]
    # }
}

def extract_h1_and_paragraphs(soup_obj):
    """
    Extracts each <h1> and its following <p> siblings from a BeautifulSoup object.
    Returns a list of dicts with 'heading' and 'summary'.
    """
    entries = []
    # Find all h1 and p elements in the soup object
    elements = soup_obj.find_all(['h1', 'p'])
    i = 0
    while i < len(elements):
        el = elements[i]
        if el.name == 'h1':
            heading = el.get_text(strip=True)
            i += 1
            summary_parts = []
            # Collect all following p siblings
            while i < len(elements) and elements[i].name == 'p':
                summary_parts.append(elements[i].get_text(strip=True))
                i += 1
            if summary_parts:
                entries.append({'heading': heading, 'summary': ' '.join(summary_parts)})
        else:
            i += 1
    return entries

def extract_from_structure_map(soup_obj, structure_map):
    """
    Extracts content blocks based on a predefined structure map.
    Returns a list of dicts with 'heading' (as 'Content Block N') and 'summary'.
    """
    content_blocks_data = []
    base_element = soup_obj.select_one(structure_map['selector'])

    if not base_element:
        print(f"Warning: Base element not found for selector: {structure_map['selector']}")
        return []

    # Get all direct children of the base_element that are relevant for parsing
    # This filters out NavigableString (e.g., whitespace) and targets tag elements
    direct_children_tags = [child for child in base_element.children if child.name is not None]

    for rule in structure_map['rules']:
        try:
            # Ensure the index is valid for direct children
            if rule['index'] >= len(direct_children_tags):
                print(f"Warning: Rule index {rule['index']} out of bounds for direct children in selector: {structure_map['selector']}")
                continue

            target_element = direct_children_tags[rule['index']]

            # Further check if the tag of the target element matches the rule's tag
            if target_element.name != rule['tag']:
                print(f"Warning: Expected tag '{rule['tag']}' at index {rule['index']} but found '{target_element.name}'. Skipping rule.")
                continue

            if rule['type'] == '+':
                # Extract text from this element and create an entry
                summary_text = target_element.get_text(strip=True)
                if summary_text: # Only add if content is not empty
                    content_blocks_data.append({'heading': '', 'summary': summary_text})
            elif rule['type'] == '*':
                # Extract text from its specified direct child elements
                for child_block in target_element.find_all(rule['child_tag'], recursive=False):
                    summary_text = child_block.get_text(strip=True)
                    if summary_text: # Only add if content is not empty
                        content_blocks_data.append({'heading': '', 'summary': summary_text})
            # If type is '-', do nothing
        except Exception as e:
            print(f"Error processing rule {rule} for {structure_map['selector']}: {e}")
            continue

    # Assign generic headings "Content Block N" and filter out empty summaries
    final_entries = []
    block_counter = 1
    for entry in content_blocks_data:
        if entry['summary']: # Ensure there is actual content
            entry['heading'] = f"Content Block {block_counter}"
            final_entries.append(entry)
            block_counter += 1

    return final_entries


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
        print(f"\nProcessing {url}")

        response = None
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        page_entries = []

        if name in PAGE_SPECIFIC_STRUCTURES:
            print(f"  Using custom parser for '{name}'...")
            page_entries = extract_from_structure_map(soup, PAGE_SPECIFIC_STRUCTURES[name])
        else:
            print(f"  Using default parser for '{name}'...")
            page_entries = extract_h1_and_paragraphs(soup) # Pass the soup object directly

        if not page_entries:
            print(f"  No entries found for '{name}', skipping.")
            continue

        # Assign unique integer ID to each entry
        for entry in page_entries:
            entry['id'] = id_counter
            id_counter += 1

        # Save to JSON file
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        save_to_json(page_entries, output_path)


if __name__ == '__main__':
    main()