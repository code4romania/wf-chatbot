import requests
from bs4 import BeautifulSoup
import json
import os

PAGE_NAMES= ['prima-pagina', 'titlu', 'faq', 'entering-romania', 'my-rights', 'get-help', 'about', 'info-on-romania', 'stay-safe', 'dopomohaapp', 'applying-for-asylum', 'useful-resources', 'phrase-book', 'share', 'prima-pagina-1', 'support', 'info', 'regional-centers-for-procedures-and-accommodation-for-asylum-seekers', 'transportation', 'issue-of-the-residence-permit', 'legal-status', 'health', 'accommodation', 'call-center', 'sprijin', 'education', 'housing', 'financial-support-from-unhcr', 'main-page', 'sanatate', 'information', 'temporary-protection', 'short-stay', 'support-for-drug-users', 'childrens-nutrition', 'vaccination', 'psychological-support-for-children', 'legal-help', 'hygiene', 'childrens-security', 'taking-care-of-children-with-disabilities', 'the-5020-program', 'romanian-language-courses', 'free-medical-care-for-refugees-from-ukraine', 'job-for-refugees-from-ukraine-in-romania', 'romanian-eng', 'family-doctor-for-refugees-from-ukraine', 'buses-from-ukraine', 'laboratory-analysis', 'specialist-doctor', 'assistance-in-the-hospital', 'dentist', 'medicine-and-rehabilitation-services', 'medical-recovery-in-hospitals', 'emergency-consultations', 'medicines', 'medical-devices', 'patients-with-covid-19', 'pregnant-women-from-ukraine', 'national-curative-health-programs', 'medical-assistance-for-refugees', 'enrolment-of-ukrainian-children', 'education-useful-information', 'medzirka', 'health-promotion-center', 'free-medical-care', 'dopomoha-brasov', 'dopomoha-timisoara', 'dopomoha-constanta', 'dopomoha-galati', 'dopomoha-suceava', 'dopomoha-cluj-napoca', 'dopomoha-satu-mare', 'dopomoha-satu-mare-1', 'dopomoha-iasi', 'dopomoha-braila', 'dopomoha-oradea', 'terms-and-conditions', 'cookie-policy', 'dopomoha-sibiu', 'psychological-support-for-ukrainian-refugees', 'human-trafficking', 'integration', 'ptsd-help', 'social-benefits-for-refugees']

BASE_URL = 'https://dopomoha.ro/en'
USE_WHOLE_PAGE_MODE = True

OUTPUT_DIR = './data_whole_page/dopomoha_parsed'
PARSING_RULES_FILE = './data/parsing_rules/parsing_rules.json' # File for custom parsing rules
FAILS_LOG_FILE = os.path.join(OUTPUT_DIR, 'notes', 'fails.json') # New file for parsing failures


def load_parsing_rules(filename):
    """
    Loads custom page parsing rules from a JSON file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        print(f"Successfully loaded parsing rules from {filename}")
        return rules
    except FileNotFoundError:
        print(f"Error: Parsing rules file '{filename}' not found. Please create it.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filename}': {e}")
        return {}

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
    It now assigns generic "Content Block N" headings to all extracted blocks.
    Returns a list of dicts with 'heading' and 'summary'.
    """
    content_blocks_data = []
    base_element = soup_obj.select_one(structure_map['selector'])

    if not base_element:
        print(f"Warning: Base element not found for selector: {structure_map['selector']}")
        return []

    direct_children_tags = [child for child in base_element.children if child.name is not None]

    for rule in structure_map['rules']:
        try:
            if rule['index'] >= len(direct_children_tags):
                print(f"Warning: Rule index {rule['index']} out of bounds for direct children in selector: {structure_map['selector']}")
                continue

            target_element = direct_children_tags[rule['index']]

            if target_element.name != rule['tag']:
                print(f"Warning: Expected tag '{rule['tag']}' at index {rule['index']} but found '{target_element.name}'. Skipping rule.")
                continue

            if rule['type'] == '+':
                summary_text = target_element.get_text(strip=True)
                if summary_text:
                    content_blocks_data.append({'heading': '', 'summary': summary_text}) # Heading will be set later
            elif rule['type'] == '*':
                elements_to_process = []
                if 'child_relative_path' in rule:
                    container_element = target_element.select_one(rule['child_relative_path'])
                    if container_element:
                        elements_to_process = container_element.find_all(rule['child_tag'], recursive=False)
                    else:
                        print(f"Warning: child_relative_path '{rule['child_relative_path']}' not found within target_element.")
                else:
                    elements_to_process = target_element.find_all(rule['child_tag'], recursive=False)

                for child_block in elements_to_process:
                    summary_text = child_block.get_text(strip=True)
                    if summary_text:
                        content_blocks_data.append({'heading': '', 'summary': summary_text}) # Heading will be set later
            # If type is '-', do nothing
        except Exception as e:
            print(f"Error processing rule {rule} for {structure_map['selector']}: {e}")
            continue

    # Assign generic headings "Content Block N" and filter out empty summaries.
    final_entries = []
    block_counter = 1
    for entry in content_blocks_data:
        if entry['summary']: # Ensure there is actual content
            entry['heading'] = f"Content Block {block_counter}" # Always assign generic heading
            final_entries.append(entry)
            block_counter += 1

    return final_entries

def extract_whole_page_content(soup_obj):
    """
    Extracts the entire text content of the <body> tag as a single content block.
    Returns a list containing one dict with 'heading' and 'summary'.
    """
    body_content = soup_obj.find('body')
    if body_content:
        # Get all text, strip excess whitespace, and replace multiple newlines with single spaces
        # to create a more compact summary.
        full_text = body_content.get_text(separator=' ', strip=True)
        return [{'heading': 'Whole Page Content', 'summary': full_text}]
    return []

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
    page_specific_structures = load_parsing_rules(PARSING_RULES_FILE)
    if not page_specific_structures and not USE_WHOLE_PAGE_MODE:
        print("Exiting: No parsing rules found and not in whole page mode.")
        return

    id_counter = 1
    not_found_pages = []
    parsing_failed_pages = []

    # Determine which pages to process based on USE_WHOLE_PAGE_MODE or parsing rules
    if USE_WHOLE_PAGE_MODE:
        pages_to_process = PAGE_NAMES
    else:
        # If not whole page mode, only process pages for which we have specific parsing rules
        pages_to_process = list(page_specific_structures.keys())
        # Add any pages from PAGE_NAMES that are NOT in page_specific_structures if default parsing is desired
        # for them (which is not the explicit instruction here, so we stick to rules-only).
        # However, the previous logic was to use default h1/p if no specific rule,
        # so for consistency with the spirit of the previous code,
        # we should still process all PAGE_NAMES but apply rules if available.
        # Let's adjust to match the requested behavior: only process pages with rules if USE_WHOLE_PAGE_MODE is False.


    for name in pages_to_process:
        url = f"{BASE_URL}/{name}"
        print(f"\nProcessing {url}")

        response = None
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            not_found_pages.append(url)
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        page_entries = []

        if USE_WHOLE_PAGE_MODE:
            print(f"  Using 'whole_page' parser for '{name}'...")
            page_entries = extract_whole_page_content(soup)
        elif name in page_specific_structures:
            print(f"  Using custom parser for '{name}'...")
            try:
                page_entries = extract_from_structure_map(soup, page_specific_structures[name])
                if not page_entries: # If custom parsing yielded no results, consider it a failure
                    print(f"  Custom parsing for '{name}' yielded no entries. Logging as failure.")
                    parsing_failed_pages.append({'page': name, 'url': url, 'reason': 'No entries extracted by custom rule'})
            except Exception as e:
                print(f"  Error during custom parsing for '{name}': {e}. Logging as failure.")
                parsing_failed_pages.append({'page': name, 'url': url, 'reason': str(e)})
                continue # Skip saving if parsing failed
        else:
            # This 'else' block implies a default parser for pages *without* a specific rule
            # when USE_WHOLE_PAGE_MODE is False.
            # Based on the new instruction ("use the pages for which we have a parsing rule"),
            # this 'else' block would ideally not be reached if `pages_to_process` only contains pages with rules.
            # However, if `pages_to_process` remains `PAGE_NAMES` for some reason while USE_WHOLE_PAGE_MODE is False,
            # this would catch pages without specific rules. For now, it's consistent with previous default behavior.
            print(f"  Using default parser for '{name}' (no custom rule specified)...")
            page_entries = extract_h1_and_paragraphs(soup)


        if not page_entries:
            print(f"  No entries found for '{name}', skipping save.")
            continue

        # Assign unique integer ID to each entry
        for entry in page_entries:
            entry['id'] = id_counter
            id_counter += 1

        # Save to JSON file
        output_path = os.path.join(OUTPUT_DIR, f"{name}.json")
        save_to_json(page_entries, output_path)

    # Save the list of not found pages to not_found.json
    not_found_output_path = os.path.join(OUTPUT_DIR, 'notes', 'not_found.json')
    save_to_json(not_found_pages, not_found_output_path)
    print(f"\nSaved {len(not_found_pages)} not found pages to {not_found_output_path}")

    # Save the list of parsing failed pages to fails.json
    save_to_json(parsing_failed_pages, FAILS_LOG_FILE)
    print(f"Saved {len(parsing_failed_pages)} parsing failed pages to {FAILS_LOG_FILE}")

if __name__ == '__main__':
    main()