import requests
from bs4 import BeautifulSoup
import json
import os
import logging

# Set up logging to a file
logging.basicConfig(filename='processing.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_generation_params(filename):
    """
    Loads all configuration parameters from a JSON file.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            params = json.load(f)
        logging.info(f"Successfully loaded parameters from {filename}")
        return params
    except FileNotFoundError:
        logging.error(f"Error: Parameter file '{filename}' not found. Please create it.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{filename}': {e}")
        return None

def extract_whole_page_content(soup_obj):
    """
    Extracts the entire text content of the <body> tag as a single content block.
    Returns a list containing one dict with 'heading' and 'summary'.
    """
    body_content = soup_obj.find('body')
    if body_content:
        full_text = body_content.get_text(separator=' ', strip=True)
        return [{'heading': 'Whole Page Content', 'summary': full_text}]
    return []

def save_to_json(entries, path):
    """
    Save `entries` to a JSON file at `path`, creating directories as needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved {len(entries)} entries to {path}")
    except IOError as e:
        logging.error(f"Error writing {path}: {e}")

def process_summary(text, start_phrase, end_phrase):
    """
    Finds and returns the text between the start and end phrases.
    Supports None values for start_phrase or end_phrase.
    """
    start_idx = 0
    if start_phrase is not None and start_phrase.strip():
        start_idx = text.find(start_phrase)
        if start_idx == -1:
            return None
        start_idx += len(start_phrase)

    end_idx = len(text)
    if end_phrase is not None and end_phrase.strip():
        end_idx = text.find(end_phrase, start_idx)
        if end_idx == -1:
            return None

    return text[start_idx:end_idx].strip()

def scrape():
    """
    Main function to orchestrate the scraping and stripping process.
    """
    params = load_generation_params('GenerationParams.json')
    if not params:
        print("Failed to load parameters. Exiting.")
        return

    # --- Scraping, Processing, and Storing ---
    print("\n--- Starting Web Scraping and Processing ---")
    scraping_params = params.get('ScrapingParams', {})
    page_names = scraping_params.get('PAGE_NAMES', [])
    base_url = scraping_params.get('BASE_URL')
    start_phrase = scraping_params.get('START_PHRASE')
    end_phrase = scraping_params.get('END_PHRASE')
    output_dir = params.get('OUTPUT_DIR')

    if not page_names or not base_url or not output_dir:
        logging.error("Missing required parameters.")
        return
    
    final_output_path = os.path.join('corpus', output_dir)
    os.makedirs(final_output_path, exist_ok=True)

    id_counter = 1
    for name in page_names:
        url = f"{base_url}/{name}"
        logging.info(f"Processing URL: {url}")
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            page_entries = extract_whole_page_content(soup)

            if not page_entries:
                logging.warning(f"No content found for '{name}', skipping save.")
                continue

            # Process the summary immediately after scraping
            entry = page_entries[0]
            summary = entry.get('summary', '')
            processed_summary = process_summary(summary, start_phrase, end_phrase)
            
            if processed_summary is not None:
                entry['summary'] = processed_summary
                entry['id'] = id_counter
                id_counter += 1
                
                # Save the processed content directly
                file_path = os.path.join(final_output_path, f"{name}.json")
                save_to_json([entry], file_path)
            else:
                logging.info(f"Stripping phrases not found in {name}. Skipping save for this page.")

        except requests.RequestException as e:
            logging.error(f"Error fetching {url}: {e}")
            continue
    
    print("\nScraping and processing complete. Check processing.log for details.")

if __name__ == '__main__':
    scrape()