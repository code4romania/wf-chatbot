import requests
from bs4 import BeautifulSoup
import json
import re
import os
import time

# Import your Gemini LLM class
from Gemini import GeminiChat # Assuming GeminiChat is your wrapper for Gemini API/model
from LLM import ChatSession # Import the base class for type hinting/clarity

# --- Configuration ---
BASE_URL = 'https://dopomoha.ro/en'
OUTPUT_RULES_DIR = './data/parsing_rules'
OUTPUT_RULES_FILE = os.path.join(OUTPUT_RULES_DIR, 'parsing_rules.json')
NOTES_DIR = os.path.join(OUTPUT_RULES_DIR, 'notes')
NOT_FOUND_PAGES_FILE = os.path.join(NOTES_DIR, '404.json')
FAILED_RULES_FILE = os.path.join(NOTES_DIR, 'fails.json')
EXAMPLE_RULES_FILE = 'example_parsing_rules.json' # File containing example rules for LLM
MAX_RETRIES_LLM = 3 # Max attempts to get valid JSON from LLM
DEFAULT_NUM_EXAMPLES_FOR_LLM = 2 # Number of example rules to include in the prompt


# Predefined list of page names
PAGE_NAMES = [
    'prima-pagina', 'titlu', 'faq', 'entering-romania', 'my-rights', 'get-help', 'about',
    'info-on-romania', 'stay-safe', 'dopomohaapp', 'applying-for-asylum', 'useful-resources',
    'phrase-book', 'share', 'prima-pagina-1', 'support', 'info',
    'regional-centers-for-procedures-and-accommodation-for-asylum-seekers', 'transportation',
    'issue-of-the-residence-permit', 'legal-status', 'health', 'accommodation', 'call-center',
    'sprijin', 'education', 'housing', 'financial-support-from-unhcr', 'main-page', 'sanatate',
    'information', 'temporary-protection', 'short-stay', 'support-for-drug-users',
    'childrens-nutrition', 'vaccination', 'psychological-support-for-children', 'legal-help',
    'hygiene', 'childrens-security', 'taking-care-of-children-with-disabilities',
    'the-5020-program', 'romanian-language-courses',
    'free-medical-care-for-refugees-from-ukraine', 'job-for-refugees-from-ukraine-in-romania',
    'romanian-eng', 'family-doctor-for-refugees-from-ukraine', 'buses-from-ukraine',
    'laboratory-analysis', 'specialist-doctor', 'assistance-in-the-hospital', 'dentist',
    'medicine-and-rehabilitation-services', 'medical-recovery-in-hospitals',
    'emergency-consultations', 'medicines', 'medical-devices', 'patients-with-covid-19',
    'pregnant-women-from-ukraine', 'national-curative-health-programs',
    'medical-assistance-for-refugees', 'enrolment-of-ukrainian-children',
    'education-useful-information', 'medzirka', 'health-promotion-center', 'free-medical-care',
    'dopomoha-brasov', 'dopomoha-timisoara', 'dopomoha-constanta', 'dopomoha-galati',
    'dopomoha-suceava', 'dopomoha-cluj-napoca', 'dopomoha-satu-mare', 'dopomoha-satu-mare-1',
    'dopomoha-iasi', 'dopomoha-braila', 'dopomoha-oradea', 'terms-and-conditions',
    'cookie-policy', 'dopomoha-sibiu', 'psychological-support-for-ukrainian-refugees',
    'human-trafficking', 'integration', 'ptsd-help', 'social-benefits-for-refugees'
]

# --- LLM Setup (adjust as needed for your environment) ---
llm: ChatSession | None = None # Type hint for clarity
try:
    llm = GeminiChat()
    print("Gemini model initialized successfully.")
except Exception as e:
    print(f"Error initializing Gemini model: {e}")
    print("Please ensure your GeminiChat class is correctly set up and authenticated.")
    # llm remains None if initialization fails, which will be checked in main()


# --- Helper Functions for Incremental Saving ---

def _load_data(filepath: str, default_value):
    """Loads JSON data from a file. Returns default_value if file doesn't exist or is invalid."""
    if not os.path.exists(filepath) or os.stat(filepath).st_size == 0:
        print(f"No existing data found at {filepath}. Initializing with default.")
        return default_value
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded existing data from {filepath}.")
            return data
    except json.JSONDecodeError as e:
        print(f"Warning: Could not decode JSON from {filepath}: {e}. Overwriting with default.")
        return default_value
    except Exception as e:
        print(f"Error loading {filepath}: {e}. Initializing with default.")
        return default_value

def _save_data(filepath: str, data):
    """Saves data to a file as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Data saved to {filepath}")
    except Exception as e:
        print(f"Error saving data to {filepath}: {e}")

# --- Original load_example_rules function, re-added ---
def load_example_rules(filename: str) -> dict:
    """Loads example parsing rules from a JSON file.
    The format is expected to be a dictionary where keys are page names and values are rule objects.
    """
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: Example rules file '{filename}' not found. LLM will have no examples.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{filename}': {e}")
        return {}

def get_html_content(url: str, not_found_pages_list: list[str]) -> str | None:
    """
    Fetches HTML content from a given URL.
    Records 404 errors in the provided list.
    """
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            print(f"404 Not Found: {url}")
            if url not in not_found_pages_list: # Avoid duplicates if same URL is tried multiple times
                not_found_pages_list.append(url)
        else:
            print(f"HTTP Error fetching {url}: {e}")
        return None
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_json_from_response(text: str) -> str | None:
    """
    Extracts a JSON object string from the LLM's response using regex.
    It prioritizes JSON within ```json...``` blocks.
    """
    # Priority 1: JSON within ```json``` block
    match = re.search(r'```json\s*({.*?})\s*```', text, re.DOTALL)
    if match:
        return match.group(1)
    
    # If no ```json``` block is found, return None.
    return None

def generate_prompt(html_content: str, page_name: str, all_example_rules: dict, not_found_pages_list: list[str], num_examples: int = DEFAULT_NUM_EXAMPLES_FOR_LLM) -> str:
    """
    Generates the prompt for the LLM, providing detailed instructions and a specified number of examples.
    """
    rule_structure_desc = """
    You are an expert HTML parsing rule generator. Your task is to analyze the provided HTML content and generate a JSON parsing rule for it.
    The rule should define how to extract the main content blocks from the page.

    The JSON rule MUST be formatted as a single object with the page name as the key, like this:
    ```json
    {
      "your_page_name": {
        "selector": "CSS selector for the main container of content blocks",
        "rules": [
          {
            "tag": "HTML tag of the direct child element",
            "index": "0-based index of this direct child element within the 'selector' element",
            "type": "'+' for a content block, '-' to ignore, '*' if its children are content blocks",
            "child_relative_path": "Optional: CSS selector relative to this element, if 'type' is '*' and its content blocks are nested deeper (e.g., 'div.inner-wrapper')",
            "child_tag": "Required if 'type' is '*': HTML tag of the actual content blocks (e.g., 'p', 'div') within 'child_relative_path' or directly under this element"
          }
          // ... include more rule objects for other direct children as needed
        ]
      }
    }

    Explanation of 'rules' list:
    - The 'rules' array describes the direct children of the element identified by 'selector'.
    - Each rule object corresponds to a direct child HTML tag.
    - 'tag': The HTML tag name of the direct child (e.g., 'div', 'p', 'h1').
    - 'index': The 0-based position of this direct child element.
    - 'type':
        - '+': This direct child element itself contains content that should be extracted.
        - '-': This direct child element should be ignored (e.g., navigation, sidebars, headers, footers, empty divs).
        - '*': This direct child element is a CONTAINER whose direct children (or children specified by 'child_relative_path') are the actual content blocks.
    - 'child_relative_path' (optional for 'type: "*"'): Use this if the actual content blocks are *nested one level deeper* inside the current element, within a specific container (e.g., if target_element has <div class="container"><p>content</p></div>, then child_relative_path would be "div.container").
    - 'child_tag' (required for 'type: "*"'): The HTML tag of the actual content blocks within the container specified by 'child_relative_path' or directly under the current element.

    Aim for a 'selector' that is general enough to capture the main content but specific enough to exclude irrelevant parts like headers, footers, or sidebars.
    Analyze the HTML structure carefully to identify the correct 'selector' and the 'tag', 'index', and 'type' for each relevant direct child.

    Provide ONLY the JSON object. Do NOT include any introductory or concluding text, explanations, or comments outside the JSON block.
    """
    
    example_str = ""
    if all_example_rules and num_examples > 0:
        example_keys = list(all_example_rules.keys())
        # Filter out the current page if it happens to be in example_keys
        filtered_example_keys = [k for k in example_keys if k != page_name]
        selected_examples = filtered_example_keys[:min(num_examples, len(filtered_example_keys))] # Select up to num_examples
        
        example_str += "\nHere are some examples of well-formed parsing rules for other pages to guide you:\n"
        for ex_page_name in selected_examples:
            ex_rule = all_example_rules[ex_page_name]
            ex_html_url = f"{BASE_URL}/{ex_page_name}"
            # Pass not_found_pages_list to get_html_content for example HTML fetches too
            ex_html_content = get_html_content(ex_html_url, not_found_pages_list)

            if ex_html_content:
                example_str += f"\nFor page '{ex_page_name}':\n"
                example_str += json.dumps({ex_page_name: ex_rule}, indent=2) + "\n"
                example_str += f"HTML Content for '{ex_page_name}':\n"
                example_str += f"```html\n{ex_html_content}\n```\n"
            else:
                print(f"Warning: Could not fetch HTML for example page '{ex_page_name}'. Skipping this example.")

    prompt = f"""
    {rule_structure_desc}
    {example_str}

    Now, generate a parsing rule for the page '{page_name}'.
    Analyze the HTML content below to determine the appropriate 'selector' and 'rules' array.
    Focus on identifying the main textual content blocks.

    HTML Content for '{page_name}':
    ```html
    {html_content}
    ```
    Your JSON output for '{page_name}' should be in the format described above. Ensure it's valid JSON and strictly adheres to the requested structure.
    """
    return prompt

def main():
    if llm is None: # Check if LLM initialization failed
        print("LLM model not initialized. Exiting.")
        return

    # Create output directories if they don't exist
    os.makedirs(OUTPUT_RULES_DIR, exist_ok=True)
    os.makedirs(NOTES_DIR, exist_ok=True)

    # --- Load existing data for incremental saving ---
    generated_rules = _load_data(OUTPUT_RULES_FILE, {})
    not_found_pages = _load_data(NOT_FOUND_PAGES_FILE, [])
    failed_rules_pages = _load_data(FAILED_RULES_FILE, [])

    # Use the predefined list of page names.
    # Filter out pages already processed in previous runs for rules and failures.
    # Also ensure we don't re-process pages that were 404 in previous runs.
    processed_pages_keys = set(generated_rules.keys()).union(failed_rules_pages).union(not_found_pages)
    
    discovered_page_names_to_process = [p for p in PAGE_NAMES if p not in processed_pages_keys]
    
    if not discovered_page_names_to_process:
        print("All predefined pages have already been processed or logged as 404/failed. Exiting.")
        return

    print(f"Processing {len(discovered_page_names_to_process)} new/unprocessed page names out of {len(PAGE_NAMES)} total.")

    # Load example rules to provide context to the LLM
    all_example_rules = load_example_rules(EXAMPLE_RULES_FILE)

    for page_name in discovered_page_names_to_process:
        page_url = f"{BASE_URL}/{page_name}"
        print(f"\n--- Processing page: {page_name} ({page_url}) ---")

        html_content = get_html_content(page_url, not_found_pages)
        
        # Always save not_found_pages after trying to fetch HTML for the current page
        # as get_html_content might have added a 404 entry.
        _save_data(NOT_FOUND_PAGES_FILE, not_found_pages)

        if not html_content:
            # If html_content is None, it means there was a fetch error or 404.
            # 404s are already logged to not_found_pages and saved above.
            print(f"Skipping {page_name} due to failure to fetch HTML.")
            # No need to add to failed_rules_pages here, as it's a fetch error, not LLM rule generation failure
            continue

        prompt = generate_prompt(
            html_content, 
            page_name, 
            all_example_rules, 
            not_found_pages, # Pass not_found_pages to collect example HTML fetch errors
            DEFAULT_NUM_EXAMPLES_FOR_LLM
        )
        
        # Save not_found_pages again, as generate_prompt might have added 404s from example fetches
        _save_data(NOT_FOUND_PAGES_FILE, not_found_pages)

        rule_data = None
        for attempt in range(MAX_RETRIES_LLM):
            print(f"Attempt {attempt + 1}/{MAX_RETRIES_LLM} to get JSON for {page_name}...")
            
            llm_response_text, error_code = llm.send(prompt)
            
            if error_code:
                print(f"LLM communication error for {page_name} (attempt {attempt + 1}): {error_code}. Retrying...")
                prompt += (f"\n\nPrevious attempt resulted in an LLM error: {error_code}. "
                            "Please try to generate the rule again.")
                time.sleep(2) # Small delay before retrying LLM call
                continue # Skip to next attempt

            json_str = extract_json_from_response(llm_response_text)
            
            if json_str:
                try:
                    # Attempt to parse the extracted JSON
                    parsed_json = json.loads(json_str)
                    
                    # Validate if the key exists and structure is somewhat correct
                    if page_name in parsed_json and \
                    isinstance(parsed_json[page_name], dict) and \
                    'selector' in parsed_json[page_name] and \
                    'rules' in parsed_json[page_name] and \
                    isinstance(parsed_json[page_name]['rules'], list):
                        
                        rule_data = parsed_json[page_name]
                        print(f"Successfully generated rule for {page_name}.")
                        break # Exit retry loop on success
                    else:
                        print(f"LLM response JSON structure invalid for {page_name}. Retrying...")
                        prompt += (f"\n\nPrevious response had invalid structure or missing key: {json_str}"
                                f"\nPlease ensure the JSON is valid and the key '{page_name}' exists "
                                "with 'selector' (string) and 'rules' (list of objects).")
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error for {page_name}: {e}. Retrying...")
                    prompt += (f"\n\nPrevious response was not valid JSON: {json_str}"
                            f"\nPlease provide ONLY valid JSON within the ```json``` block.")
            else:
                print(f"Could not extract JSON from LLM response for {page_name}. Retrying...")
                prompt += (f"\n\nCould not find a valid JSON block in your previous response. "
                        "Please ensure your output is ONLY the JSON object, enclosed in ```json...```.")

            time.sleep(2) # Small delay before retrying LLM call

        if rule_data:
            generated_rules[page_name] = rule_data
            _save_data(OUTPUT_RULES_FILE, generated_rules) # Save rules immediately
        else:
            print(f"Failed to generate a valid parsing rule for {page_name} after {MAX_RETRIES_LLM} attempts.")
            if page_name not in failed_rules_pages: # Prevent duplicates
                failed_rules_pages.append(page_name)
            _save_data(FAILED_RULES_FILE, failed_rules_pages) # Save failures immediately

    print("\n--- Processing complete ---")
    print(f"Total pages with rules: {len(generated_rules)}")
    print(f"Total 404 pages: {len(not_found_pages)}")
    print(f"Total failed rule generations: {len(failed_rules_pages)}")

if __name__ == '__main__':
    main()