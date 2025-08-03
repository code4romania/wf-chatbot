import os
import sys
import argparse
import json
import logging

# Import the core function from each stage-specific file
from scrape import scrape
from generate_questions import generate_questions
from generate_answers import answer_generation # Correctly imports your new function

def main():
    """
    Main function to orchestrate the corpus generation pipeline by calling imported functions.
    """
    # --- Pre-flight Checks ---
    if not os.path.exists('GenerationParams.json'):
        print("Error: `GenerationParams.json` not found. This file is required to run.")
        sys.exit(1)
        
    if 'GOOGLE_API_KEY' not in os.environ:
        print("Error: `GOOGLE_API_KEY` environment variable not set.")
        print("Please get an API key from Google AI Studio and set it.")
        sys.exit(1)

    # --- CLI Argument Parsing ---
    parser = argparse.ArgumentParser(description="Corpus Generation Pipeline for Q&A creation.")
    parser.add_argument('-s', '--scrape', action='store_true', help="Run only the web scraping process.")
    parser.add_argument('-q', '--questions', action='store_true', help="Run only the question generation process.")
    parser.add_argument('-a', '--answers', action='store_true', help="Run only the answer generation process.")
    parser.add_argument('-qa', action='store_true', help="Run question generation followed by answer generation.")
    
    args = parser.parse_args()

    # Determine which actions to run
    run_s = args.scrape
    run_q = args.questions or args.qa
    run_a = args.answers or args.qa

    # Default action: if no flags are given, run everything
    if not any([run_s, run_q, run_a]):
        run_s, run_q, run_a = True, True, True
        print("No specific stage selected. Running the full pipeline: Scrape -> Questions -> Answers")

    # --- Execute Pipeline Stages ---
    try:
        if run_s:
            scrape()
        
        if run_q:
            # Check if scraped data exists before running
            with open('GenerationParams.json') as f:
                params = json.load(f)
            scraped_dir = os.path.join('corpus', params.get('OUTPUT_DIR'))
            if not os.path.isdir(scraped_dir) or not os.listdir(scraped_dir):
                 print(f"Error: Scraped data not found in '{scraped_dir}'. Please run --scrape first.")
                 sys.exit(1)
            generate_questions()

        if run_a:
            # Check if question data exists before running
            with open('GenerationParams.json') as f:
                params = json.load(f)
            q_dir = os.path.join('corpus', params.get('OUTPUT_DIR'), params.get('QuestionGenerationParams').get('question_directory'))
            if not os.path.isdir(q_dir) or not os.listdir(q_dir):
                 print(f"Error: Question data not found in '{q_dir}'. Please run --questions or -q first.")
                 sys.exit(1)
            answer_generation() # Calling your new function
            
        print("\n✅ Pipeline execution finished successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred in the pipeline: {e}", exc_info=True)
        print(f"❌ An unexpected error occurred. Check 'pipeline.log' for details. Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()