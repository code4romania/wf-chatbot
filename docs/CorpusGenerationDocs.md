# Corpus Generation Pipeline

This documentation explains the automatred generation of the Q&A corpus. It works in three main stages:
1. **Scrape**: Extracts text content from a predefined list of web pages.
2. **Generate Questions**: Uses the Google Gemini model to generate relevant questions based on the scraped text.
3. **Generate Answers**: Uses the same AI model to answer the generated questions using the original text as context.

The entire pipeline is orchestrated by `backend/CorpusGeneration/corpus_generator.py` and configured via `backend/CorpusGeneration/GenerationParams.json`.
## Table of Contents

- [Setup](#setup)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
  - [API Key Configuration](#api-key-configuration)  
- [Configuration](#configuration)  
  - [`OUTPUT_DIR`](#output_dir)  
  - [`ScrapingParams`](#scrapingparams)  
  - [`QuestionGenerationParams`](#questiongenerationparams)  
  - [`AnswerGenerationParams`](#answergenerationparams)  
  - [Example Workflow Modification](#example-workflow-modification)  
- [Usage](#usage)  
  - [Run the Full Pipeline](#run-the-full-pipeline)  
  - [Run Specific Stages](#run-specific-stages)  

---

## Setup

### Prerequisites
* Python 3.8+

### Installation
1. Clone this repository to your local machine.
2. Navigate to the `backend/CorpusGeneration` directory.
3. Create a file named `requirements.txt` with the following content:
   ```
   requests
   beautifulsoup4
   google-generativeai
   ```
4. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### API Key Configuration

This project requires access to the Google Gemini API.

2. **Set the Environment Variable**: You must set your API key as an environment variable named `GOOGLE_API_KEY`.

**On macOS / Linux:**
```bash
export GOOGLE_API_KEY='YOUR_API_KEY_HERE'
```
*To make this permanent, add the line to your `~/.bashrc`, `~/.zshrc`, or other shell configuration file.*

**On Windows (Command Prompt):**
```powershell
setx GOOGLE_API_KEY "YOUR_API_KEY_HERE"
```
*You may need to restart your terminal for the change to take effect.*

---

## Configuration

ConfigParams.json JSON file controls how each stage of the pipeline runs. It contains several important sections:

### `OUTPUT_DIR`
- The root output folder (under `corpus/`) where scraped content, questions, and answers will be stored.

---

### `ScrapingParams`
Controls which pages are scraped and how content is extracted.

- `BASE_URL`: The base URL for all pages.
- `START_PHRASE` and `END_PHRASE`: Define the start and end boundaries for relevant text content extraction.
- `PAGE_NAMES`: A list of page slugs to scrape, relative to the `BASE_URL`.

---

### `QuestionGenerationParams`

Controls how questions are generated from scraped content.

- `question_prompt`: Template sent to the Gemini model to generate questions.
- `n_questions`: Minimum number of questions to generate per page.
- `max_questions`: Maximum number of questions to allow per page.
- `question_directory`: The directory **to write the generated questions to** under `corpus/OUTPUT_DIR/`. You can change this value to experiment with different question generation settings and have multiple question sets.


---

### `AnswerGenerationParams`

Controls how answers are generated from questions and scraped content.

- `answer_prompt_template`: Template used to ask Gemini to generate answers to questions.
- `answers_directory`: The directory **to write the generated answers to** under `corpus/OUTPUT_DIR/`. Change this to test different answer generation strategies.
- `questions_directory`: The directory **to read questions from**, relative to `corpus/OUTPUT_DIR/`. This allows you to pair different sets of questions with the same content.
---

### Example Workflow Modification

To experiment with new question generation settings without overwriting old outputs:

1. In `QuestionGenerationParams`, set:
   ```json
   "question_directory": "questions_v2"
   ```

2. In `AnswerGenerationParams`, point to the new questions:
   ```json
   "questions_directory": "questions_v2",
   "answers_directory": "answers_v2"
   ```

This way you can run the same scraped content through different question or answer generation configurations and store the outputs separately.

---




## Usage

Run from the `backend/CorpusGeneration` directory.

### Run the Full Pipeline

To run all three stages (Scrape → Generate Questions → Generate Answers):
```bash
python corpus_generator.py
```

### Run Specific Stages

* **Scrape only:**
```bash
python corpus_generator.py --scrape
```
*(or `python corpus_generator.py -s`)*

* **Generate Questions only:** (Requires scraped content)
```bash
python corpus_generator.py --questions
```
*(or `python corpus_generator.py -q`)*

* **Generate Answers only:** (Requires questions to exist)
```bash
python corpus_generator.py --answers
```
*(or `python corpus_generator.py -a`)*

* **Generate Questions and Answers:**
```bash
python corpus_generator.py -qa
```
