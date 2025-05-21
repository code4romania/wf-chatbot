"""
Script: generate.py
Reads scraped JSON data and generates Q&A in multiple languages using DeepSeek.
Usage: python generate.py <input_json> [output_prefix]
"""
import json
import sys
import re
#from DeepSeek import DeepSeek
from Gemini import GeminiChat

chat = GeminiChat()

language_map = {'ro': 'Romanian', 'ru': 'Russian', 'uk': 'Ukrainian', 'en': 'English'}
language_map = {'en': 'English'}


def load_scraped(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_for_language(scraped, code, language):
    training = []
    


    for entry in scraped:

        base_prompt = (
        f"Given the title: '{entry['heading']}' and the summary: '{entry['summary']}', "
        f"generate 5 short, distinct questions in {language}. "
        "You are creating a database of frequently asked questions where each question must be stand-alone and self-explanatory. Threfore avoid saying This or These!"
        "Someone searching this database should understand the topic of the question just by reading the question itself, without referring to any title or summary. "
        "For example, if the topic is 'Benefits of Regular Exercise', a POOR question is 'How often should one do it?'. A BETTER question is 'How often should one engage in regular exercise?'. "
        "The questions should be things a person curious about '{entry['heading']}' might ask before knowing details. "
        "Do not include the words “title”, 'these','this' or “summary” in your questions. "
        "Return only a Python list literal of exactly 5 items."
        )
        history = []
        questions = []
        # attempt loop
        for _ in range(4):
            # initial
            resp1, _ = chat.send(base_prompt)
            history += [('user', base_prompt), ('assistant', resp1)]
            match = re.search(r"(\[.*\])", resp1, re.S)
            if match:
                try:
                    lst = eval(match.group(1))
                    if isinstance(lst, list) and len(lst) == 5:
                        questions = lst
                        break
                except:
                    pass
            # corrective
            correction = "Response couln't parsed as a python list with 5 values. Make sure you return a valid Python list literal with 5 items, e.g. ['a','b','c','d','e']."
            history.append(('user', correction))
            convo = ''.join([f"{('User' if r=='user' else 'Bot')}: {msg}\n" for r,msg in history])
            resp2, _ = chat.send(convo)
            history.append(('assistant', resp2))
            match2 = re.search(r"(\[.*\])", resp2, re.S)
            if match2:
                try:
                    lst2 = eval(match2.group(1))
                    if isinstance(lst2, list) and len(lst2)==5:
                        questions = lst2
                        break
                except:
                    pass
        if not questions:
            continue
        # answers
        for q in questions:
            apr = q
            ans_prompt = f"Answer in {language} using only the summary: '{entry['summary']}'.\nQuestion: {q}\n Give a concise answer of the question while providing details. Don't just give a yes/no answer, give a full answer from the summary. Don't mention that the response is coming from the summary.Don't say anything like according to the summary. Return only the answer text."
            ans, _ = chat.send(ans_prompt)
            training.append({
                'aPrompt': apr,
                'bResponse': ans.strip() + f" For more details, visit {entry.get('url','')}",
                'cSubject': entry['heading'],
                'dLanguage': language,
                'eVerified Translation': 'No',
                'fStatus': 'Scraped'
            })
    return training


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python generate.py <input_json> [output_prefix]")
        sys.exit(1)
    input_file = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv)>2 else 'scrape_data'
    scraped = load_scraped(input_file)
    for code, lang in language_map.items():
        out = generate_for_language(scraped, code, lang)
        fname = f"{prefix}_{code}.json"
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump({'conversation': out}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {fname}")