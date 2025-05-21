# count_json_items.py
"""
Utility to count list items in a JSON file.
Supports files where the top-level is a list, or where a key (like 'conversation') maps to a list.
Usage:
    python count_json_items.py path/to/file.json
"""
import json
import sys
import argparse

def count_items_in_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # If top-level is a list
    if isinstance(data, list):
        return len(data)
    # If top-level is a dict with a list value
    if isinstance(data, dict):
        # Try common keys
        for key in ('conversation', 'data', 'items'):
            if key in data and isinstance(data[key], list):
                return len(data[key])
        # If exactly one list value
        list_values = [v for v in data.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return len(list_values[0])
    raise ValueError("No list found at top-level in JSON file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count number of list items in a JSON file.")
    parser.add_argument('json_file', help='Path to the JSON file')
    args = parser.parse_args()
    try:
        count = count_items_in_json(args.json_file)
        print(f"Found {count} items in {args.json_file}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
