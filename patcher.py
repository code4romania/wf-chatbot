# patcher.py
import sys
import os
import re

def apply_patch(patch_details, patch_num):
    """Applies a single, parsed patch to the target file."""
    file_path = patch_details.get("FILE")
    operation_type = patch_details.get("TYPE")
    description = patch_details.get("DESCRIPTION", "No description")
    old_content = patch_details.get("OLD_BLOCK")
    new_content = patch_details.get("NEW_BLOCK")

    print(f"--- Applying Patch #{patch_num} ---")
    print(f"  Description: {description}")
    print(f"  File: {file_path}")
    print(f"  Type: {operation_type}")

    if not all([file_path, operation_type, old_content is not None, new_content is not None]):
        print("  [ERROR] Patch is missing required fields (FILE, TYPE, OLD_BLOCK, NEW_BLOCK). Skipping.")
        return

    if not os.path.exists(file_path):
        print(f"  [ERROR] Target file not found: '{file_path}'. Skipping.")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            target_content = f.read()
    except Exception as e:
        print(f"  [ERROR] Could not read target file '{file_path}': {e}. Skipping.")
        return

    if old_content not in target_content:
        print(f"  [ERROR] OLD_BLOCK content not found in '{file_path}'. The file may have been modified already. Skipping.")
        return

    if operation_type.upper() == "REPLACE":
        # The 'count=1' argument ensures only the first occurrence is replaced.
        # Remove this if you want to replace all occurrences.
        modified_content = target_content.replace(old_content, new_content, 1)
    else:
        print(f"  [ERROR] Unsupported operation type: '{operation_type}'. Skipping.")
        return
        
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        print(f"  [SUCCESS] Patch applied and '{file_path}' has been updated.")
    except Exception as e:
        print(f"  [ERROR] Could not write changes to '{file_path}': {e}. Skipping.")

def parse_cdiff(file_content):
    """Parses the content of a .cdiff file and returns a list of patches."""
    # Use regex to find all patch blocks
    patch_blocks = re.findall(r'--- PATCH_START ---(.*?)--- PATCH_END ---', file_content, re.DOTALL)
    
    parsed_patches = []
    for block in patch_blocks:
        patch = {}
        # Extract headers
        headers = re.findall(r'^\s*([A-Z]+):\s*(.*)', block, re.MULTILINE)
        for key, value in headers:
            patch[key.upper()] = value.strip()
            
        # Extract content blocks
        old_block_match = re.search(r'OLD_BLOCK_START\n(.*?)\nOLD_BLOCK_END', block, re.DOTALL)
        if old_block_match:
            patch['OLD_BLOCK'] = old_block_match.group(1)

        new_block_match = re.search(r'NEW_BLOCK_START\n(.*?)\nNEW_BLOCK_END', block, re.DOTALL)
        if new_block_match:
            patch['NEW_BLOCK'] = new_block_match.group(1)
            
        parsed_patches.append(patch)
        
    return parsed_patches


def main():
    """Main function to run the patcher."""
    if len(sys.argv) < 2:
        print("Usage: python patcher.py <your_diff_file.cdiff>")
        sys.exit(1)

    cdiff_file = sys.argv[1]
    if not os.path.exists(cdiff_file):
        print(f"Error: The specified diff file '{cdiff_file}' was not found.")
        sys.exit(1)

    print(f"Reading patch file: {cdiff_file}\n")
    try:
        with open(cdiff_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error: Could not read the diff file: {e}")
        sys.exit(1)

    patches = parse_cdiff(content)
    
    if not patches:
        print("No valid patch blocks found in the file.")
        sys.exit(0)

    print(f"Found {len(patches)} patch(es) to process.\n")
    
    for i, patch in enumerate(patches, 1):
        apply_patch(patch, i)
        print("-" * 30)

if __name__ == "__main__":
    main()