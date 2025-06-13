# `.cdiff` File Syntax Guide

## Overview

The `.cdiff` (Custom Diff) format is a plain text specification for describing targeted changes within source code files. It is designed to be human-readable and version-controllable. The format is processed by a patcher script that applies the described changes in a literal and precise manner.

This system is ideal for small, atomic updates, such as changing configuration values, updating styling, or modifying specific code blocks, where the original content is known.

## Core Structure

A `.cdiff` file consists of one or more **Patch Blocks**. Each block defines a single, self-contained operation on a target file.

- **Patch Block Delimiters**: Every patch block must begin with `--- PATCH_START ---` and end with `--- PATCH_END ---` on their own lines.

- **Comments**: Lines starting with a hash (`#`) outside of `OLD_BLOCK` and `NEW_BLOCK` sections are treated as comments and are ignored by the parser.

### Patch Block Components

Inside each `PATCH_START`/`PATCH_END` block, there are three components: Headers, the Old Content Block, and the New Content Block.

#### 1. Headers

Headers are key-value pairs that provide metadata for the patch.

- `FILE: <target_filename>` (Required)
  - Specifies the path to the file that needs to be modified. The path should be relative to the location of the patcher script.
  - Example: `FILE: src/config/settings.py`

- `TYPE: <operation_type>` (Required)
  - Defines the action to perform. Currently, the primary supported type is `REPLACE`.
  - Example: `TYPE: REPLACE`

- `DESCRIPTION: <text>` (Optional)
  - A brief, human-readable description of what the patch does.
  - Example: `DESCRIPTION: Update the welcome message on the main page.`

#### 2. Old Content Block

This block defines the exact content to find in the target file.

- **Start Delimiter**: `OLD_BLOCK_START` on its own line.
- **End Delimiter**: `OLD_BLOCK_END` on its own line.
- **Content**: Everything between the start and end delimiters, including all whitespace, indentation, and newline characters, is treated as part of the literal string to be matched.

#### 3. New Content Block

This block defines the content that will replace the `OLD_BLOCK`.

- **Start Delimiter**: `NEW_BLOCK_START` on its own line.
- **End Delimiter**: `NEW_BLOCK_END` on its own line.
- **Content**: Everything between the start and end delimiters, including all whitespace, indentation, and newlines, will be inserted into the file.

---

## Full Example

Here is a complete example of a `.cdiff` file containing two distinct patches for a Python file that generates HTML.

```
# report_updates.cdiff
# This file contains patches to update the visual style and heading of the report.

--- PATCH_START ---
FILE: report.py
TYPE: REPLACE
DESCRIPTION: Change the main report heading from H1 to H2 with a new class.

OLD_BLOCK_START
html += "<h1>Financial Report</h1>"
OLD_BLOCK_END

NEW_BLOCK_START
html += '<h2 class="report-title">Quarterly Financial Report</h2>'
NEW_BLOCK_END
--- PATCH_END ---

--- PATCH_START ---
FILE: report.py
TYPE: REPLACE
DESCRIPTION: Update the CSS style block to add a new color for the title.

OLD_BLOCK_START
<style>
  body { font-family: Arial, sans-serif; }
  table { border-collapse: collapse; }
</style>
OLD_BLOCK_END

NEW_BLOCK_START
<style>
  body { font-family: Arial, sans-serif; }
  table { border-collapse: collapse; }
  .report-title { color: #4A90E2; }
</style>
NEW_BLOCK_END
--- PATCH_END ---

```