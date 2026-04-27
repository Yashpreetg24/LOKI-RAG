
import os

with open('/Users/yashpreetgupta/Desktop/LOKI-RAG/app/static/app.js', 'r') as f:
    lines = f.readlines()

header_lines = []
in_header = False
for line in lines:
    if 'pre.textContent = [' in line:
        in_header = True
        continue
    if in_header:
        if '].join' in line:
            in_header = False
            break
        header_lines.append(line.strip().strip("',"))

for i, hl in enumerate(header_lines):
    print(f"Line {i}: {len(hl)} chars - '{hl}'")
