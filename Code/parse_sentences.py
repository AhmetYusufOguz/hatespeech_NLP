
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Read the file
file_path = os.path.join(script_dir, "Niyet cümleleri.txt")
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start of the sentences
start_index = -1
for i, line in enumerate(lines):
    if "Eklenen 50 Yeni Cümle:" in line:
        start_index = i + 1
        break

# Extract sentences
sentences = []
if start_index != -1:
    for line in lines[start_index:]:
        stripped_line = line.strip()
        if stripped_line:
            sentences.append(stripped_line)

# Print the sentences to verify
for sentence in sentences:
    print(sentence)
