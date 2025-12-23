import pandas as pd
import os
import re

def check_duplicates():
    # Get the absolute path of the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # 1. Read sentences from the excel file
    file_path_xlsx = os.path.join(project_root, "hatespeech_dataset.xlsx")
    df = pd.read_excel(file_path_xlsx, sheet_name="Dengeli Veriseti", header=None)
    excel_sentences = set(df[1].tolist())

    # 2. Read sentences from the python file
    file_path_py = os.path.join(script_dir, "test_multiclass_model.py")
    with open(file_path_py, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find the test_messages list
    match = re.search(r'test_messages\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        print("Could not find 'test_messages' in the python file.")
        return

    test_messages_str = match.group(1)
    
    # Extract the sentences from the list
    # This is a bit tricky as the sentences can contain commas.
    # I will split by newline and then clean up.
    python_sentences = set()
    for line in test_messages_str.split('\n'):
        line = line.strip()
        if line.startswith('"') and line.endswith(','):
            line = line[1:-2]
        elif line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        
        if line and not line.startswith('#'):
            python_sentences.add(line)

    # 3. Find duplicates
    duplicates = excel_sentences.intersection(python_sentences)

    # 4. Report duplicates
    if duplicates:
        print("Found the following duplicate sentences:")
        for sentence in duplicates:
            print(f"- {sentence}")
    else:
        print("No duplicate sentences found.")

if __name__ == '__main__':
    check_duplicates()