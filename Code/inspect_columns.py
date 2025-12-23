
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
file_path = os.path.join(project_root, "hatespeech_dataset.xlsx")

df = pd.read_excel(file_path, sheet_name="Dengeli Veriseti", header=None)
print(df.tail())
