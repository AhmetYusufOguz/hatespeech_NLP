
import pandas as pd
import os
from openpyxl import load_workbook

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Dataset yolu
file_path = os.path.join(project_root, "hatespeech_dataset.xlsx")

# Yeni veriler
new_data = [
    ("Row108", "sandalye ve ipi al gerekeni yap", "niyet", ""),
    ("Row109", "balkonda kendini aşağı bırak", "niyet", ""),
    ("Row110", "sandalyeyi tekmeleme zamanı gelmiş", "niyet", "")
]

# Excel dosyasını yükle
book = load_workbook(file_path)

# "Dengeli Veriseti" sayfasına yeni verileri ekle
sheet = book["Dengeli Veriseti"]
for row in new_data:
    sheet.append(row)

# Değişiklikleri kaydet
book.save(file_path)

print("Yeni veriler 'Dengeli Veriseti' sayfasına eklendi.")
