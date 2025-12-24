from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Model ve veriyi yükleme
model_path = os.path.join(script_dir, "binary_hatespeech_model")
tokenizer_path = os.path.join(script_dir, "binary_hatespeech_tokenizer")
file_path = os.path.join(project_root, "hatespeech_dataset.xlsx")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
df = pd.read_excel(file_path, sheet_name="Dengeli Veriseti", header=None)

def predict_message(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch. argmax(predictions, dim=-1).item()
    
    class_names = {0: "Zararsız", 1: "Zararlı"} 
    confidence = predictions[0][predicted_class].item()
    return predicted_class, class_names[predicted_class], confidence

print("Gerçek Dataset ile İkili Model Testi:")
print("=" * 80)

# Her kategoriden örnekler alalım
zararsiz_samples = df[df[2] == 'hiçbiri'][1].head(5)
nefret_samples = df[df[2] == 'nefret'][1].head(5)
saldirgan_samples = df[df[2] == 'saldırgan'][1].head(5)

print("ZARARSIZ örnekleri:")
print("-" * 40)
zararsiz_dogru = 0
for i, tweet in enumerate(zararsiz_samples, 1):
    pred_class, pred_name, confidence = predict_message(tweet)
    correct = "✓" if pred_name == "Zararsız" else "✗"
    if pred_name == "Zararsız": 
        zararsiz_dogru += 1
    print(f"{i}. {correct} Tahmin: {pred_name} ({confidence:.1%})")
    print(f"   Tweet: {tweet[:80]}...")
    print()

print("NEFRET örnekleri (zararlı olmalı):")
print("-" * 40)
nefret_dogru = 0
for i, tweet in enumerate(nefret_samples, 1):
    pred_class, pred_name, confidence = predict_message(tweet)
    correct = "✓" if pred_name == "Zararlı" else "✗"
    if pred_name == "Zararlı":
        nefret_dogru += 1
    print(f"{i}. {correct} Tahmin: {pred_name} ({confidence:.1%})")
    print(f"   Tweet: {tweet[:80]}...")
    print()

print("SALDIRGAN örnekleri (zararlı olmalı):")
print("-" * 40)
saldirgan_dogru = 0
for i, tweet in enumerate(saldirgan_samples, 1):
    pred_class, pred_name, confidence = predict_message(tweet)
    correct = "✓" if pred_name == "Zararlı" else "✗"
    if pred_name == "Zararlı": 
        saldirgan_dogru += 1
    print(f"{i}. {correct} Tahmin: {pred_name} ({confidence:.1%})")  # Format hatası düzeltildi
    print(f"   Tweet:  {tweet[:80]}...")
    print()

# Genel performans
print("PERFORMANS ÖZETI:")
print("-" * 40)
print(f"Zararsız: {zararsiz_dogru}/5 doğru ({zararsiz_dogru/5:.1%})")
print(f"Nefret: {nefret_dogru}/5 doğru ({nefret_dogru/5:.1%})")
print(f"Saldırgan: {saldirgan_dogru}/5 doğru ({saldirgan_dogru/5:.1%})")
print(f"Toplam zararlı tespit:  {nefret_dogru + saldirgan_dogru}/10 ({(nefret_dogru + saldirgan_dogru)/10:.1%})")