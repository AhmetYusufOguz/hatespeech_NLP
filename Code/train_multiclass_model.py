from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import os
import numpy as np

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Dataset yolu
file_path = os.path.join(project_root, "hatespeech_dataset.xlsx")

# Dataset'i yükleme ve düzenleme
df = pd.read_excel(file_path, sheet_name="Dengeli Veriseti", header=None)
df = df.dropna(subset=[2])
inputs = df[1].tolist()

# Çok sınıflı sınıflandırma: 0=Hiçbiri, 1=Nefret (group-based), 2=Saldırgan (insult), 3=Tehdit (violence to others), 4=Niyet (self-harm)
labels = df[2].str.lower().map({"hiçbiri": 0, "nefret": 1, "saldırgan": 2, "tehdit": 3, "niyet": 4}).tolist()

print(f"Toplam veri sayısı: {len(inputs)}")
print(f"Etiket dağılımı:")
print("Hiçbiri (0):", labels.count(0))
print("Nefret (1):", labels.count(1))
print("Saldırgan (2):", labels.count(2))
print("Tehdit (3):", labels.count(3))
print("Niyet (4):", labels.count(4))

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Eğitim ve doğrulama datası ayırma
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Model ve tokenizer seçimi - num_labels=5 (çok sınıflı sınıflandırma)
model_name = "loodos/bert-base-turkish-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)

# Inputları tokenize etme
train_encodings = tokenizer(train_inputs, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_inputs, truncation=True, padding=True, max_length=128)

# PyTorch Dataset sınıfı
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

# Custom Trainer
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro")
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(project_root, "results_multiclass"),
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=8,
    learning_rate=2e-5,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
)

# Trainer'ı oluşturuyoruz
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Çok sınıflı sınıflandırma modeli eğitimi başlıyor...")
# Trainer ile model eğitme
trainer.train()

print("Eğitim tamamlandı, model kaydediliyor...")
# Eğitilen modeli kaydetme - yeni isimle
model.save_pretrained(os.path.join(script_dir, "multiclass_hatespeech_model"))
tokenizer.save_pretrained(os.path.join(script_dir, "multiclass_hatespeech_tokenizer"))

print("Çok sınıflı sınıflandırma modeli başarıyla kaydedildi!")
print("\nYeni model dosyaları:")
print("Model:", os.path.join(script_dir, "multiclass_hatespeech_model"))
print("Tokenizer:", os.path.join(script_dir, "multiclass_hatespeech_tokenizer"))
