from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

# Dataset yolu
file_path = "C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/hatespeech_dataset.xlsx"

# Dataset'i yükleme ve düzenleme
df = pd.read_excel(file_path, sheet_name="1000 Tweet")
inputs = df['Tweet'].tolist()

# İkili sınıflandırma: "nefret" ve "saldırgan" -> 1 (zararlı), "hiçbiri" -> 0 (zararsız)
labels = df['Etiket'].map({"hiçbiri":  0, "nefret": 1, "saldırgan": 1}).tolist()

print(f"Toplam veri sayısı: {len(inputs)}")
print(f"Etiket dağılımı:")
print("Zararsız (0):", labels.count(0))
print("Zararlı (1):", labels.count(1))

# Eğitim ve doğrulama datası ayırma
train_inputs, val_inputs, train_labels, val_labels = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Model ve tokenizer seçimi - num_labels=2 (ikili sınıflandırma)
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Inputları tokenize etme
train_encodings = tokenizer(train_inputs, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_inputs, truncation=True, padding=True, max_length=128)

# PyTorch Dataset sınıfı
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self. labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = HateSpeechDataset(train_encodings, train_labels)
val_dataset = HateSpeechDataset(val_encodings, val_labels)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results_binary",
    per_device_train_batch_size=4,
    num_train_epochs=3,  # Biraz daha fazla epoch
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
)

# Trainer'ı oluşturuyoruz
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("İkili sınıflandırma modeli eğitimi başlıyor...")
# Trainer ile model eğitme
trainer.train()

print("Eğitim tamamlandı, model kaydediliyor...")
# Eğitilen modeli kaydetme - yeni isimle
model.save_pretrained("C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/Code/binary_hatespeech_model")
tokenizer.save_pretrained("C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/Code/binary_hatespeech_tokenizer")

print("İkili sınıflandırma modeli başarıyla kaydedildi!")
print("\nYeni model dosyaları:")
print("Model: C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/Code/binary_hatespeech_model")
print("Tokenizer: C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/Code/binary_hatespeech_tokenizer")