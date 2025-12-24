from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Yeni çok sınıflı modeli yükleme
model_path = os.path.join(script_dir, "multiclass_hatespeech_model")
tokenizer_path = os.path.join(script_dir, "multiclass_hatespeech_tokenizer")

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Test fonksiyonu
def predict_message(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class_tensor = torch.max(predictions, dim=-1)
        predicted_class = predicted_class_tensor.item()
        confidence = confidence.item()

    if confidence < 0.80:
        predicted_class = 0  # Hiçbiri

    # Çok sınıflı sınıflandırma: 0=Hiçbiri, 1=Nefret, 2=Saldırgan, 3=Tehdit, 4=Niyet
    class_names = {0: "Hiçbiri", 1: "Nefret", 2: "Saldırgan", 3: "Tehdit", 4: "Niyet"}
    
    return class_names[predicted_class], confidence

# Test mesajları
test_messages = [
    # Hiçbiri mesajlar
    "Merhaba, nasılsın?",
    "Bu ürün gerçekten çok güzel",
    "Bugün hava çok güzel.",
    "Yemek harikaydı.",
    "Kitap okumayı seviyorum.",
    "Bu aptalca bir fikir.",

    # Nefret içeren mesajlar
    "Bu insanları hiç sevmiyorum, hepsinden nefret ediyorum",
    "Seni dövmek istiyorum", # This could be Tehdit too. But as a general rule, if it's not specific to a person, it's Nefret
    "Ondan nefret ediyorum.",
    "Hepinizden iğreniyorum.",

    # Saldırgan içeren mesajlar
    "Bu ne saçmalık?",
    "İşini düzgün yap.",

    # Tehdit içeren mesajlar (violence to others)
    "Seni mahvederim.",
    "Bunu bir daha yapma.",
    "Sonun kötü olur.",

    # Niyet içeren mesajlar (self-harm)
    "Kendini boğmak için bir nedenin var.",
    "balkonda kendini aşağı bırak",
    "Çatıdan atlamak için güzel bir gün.",
    "Bıçağı al ve bitir bu işi.",
    "Artık nefes almanın bir anlamı yok.",
]

# Beklenen etiketler
expected_labels = [
    "Hiçbiri", "Hiçbiri", "Hiçbiri", "Hiçbiri", "Hiçbiri", "Hiçbiri",
    "Nefret", "Nefret", "Nefret", "Nefret",
    "Saldırgan", "Saldırgan",
    "Tehdit", "Tehdit", "Tehdit",
    "Niyet", "Niyet", "Niyet", "Niyet", "Niyet",
]

print("Çok Sınıflı Sınıflandırma Model Test Sonuçları:")
print("=" * 70)

correct_predictions = 0
for i, message in enumerate(test_messages):
    prediction, confidence = predict_message(message)
    
    expected = expected_labels[i]
    correct = "✓" if prediction == expected else "✗"
    
    if prediction == expected:
        correct_predictions += 1
    
    print(f"{correct} Mesaj: '{message}'")
    print(f"   Beklenen: {expected} | Tahmin: {prediction} (Güven: {confidence:.1%})")
    print("-" * 70)

print(f"\nPerformans Özeti:")
print(f"Genel doğruluk: {correct_predictions}/{len(test_messages)} ({(correct_predictions/len(test_messages)):.1%})")
