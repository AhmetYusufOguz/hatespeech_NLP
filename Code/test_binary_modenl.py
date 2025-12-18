from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Yeni ikili modeli yükleme
model_path = "C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/Code/binary_hatespeech_model"
tokenizer_path = "C:/Users/ayogu/Desktop/Okul/4.Year/1.Semester/NLP/Project/Code/binary_hatespeech_tokenizer"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Test fonksiyonu
def predict_message(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch. nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch. argmax(predictions, dim=-1).item()
    
    # İkili sınıflandırma: 0=Zararsız, 1=Zararlı
    class_names = {0: "Zararsız", 1: "Zararlı"}
    confidence = predictions[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# Test mesajları
test_messages = [
    # Zararsız mesajlar
    "Merhaba, nasılsın?",
    "Bu ürün gerçekten çok güzel", 
    "İyi günler diliyorum",
    "Bugün hava çok güzel",
    "Teşekkür ederim",
    
    # Zararlı içerik (nefret)
    "Bu insanları hiç sevmiyorum, hepsinden nefret ediyorum",
    "Şu tür insanlar ülkemizde olmasın", 
    "Bu gruptan olan herkesi kovmak lazım",
    
    # Zararlı içerik (saldırgan)
    "Seni dövmek istiyorum",
    "Kafanı kırarım",
    "Canını yakarım", 
    "Gebermeli bunlar",
]

print("İkili Sınıflandırma Model Test Sonuçları:")
print("=" * 70)

zararsiz_dogru = 0
zararli_dogru = 0
zararsiz_toplam = 5  # İlk 5 mesaj zararsız
zararli_toplam = 7   # Son 7 mesaj zararlı

for i, message in enumerate(test_messages):
    prediction, confidence = predict_message(message)
    
    # Beklenen sonuç
    expected = "Zararsız" if i < 5 else "Zararlı"
    correct = "✓" if prediction == expected else "✗"
    
    # Doğruluk sayımı
    if expected == "Zararsız" and prediction == "Zararsız":
        zararsiz_dogru += 1
    elif expected == "Zararlı" and prediction == "Zararlı":
        zararli_dogru += 1
    
    print(f"{correct} Mesaj: '{message}'")
    print(f"   Beklenen: {expected} | Tahmin: {prediction} (Güven: {confidence:.1%})")
    print("-" * 70)

print(f"\nPerformans Özeti:")
print(f"Zararsız mesajlar: {zararsiz_dogru}/{zararsiz_toplam} doğru ({zararsiz_dogru/zararsiz_toplam:.1%})")
print(f"Zararlı mesajlar: {zararli_dogru}/{zararli_toplam} doğru ({zararli_dogru/zararli_toplam:.1%})")
print(f"Genel doğruluk: {(zararsiz_dogru + zararli_dogru)}/{len(test_messages)} ({(zararsiz_dogru + zararli_dogru)/len(test_messages):.1%})")