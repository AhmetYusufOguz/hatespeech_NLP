# Turkish Hate Speech Detection with BERT

Bu proje, Türkçe metinlerde nefret söylemini tespit etmek için BERT tabanlı derin öğrenme modeli kullanır. Model, metinleri "Zararsız" ve "Zararlı" olarak ikili sınıflandırma yapar.

## 📁 Proje Yapısı

```
Project/
├── hatespeech_dataset.xlsx           # Eğitim veri seti (1000 tweet)
├── Code/
│   ├── train_binary_model.py         # Model eğitim scripti
│   ├── test_binary_model.py          # Örnek mesajlarla test scripti  
│   ├── test_real_data.py             # Gerçek veri seti ile test scripti
│   ├── binary_hatespeech_model/      # Eğitilmiş BERT modeli
│   └── binary_hatespeech_tokenizer/  # BERT tokenizer
└── README.md                         # Bu dosya
```

## 🔧 Gereksinimler

### Kütüphaneler
```bash
pip install transformers[torch]
pip install accelerate>=0.26.0
pip install torch
pip install pandas
pip install scikit-learn
pip install openpyxl
```

### Sistem Gereksinimleri
- Python 3.8+
- En az 8GB RAM
- CUDA destekli GPU (önerilen, CPU ile de çalışır)

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/AhmetYusufOguz/hatespeech_NLP.git
cd hatespeech_NLP
```

### 2. Kütüphaneleri Yükleyin
```bash
pip install transformers[torch] accelerate>=0.26.0 pandas scikit-learn openpyxl
```

### 3. Model Eğitimi (İsteğe Bağlı)
Eğer modeli sıfırdan eğitmek istiyorsanız: 
```bash
python Code/train_binary_model.py
```

### 4. Model Testi
```bash
# Örnek mesajlarla test
python Code/test_binary_model.py

# Gerçek veri seti ile test  
python Code/test_real_data.py
```

## 📋 Dosya Açıklamaları

### `train_binary_model.py`
**Amaç:** BERT modelini Türkçe nefret söylemi tespiti için eğitir.

**İşlevler:**
- Dataset'i yükler (`hatespeech_dataset.xlsx`)
- Etiketleri ikili sınıflandırma için dönüştürür (hiçbiri→0, nefret→1, saldırgan→1)
- BERTurk (`dbmdz/bert-base-turkish-cased`) modelini yükler
- Metinleri tokenize eder
- 3 epoch boyunca modeli eğitir
- Eğitilen modeli `binary_hatespeech_model/` klasörüne kaydeder

**Çalıştırma süresi:** ~10-15 dakika

### `test_binary_model.py`
**Amaç:** Eğitilen modeli örnek mesajlarla test eder.

**İşlevler:**
- Önceden eğitilmiş modeli yükler
- 12 farklı örnek mesaj test eder (zararsız ve zararlı içerik)
- Her mesaj için tahmin ve güven skorunu gösterir
- Genel performans özetini çıkarır

**Örnek çıktı:**
```
✓ Mesaj: 'Merhaba, nasılsın?'
   Beklenen: Zararsız | Tahmin: Zararsız (Güven: 99.7%)
```

### `test_real_data.py`
**Amaç:** Modeli gerçek veri seti örnekleriyle test eder.

**İşlevler:**
- Orijinal dataset'ten her kategoriden 5'er örnek seçer
- Model performansını gerçek verilerle değerlendirir
- Kategori bazında doğruluk oranlarını hesaplar
- Detaylı performans raporu sunar

## 📊 Model Performansı

### Eğitim Sonuçları
- **Model:** dbmdz/bert-base-turkish-cased (Türkçe BERT)
- **Veri seti:** 1000 tweet (664 zararsız, 336 zararlı)
- **Eğitim süresi:** 3 epoch
- **Final loss:** 0.20 (başlangıç: 0.66)

### Test Sonuçları
- **Zararsız içerik tespiti:** %95+ doğruluk
- **Zararlı içerik tespiti:** %90+ doğruluk  
- **Genel performans:** %90+ doğruluk

## 🔍 Model Kullanımı

### Python Kodu ile Kullanım
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Model yükle
tokenizer = AutoTokenizer.from_pretrained("./Code/binary_hatespeech_tokenizer")
model = AutoModelForSequenceClassification.from_pretrained("./Code/binary_hatespeech_model")

def predict_message(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch. argmax(predictions, dim=-1).item()
    
    class_names = {0: "Zararsız", 1: "Zararlı"}
    confidence = predictions[0][predicted_class].item()
    
    return class_names[predicted_class], confidence

# Kullanım
result, confidence = predict_message("Test mesajınız")
print(f"Sonuç: {result} (Güven: {confidence:.1%})")
```

## 📚 Veri Seti

### `hatespeech_dataset.xlsx`
- **Kaynak:** Türkçe sosyal medya metinleri
- **Boyut:** 1000 örnek
- **Sütunlar:**
  - `row ID`: Benzersiz kimlik
  - `Tweet`: Metin içeriği
  - `Etiket`: Ana kategori (hiçbiri/nefret/saldırgan)
  - `Alt Etiket`: Alt kategoriler (etnik, dini, ideolojik vb.)

### Etiket Dağılımı
- **hiçbiri:** 664 örnek (%66.4)
- **nefret:** 276 örnek (%27.6) 
- **saldırgan:** 60 örnek (%6.0)

## 🎯 Uygulama Alanları

- **Sosyal medya moderasyonu**
- **Yorum filtreleme sistemleri**
- **Mobil uygulama içerik kontrolü**
- **Online topluluk yönetimi**

## 🔄 Gelecek Geliştirmeler

- [ ] REST API entegrasyonu
- [ ] Gerçek zamanlı metin analizi
- [ ] Mobil uygulama entegrasyonu
- [ ] Çoklu dil desteği
- [ ] Model performans iyileştirmeleri

## 👥 Katkıda Bulunanlar

- **Ahmet Yusuf Oğuz** - Proje geliştiricisi

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🔗 İlgili Kaynaklar

- [Transformers Kütüphanesi](https://huggingface.co/transformers/)
- [BERTurk Modeli](https://huggingface.co/dbmdz/bert-base-turkish-cased)
- [PyTorch](https://pytorch.org/)