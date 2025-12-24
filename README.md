# Turkish Hate Speech Detection with BERT

Bu proje, Türkçe metinlerde nefret söylemini tespit etmek için BERT tabanlı derin öğrenme modeli kullanır.  Proje hem ikili (Zararsız/Zararlı) hem de 5 sınıflı detaylı sınıflandırma sunar ve FastAPI ile REST API hizmeti sağlar. 

## 📁 Proje Yapısı

```
Project/
├── hatespeech_dataset.xlsx                    # Eğitim veri seti (441 örnek)
├── Code/
│   ├── train_binary_model.py                  # İkili model eğitim scripti
│   ├── train_multiclass_model.py              # 5 sınıflı model eğitim scripti
│   ├── test_binary_model.py                   # İkili model test scripti  
│   ├── test_multiclass_model.py               # 5 sınıflı model test scripti
│   ├── test_real_data.py                      # Gerçek veri seti ile test
│   ├── multiclass_hatespeech_model/           # 5 sınıflı BERT modeli
│   ├── multiclass_hatespeech_tokenizer/       # 5 sınıflı BERT tokenizer
│   ├── api/                                   # FastAPI REST API
│   │   ├── __init__.py
│   │   ├── config.py                          # API konfigürasyonu
│   │   ├── model_manager.py                   # Model yönetimi
│   │   └── app.py                             # FastAPI ana uygulama
│   ├── auto_label.py                          # Otomatik etiket kontrolü
│   ├── add_new_data.py                        # Yeni veri ekleme
│   ├── check_duplicates.py                    # Duplikat kontrol
│   └── (diğer yardımcı scriptler)
├── model.py                                   # Model boyutu hesaplayıcı
└── README.md                                  # Bu dosya
```

## 🎯 Model Sınıfları

### 5 Sınıflı Detaylı Model (Önerilen)
- **0: Hiçbiri** - Zararsız içerik
- **1: Nefret** - Grup bazlı nefret söylemi  
- **2: Saldırgan** - Hakaret/saldırgan dil
- **3: Tehdit** - Başkalarına yönelik şiddet tehdidi
- **4: Niyet** - Kendine zarar verme niyeti

### İkili Model (Basit)
- **0: Zararsız** - Normal içerik
- **1: Zararlı** - Nefret/hakaret/tehdit içeren

## 🔧 Gereksinimler

### Kütüphaneler
```bash
# ML/AI kütüphaneleri
pip install transformers[torch]
pip install accelerate>=0.26.0
pip install torch
pip install pandas
pip install scikit-learn
pip install openpyxl

# API kütüphaneleri
pip install fastapi
pip install uvicorn
pip install python-multipart
```

### Sistem Gereksinimleri
- Python 3.8+
- En az 8GB RAM
- CUDA destekli GPU (önerilen, CPU ile de çalışır)

## 🚀 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/AhmetYusufOguz/hatespeech_NLP. git
cd hatespeech_NLP
```

### 2. Kütüphaneleri Yükleyin
```bash
pip install transformers[torch] accelerate>=0.26.0 pandas scikit-learn openpyxl fastapi uvicorn python-multipart
```

### 3. Model Eğitimi
```bash
# 5 sınıflı model eğitimi (önerilen)
python Code/train_multiclass_model.py

# Veya ikili model eğitimi
python Code/train_binary_model.py
```

### 4. Model Testi
```bash
# 5 sınıflı model testi
python Code/test_multiclass_model.py

# İkili model testi
python Code/test_binary_model. py
```

### 5. REST API Başlatma
```bash
# API sunucusunu başlat
cd Code
python -m uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

# Tarayıcıda test et
# http://localhost:8000/docs - Swagger UI
# http://localhost:8000/ - Ana sayfa
```

## 🌐 REST API Kullanımı

### API Endpoint'leri
- **GET /** - Ana sayfa ve API bilgisi
- **POST /predict** - Tek metin analizi
- **POST /predict/batch** - Toplu metin analizi (max 10)
- **GET /health** - API sağlık kontrolü
- **GET /model/info** - Model detayları
- **GET /classes** - Desteklenen sınıflar
- **POST /test** - Hızlı API testi

### Örnek Kullanım

#### Tek Metin Analizi
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text":  "Bu aptal bir fikir!"}'
```

#### Python ile Kullanım
```python
import requests

# Tek tahmin
response = requests. post('http://localhost:8000/predict', 
                        json={'text': 'Analiz edilecek metin'})
result = response.json()

print(f"Tahmin:  {result['prediction']}")
print(f"Güven: {result['confidence_percentage']}%")
print(f"Açıklama: {result['description']}")
```

#### JavaScript ile Kullanım
```javascript
fetch('http://localhost:8000/predict', {
  method:  'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'Kontrol edilecek yorum'})
})
.then(response => response.json())
.then(data => {
  console.log('Sonuç:', data. prediction);
  console.log('Zararlı mı:', data.is_harmful);
});
```

### API Yanıt Örneği
```json
{
  "text": "Bu aptal bir fikir! ",
  "prediction": "Saldırgan",
  "prediction_id": 2,
  "confidence": 0.8234,
  "confidence_percentage": 82.34,
  "description": "Hakaret/saldırgan dil",
  "is_harmful": true,
  "original_prediction": "Saldırgan",
  "original_confidence": 0.8234,
  "threshold_applied": false,
  "model_version": "v1.1",
  "device": "cpu",
  "timestamp":  "2025-01-24T15:30:45"
}
```

## 📊 Model Performansı

### 5 Sınıflı Model (v1.1)
- **Model:** loodos/bert-base-turkish-uncased
- **Veri seti:** 441 örnek (dengeli dağıtım)
- **Eğitim süresi:** 3 epoch (~5 dakika)
- **Final F1 Score:** 0.73 (macro average)
- **Test Doğruluğu:** %65+ (5 sınıf için çok iyi)
- **Güven Eşiği:** 0.80 (düşük güvenli tahminler "Hiçbiri" olur)

### Sınıf Dağılımı
- **Hiçbiri:** 104 örnek (%23.6)
- **Nefret:** 100 örnek (%22.7)
- **Saldırgan:** 102 örnek (%23.1)
- **Tehdit:** 42 örnek (%9.5)
- **Niyet:** 93 örnek (%21.1)

## 🔍 Kod ile Model Kullanımı

### Direkt Model Kullanımı
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Model yükle
tokenizer = AutoTokenizer.from_pretrained("./Code/multiclass_hatespeech_tokenizer", local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained("./Code/multiclass_hatespeech_model", local_files_only=True)

def predict_multiclass(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence, predicted_class = torch. max(predictions, dim=-1)
        predicted_class = predicted_class.item()
        confidence = confidence.item()
    
    # Düşük güvenli tahminleri "Hiçbiri" yap
    if confidence < 0.80:
        predicted_class = 0
    
    class_names = {0: "Hiçbiri", 1: "Nefret", 2: "Saldırgan", 3: "Tehdit", 4: "Niyet"}
    return class_names[predicted_class], confidence

# Kullanım
result, confidence = predict_multiclass("Test mesajınız")
print(f"Sonuç: {result} (Güven: {confidence:.1%})")
```

## 📚 Veri Seti Detayları

### `hatespeech_dataset.xlsx`
- **Kaynak:** Türkçe sosyal medya metinleri
- **Boyut:** 441 örnek (dengeli)
- **Sütunlar:**
  - `row ID`: Benzersiz kimlik
  - `Tweet`: Metin içeriği  
  - `Etiket`: Ana kategori (hiçbiri/nefret/saldırgan/tehdit/niyet)

### Veri İşleme Araçları
- `auto_label.py` - Otomatik etiket kontrolü ve bayraklama
- `add_new_data. py` - Yeni veri ekleme
- `check_duplicates.py` - Duplikat tespit ve temizleme
- `update_labels.py` - Etiket güncelleme

## 🎯 Uygulama Alanları

- **Sosyal medya moderasyonu** - Otomatik içerik filtreleme
- **E-ticaret yorum sistemleri** - Müşteri yorumu kontrolü
- **Online topluluk yönetimi** - Forum/chat moderasyonu
- **Mobil uygulama güvenliği** - Kullanıcı içeriği kontrolü
- **Haber sitesi yorum filtreleme** - Zararlı yorum engelleme
- **Oyun içi chat moderasyonu** - Gerçek zamanlı metin kontrolü

## 🛠️ Katkıda Bulunma

### Geliştirme Ortamı Kurulumu
```bash
# Depoyu fork edin ve klonlayın
git clone https://github.com/YOURUSERNAME/hatespeech_NLP.git
cd hatespeech_NLP

# Development branch oluşturun
git checkout -b feature/your-feature-name

# Değişiklikleri yapın ve test edin
python Code/test_multiclass_model.py
python -m uvicorn api. app:app --reload

# Pull request gönderin
```

### Yeni Veri Ekleme
1. `hatespeech_dataset.xlsx` dosyasına yeni örnekler ekleyin
2. `python Code/check_duplicates.py` ile duplikatları kontrol edin
3. `python Code/train_multiclass_model.py` ile modeli yeniden eğitin
4. `python Code/test_multiclass_model.py` ile performansı test edin

## 👥 Katkıda Bulunanlar

- **Ahmet Yusuf Oğuz**
- **Sefa Akgün**
- **Yusuf Alperen Dönmez**

## 📈 Performans Karşılaştırması

| Model | Sınıf Sayısı | F1 Score | Doğruluk | Eğitim Süresi |
|-------|-------------|----------|----------|---------------|
| İkili Model | 2 | 0.85 | %90+ | ~10 dk |
| Çok Sınıflı | 5 | 0.73 | %65+ | ~5 dk |

## 🔐 Güvenlik ve Gizlilik

- API anahtarı gerektirmez (geliştirme ortamında)
- Gelen metinler loglanmaz
- Kişisel veri işlenmez
- CORS desteği ile güvenli web entegrasyonu

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🔗 İlgili Kaynaklar

- [Transformers Kütüphanesi](https://huggingface.co/transformers/)
- [Loodos BERT Türkçe](https://huggingface.co/loodos/bert-base-turkish-uncased)
- [FastAPI Dokümantasyonu](https://fastapi.tiangolo.com/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)