import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from . config import Config

# Logging ayarla
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

class ModelManager:
    """5 sınıflı BERT modelini yükler ve yönetir"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_version = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 5 sınıflı model için class names
        self. class_names = Config.CLASS_NAMES["multiclass"]
        self.class_descriptions = Config. CLASS_DESCRIPTIONS
        
        # Model yüklemeyi dene
        try:  
            self.load_model()
        except FileNotFoundError as e:
            logger.warning(f"Model dosyaları bulunamadı: {e}")
            logger.info("Model bulunamadı, lütfen modeli eğitip Code/ klasörüne yerleştirin.")
    
    def load_model(self):
        """Code/ klasöründeki modeli yükler"""
        model_paths = Config.get_model_paths()
        model_dir = model_paths["model"]
        tokenizer_dir = model_paths["tokenizer"]
        
        logger.info(f"Model aranıyor: {model_dir}")
        logger.info(f"Tokenizer aranıyor: {tokenizer_dir}")
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model klasörü bulunamadı:  {model_dir}")
        
        if not tokenizer_dir.exists():
            raise FileNotFoundError(f"Tokenizer klasörü bulunamadı: {tokenizer_dir}")
        
        logger.info("Model yükleniyor...")
        
        try:
            # Tokenizer yükle - local_files_only Windows sorununu önler
            self.tokenizer = AutoTokenizer. from_pretrained(
                str(tokenizer_dir), 
                local_files_only=True
            )
            
            # Model yükle
            self. model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), 
                local_files_only=True
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.current_version = Config.DEFAULT_MODEL_VERSION
            logger.info(f"Model başarıyla yüklendi:  {self.current_version} ({self.device})")
            
        except Exception as e: 
            logger.error(f"Model yükleme hatası: {str(e)}")
            raise
    
    def is_model_loaded(self):
        """Model yüklenmiş mi kontrol eder"""
        return self.model is not None and self.tokenizer is not None
    
    def predict(self, text: str):
        """Metin için 5 sınıflı nefret söylemi tahmini yapar"""
        if not self. is_model_loaded():
            raise RuntimeError("Model henüz yüklenmemiş.")
        
        try:
            # Metni tokenize et
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_LENGTH
            )
            
            # GPU'ya taşı
            inputs = {k: v.to(self.device) for k, v in inputs. items()}
            
            # Tahmin yap
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                confidence, predicted_class_tensor = torch.max(predictions, dim=-1)
                predicted_class = predicted_class_tensor.item()
                confidence = confidence.item()
            
            # Düşük güvenli tahminleri "Hiçbiri" olarak ayarla
            original_prediction = predicted_class
            original_confidence = confidence
            
            if confidence < Config.CONFIDENCE_THRESHOLD:
                predicted_class = 0  # Hiçbiri
            
            result = {
                "text": text,
                "prediction": self.class_names[predicted_class],
                "prediction_id": predicted_class,
                "confidence": round(confidence, 4),
                "confidence_percentage": round(confidence * 100, 2),
                "description": self.class_descriptions[predicted_class],
                "is_harmful": predicted_class > 0,  # 0=Hiçbiri, diğerleri zararlı
                "original_prediction":  self.class_names[original_prediction],
                "original_confidence": round(original_confidence, 4),
                "threshold_applied": confidence < Config.CONFIDENCE_THRESHOLD,
                "model_version": self.current_version,
                "device": str(self.device)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            raise
    
    def batch_predict(self, texts: list):
        """Birden fazla metin için toplu tahmin yapar"""
        if not self.is_model_loaded():
            raise RuntimeError("Model henüz yüklenmemiş.")
            
        results = []
        for text in texts: 
            try:
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                results.append({
                    "text": text,
                    "error": str(e),
                    "model_version": self.current_version
                })
        return results
    
    def get_model_info(self):
        """Mevcut model hakkında bilgi döndürür"""
        model_paths = Config.get_model_paths()
        return {
            "current_version":  self.current_version,
            "supported_versions": Config. SUPPORTED_VERSIONS,
            "device": str(self.device),
            "model_loaded": self.is_model_loaded(),
            "class_names": self. class_names,
            "class_descriptions": self.class_descriptions,
            "confidence_threshold":  Config.CONFIDENCE_THRESHOLD,
            "model_type": "multiclass (5 classes)",
            "model_path": str(model_paths["model"]),
            "tokenizer_path": str(model_paths["tokenizer"])
        }

# Global model manager instance
model_manager = ModelManager()