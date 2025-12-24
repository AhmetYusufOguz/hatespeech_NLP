import os
from pathlib import Path

class Config: 
    # Proje ana dizini
    BASE_DIR = Path(__file__).parent.parent. parent
    
    # Model dizinleri - CODE klasöründe
    CODE_DIR = BASE_DIR / "Code"
    
    # Varsayılan model versiyonu
    DEFAULT_MODEL_VERSION = "v1.1"
    
    # Desteklenen model versiyonları
    SUPPORTED_VERSIONS = ["v1.1"]
    
    # Model tipleri
    MODEL_TYPES = {
        "v1.1": "multiclass"
    }
    
    # 5 sınıflı model için sınıf isimleri
    CLASS_NAMES = {
        "multiclass": {
            0: "Hiçbiri", 
            1: "Nefret", 
            2: "Saldırgan", 
            3: "Tehdit", 
            4: "Niyet"
        }
    }
    
    # Sınıf açıklamaları
    CLASS_DESCRIPTIONS = {
        0: "Zararsız içerik",
        1: "Grup bazlı nefret söylemi", 
        2: "Hakaret/saldırgan dil",
        3: "Başkalarına yönelik tehdit/şiddet",
        4: "Kendine zarar verme niyeti"
    }
    
    # API ayarları
    API_TITLE = "Turkish Hate Speech Detection API (5-Class)"
    API_DESCRIPTION = "BERT tabanlı 5 sınıflı Türkçe nefret söylemi tespit API'si"
    API_VERSION = "2.0.0"
    
    # Server ayarları
    HOST = "127.0.0.1"
    PORT = 8000
    DEBUG = True
    
    # Model ayarları
    MAX_LENGTH = 128
    CONFIDENCE_THRESHOLD = 0.8  # Test scriptindeki threshold
    
    # Loglar
    LOG_LEVEL = "INFO"
    
    @classmethod
    def get_model_paths(cls):
        """Model ve tokenizer yollarını döndürür - Code/ klasöründe"""
        return {
            "model": cls.CODE_DIR / "multiclass_hatespeech_model",
            "tokenizer":  cls.CODE_DIR / "multiclass_hatespeech_tokenizer"
        }