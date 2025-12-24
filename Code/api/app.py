from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
from datetime import datetime
import uvicorn

from .model_manager import model_manager
from .config import Config

# Logging ayarla
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))
logger = logging.getLogger(__name__)

# FastAPI uygulaması
app = FastAPI(
    title=Config.API_TITLE,
    description=Config.API_DESCRIPTION,
    version=Config.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS ayarları - Web tarayıcılarından erişim için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Geliştirme için, production'da kısıtla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic modelleri - API input/output şemaları
class TextInput(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="Analiz edilecek Türkçe metin",
        example="Bu bir test mesajıdır."
    )
    
class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ..., 
        min_items=1, 
        max_items=10, 
        description="Analiz edilecek metinler listesi"
    )

class PredictionResponse(BaseModel):
    text: str
    prediction:  str
    prediction_id: int 
    confidence: float
    confidence_percentage: float
    description:  str
    is_harmful: bool
    original_prediction: str
    original_confidence: float
    threshold_applied: bool
    model_version:  str
    device: str
    timestamp: datetime = Field(default_factory=datetime. now)

class BatchPredictionResponse(BaseModel):
    results: List[dict]
    total_count: int
    successful_count: int
    error_count: int
    harmful_count: int
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelInfo(BaseModel):
    current_version: str
    supported_versions: List[str]
    device: str
    model_loaded: bool
    class_names:  dict
    class_descriptions: dict
    confidence_threshold: float
    model_type: str

# API Endpoint'leri

@app.get("/")
async def root():
    """API ana sayfası - temel bilgiler"""
    return {
        "message": "🤖 Turkish Hate Speech Detection API (5-Class)",
        "version": Config. API_VERSION,
        "description": "5 sınıflı Türkçe nefret söylemi tespit API'si",
        "classes": Config.CLASS_NAMES["multiclass"],
        "docs":  "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "model_version": model_manager.current_version,
        "model_loaded": model_manager.is_model_loaded()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_text(input_data: TextInput):
    """
    Tek metin için 5 sınıflı nefret söylemi analizi yapar
    
    **Sınıflar:**
    - 0: Hiçbiri (Zararsız)
    - 1: Nefret (Grup bazlı nefret söylemi)  
    - 2: Saldırgan (Hakaret/saldırgan dil)
    - 3: Tehdit (Başkalarına yönelik şiddet)
    - 4: Niyet (Kendine zarar verme)
    """
    try:
        if not model_manager.is_model_loaded():
            raise HTTPException(
                status_code=503, 
                detail="Model henüz yüklenmemiş.  Lütfen daha sonra tekrar deneyin."
            )
        
        result = model_manager.predict(input_data.text)
        result["timestamp"] = datetime.now()
        logger.info(f"Tahmin yapıldı: '{input_data.text}' -> {result['prediction']}")
        return result
        
    except Exception as e:
        logger.error(f"Tahmin hatası:  {str(e)}")
        raise HTTPException(status_code=500, detail=f"Tahmin hatası: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(input_data: BatchTextInput):
    """
    Birden fazla metin için toplu analiz yapar
    
    **Maksimum 10 metin** aynı anda işlenebilir. 
    """
    try: 
        if not model_manager.is_model_loaded():
            raise HTTPException(
                status_code=503, 
                detail="Model henüz yüklenmemiş. Lütfen daha sonra tekrar deneyin."
            )
            
        results = model_manager. batch_predict(input_data. texts)
        
        # İstatistikleri hesapla
        successful_count = len([r for r in results if "error" not in r])
        error_count = len([r for r in results if "error" in r])
        harmful_count = len([r for r in results if "error" not in r and r. get("is_harmful", False)])
        
        response = {
            "results": results,
            "total_count": len(results),
            "successful_count": successful_count,
            "error_count":  error_count,
            "harmful_count": harmful_count,
            "model_version": model_manager.current_version,
            "timestamp": datetime.now()
        }
        
        logger. info(f"Toplu tahmin:  {len(input_data.texts)} metin, {successful_count} başarılı")
        return response
        
    except Exception as e:
        logger.error(f"Toplu tahmin hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Toplu tahmin hatası: {str(e)}")

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Mevcut model hakkında detaylı bilgi döndürür"""
    try: 
        return model_manager.get_model_info()
    except Exception as e:
        logger.error(f"Model bilgisi alınamadı: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model bilgisi alınamadı: {str(e)}")

@app.get("/health")
async def health_check():
    """API sağlık kontrolü - sistem durumu"""
    model_loaded = model_manager.is_model_loaded()
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "timestamp": datetime.now(),
        "model_version": model_manager.current_version,
        "model_loaded":  model_loaded,
        "device": str(model_manager.device) if model_loaded else "unknown",
        "api_version": Config.API_VERSION,
        "uptime": "running"
    }

@app.get("/classes")
async def get_classes():
    """Desteklenen sınıflar ve açıklamaları"""
    return {
        "classes": Config.CLASS_NAMES["multiclass"],
        "descriptions": Config.CLASS_DESCRIPTIONS,
        "total_classes": len(Config.CLASS_NAMES["multiclass"]),
        "model_type": "multiclass"
    }

# Test endpoint'i - geliştirme için
@app.post("/test")
async def quick_test():
    """Hızlı test - örnek metinlerle API'yi test et"""
    test_messages = [
        "Merhaba, nasılsın?",
        "Bu aptal bir fikir!",
        "Seni öldüreceğim! ",
        "Kendimi öldürmek istiyorum",
        "Bu gruptan nefret ediyorum"
    ]
    
    try:
        if not model_manager.is_model_loaded():
            return {"error": "Model yüklenmemiş"}
            
        results = []
        for text in test_messages:
            result = model_manager.predict(text)
            results.append({
                "text": text,
                "prediction": result["prediction"],
                "confidence": result["confidence_percentage"]
            })
            
        return {
            "test_results": results,
            "status": "success",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        return {"error": str(e), "status": "failed"}

# Uygulama başlatma fonksiyonu
if __name__ == "__main__":
    print(f"🚀 {Config.API_TITLE} başlatılıyor...")
    print(f"📊 Model: {model_manager.current_version if model_manager.is_model_loaded() else 'YÜKLENMEDİ'}")
    print(f"🌐 Swagger Docs: http://{Config.HOST}:{Config. PORT}/docs")
    
    uvicorn.run(
        "app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=Config.DEBUG,
        log_level=Config.LOG_LEVEL. lower()
    )