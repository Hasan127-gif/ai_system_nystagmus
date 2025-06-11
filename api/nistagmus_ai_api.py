"""
Nistagmus AI API
---------------
Bu modül, nistagmus tespiti için FastAPI tabanlı bir REST API sağlar.
Görüntü ve video analizi için endpoint'ler içerir.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import traceback

# Logging yapılandırması
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nistagmus_ai_api')

# FastAPI
from fastapi import FastAPI, HTTPException, Depends, File, UploadFile, Form, Body, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Modeli import et
try:
    # Basitleştirilmiş detector'ı kullan
    from simple_detector import NistagmusDetector
    logger.info("Basitleştirilmiş NistagmusDetector modülü içe aktarıldı.")
except ImportError:
    try:
        # Projenin kök dizinini import path'ine ekle
        root_dir = Path(__file__).parent.parent.parent
        sys.path.append(str(root_dir))
        logger.info(f"Import path'e eklendi: {root_dir}")
        
        # Tekrar import etmeyi dene
        try:
            from ai_system.simple_detector import NistagmusDetector
            logger.info("Basitleştirilmiş NistagmusDetector modülü içe aktarıldı.")
        except Exception as e:
            logger.error(f"Detector modülü içe aktarılamadı: {str(e)}")
            NistagmusDetector = None
            traceback.print_exc()
    except Exception as e:
        logger.error(f"Import path ayarlanırken hata: {str(e)}")
        NistagmusDetector = None
        traceback.print_exc()

# API metadata
API_VERSION = "1.0.0"
API_TITLE = "Nistagmus AI API"
API_DESCRIPTION = "Göz hareketlerinin analizi ve nistagmus tespiti için AI API"

# Numpy bool_ tiplerini Python bool tipine dönüştüren yardımcı fonksiyon
def convert_numpy_types(obj):
    """
    Numpy tiplerini (özellikle bool_) Python tiplerine dönüştürür.
    """
    import numpy as np
    
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    else:
        return obj

# FastAPI uygulaması
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver (production için sınırlandırın)
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*", "Content-Type", "Authorization", "X-API-Key"],
    expose_headers=["*"],
    max_age=600,  # 10 dakika önbellekleme
)

# API Key güvenliği (opsiyonel)
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("NISTAGMUS_API_KEY", "test-api-key")  # Varsayılan key (gerçek projede değiştirin)

api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Nistagmus detector
nistagmus_detector = None

# Veri modelleri
class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    timestamp: str
    model_status: bool

class ImageAnalysisRequest(BaseModel):
    image_data: str = Field(..., description="Base64 kodlu görüntü veya görüntü URL'si")

class VideoUploadRequest(BaseModel):
    file_key: str = Field(..., description="Video dosyası form anahtarı")

class AnalysisResponse(BaseModel):
    status: str
    results: Dict[str, Any]
    timestamp: str

class FeedbackRequest(BaseModel):
    analysis_id: str
    correct: bool
    comments: Optional[str] = None

# API key doğrulama
async def get_api_key(api_key: str = Depends(api_key_header)):
    # Debug log
    logger.info(f"Gelen API anahtarı: '{api_key}'")
    logger.info(f"Beklenen API anahtarı: '{API_KEY}'")
    
    # API doğrulamasını devre dışı bırak (geçici)
    if True:  # Tüm istekleri kabul et (geçici çözüm)
        return "test-api-key"
    
    if API_KEY == "":  # API key doğrulaması devre dışı bırakıldı
        return ""
        
    if api_key == API_KEY:
        return api_key
    
    # API anahtarı eşleşmedi hatası
    raise HTTPException(
        status_code=401,
        detail="Geçersiz API anahtarı",
        headers={"WWW-Authenticate": "ApiKey"},
    )

# API başlangıç zamanı
start_time = time.time()

# API olayları
@app.on_event("startup")
async def startup_event():
    """API başlatıldığında çalışır"""
    global nistagmus_detector
    
    try:
        # Modeli oluştur
        custom_config = {
            "nistagmus_threshold": 0.25,  # Düşük eşik değeri (orijinal: 0.65)
            "feature_importance_boost": {
                "left_fast_phases": 1.5,  # Fast fazlara daha fazla ağırlık ver
                "right_fast_phases": 1.5,
                "left_max_velocity": 1.3,  # Hıza daha fazla ağırlık ver
                "right_max_velocity": 1.3
            },
            "sensitivity_boost": True,
            "detection_mode": "sensitive"  # Daha hassas tespit için
        }
        
        # Modeli yükle
        nistagmus_detector = NistagmusDetector(config=custom_config)
        logger.info("Nistagmus detector model başarıyla yüklendi")
        model_status = True
    except Exception as e:
        nistagmus_detector = None
        model_status = False
        logger.error(f"Nistagmus detector model yüklenemedi: {str(e)}")
        traceback.print_exc()

@app.on_event("shutdown")
async def shutdown_event():
    """API kapandığında çalışır"""
    logger.info("API kapatılıyor")

# Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Tüm istekleri loglar"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    return response

# API rotaları
@app.get("/")
async def root():
    """API kök endpoint'i"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """API sağlık kontrolü"""
    global nistagmus_detector
    
    return {
        "status": "ok",
        "version": API_VERSION,
        "uptime": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
        "model_status": nistagmus_detector is not None and nistagmus_detector.is_initialized
    }

@app.post("/analyze/image", response_model=AnalysisResponse)
async def analyze_image(request: ImageAnalysisRequest, api_key: str = Depends(get_api_key)):
    """
    Görüntü analizi yaparak nistagmus tespiti yapar.
    """
    logger.info(f"POST /analyze/image")
    
    start_time = time.time()
    
    # Tensor model devre dışıysa hata döndür
    if nistagmus_detector is None:
        error_msg = "Nistagmus dedektör modeli başlatılamadı. Servis hazır değil."
        logger.error(error_msg)
        return JSONResponse(
            status_code=503,
            content={"detail": error_msg, "status": "error"}
        )
    
    try:
        # Görüntüyü analiz et
        result = nistagmus_detector.analyze_image(request.image_data)
        
        # Süreyi ölç ve logla
        elapsed_time = time.time() - start_time
        logger.info(f"Görüntü analizi tamamlandı, süre: {elapsed_time:.2f}s")
        
        # Durumu kontrol et ve döndür
        if "error" in result:
            logger.error(f"Analiz hatası: {result['error']}")
            return JSONResponse(
                status_code=400,
                content={"detail": f"Analiz başarısız: {result['error']}", "status": "error"}
            )
        
        # Başarılı bir şekilde tamamlandı
        return JSONResponse(
            status_code=200,
            content={"status": "success", "results": result}
        )
    except Exception as e:
        error_msg = f"Görüntü analizi hatası: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": error_msg, "status": "error"}
        )

@app.post("/analyze/video", response_model=AnalysisResponse)
async def analyze_video(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    """
    Video analizi yaparak nistagmus tespiti yapar.
    İyileştirilmiş: Ortak analysis_utils fonksiyonlarını kullanır.
    """
    logger.info(f"POST /analyze/video")
    
    start_time = time.time()
    
    # Tensor model devre dışıysa hata döndür
    if nistagmus_detector is None:
        error_msg = "Nistagmus dedektör modeli başlatılamadı. Servis hazır değil."
        logger.error(error_msg)
        return JSONResponse(
            status_code=503,
            content={"detail": error_msg, "status": "error"}
        )
    
    try:
        # Analysis utils'i import et
        from analysis_utils import analyze_video_file, format_results_for_api, validate_video_file
        
        # Video dosyasını geçici olarak kaydet
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        supported_extensions = ['.mp4', '.avi', '.mov', '.webm']
        
        # Desteklenmeyen video formatı kontrolü
        if file_extension not in supported_extensions:
            # Desteklenmeyen formatsa, convertible mi kontrol et
            if file_extension in ['.hevc', '.mkv', '.flv', '.wmv', '.3gp', '.m4v']:
                # Dönüştürülebilir, ama şu an desteklenmiyor
                logger.warning(f"Desteklenmeyen video formatı: {file_extension}. Dönüştürme gerekiyor.")
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error", 
                        "detail": f"Desteklenmeyen video formatı: {file_extension}. Lütfen MP4, AVI, MOV veya WEBM formatı kullanın."
                    }
                )
            else:
                # Bu format hiçbir şekilde desteklenmiyor
                logger.error(f"Desteklenmeyen ve dönüştürülemeyen video formatı: {file_extension}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error", 
                        "detail": f"Bu video formatı desteklenmiyor: {file_extension}. Lütfen MP4, AVI, MOV veya WEBM formatı kullanın."
                    }
                )
        
        # Dosyayı temp klasöre kaydet
        file_path = temp_dir / f"temp_video_{int(time.time())}{file_extension}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        
        logger.info(f"Video geçici olarak kaydedildi: {file_path}")
        
        # Video dosyasını doğrula
        validation_result = validate_video_file(str(file_path))
        if not validation_result["valid"]:
            # Geçici dosyayı sil
            try:
                os.remove(file_path)
            except:
                pass
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error", 
                    "detail": f"Video doğrulama hatası: {validation_result['error']}"
                }
            )
        
        # Maksimum frame sayısı
        max_frames = 300  # Varsayılan değer
        
        # REFAKTÖR EDİLMİŞ: Ortak video analizi fonksiyonunu kullan
        raw_result = analyze_video_file(str(file_path), nistagmus_detector, max_frames=max_frames)
        
        # Geçici dosyayı sil
        try:
            os.remove(file_path)
            logger.info(f"Geçici video dosyası silindi: {file_path}")
        except Exception as e:
            logger.warning(f"Geçici dosya silinirken hata: {str(e)}")
        
        # Süreyi ölç ve logla
        elapsed_time = time.time() - start_time
        logger.info(f"Video analizi tamamlandı, süre: {elapsed_time:.2f}s")
        
        # Sonuçları API formatına çevir
        formatted_result = format_results_for_api(raw_result)
        
        # Durumu kontrol et ve döndür
        if formatted_result.get("status") == "error":
            logger.error(f"Analiz hatası: {formatted_result.get('message')}")
            return JSONResponse(
                status_code=400,
                content={"detail": f"Analiz başarısız: {formatted_result.get('message')}", "status": "error"}
            )
        
        # Başarılı bir şekilde tamamlandı
        return JSONResponse(
            status_code=200,
            content={"status": "success", "results": formatted_result}
        )
        
    except Exception as e:
        # Geçici dosyayı temizle (hata durumunda)
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
            
        error_msg = f"Video analizi hatası: {str(e)}"
        logger.error(error_msg)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"detail": error_msg, "status": "error"}
        )

@app.post("/reload_model")
async def reload_model(api_key: str = Depends(get_api_key)):
    """Nistagmus modelini yeniden yükle"""
    global nistagmus_detector
    
    try:
        # Mevcut detektörü temizle
        if nistagmus_detector is not None:
            del nistagmus_detector
        
        # Yeni detektör oluştur
        nistagmus_detector = NistagmusDetector()
        logger.info("Nistagmus detector yeniden başlatıldı")
        
        return {
            "status": "success",
            "message": "Model başarıyla yeniden yüklendi",
            "timestamp": datetime.now().isoformat(),
            "model_status": nistagmus_detector is not None and nistagmus_detector.is_initialized
        }
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Model yeniden yükleme hatası: {error_msg}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"Model yeniden yüklenirken hata oluştu: {error_msg}",
                "timestamp": datetime.now().isoformat()
            }
        )

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest, api_key: str = Depends(get_api_key)):
    """Analiz sonuçları için geri bildirim gönder"""
    try:
        # Geri bildirimi kaydet
        feedback_file = Path("feedback") / f"{feedback.analysis_id}.json"
        os.makedirs(feedback_file.parent, exist_ok=True)
        
        with open(feedback_file, "w") as f:
            json.dump({
                "analysis_id": feedback.analysis_id,
                "correct": feedback.correct,
                "comments": feedback.comments,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        return {"status": "success", "message": "Geri bildirim için teşekkürler"}
    except Exception as e:
        logger.error(f"Geri bildirim hatası: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Hata yöneticileri
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP hata işleyicisi"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Genel hata işleyicisi"""
    logger.error(f"Beklenmeyen hata: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Beklenmeyen bir hata oluştu"},
    )

# UVicorn doğrudan çalıştırma
if __name__ == "__main__":
    import uvicorn
    import os
    from dotenv import load_dotenv
    
    # .env dosyasını yükle
    load_dotenv()
    
    # Ortam değişkenlerinden port değerini al, varsayılan olarak 8001 kullan
    port = int(os.getenv("VITE_AI_API_PORT", 8001))
    
    # Debug log
    print(f"AI API sunucusu başlatılıyor - Port: {port}")
    
    # Sunucuyu başlat
    uvicorn.run("nistagmus_ai_api:app", host="0.0.0.0", port=port, reload=True) 