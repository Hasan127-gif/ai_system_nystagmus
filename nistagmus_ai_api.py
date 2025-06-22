#!/usr/bin/env python3
"""
ROBUST NİSTAGMUS AI API SİSTEMİ
==============================
Gelişmiş hata yönetimi ve geri dönüş mekanizmalı FastAPI sistemi.
"""

import asyncio
import traceback
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import json
import logging
import time
import os

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ValidationError, validator
import uvicorn

# Custom exceptions
class NistagmusAPIError(Exception):
    """Base nistagmus API exception."""
    pass

class VideoProcessingError(NistagmusAPIError):
    """Video işleme hatası."""
    pass

class ModelInferenceError(NistagmusAPIError):
    """Model çıkarım hatası."""
    pass

class DataValidationError(NistagmusAPIError):
    """Veri doğrulama hatası."""
    pass

class ResourceNotFoundError(NistagmusAPIError):
    """Kaynak bulunamadı hatası."""
    pass

# Pydantic models
class AnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    analysis_type: str = "nystagmus"
    privacy_level: str = "standard"
    include_explainability: bool = True
    
    @validator('analysis_type')
    def validate_analysis_type(cls, v):
        allowed = ['nystagmus', 'strabismus', 'combined']
        if v not in allowed:
            raise ValueError(f'Analysis type must be one of: {allowed}')
        return v
    
    @validator('privacy_level')
    def validate_privacy_level(cls, v):
        allowed = ['basic', 'standard', 'high', 'maximum']
        if v not in allowed:
            raise ValueError(f'Privacy level must be one of: {allowed}')
        return v

class AnalysisResponse(BaseModel):
    analysis_id: str
    patient_id: Optional[str]
    timestamp: datetime
    results: Dict[str, Any]
    processing_time_ms: float
    confidence_score: float
    status: str
    explainability: Optional[Dict[str, Any]] = None
    warnings: List[str] = []
    metadata: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    system_health: Dict[str, Any]
    last_check: datetime

# Logging konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nistagmus_api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global değişkenler
app_start_time = time.time()
analysis_cache = {}
system_stats = {
    "total_analyses": 0,
    "successful_analyses": 0,
    "failed_analyses": 0,
    "avg_processing_time": 0.0
}

# FastAPI uygulaması
app = FastAPI(
    title="🔬 Nistagmus AI Analysis API",
    description="Gelişmiş nistagmus ve şaşılık tespit sistemi - Robust hata yönetimi ile",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware konfigürasyonu
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "https://yourdomain.com"],  # Production'da spesifik domain'ler
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Production'da spesifik host'lar
)

# Dependency injection fonksiyonları
async def get_system_status() -> Dict[str, Any]:
    """Sistem durumu kontrolü."""
    try:
        # CPU ve bellek kontrolleri
        import psutil
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        
        return {
            "cpu_usage_percent": cpu_usage,
            "memory_usage_percent": memory_info.percent,
            "memory_available_gb": memory_info.available / (1024**3),
            "healthy": cpu_usage < 80 and memory_info.percent < 90
        }
    except ImportError:
        return {"healthy": True, "note": "psutil not available"}

def validate_file_upload(file: UploadFile) -> UploadFile:
    """Dosya yükleme validasyonu - Gelişmiş MIME type detection."""
    if not file:
        raise HTTPException(status_code=400, detail="Dosya yüklenmedi")
    
    # Dosya boyutu kontrolü (100MB limit)
    if hasattr(file, 'size') and file.size > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Dosya çok büyük (max 100MB)")
    
    # 🔧 GELİŞTİRİLMİŞ MIME TYPE DETECTION
    allowed_types = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv']
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    
    # 1. Content-Type kontrolü
    content_type = file.content_type
    
    # 2. Eğer content_type yoksa filename'den tespit et
    if not content_type or content_type == 'application/octet-stream':
        if file.filename:
            filename_lower = file.filename.lower()
            if filename_lower.endswith('.mp4'):
                content_type = 'video/mp4'
            elif filename_lower.endswith('.avi'):
                content_type = 'video/avi'
            elif filename_lower.endswith('.mov'):
                content_type = 'video/mov'
            elif filename_lower.endswith('.mkv'):
                content_type = 'video/mkv'
    
    # 3. Extension kontrolü (fallback)
    valid_by_extension = False
    if file.filename:
        filename_lower = file.filename.lower()
        valid_by_extension = any(filename_lower.endswith(ext) for ext in allowed_extensions)
    
    # 4. MIME type veya extension'dan biri geçerliyse kabul et
    valid_by_content_type = content_type in allowed_types
    
    if not (valid_by_content_type or valid_by_extension):
        raise HTTPException(
            status_code=415, 
            detail=f"Desteklenmeyen dosya tipi. "
                   f"Content-Type: {content_type}, "
                   f"Filename: {file.filename}. "
                   f"İzin verilen tipler: {allowed_types} veya uzantılar: {allowed_extensions}"
        )
    
    # 5. Content-type'ı düzelt
    if not content_type or content_type == 'application/octet-stream':
        if file.filename:
            filename_lower = file.filename.lower()
            if filename_lower.endswith('.mp4'):
                file.content_type = 'video/mp4'
            elif filename_lower.endswith('.avi'):
                file.content_type = 'video/avi'
            elif filename_lower.endswith('.mov'):
                file.content_type = 'video/mov'
            elif filename_lower.endswith('.mkv'):
                file.content_type = 'video/mkv'
    
    logger.info(f"✅ Dosya validation başarılı: {file.filename} (type: {file.content_type})")
    return file

# Hata yönetimi middleware'i
@app.middleware("http")
async def error_handling_middleware(request, call_next):
    """Global hata yakalama middleware."""
    try:
        start_time = time.time()
        response = await call_next(request)
        
        # İşlem süresini log'la
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}\n{traceback.format_exc()}")
        
        # Hata türüne göre uygun yanıt
        if isinstance(e, HTTPException):
            raise e
        elif isinstance(e, VideoProcessingError):
            raise HTTPException(status_code=422, detail=f"Video işleme hatası: {str(e)}")
        elif isinstance(e, ModelInferenceError):
            raise HTTPException(status_code=500, detail=f"Model çıkarım hatası: {str(e)}")
        elif isinstance(e, DataValidationError):
            raise HTTPException(status_code=400, detail=f"Veri doğrulama hatası: {str(e)}")
        elif isinstance(e, ResourceNotFoundError):
            raise HTTPException(status_code=404, detail=f"Kaynak bulunamadı: {str(e)}")
        else:
            raise HTTPException(status_code=500, detail="Bilinmeyen bir sistem hatası oluştu")

# Ana endpoint'ler
@app.get("/", response_class=JSONResponse)
async def root():
    """Ana sayfa."""
    return {
        "message": "🔬 Nistagmus AI Analysis API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(system_status: Dict[str, Any] = Depends(get_system_status)):
    """Detaylı sistem sağlık kontrolü."""
    uptime = time.time() - app_start_time
    
    # Bileşen sağlık kontrolleri
    component_health = {
        "database": True,  # Gerçek DB bağlantı kontrolü
        "model": True,     # Model yükleme kontrolü
        "storage": True,   # Dosya sistemi kontrolü
        "memory": system_status.get("healthy", True)
    }
    
    overall_healthy = all(component_health.values())
    
    return HealthResponse(
        status="healthy" if overall_healthy else "degraded",
        version="2.0.0",
        uptime_seconds=uptime,
        system_health={
            "components": component_health,
            "system_stats": system_stats,
            **system_status
        },
        last_check=datetime.now()
    )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(validate_file_upload),
    
    # 🔧 FORM DATA DESTEĞİ EKLENDİ
    analysis_type: str = Form(default="combined"),
    include_explainability: str = Form(default="false"),
    patient_id: str = Form(default=""),
    privacy_level: str = Form(default="standard")
):
    """
    Ana video analiz endpoint'i - Form data ve JSON desteği ile.
    """
    # Form data'dan AnalysisRequest oluştur
    request_data = AnalysisRequest(
        analysis_type=analysis_type,
        include_explainability=include_explainability.lower() == "true",
        patient_id=patient_id,
        privacy_level=privacy_level
    )
    
    analysis_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(f"🎬 Analiz başlatılıyor: {analysis_id} (patient: {patient_id})")
    
    try:
        # Global stats güncelle
        system_stats["total_analyses"] += 1
        
        # Dosya kaydetme
        temp_file_path = await save_uploaded_file(file, analysis_id)
        
        try:
            # Video analizi
            analysis_results = await process_video_analysis(
                temp_file_path, 
                request_data,
                analysis_id
            )
            
            # Başarı stats
            system_stats["successful_analyses"] += 1
            
            # İşlem süresini hesapla
            processing_time = (time.time() - start_time) * 1000
            system_stats["avg_processing_time"] = (
                (system_stats["avg_processing_time"] * (system_stats["total_analyses"] - 1) + processing_time) 
                / system_stats["total_analyses"]
            )
            
            # Açıklanabilirlik analizi
            explainability_data = None
            if request_data.include_explainability:
                try:
                    explainability_data = await generate_explainability_analysis(
                        analysis_results, analysis_id
                    )
                except Exception as e:
                    logger.warning(f"Explainability hatası: {e}")
                    # Açıklanabilirlik hatası ana işlemi etkilemez
            
            # Background temizlik işlemi
            background_tasks.add_task(cleanup_temp_files, temp_file_path)
            
            # Sonuç hazırla
            response = AnalysisResponse(
                analysis_id=analysis_id,
                patient_id=request_data.patient_id,
                timestamp=datetime.now(),
                results=analysis_results,
                processing_time_ms=processing_time,
                confidence_score=analysis_results.get("confidence", 0.0),
                status="completed",
                explainability=explainability_data,
                warnings=analysis_results.get("warnings", []),
                metadata={
                    "file_name": file.filename,
                    "file_size": getattr(file, 'size', 0),
                    "analysis_type": request_data.analysis_type,
                    "privacy_level": request_data.privacy_level
                }
            )
            
            # Cache'e kaydet
            analysis_cache[analysis_id] = response.dict()
            
            logger.info(f"✅ Analiz tamamlandı: {analysis_id} ({processing_time:.1f}ms)")
            return response
            
        except VideoProcessingError as e:
            system_stats["failed_analyses"] += 1
            logger.error(f"Video işleme hatası: {e}")
            raise HTTPException(status_code=422, detail=f"Video işleme hatası: {str(e)}")
            
        except ModelInferenceError as e:
            system_stats["failed_analyses"] += 1
            logger.error(f"Model hatası: {e}")
            raise HTTPException(status_code=500, detail=f"Model çıkarım hatası: {str(e)}")
            
        finally:
            # Her durumda temp dosyayı temizle
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
            except Exception as e:
                logger.warning(f"Temp dosya temizlik hatası: {e}")
    
    except HTTPException:
        raise  # HTTPException'ları tekrar fırlat
    except Exception as e:
        system_stats["failed_analyses"] += 1
        logger.error(f"Beklenmeyen hata: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Beklenmeyen bir sistem hatası oluştu")

@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_result(analysis_id: str):
    """Analiz sonucunu ID ile getir."""
    try:
        if analysis_id not in analysis_cache:
            raise ResourceNotFoundError(f"Analiz bulunamadı: {analysis_id}")
        
        result = analysis_cache[analysis_id]
        return AnalysisResponse(**result)
        
    except ResourceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Analiz bulunamadı: {analysis_id}")
    except Exception as e:
        logger.error(f"Analiz getirme hatası: {e}")
        raise HTTPException(status_code=500, detail="Analiz getirme hatası")

@app.get("/metrics", response_class=JSONResponse)
async def get_system_metrics(system_status: Dict[str, Any] = Depends(get_system_status)):
    """Sistem metrikleri."""
    uptime = time.time() - app_start_time
    
    return {
        "uptime_seconds": uptime,
        "uptime_formatted": f"{uptime/3600:.1f} saat",
        "analysis_stats": system_stats,
        "system_health": system_status,
        "cache_size": len(analysis_cache),
        "api_version": "2.0.0"
    }

@app.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Analiz sonucunu sil."""
    try:
        if analysis_id not in analysis_cache:
            raise ResourceNotFoundError(f"Analiz bulunamadı: {analysis_id}")
        
        del analysis_cache[analysis_id]
        logger.info(f"🗑️ Analiz silindi: {analysis_id}")
        
        return {"message": f"Analiz silindi: {analysis_id}"}
        
    except ResourceNotFoundError:
        raise HTTPException(status_code=404, detail=f"Analiz bulunamadı: {analysis_id}")
    except Exception as e:
        logger.error(f"Analiz silme hatası: {e}")
        raise HTTPException(status_code=500, detail="Analiz silme hatası")

# Yardımcı fonksiyonlar
async def save_uploaded_file(file: UploadFile, analysis_id: str) -> str:
    """Dosyayı güvenli şekilde kaydet."""
    try:
        # Güvenli dosya adı oluştur
        safe_filename = f"temp_{analysis_id}_{int(time.time())}.mp4"
        temp_path = f"/tmp/{safe_filename}"
        
        # Dosyayı kaydet
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"📁 Dosya kaydedildi: {temp_path}")
        return temp_path
        
    except Exception as e:
        raise VideoProcessingError(f"Dosya kaydetme hatası: {str(e)}")

async def process_video_analysis(
    video_path: str, 
    request_data: AnalysisRequest,
    analysis_id: str
) -> Dict[str, Any]:
    """Video analiz işlemi."""
    try:
        logger.info(f"🔍 Video analiz ediliyor: {analysis_id}")
        
        # Mock analiz (gerçek implementasyon buraya gelecek)
        await asyncio.sleep(0.1)  # Simulated processing time
        
        # Gerçek nistagmus detection kodu burada olacak
        try:
            # from detector import Detector
            # detector = Detector()
            # results = detector.analyze_video(video_path)
            
            # Mock sonuç
            results = {
                "nystagmus_detected": True,
                "nystagmus_frequency": 3.2,
                "movement_amplitude": 0.75,
                "regularity": 0.68,
                "strabismus_detected": False,
                "strabismus_angle": 1.1,
                "strabismus_stability": 0.92,
                "confidence": 0.87,
                "frame_count": 150,
                "face_detection_rate": 0.94,
                "analysis_quality": "high",
                "warnings": []
            }
            
            # Ek validasyonlar
            if results["face_detection_rate"] < 0.5:
                results["warnings"].append("Düşük yüz tespit oranı - sonuçlar güvenilir olmayabilir")
            
            if results["confidence"] < 0.7:
                results["warnings"].append("Düşük güven skoru - ek inceleme gerekebilir")
            
            return results
            
        except Exception as e:
            raise ModelInferenceError(f"Model çıkarım hatası: {str(e)}")
            
    except Exception as e:
        raise VideoProcessingError(f"Video analiz hatası: {str(e)}")

async def generate_explainability_analysis(
    analysis_results: Dict[str, Any],
    analysis_id: str
) -> Dict[str, Any]:
    """Açıklanabilirlik analizi oluştur."""
    try:
        # Mock açıklanabilirlik (gerçek implementasyon buraya gelecek)
        from advanced_explainability import create_combined_explainer
        
        explainability = {
            "method": "SHAP + GradCAM",
            "analysis_id": analysis_id,
            "key_factors": [
                {
                    "factor": "Nistagmus Frekansı",
                    "importance": 0.42,
                    "value": analysis_results.get("nystagmus_frequency", 0),
                    "contribution": "Ana belirleyici faktör"
                },
                {
                    "factor": "Hareket Düzenliliği", 
                    "importance": 0.28,
                    "value": analysis_results.get("regularity", 0),
                    "contribution": "Önemli destekleyici faktör"
                }
            ],
            "visual_explanation_path": f"explainability_{analysis_id}.png",
            "clinical_interpretation": [
                "🔴 Yüksek frekans nistagmus tespit edildi",
                "🟡 Orta düzeyde düzensizlik gözlendi",
                "✅ Göz hizalanması normal sınırlarda"
            ],
            "confidence_breakdown": {
                "feature_quality": 0.88,
                "model_certainty": 0.85,
                "data_completeness": 0.94
            }
        }
        
        return explainability
        
    except Exception as e:
        logger.warning(f"Explainability hatası: {e}")
        return {
            "error": str(e),
            "fallback": "Basit açıklama mevcut değil"
        }

def cleanup_temp_files(file_path: str):
    """Geçici dosyaları temizle."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"🧹 Temp dosya temizlendi: {file_path}")
    except Exception as e:
        logger.warning(f"Temp dosya temizlik hatası: {e}")

# Custom exception handler'lar
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    """Pydantic validation hatası handler'ı."""
    logger.warning(f"Validation hatası: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Veri doğrulama hatası",
            "details": exc.errors(),
            "message": "Gönderilen veri formatı geçersiz"
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """ValueError handler'ı."""
    logger.warning(f"Value hatası: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error": "Geçersiz değer",
            "message": str(exc)
        }
    )

# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    """Uygulama başlatma eventi."""
    logger.info("🚀 Nistagmus AI API başlatılıyor...")
    
    # Sistem kontrollerini yap
    try:
        # Model yükleme kontrolü
        logger.info("🧠 Model kontrol ediliyor...")
        # model = load_model()  # Gerçek model yükleme
        
        # Cache temizleme
        analysis_cache.clear()
        
        logger.info("✅ Sistem başarıyla başlatıldı")
        
    except Exception as e:
        logger.error(f"❌ Başlatma hatası: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Uygulama kapatma eventi."""
    logger.info("🛑 Nistagmus AI API kapatılıyor...")
    
    # Temizlik işlemleri
    analysis_cache.clear()
    
    logger.info("✅ Sistem güvenli şekilde kapatıldı")

if __name__ == "__main__":
    # Production-ready server ayarları
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,  # Production'da artırılabilir
        "log_level": "info",
        "access_log": True,
        "reload": False  # Production'da False
    }
    
    logger.info(f"🌐 Server başlatılıyor: http://{config['host']}:{config['port']}")
    uvicorn.run(app, **config) 