"""
Basit FastAPI uygulaması - Test amaçlı
"""

import logging
import time
import random
import json
from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import base64
from typing import Dict, Any, Optional

# Logging yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_api')

# FastAPI uygulaması
app = FastAPI(
    title="Basit Test API",
    description="Test amaçlı basit API",
    version="1.0.0",
)

# CORS middleware - tüm kaynaklara izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm kaynaklara izin ver (test amaçlı)
    allow_credentials=True,
    allow_methods=["*"],  # Tüm HTTP metotlarına izin ver
    allow_headers=["*"],  # Tüm başlıklara izin ver
    expose_headers=["*"]
)

# API başlangıç zamanı
start_time = time.time()

@app.get("/")
async def root():
    """API kök endpoint'i"""
    logger.info("Kök endpoint çağrıldı")
    return {
        "name": "Basit Test API",
        "version": "1.0.0",
        "description": "Test amaçlı basit API",
    }

@app.get("/health")
async def health_check(x_api_key: Optional[str] = Header(None)):
    """API sağlık kontrolü"""
    logger.info("Sağlık kontrolü çağrıldı")
    
    # API key doğrulama (opsiyonel)
    if x_api_key and x_api_key != "test-api-key":
        logger.warning(f"Geçersiz API anahtarı: {x_api_key}")
        # API key'i logla ama hata döndürme (test için)
    
    return {
        "status": "ok",
        "version": "1.0.0",
        "uptime": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
        "model_status": True
    }

@app.post("/analyze/image")
async def analyze_image(request: Request, x_api_key: Optional[str] = Header(None)):
    """Görüntü analizi - demo amaçlı"""
    logger.info("Görüntü analizi endpoint'i çağrıldı")
    
    # Request body'yi oku
    body = await request.json()
    
    # Base64 görüntü var mı?
    image_data = body.get("image_data", "")
    if not image_data:
        raise HTTPException(status_code=400, detail="image_data gerekli")
    
    # Demo amaçlı bekleme ekle
    await asyncio.sleep(1)
    
    # Demo sonuç oluştur
    result = create_dummy_analysis_result()
    
    return {
        "status": "success",
        "results": result,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/analyze/video")
async def analyze_video(request: Request, x_api_key: Optional[str] = Header(None)):
    """Video analizi - demo amaçlı"""
    logger.info("Video analizi endpoint'i çağrıldı")
    
    # Video analizi için ek bekleme
    await asyncio.sleep(2)
    
    # Demo sonuç oluştur (video için daha kapsamlı)
    result = create_dummy_analysis_result()
    
    # Video analizine özel alanlar ekle
    result["data"]["frames_processed"] = 180
    result["data"]["faces_detected"] = 175
    result["data"]["frames_processed_ratio"] = 0.97
    
    return {
        "status": "success",
        "results": result,
        "timestamp": datetime.now().isoformat()
    }

@app.options("/{path:path}")
async def options_route(path: str):
    """CORS preflight istekleri için global OPTIONS handler"""
    logger.info(f"OPTIONS isteği alındı: /{path}")
    return Response(status_code=200)

# Demo analiz sonucu oluşturma
def create_dummy_analysis_result():
    return {
        "data": {
            "is_nystagmus": random.random() > 0.3,  # %70 pozitif, %30 negatif
            "confidence": random.random() * 0.5 + 0.5,  # 0.5-1.0 arası güven skoru
            "eye_movements": {
                "left_eye": {
                    "movement": {
                        "position_std_x": round(random.random() * 1.5 + 1.0, 2),
                        "position_std_y": round(random.random() * 0.8 + 0.2, 2),
                        "velocity_std_x": round(random.random() * 15 + 10, 2),
                        "velocity_std_y": round(random.random() * 5 + 2, 2),
                        "max_velocity": round(random.random() * 25 + 20, 2),
                        "fast_phases": round(random.random() * 15 + 10)
                    },
                    "spectral": {
                        "dominant_frequency": round(random.random() * 4 + 2, 2)
                    }
                },
                "right_eye": {
                    "movement": {
                        "position_std_x": round(random.random() * 1.5 + 1.0, 2),
                        "position_std_y": round(random.random() * 0.8 + 0.2, 2),
                        "velocity_std_x": round(random.random() * 15 + 10, 2),
                        "velocity_std_y": round(random.random() * 5 + 2, 2),
                        "max_velocity": round(random.random() * 25 + 20, 2),
                        "fast_phases": round(random.random() * 15 + 10)
                    },
                    "spectral": {
                        "dominant_frequency": round(random.random() * 4 + 2, 2)
                    }
                }
            },
            "static_assessment": "Nistagmus analizi tamamlandı",
            "timestamp": time.time()
        }
    }

# UVicorn doğrudan çalıştırma
if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    logger.info("Basit API başlatılıyor...")
    uvicorn.run("simple_api:app", host="0.0.0.0", port=8001, reload=True)
else:
    # Module olarak import edildiğinde asyncio modülünü yükle
    import asyncio 