#!/usr/bin/env python3
"""
Ortak fonksiyonlar - duplikasyon önleme
"""

import time
import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def create_test_video(output_path: str = "test_video.mp4", 
                     duration: int = 5, 
                     fps: int = 30,
                     with_nystagmus: bool = True) -> str:
    """
    Standart test videosu oluşturma fonksiyonu.
    Tüm duplikasyonları bu fonksiyona yönlendir.
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    frames = duration * fps
    for i in range(frames):
        # Basit test karesi oluştur
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Gri arka plan
        
        if with_nystagmus:
            # Basit nistagmus simülasyonu - yatay hareket
            offset = int(20 * np.sin(i * 0.3))
            cv2.circle(frame, (320 + offset, 240), 20, (255, 255, 255), -1)  # Sol göz
            cv2.circle(frame, (400 + offset, 240), 20, (255, 255, 255), -1)  # Sağ göz
        else:
            # Statik gözler
            cv2.circle(frame, (320, 240), 20, (255, 255, 255), -1)
            cv2.circle(frame, (400, 240), 20, (255, 255, 255), -1)
        
        out.write(frame)
    
    out.release()
    return output_path

def standard_main_function():
    """
    Standart main fonksiyonu - tüm duplikasyonları buraya yönlendir.
    """
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            print("🧪 Test modu aktif")
            return test_mode()
        elif sys.argv[1] == "--help":
            print_help()
            return
    
    print("🚀 Nistagmus AI Sistemi")
    print("Kullanım: python script.py [--test|--help]")

def test_mode():
    """Test modunu çalıştır."""
    print("✅ Test modu başlatıldı")
    # Test kodları buraya

def print_help():
    """Yardım metni."""
    print("""
🏥 Nistagmus AI Sistemi Yardım
=============================
    
Kullanılabilir komutlar:
  --test    : Test modunu başlat
  --help    : Bu yardım metnini göster
  
Örnekler:
  python detector.py --test
  python analysis.py --help
""")

def cleanup_temp_files():
    """
    Geçici dosyaları temizle - duplikasyon önleme.
    """
    temp_patterns = ["*.tmp", "*_temp.*", "temp_*"]
    removed_count = 0
    
    for pattern in temp_patterns:
        import glob
        for file in glob.glob(pattern):
            try:
                os.remove(file)
                removed_count += 1
            except Exception as e:
                logger.warning(f"Temp dosya silinemedi: {file} - {e}")
    
    if removed_count > 0:
        logger.info(f"{removed_count} geçici dosya temizlendi")

def get_system_stats() -> Dict[str, Any]:
    """Sistem istatistikleri - merkezi fonksiyon."""
    import psutil
    
    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_usage": psutil.disk_usage("/").percent,
        "timestamp": time.time()
    }

# Export edilen fonksiyonlar
__all__ = [
    'create_test_video',
    'standard_main_function', 
    'test_mode',
    'cleanup_temp_files',
    'get_system_stats'
]
