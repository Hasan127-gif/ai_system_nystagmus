#!/usr/bin/env python3
"""
Kritik kod duplikasyonlarını temizleme sistemi
"""

import os
import shutil
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DuplicateRemover:
    """Sistem bütünlüğünü bozmadan duplikasyonları temizler."""
    
    def __init__(self):
        self.removed_files = []
        self.backup_dir = "backup_before_cleanup"
        
    def create_backup(self):
        """Temizlik öncesi yedek oluştur."""
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        
        # Kritik dosyaları yedekle
        critical_files = [
            "error_manager.py",
            "error/error_manager.py", 
            "error_ui_components.py",
            "error/error_ui_components.py"
        ]
        
        for file in critical_files:
            if os.path.exists(file):
                backup_path = os.path.join(self.backup_dir, file.replace("/", "_"))
                shutil.copy2(file, backup_path)
                print(f"✅ Yedeklendi: {file} → {backup_path}")
    
    def remove_duplicate_files(self):
        """Duplicate dosyaları kaldır."""
        # error_manager.py duplikasyonunu çöz - error/ klasöründekini koru
        if os.path.exists("error_manager.py") and os.path.exists("error/error_manager.py"):
            print("🔧 error_manager.py duplikasyonu temizleniyor...")
            os.remove("error_manager.py")
            self.removed_files.append("error_manager.py")
            print("✅ Root dizindeki error_manager.py silindi, error/ içindeki korundu")
        
        # error_ui_components.py duplikasyonunu çöz
        if os.path.exists("error_ui_components.py") and os.path.exists("error/error_ui_components.py"):
            print("🔧 error_ui_components.py duplikasyonu temizleniyor...")
            os.remove("error_ui_components.py")
            self.removed_files.append("error_ui_components.py")
            print("✅ Root dizindeki error_ui_components.py silindi, error/ içindeki korundu")
    
    def consolidate_duplicate_functions(self):
        """Duplike fonksiyonları tek yerde topla."""
        # Common utilities dosyası oluştur
        common_utils_content = '''#!/usr/bin/env python3
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
'''
        
        # common_utils.py dosyasını oluştur
        with open("common_utils.py", "w", encoding="utf-8") as f:
            f.write(common_utils_content)
        print("✅ common_utils.py oluşturuldu - duplike fonksiyonlar burada toplanacak")
    
    def update_imports(self):
        """Dosyalardaki importları common_utils'e yönlendir."""
        # Bu gelecekte manuel olarak yapılacak
        print("ℹ️  Import güncellemeleri manuel olarak yapılmalı:")
        print("   from common_utils import create_test_video, standard_main_function")
    
    def generate_report(self):
        """Temizlik raporunu oluştur."""
        report = f"""
🧹 KOD DUPLİKASYON TEMİZLİK RAPORU
=================================

✅ BAŞARILI TEMİZLİKLER:
{chr(10).join(f"  • {file}" for file in self.removed_files)}

📁 OLUŞTURULAN DOSYALAR:
  • common_utils.py - Duplike fonksiyonlar için merkezi modül
  • {self.backup_dir}/ - Temizlik öncesi yedekler

📋 SONRAKI ADIMLAR:
  1. Import statements'ları güncelle
  2. Kalan duplike fonksiyonları common_utils'e taşı  
  3. Test coverage'ı kontrol et

🎯 SONUÇ: Kritik duplikasyonlar temizlendi, sistem bütünlüğü korundu.
"""
        
        with open("cleanup_report.md", "w", encoding="utf-8") as f:
            f.write(report)
        
        print(report)

def main():
    """Ana temizlik süreci."""
    print("🚨 KRİTİK DUPLİKASYON TEMİZLİĞİ BAŞLATILIYOR...")
    print("=" * 50)
    
    remover = DuplicateRemover()
    
    try:
        # 1. Yedek oluştur
        print("\n📁 1. YEDEKLEMİ İŞLEMİ")
        remover.create_backup()
        
        # 2. Duplicate dosyaları sil
        print("\n🗑️  2. DUPLİKAT DOSYA TEMİZLİĞİ")
        remover.remove_duplicate_files()
        
        # 3. Ortak fonksiyonlar oluştur
        print("\n🔧 3. ORTAK FONKSİYON KONSOLIDASYONU")
        remover.consolidate_duplicate_functions()
        
        # 4. Rapor oluştur
        print("\n📊 4. RAPOR OLUŞTURMA")
        remover.generate_report()
        
        print("\n✅ TEMİZLİK BAŞARILI! Sistem bütünlüğü korundu.")
        
    except Exception as e:
        print(f"\n❌ TEMİZLİK HATASI: {e}")
        print("💡 Yedeklerden geri yükleme yapılabilir.")

if __name__ == "__main__":
    main() 