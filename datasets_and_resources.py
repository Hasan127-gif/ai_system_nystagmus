#!/usr/bin/env python3
"""
AÇIK KAYNAK VERİ SETLERİ VE KOD ÖRNEKLERİ
=========================================
Eğitimi tekrarlanabilir yapmak ve araştırmacıların projeyi denemesi için kaynaklar.
"""

import os
import urllib.request
import zipfile
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """Açık kaynak veri setlerini yönetir."""
    
    def __init__(self, base_path: str = "datasets"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def get_available_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Mevcut açık kaynak veri setlerini listeler."""
        return {
            "UnityEyes": {
                "description": "Sentetik göz hareketi verisi - çok çeşitli göz pozisyonları",
                "size": "~5GB",
                "format": "PNG images + JSON annotations",
                "use_case": "Göz tracking modeli eğitimi",
                "github": "https://github.com/swook/UnityEyes",
                "paper": "https://arxiv.org/abs/1512.04916",
                "download_command": "git clone https://github.com/swook/UnityEyes.git",
                "license": "MIT",
                "quality": "⭐⭐⭐⭐⭐",
                "samples": 1000000
            },
            "OpenEDS": {
                "description": "Facebook'un açık göz-izleme veri seti",
                "size": "~10GB", 
                "format": "Video sequences + eye landmarks",
                "use_case": "Gerçek göz hareketi analizi",
                "github": "https://github.com/facebookresearch/openeds",
                "paper": "https://arxiv.org/abs/1905.03702",
                "download_command": "git clone https://github.com/facebookresearch/openeds.git",
                "license": "CC BY-NC 4.0",
                "quality": "⭐⭐⭐⭐⭐",
                "samples": 152
            },
            "MPIIGaze": {
                "description": "Gerçek göz bakış yönü veri seti",
                "size": "~3GB",
                "format": "Images + gaze directions",
                "use_case": "Göz bakış tahmini",
                "website": "https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/",
                "download_command": "# İzin gerekir, website'den başvurun",
                "license": "Academic use only",
                "quality": "⭐⭐⭐⭐",
                "samples": 213659
            },
            "EyeTrackingData": {
                "description": "Çeşitli göz tracking görevleri veri seti",
                "size": "~1GB",
                "format": "CSV + video files", 
                "use_case": "Sakkad ve fiksasyon analizi",
                "github": "https://github.com/esdalmaijer/PyGazeAnalyser",
                "download_command": "git clone https://github.com/esdalmaijer/PyGazeAnalyser.git",
                "license": "GPL-3.0",
                "quality": "⭐⭐⭐",
                "samples": 50000
            },
            "MediaPipe_Samples": {
                "description": "MediaPipe demo videoları ve kod örnekleri",
                "size": "~500MB",
                "format": "MP4 videos + Python scripts",
                "use_case": "MediaPipe entegrasyonu test",
                "github": "https://github.com/google/mediapipe",
                "download_command": "git clone https://github.com/google/mediapipe.git",
                "license": "Apache 2.0",
                "quality": "⭐⭐⭐⭐⭐",
                "samples": 100
            }
        }
    
    def download_sample_datasets(self):
        """Örnek veri setlerini indirir."""
        print("📥 Örnek veri setleri indiriliyor...")
        
        # 1. MediaPipe örnek veriler
        self._download_mediapipe_samples()
        
        # 2. Nistagmus test verileri oluştur
        self._create_synthetic_nystagmus_data()
        
        print("✅ Örnek veri setleri hazır!")
    
    def _download_mediapipe_samples(self):
        """MediaPipe örnek verilerini indir."""
        try:
            import subprocess
            
            mediapipe_path = os.path.join(self.base_path, "mediapipe_samples")
            if not os.path.exists(mediapipe_path):
                print("MediaPipe örnekleri klonlanıyor...")
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    "https://github.com/google/mediapipe.git",
                    mediapipe_path
                ], check=True)
                print("✅ MediaPipe örnekleri indirildi")
            else:
                print("ℹ️ MediaPipe örnekleri zaten mevcut")
                
        except Exception as e:
            logger.error(f"MediaPipe örnekleri indirme hatası: {e}")
    
    def _create_synthetic_nystagmus_data(self):
        """Sentetik nistagmus verisi oluştur."""
        try:
            import cv2
            import numpy as np
            
            synthetic_path = os.path.join(self.base_path, "synthetic_nystagmus")
            os.makedirs(synthetic_path, exist_ok=True)
            
            # Farklı nistagmus tiplerinde test videoları oluştur
            nystagmus_types = {
                "normal": {"freq": 0.5, "amplitude": 5},
                "mild_nystagmus": {"freq": 3.0, "amplitude": 15},
                "severe_nystagmus": {"freq": 8.0, "amplitude": 30},
                "strabismus": {"freq": 1.0, "amplitude": 10, "offset": 20}
            }
            
            for nys_type, params in nystagmus_types.items():
                video_path = os.path.join(synthetic_path, f"{nys_type}.mp4")
                
                if not os.path.exists(video_path):
                    print(f"🎥 {nys_type} videosu oluşturuluyor...")
                    self._create_nystagmus_video(video_path, **params)
            
            # Metadata oluştur
            metadata = {
                "dataset_info": {
                    "name": "Synthetic Nystagmus Dataset",
                    "version": "1.0",
                    "created_by": "Nystagmus Detection System",
                    "samples": len(nystagmus_types),
                    "format": "MP4 videos with simulated eye movements"
                },
                "samples": nystagmus_types
            }
            
            with open(os.path.join(synthetic_path, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print("✅ Sentetik nistagmus verisi oluşturuldu")
            
        except Exception as e:
            logger.error(f"Sentetik veri oluşturma hatası: {e}")
    
    def _create_nystagmus_video(self, output_path: str, freq: float, amplitude: float, offset: int = 0):
        """Belirli parametrelerle nistagmus videosu oluştur."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
        
        for i in range(300):  # 10 saniye
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Yüz çiz
            cv2.circle(frame, (320, 240), 100, (200, 150, 100), -1)
            
            # Nistagmus hareketi simülasyonu
            left_x = 300 + int(amplitude * np.sin(i * freq * 0.2))
            right_x = 340 + int(amplitude * np.sin(i * freq * 0.2)) + offset
            
            # Gözleri çiz
            cv2.circle(frame, (left_x, 220), 12, (255, 255, 255), -1)
            cv2.circle(frame, (right_x, 220), 12, (255, 255, 255), -1)
            cv2.circle(frame, (left_x, 220), 5, (0, 0, 0), -1)
            cv2.circle(frame, (right_x, 220), 5, (0, 0, 0), -1)
            
            out.write(frame)
        
        out.release()

def get_code_examples() -> Dict[str, str]:
    """Kullanışlı kod örnekleri döndürür."""
    return {
        "mediapipe_installation": """
# MediaPipe kurulumu
pip install mediapipe opencv-python

# Veya kaynak koddan
git clone https://github.com/google/mediapipe.git
cd mediapipe
python setup.py install
        """,
        
        "advanced_eye_tracking": """
# Gelişmiş göz tracking için ConVNG
pip install git+https://github.com/guanfuchen/ConVNG.git

# Veya manuel kurulum
git clone https://github.com/guanfuchen/ConVNG.git
cd ConVNG
pip install -r requirements.txt
python setup.py install
        """,
        
        "performance_profiling": """
import time
import cv2
from detector import NystagmusDetector

def profile_detection_speed():
    detector = NystagmusDetector()
    
    # Test videosu yükle
    cap = cv2.VideoCapture('test_video.mp4')
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Iris tespit et
        detector.detect_iris_centers(frame)
        frame_count += 1
        
        if frame_count >= 100:  # 100 kare test
            break
    
    end_time = time.time()
    fps = frame_count / (end_time - start_time)
    
    print(f"FPS: {fps:.2f}")
    print(f"Frame başına süre: {1000/fps:.2f}ms")
    
    cap.release()
    return fps

# Performans testi çalıştır
if __name__ == "__main__":
    fps = profile_detection_speed()
    if fps >= 25:
        print("✅ Gerçek zamanlı kullanım için uygun")
    else:
        print("⚠️ Optimizasyon gerekli")
        """,
        
        "dataset_preprocessing": """
import os
import json
import cv2
import numpy as np

def preprocess_eye_tracking_dataset(dataset_path):
    \"\"\"Göz tracking veri setini ön işle.\"\"\"
    
    processed_data = []
    
    for video_file in os.listdir(dataset_path):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(dataset_path, video_file)
            
            # Video yükle
            cap = cv2.VideoCapture(video_path)
            
            frame_data = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Göz tespiti yap
                # (Burada kendi detector'ınızı kullanın)
                
                frame_data.append({
                    "timestamp": cap.get(cv2.CAP_PROP_POS_MSEC),
                    "features": "extracted_features"
                })
            
            cap.release()
            
            processed_data.append({
                "video": video_file,
                "frames": frame_data
            })
    
    # Sonuçları kaydet
    with open('processed_dataset.json', 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    return processed_data
        """,
        
        "data_augmentation": """
import cv2
import numpy as np
from typing import Tuple

def augment_eye_video(video_path: str, output_path: str):
    \"\"\"Göz videosunu çeşitli augmentasyonlarla çoğalt.\"\"\"
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Farklı augmentasyon türleri
    augmentations = [
        ("brightness", lambda x: cv2.convertScaleAbs(x, alpha=1.2, beta=30)),
        ("contrast", lambda x: cv2.convertScaleAbs(x, alpha=1.5, beta=0)),
        ("blur", lambda x: cv2.GaussianBlur(x, (3, 3), 0)),
        ("noise", lambda x: add_noise(x))
    ]
    
    for aug_name, aug_func in augmentations:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Başa sar
        
        out_path = output_path.replace('.mp4', f'_{aug_name}.mp4')
        out = cv2.VideoWriter(out_path, fourcc, fps, (640, 480))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            augmented_frame = aug_func(frame)
            out.write(augmented_frame)
        
        out.release()
        print(f"✅ {aug_name} augmentasyonu tamamlandı: {out_path}")
    
    cap.release()

def add_noise(image: np.ndarray) -> np.ndarray:
    \"\"\"Görüntüye gürültü ekle.\"\"\"
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    return cv2.add(image, noise)
        """
    }

def print_dataset_installation_guide():
    """Veri seti kurulum rehberi yazdır."""
    
    print("=" * 80)
    print("📊 AÇIK KAYNAK VERİ SETLERİ KURULUM REHBERİ")
    print("=" * 80)
    
    datasets = DatasetManager().get_available_datasets()
    
    for name, info in datasets.items():
        print(f"\n🎯 {name}")
        print(f"   📝 Açıklama: {info['description']}")
        print(f"   💾 Boyut: {info['size']}")
        print(f"   📁 Format: {info['format']}")
        print(f"   🎯 Kullanım: {info['use_case']}")
        print(f"   ⭐ Kalite: {info['quality']}")
        print(f"   📊 Örnek sayısı: {info['samples']}")
        print(f"   📄 Lisans: {info['license']}")
        print(f"   📥 İndirme:")
        print(f"      {info['download_command']}")
        
        if 'github' in info:
            print(f"   🔗 GitHub: {info['github']}")
        if 'paper' in info:
            print(f"   📚 Makale: {info['paper']}")
        if 'website' in info:
            print(f"   🌐 Website: {info['website']}")
        
        print("-" * 50)

if __name__ == "__main__":
    print_dataset_installation_guide()
    
    # Örnek veri seti yöneticisi
    manager = DatasetManager()
    manager.download_sample_datasets() 