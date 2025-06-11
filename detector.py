#!/usr/bin/env python3
"""
Nistagmus Tespit Sistemi - Klinik Entegrasyonlu
===============================================
Bu modül, gelişmiş nistagmus ve şaşılık tespiti yapan ana sınıfı içerir.
Klinik karar destek sistemi ile entegre edilmiştir.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

# Birleştirilmiş analiz fonksiyonları
from analysis_utils import (
    analyze_video_file,
    detect_iris_centers_unified,
    calculate_nystagmus_frequency_unified,
    calculate_strabismus_angle_instant,
    format_results_for_api,
    CLINICAL_SUPPORT_AVAILABLE
)

# Klinik karar destek sistemi (varsa)
if CLINICAL_SUPPORT_AVAILABLE:
    from decision import create_structured_report, classify_findings, quick_assessment
    from config import get_clinical_threshold

logger = logging.getLogger('nistagmus_detector')

class NistagmusDetector:
    """
    Gelişmiş nistagmus ve şaşılık tespit sistemi.
    Klinik değerlendirme özellikleri ile entegre edilmiştir.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Detector'ı başlatır.
        
        Args:
            config: Konfigürasyon parametreleri
        """
        self.config = config or {}
        
        # MediaPipe FaceMesh'i başlat
        try:
            import mediapipe as mp
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_drawing = mp.solutions.drawing_utils
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.is_initialized = True
            logger.info("MediaPipe FaceMesh başlatıldı")
        except ImportError:
            logger.warning("MediaPipe bulunamadı, temel tespit modu aktif")
            self.face_mesh = None
            self.is_initialized = False
        
        # Klinik eşikler (varsa)
        self.clinical_thresholds = {}
        if CLINICAL_SUPPORT_AVAILABLE:
            try:
                self.clinical_thresholds = {
                    "nystagmus_normal_max": get_clinical_threshold("nystagmus_freq_hz", "normal_max"),
                    "strabismus_normal_max": get_clinical_threshold("strabismus_pd", "normal_max")
                }
                logger.info("Klinik eşikler yüklendi")
            except Exception as e:
                logger.warning(f"Klinik eşikler yüklenemedi: {e}")
        
        # İstatistik takibi
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "pathological_cases": 0
        }
        
        logger.info("NistagmusDetector başlatıldı")
    
    def analyze_video(self, video_path: str, patient_age: float = None, max_frames: int = 300) -> Dict[str, Any]:
        """
        Video dosyasını analiz eder ve klinik değerlendirme yapar.
        
        Args:
            video_path: Video dosya yolu
            patient_age: Hasta yaşı (klinik değerlendirme için)
            max_frames: Maksimum işlenecek kare sayısı
            
        Returns:
            dict: Kapsamlı analiz sonuçları
        """
        if not self.is_initialized:
            return {"error": "Model henüz başlatılmadı veya MediaPipe bulunamadı"}
        
        self.analysis_stats["total_analyses"] += 1
        
        try:
            logger.info(f"Video analizi başlatılıyor: {video_path}")
            
            # Temel video analizi (analysis_utils kullanarak)
            basic_results = analyze_video_file(video_path, detector=self, max_frames=max_frames)
            
            if "error" in basic_results:
                logger.error(f"Video analizi başarısız: {basic_results['error']}")
                return basic_results
            
            # Başarılı analiz sayacını artır
            self.analysis_stats["successful_analyses"] += 1
            
            # KLİNİK DEĞERLENDİRME (eğer sistem mevcutsa)
            if CLINICAL_SUPPORT_AVAILABLE and patient_age is not None:
                try:
                    # Yaş faktörlü klinik değerlendirme
                    nystagmus_freq = basic_results.get("nistagmus_frequency", 0.0)
                    strabismus_angle = basic_results.get("strabismus_angle", 0.0)
                    face_detection_rate = basic_results.get("face_detection_rate", 0.0)
                    
                    # Yaş spesifik klinik değerlendirme
                    age_specific_evaluation = classify_findings(
                        nystagmus_freq=nystagmus_freq,
                        strabismus_angle=strabismus_angle,
                        face_detection_rate=face_detection_rate,
                        age=patient_age
                    )
                    
                    # Patoloji istatistiği
                    if age_specific_evaluation.get("overall_assessment", {}).get("has_pathology", False):
                        self.analysis_stats["pathological_cases"] += 1
                    
                    # Sonuçları birleştir
                    basic_results["clinical_evaluation_with_age"] = age_specific_evaluation
                    basic_results["patient_age"] = patient_age
                    
                    logger.info(f"Yaş spesifik klinik değerlendirme tamamlandı (yaş: {patient_age})")
                    
                except Exception as e:
                    logger.warning(f"Yaş spesifik klinik değerlendirme hatası: {str(e)}")
                    basic_results["age_evaluation_error"] = str(e)
            
            # API formatına çevir
            api_formatted = format_results_for_api(basic_results)
            basic_results["api_format"] = api_formatted
            
            # İstatistikleri ekle
            basic_results["detector_stats"] = self.get_analysis_statistics()
            
            logger.info("Video analizi başarıyla tamamlandı")
            return basic_results
            
        except Exception as e:
            logger.error(f"Video analizi sırasında beklenmeyen hata: {str(e)}")
            return {"error": f"Analiz hatası: {str(e)}"}
    
    def analyze_frame_sequence(self, frames: List[np.ndarray], fps: float = 30.0) -> Dict[str, Any]:
        """
        Kare dizisini analiz eder.
        
        Args:
            frames: Video kareleri listesi
            fps: Frame rate
            
        Returns:
            dict: Analiz sonuçları
        """
        try:
            if len(frames) < 5:
                return {"error": "Yetersiz kare sayısı (minimum 5 gerekli)"}
            
            # Veri toplama
            left_positions_y = []
            strabismus_angles = []
            valid_frames = 0
            
            for frame in frames:
                left_center, right_center = self.detect_iris_centers(frame)
                
                if left_center is not None and right_center is not None:
                    valid_frames += 1
                    left_positions_y.append(left_center[1])
                    
                    # Şaşılık açısı hesapla
                    angle = calculate_strabismus_angle_instant(left_center, right_center)
                    strabismus_angles.append(angle)
            
            if valid_frames < 3:
                return {"error": "Yetersiz geçerli kare (minimum 3 gerekli)"}
            
            # Nistagmus frekansı
            nystagmus_freq = calculate_nystagmus_frequency_unified(left_positions_y, fps)
            
            # Ortalama şaşılık açısı
            avg_strabismus = np.mean(strabismus_angles) if strabismus_angles else 0.0
            
            # Sonuçlar
            results = {
                "nistagmus_frequency": float(nystagmus_freq),
                "strabismus_angle": float(avg_strabismus),
                "valid_frames": valid_frames,
                "total_frames": len(frames),
                "face_detection_rate": valid_frames / len(frames)
            }
            
            # Klinik değerlendirme (varsa)
            if CLINICAL_SUPPORT_AVAILABLE:
                clinical_eval = classify_findings(
                    nystagmus_freq=nystagmus_freq,
                    strabismus_angle=avg_strabismus,
                    face_detection_rate=results["face_detection_rate"]
                )
                results["clinical_evaluation"] = clinical_eval
            
            return results
            
        except Exception as e:
            logger.error(f"Kare dizisi analizi hatası: {str(e)}")
            return {"error": str(e)}
    
    def detect_iris_centers(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """
        Kareden iris merkezlerini tespit eder.
        
        Args:
            frame: BGR formatında görüntü karesi
            
        Returns:
            tuple: (left_center, right_center) koordinatları
        """
        if not self.is_initialized or self.face_mesh is None:
            return None, None
            
        try:
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                
                # Sol ve sağ iris merkezleri (MediaPipe indeksleri)
                left_iris = landmarks[468]
                right_iris = landmarks[473]
                
                # Normalleştirilmiş koordinatları piksel değerlerine çevir
                left_center = (int(left_iris.x * w), int(left_iris.y * h))
                right_center = (int(right_iris.x * w), int(right_iris.y * h))
                
                return left_center, right_center
            
            return None, None
            
        except Exception as e:
            logger.warning(f"Iris tespit hatası: {str(e)}")
            return None, None
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """
        Analiz istatistiklerini döndürür.
        
        Returns:
            dict: İstatistik bilgileri
        """
        stats = self.analysis_stats.copy()
        
        if stats["total_analyses"] > 0:
            stats["success_rate"] = stats["successful_analyses"] / stats["total_analyses"]
            stats["pathology_rate"] = stats["pathological_cases"] / stats["successful_analyses"] if stats["successful_analyses"] > 0 else 0.0
        else:
            stats["success_rate"] = 0.0
            stats["pathology_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Analiz istatistiklerini sıfırlar."""
        self.analysis_stats = {
            "total_analyses": 0,
            "successful_analyses": 0,
            "pathological_cases": 0
        }
        logger.info("Analiz istatistikleri sıfırlandı")
    
    def get_clinical_summary(self, analysis_results: Dict[str, Any]) -> str:
        """
        Analiz sonuçlarından klinik özet oluşturur.
        
        Args:
            analysis_results: Analiz sonuçları
            
        Returns:
            str: Klinik özet metni
        """
        if not CLINICAL_SUPPORT_AVAILABLE:
            return "Klinik değerlendirme sistemi mevcut değil."
        
        try:
            if "clinical_evaluation" in analysis_results:
                return analysis_results["clinical_evaluation"].get("clinical_summary", "Özet oluşturulamadı.")
            else:
                return "Klinik değerlendirme yapılmamış."
        except Exception as e:
            return f"Özet oluşturma hatası: {str(e)}"
    
    def check_clinical_thresholds(self, nystagmus_freq: float, strabismus_angle: float) -> Dict[str, Any]:
        """
        Sonuçları klinik eşiklerle karşılaştırır.
        
        Args:
            nystagmus_freq: Nistagmus frekansı
            strabismus_angle: Şaşılık açısı
            
        Returns:
            dict: Eşik karşılaştırma sonuçları
        """
        if not CLINICAL_SUPPORT_AVAILABLE:
            return {"error": "Klinik eşikler mevcut değil"}
        
        try:
            strabismus_pd = strabismus_angle * 1.75  # Basit dönüşüm
            
            return quick_assessment(nystagmus_freq, strabismus_pd)
            
        except Exception as e:
            logger.error(f"Klinik eşik kontrolü hatası: {str(e)}")
            return {"error": str(e)}
    
    def __del__(self):
        """Detector temizleme işlemleri."""
        if hasattr(self, 'face_mesh') and self.face_mesh:
            try:
                self.face_mesh.close()
            except Exception:
                pass

# Export için ana sınıf
__all__ = ['NistagmusDetector']

# Backward compatibility için
NystagmusDetector = NistagmusDetector

def get_detector(**kwargs):
    """Factory function for detector."""
    return NistagmusDetector(**kwargs)

if __name__ == "__main__":
    # Test
    detector = NistagmusDetector()
    print(f"Detector başlatıldı: {detector.is_initialized}")
    print(f"İstatistikler: {detector.get_analysis_statistics()}") 