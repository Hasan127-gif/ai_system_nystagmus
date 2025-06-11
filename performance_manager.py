#!/usr/bin/env python3
"""
GERÇEK ZAMANLI PERFORMANS HEDEFLERİ VE İZLEME
===========================================
Klinik kullanım için performans gereksinimlerini tanımlar ve izler.
"""

import time
import psutil
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Callable
import logging
import json
from datetime import datetime
import threading
import queue

logger = logging.getLogger(__name__)

class PerformanceManager:
    """Sistem performansını izler ve optimize eder."""
    
    # KLİNİK PERFORMANS HEDEFLERİ
    PERFORMANCE_TARGETS = {
        "minimum_fps": 25,        # Gerçek zamanlı analiz için minimum FPS
        "target_fps": 30,         # Hedef FPS
        "max_latency_ms": 40,     # Maksimum gecikme (ms)
        "max_memory_mb": 2048,    # Maksimum bellek kullanımı (MB)
        "max_cpu_percent": 80,    # Maksimum CPU kullanımı (%)
        "min_accuracy": 0.85,     # Minimum doğruluk oranı
        "frame_processing_ms": 30 # Frame başına maksimum işlem süresi
    }
    
    # MİNİMUM DONANIM GEREKSİNİMLERİ
    HARDWARE_REQUIREMENTS = {
        "minimum": {
            "cpu": "Intel i5-8400 / AMD Ryzen 5 2600",
            "ram_gb": 8,
            "gpu": "Intel UHD Graphics 630 (integrated)",
            "expected_fps": "15-20",
            "use_case": "Temel klinik kullanım"
        },
        "recommended": {
            "cpu": "Intel i7-10700 / AMD Ryzen 7 3700X", 
            "ram_gb": 16,
            "gpu": "NVIDIA GTX 1060 / AMD RX 580",
            "expected_fps": "25-30",
            "use_case": "Optimal klinik deneyim"
        },
        "high_performance": {
            "cpu": "Intel i9-12900K / AMD Ryzen 9 5900X",
            "ram_gb": 32,
            "gpu": "NVIDIA RTX 3070 / AMD RX 6700 XT",
            "expected_fps": "60+",
            "use_case": "Araştırma ve geliştirme"
        }
    }
    
    def __init__(self):
        self.performance_history = []
        self.current_metrics = {}
        self.monitoring_active = False
        self.optimization_level = "balanced"  # minimal, balanced, maximum
        
    def start_monitoring(self):
        """Performans izlemeyi başlatır."""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Performans izleme başlatıldı")
    
    def stop_monitoring(self):
        """Performans izlemeyi durdurur."""
        self.monitoring_active = False
        logger.info("Performans izleme durduruldu")
    
    def _monitor_loop(self):
        """Sürekli performans izleme döngüsü."""
        while self.monitoring_active:
            try:
                metrics = self.get_current_performance()
                self.performance_history.append(metrics)
                
                # Son 100 ölçümü tut
                if len(self.performance_history) > 100:
                    self.performance_history.pop(0)
                
                # Performans uyarıları
                self._check_performance_alerts(metrics)
                
                time.sleep(1)  # 1 saniye aralıklarla ölçüm
                
            except Exception as e:
                logger.error(f"Performans izleme hatası: {e}")
                time.sleep(5)
    
    def get_current_performance(self) -> Dict[str, Any]:
        """Mevcut sistem performansını ölçer."""
        try:
            process = psutil.Process()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "memory_percent": process.memory_percent(),
                "system_memory_percent": psutil.virtual_memory().percent,
                "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {},
                "network_io": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {},
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
            
            # GPU bilgisi (eğer mevcut ise)
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    metrics["gpu"] = {
                        "name": gpu.name,
                        "load_percent": gpu.load * 100,
                        "memory_mb": gpu.memoryUsed,
                        "memory_percent": gpu.memoryUtil * 100,
                        "temperature": gpu.temperature
                    }
            except ImportError:
                metrics["gpu"] = None
            
            self.current_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Performans ölçümü hatası: {e}")
            return {}
    
    def benchmark_detection_speed(self, detector, test_video_path: str = None, 
                                 frame_count: int = 100) -> Dict[str, Any]:
        """
        Tespit hızını ölçümler - klinik kullanım için kritik.
        
        Args:
            detector: NystagmusDetector instance'ı
            test_video_path: Test videosu (yoksa sentetik oluşturur)
            frame_count: Test edilecek kare sayısı
            
        Returns:
            dict: Performans metrikleri
        """
        try:
            logger.info(f"Tespit hızı benchmark'u başlatılıyor ({frame_count} kare)")
            
            if test_video_path and cv2.VideoCapture(test_video_path).isOpened():
                cap = cv2.VideoCapture(test_video_path)
            else:
                # Sentetik test verisi oluştur
                cap = self._create_synthetic_test_stream(frame_count)
            
            # Benchmark değişkenleri
            frame_times = []
            detection_times = []
            memory_usage = []
            successful_detections = 0
            
            # Başlangıç bellek kullanımı
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            start_time = time.time()
            
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame işleme zamanını ölç
                frame_start = time.time()
                
                # Tespit yap
                detection_start = time.time()
                left_center, right_center = detector.detect_iris_centers(frame)
                detection_end = time.time()
                
                frame_end = time.time()
                
                # Zamanları kaydet
                frame_times.append((frame_end - frame_start) * 1000)  # ms
                detection_times.append((detection_end - detection_start) * 1000)  # ms
                
                # Başarılı tespit kontrolü
                if left_center is not None and right_center is not None:
                    successful_detections += 1
                
                # Bellek kullanımını kaydet
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_usage.append(current_memory)
                
                # Her 25 karede bir rapor ver
                if (i + 1) % 25 == 0:
                    elapsed = time.time() - start_time
                    current_fps = (i + 1) / elapsed
                    logger.info(f"İşlenen kare: {i+1}/{frame_count}, FPS: {current_fps:.2f}")
            
            total_time = time.time() - start_time
            cap.release()
            
            # Sonuçları hesapla
            avg_fps = frame_count / total_time
            avg_frame_time = np.mean(frame_times)
            avg_detection_time = np.mean(detection_times)
            max_memory = max(memory_usage)
            detection_rate = successful_detections / frame_count
            
            # Performans değerlendirmesi
            performance_grade = self._evaluate_performance({
                "fps": avg_fps,
                "frame_time_ms": avg_frame_time,
                "detection_time_ms": avg_detection_time,
                "memory_mb": max_memory,
                "detection_rate": detection_rate
            })
            
            benchmark_results = {
                "test_info": {
                    "frame_count": frame_count,
                    "total_time_s": total_time,
                    "test_date": datetime.now().isoformat()
                },
                "performance_metrics": {
                    "fps": round(avg_fps, 2),
                    "avg_frame_time_ms": round(avg_frame_time, 2),
                    "avg_detection_time_ms": round(avg_detection_time, 2),
                    "max_frame_time_ms": round(max(frame_times), 2),
                    "min_frame_time_ms": round(min(frame_times), 2),
                    "std_frame_time_ms": round(np.std(frame_times), 2)
                },
                "resource_usage": {
                    "initial_memory_mb": round(initial_memory, 2),
                    "max_memory_mb": round(max_memory, 2),
                    "memory_increase_mb": round(max_memory - initial_memory, 2),
                    "avg_memory_mb": round(np.mean(memory_usage), 2)
                },
                "quality_metrics": {
                    "successful_detections": successful_detections,
                    "detection_rate": round(detection_rate, 3),
                    "failed_detections": frame_count - successful_detections
                },
                "clinical_assessment": {
                    "realtime_capable": avg_fps >= self.PERFORMANCE_TARGETS["minimum_fps"],
                    "meets_target_fps": avg_fps >= self.PERFORMANCE_TARGETS["target_fps"],
                    "low_latency": avg_frame_time <= self.PERFORMANCE_TARGETS["max_latency_ms"],
                    "memory_efficient": max_memory <= self.PERFORMANCE_TARGETS["max_memory_mb"],
                    "reliable_detection": detection_rate >= self.PERFORMANCE_TARGETS["min_accuracy"]
                },
                "performance_grade": performance_grade,
                "recommendations": self._generate_performance_recommendations(performance_grade, {
                    "fps": avg_fps,
                    "memory": max_memory,
                    "detection_rate": detection_rate
                })
            }
            
            logger.info(f"Benchmark tamamlandı: {avg_fps:.2f} FPS, Grade: {performance_grade}")
            return benchmark_results
            
        except Exception as e:
            logger.error(f"Benchmark hatası: {e}")
            return {"error": str(e)}
    
    def _create_synthetic_test_stream(self, frame_count: int):
        """Sentetik test videosu oluşturur."""
        class SyntheticVideoCapture:
            def __init__(self, total_frames):
                self.total_frames = total_frames
                self.current_frame = 0
                
            def read(self):
                if self.current_frame >= self.total_frames:
                    return False, None
                
                # 640x480 sentetik kare oluştur
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                
                # Basit yüz ve göz simülasyonu
                cv2.circle(frame, (320, 240), 80, (200, 150, 100), -1)  # Yüz
                cv2.circle(frame, (300, 220), 10, (255, 255, 255), -1)  # Sol göz
                cv2.circle(frame, (340, 220), 10, (255, 255, 255), -1)  # Sağ göz
                cv2.circle(frame, (300, 220), 3, (0, 0, 0), -1)  # Sol iris
                cv2.circle(frame, (340, 220), 3, (0, 0, 0), -1)  # Sağ iris
                
                self.current_frame += 1
                return True, frame
            
            def release(self):
                pass
        
        return SyntheticVideoCapture(frame_count)
    
    def _evaluate_performance(self, metrics: Dict[str, float]) -> str:
        """Performans notunu hesaplar."""
        score = 0
        max_score = 5
        
        # FPS skoru (25% ağırlık)
        if metrics["fps"] >= 30:
            score += 1
        elif metrics["fps"] >= 25:
            score += 0.8
        elif metrics["fps"] >= 20:
            score += 0.6
        elif metrics["fps"] >= 15:
            score += 0.4
        
        # Latency skoru (20% ağırlık)
        if metrics["frame_time_ms"] <= 30:
            score += 1
        elif metrics["frame_time_ms"] <= 40:
            score += 0.8
        elif metrics["frame_time_ms"] <= 50:
            score += 0.6
        
        # Memory skoru (20% ağırlık)
        if metrics["memory_mb"] <= 1024:
            score += 1
        elif metrics["memory_mb"] <= 2048:
            score += 0.8
        elif metrics["memory_mb"] <= 4096:
            score += 0.6
        
        # Detection rate skoru (20% ağırlık)
        if metrics["detection_rate"] >= 0.95:
            score += 1
        elif metrics["detection_rate"] >= 0.85:
            score += 0.8
        elif metrics["detection_rate"] >= 0.75:
            score += 0.6
        
        # Detection speed skoru (15% ağırlık)
        if metrics["detection_time_ms"] <= 20:
            score += 1
        elif metrics["detection_time_ms"] <= 30:
            score += 0.8
        elif metrics["detection_time_ms"] <= 40:
            score += 0.6
        
        # Not hesapla
        percentage = (score / max_score) * 100
        
        if percentage >= 90:
            return "A+ (Mükemmel)"
        elif percentage >= 80:
            return "A (Çok İyi)"
        elif percentage >= 70:
            return "B (İyi)" 
        elif percentage >= 60:
            return "C (Orta)"
        elif percentage >= 50:
            return "D (Zayıf)"
        else:
            return "F (Yetersiz)"
    
    def _generate_performance_recommendations(self, grade: str, metrics: Dict[str, float]) -> List[str]:
        """Performans önerileri oluşturur."""
        recommendations = []
        
        if metrics["fps"] < 25:
            recommendations.append("🔴 FPS çok düşük - GPU yükseltmesi veya çözünürlük azaltma gerekli")
        elif metrics["fps"] < 30:
            recommendations.append("🟡 FPS optimize edilebilir - MultiThreading kullanın")
        
        if metrics["memory"] > 2048:
            recommendations.append("🔴 Bellek kullanımı yüksek - Batch size azaltın")
        elif metrics["memory"] > 1024:
            recommendations.append("🟡 Bellek optimize edilebilir - Garbage collection ekleyin")
        
        if metrics["detection_rate"] < 0.85:
            recommendations.append("🔴 Tespit oranı düşük - Model parametrelerini kontrol edin")
        
        if not recommendations:
            recommendations.append("✅ Sistem optimal performansta çalışıyor")
        
        return recommendations
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Performans uyarılarını kontrol eder."""
        alerts = []
        
        if metrics.get("cpu_percent", 0) > self.PERFORMANCE_TARGETS["max_cpu_percent"]:
            alerts.append(f"⚠️ Yüksek CPU kullanımı: %{metrics['cpu_percent']:.1f}")
        
        if metrics.get("memory_mb", 0) > self.PERFORMANCE_TARGETS["max_memory_mb"]:
            alerts.append(f"⚠️ Yüksek bellek kullanımı: {metrics['memory_mb']:.1f}MB")
        
        if metrics.get("system_memory_percent", 0) > 90:
            alerts.append(f"⚠️ Sistem belleği kritik: %{metrics['system_memory_percent']:.1f}")
        
        for alert in alerts:
            logger.warning(alert)
    
    def optimize_for_realtime(self, detector):
        """Gerçek zamanlı kullanım için optimizasyon yapar."""
        logger.info("Gerçek zamanlı optimizasyon başlatılıyor...")
        
        optimization_settings = {
            "minimal": {
                "max_faces": 1,
                "detection_confidence": 0.3,
                "tracking_confidence": 0.3,
                "frame_skip": 2
            },
            "balanced": {
                "max_faces": 1, 
                "detection_confidence": 0.5,
                "tracking_confidence": 0.5,
                "frame_skip": 1
            },
            "maximum": {
                "max_faces": 1,
                "detection_confidence": 0.7,
                "tracking_confidence": 0.7,
                "frame_skip": 0
            }
        }
        
        settings = optimization_settings[self.optimization_level]
        
        # Detector ayarlarını güncelle
        if hasattr(detector, 'face_mesh'):
            # MediaPipe parametrelerini güncelle (yeniden başlatma gerekebilir)
            logger.info(f"Optimizasyon seviyesi: {self.optimization_level}")
            logger.info(f"Ayarlar: {settings}")
        
        return settings
    
    def generate_performance_report(self, save_path: str = None) -> Dict[str, Any]:
        """Kapsamlı performans raporu oluşturur."""
        try:
            current_perf = self.get_current_performance()
            
            report = {
                "report_info": {
                    "timestamp": datetime.now().isoformat(),
                    "monitoring_duration_s": len(self.performance_history),
                    "system_info": self._get_system_info()
                },
                "current_performance": current_perf,
                "performance_targets": self.PERFORMANCE_TARGETS,
                "hardware_requirements": self.HARDWARE_REQUIREMENTS,
                "performance_history": self.performance_history[-10:],  # Son 10 ölçüm
                "recommendations": self._generate_system_recommendations(current_perf)
            }
            
            if save_path:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False)
                logger.info(f"Performans raporu kaydedildi: {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Performans raporu oluşturma hatası: {e}")
            return {"error": str(e)}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Sistem bilgilerini toplar."""
        try:
            import platform
            
            return {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_system_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Sistem önerilerini oluşturur.""" 
        recommendations = []
        
        # Memory önerileri
        memory_mb = metrics.get("memory_mb", 0)
        if memory_mb > 2048:
            recommendations.append("💾 32GB RAM yükseltmesi öneriliyor (şu an yüksek kullanım)")
        elif memory_mb > 1024:
            recommendations.append("💾 16GB RAM yeterli, optimize kullanımda")
        
        # CPU önerileri
        cpu_percent = metrics.get("cpu_percent", 0)
        if cpu_percent > 80:
            recommendations.append("🔧 Daha güçlü CPU gerekli (i7/Ryzen 7+)")
        
        # GPU önerileri
        gpu_info = metrics.get("gpu")
        if gpu_info is None:
            recommendations.append("🎮 Dedicated GPU eklemek performansı artırabilir")
        elif gpu_info.get("load_percent", 0) > 90:
            recommendations.append("🎮 GPU yükseltmesi gerekli")
        
        return recommendations

# Factory function
def create_performance_manager() -> PerformanceManager:
    """Performance manager oluştur."""
    return PerformanceManager()

# Kolay kullanım fonksiyonları
def quick_fps_test(detector, frame_count: int = 100) -> float:
    """Hızlı FPS testi."""
    manager = PerformanceManager()
    results = manager.benchmark_detection_speed(detector, frame_count=frame_count)
    return results.get("performance_metrics", {}).get("fps", 0.0)

def check_realtime_capability(detector) -> bool:
    """Gerçek zamanlı kullanım kontrolü."""
    fps = quick_fps_test(detector, 50)
    return fps >= PerformanceManager.PERFORMANCE_TARGETS["minimum_fps"]

if __name__ == "__main__":
    # Test
    manager = PerformanceManager()
    print("🚀 Performans yöneticisi test edildi")
    print(f"Hedef FPS: {manager.PERFORMANCE_TARGETS['minimum_fps']}+")
    print(f"Maks gecikme: {manager.PERFORMANCE_TARGETS['max_latency_ms']}ms")
    
    # Mevcut performansı göster
    current = manager.get_current_performance()
    print(f"\n📊 Mevcut sistem:")
    print(f"CPU: %{current.get('cpu_percent', 0):.1f}")
    print(f"Memory: {current.get('memory_mb', 0):.1f}MB") 