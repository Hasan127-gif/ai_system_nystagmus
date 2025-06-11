"""
Göz İzleme Sistemi - Performans Optimizasyon Yöneticisi

Bu modül, diğer tüm optimizasyon modüllerini koordine eden bir merkezi ara yüz sağlar.
Uygulamanın kalan kısmı bu yönetici üzerinden optimizasyon özelliklerine erişebilir.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import threading
import time

# Optimizasyon modüllerini içe aktar
from optimization.model_quantization import ModelQuantizer
from optimization.frame_optimization import FrameProcessingOptimizer
from optimization.data_structures import EfficientEyeTrackingData, LightweightModelManager

logger = logging.getLogger('eye_tracker.optimization.manager')

class PerformanceOptimizationManager:
    """Performans optimizasyon özelliklerini yöneten sınıf"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Args:
            config: Yapılandırma (None: varsayılan ayarlar)
        """
        self.config = config or {}
        
        # Alt optimizasyon bileşenleri
        self.model_quantizer = None
        self.frame_optimizer = None
        self.data_store = None
        self.model_manager = None
        
        # Optimizasyon durumu
        self.optimization_level = self.config.get('optimization_level', 0)  # 0-3 arası
        self.auto_optimize = self.config.get('auto_optimize', True)
        
        # Performans metrikleri
        self.metrics = {
            'fps': 0.0,
            'frame_drop_rate': 0.0,
            'model_inference_time': 0.0,
            'pipeline_latency': 0.0,
            'memory_usage': 0.0
        }
        
        # Alt bileşenleri başlat
        self._initialize_components()
        
        # Otomatik performans izleyici
        self.monitor_active = False
        self.monitor_thread = None
        
        # Performans optimizasyon önerileri
        self.optimization_suggestions = []
    
    def _initialize_components(self) -> None:
        """Alt optimizasyon bileşenlerini başlat"""
        # Model yöneticisi (hafif bağımlılıklar)
        self.model_manager = LightweightModelManager()
        
        # Frame işleme optimizer'ı
        buffer_size = self.config.get('frame_buffer_size', 5)
        self.frame_optimizer = FrameProcessingOptimizer(buffer_size=buffer_size)
        
        # Verimli veri yapısı
        max_history = self.config.get('data_history_size', 300)
        sampling_rate = self.config.get('sampling_rate', 30)
        self.data_store = EfficientEyeTrackingData(
            max_history=max_history,
            sampling_rate=sampling_rate
        )
        
        # Model quantizer - model yolu varsa başlat
        model_path = self.config.get('model_path', None)
        if model_path and os.path.exists(model_path):
            output_dir = self.config.get('optimized_model_dir', 'optimized_models')
            self.model_quantizer = ModelQuantizer(model_path, output_dir=output_dir)
    
    def optimize_model(self, model_path: str, optimization_type: str = 'auto') -> str:
        """
        ONNX modelini optimize et
        
        Args:
            model_path: ONNX model dosya yolu
            optimization_type: Optimizasyon türü ('int8', 'fp16', 'optimize', 'auto')
            
        Returns:
            Optimize edilmiş model dosya yolu
        """
        if not os.path.exists(model_path):
            logger.error(f"Model dosyası bulunamadı: {model_path}")
            return model_path
        
        # Model optimizer'ı yoksa oluştur
        if self.model_quantizer is None:
            output_dir = self.config.get('optimized_model_dir', 'optimized_models')
            self.model_quantizer = ModelQuantizer(model_path, output_dir=output_dir)
        else:
            # Farklı bir model için güncelle
            self.model_quantizer.original_model_path = model_path
        
        # Optimizasyon türüne göre işlem yap
        if optimization_type == 'int8':
            return self.model_quantizer.quantize_to_int8()
        elif optimization_type == 'fp16':
            return self.model_quantizer.convert_to_fp16()
        elif optimization_type == 'optimize':
            return self.model_quantizer.optimize_for_inference()
        else:
            # Otomatik mod - cihaz özelliklerine göre seç
            try:
                import psutil
                # Bellek 4GB'dan azsa INT8 quantization yap
                mem_gb = psutil.virtual_memory().total / (1024**3)
                if mem_gb < 4:
                    logger.info(f"Düşük bellek tespit edildi ({mem_gb:.1f} GB). INT8 quantization uygulanıyor.")
                    return self.model_quantizer.quantize_to_int8()
                else:
                    logger.info(f"Yeterli bellek tespit edildi ({mem_gb:.1f} GB). FP16 dönüşümü uygulanıyor.")
                    return self.model_quantizer.convert_to_fp16()
            except:
                # Bellek bilgisi alınamazsa FP16 kullan
                logger.info("Sistem bilgisi alınamadı. FP16 dönüşümü uygulanıyor.")
                return self.model_quantizer.convert_to_fp16()
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> np.ndarray:
        """
        Frame'i optimize et ve işle
        
        Args:
            frame: İşlenecek görüntü
            timestamp: Frame zaman damgası
            
        Returns:
            İşlenmiş görüntü
        """
        # Frame optimizer kullanarak işleme uygula
        return self.frame_optimizer.process_frame(frame, timestamp)
    
    def add_tracking_sample(self, 
                          timestamp: float, 
                          gaze_x: float, 
                          gaze_y: float, 
                          pupil_size: Optional[float] = None,
                          confidence: Optional[float] = None) -> None:
        """
        Yeni bir göz izleme örneği ekle
        
        Args:
            timestamp: Zaman damgası
            gaze_x: X koordinatı
            gaze_y: Y koordinatı
            pupil_size: Göz bebeği boyutu
            confidence: Güven değeri
        """
        self.data_store.add_sample(
            timestamp=timestamp,
            gaze_x=gaze_x,
            gaze_y=gaze_y,
            pupil_size=pupil_size,
            confidence=confidence
        )
    
    def get_tracking_window(self, seconds: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Göz izleme verilerinin bir penceresini al
        
        Args:
            seconds: Pencere süresi (saniye)
            
        Returns:
            Veri penceresi
        """
        return self.data_store.get_window(seconds=seconds)
    
    def get_tracking_stats(self, window_seconds: float = 1.0) -> Dict[str, float]:
        """
        Göz izleme istatistiklerini hesapla
        
        Args:
            window_seconds: Pencere süresi (saniye)
            
        Returns:
            İstatistik sözlüğü
        """
        return self.data_store.compute_statistics(window_seconds=window_seconds)
    
    def get_model_loader(self, model_path: str, use_gpu: bool = True) -> Any:
        """
        Hafif model yükleyicisi oluştur
        
        Args:
            model_path: Model dosya yolu
            use_gpu: GPU kullanılsın mı
            
        Returns:
            Model yükleyici
        """
        return self.model_manager.get_model_loader(model_path, use_gpu)
    
    def set_frame_strategy(self, strategy_name: str) -> bool:
        """
        Frame işleme stratejisini ayarla
        
        Args:
            strategy_name: Strateji adı
            
        Returns:
            Başarı durumu
        """
        return self.frame_optimizer.set_active_strategy(strategy_name)
    
    def set_optimization_level(self, level: int) -> None:
        """
        Genel optimizasyon seviyesini ayarla
        
        Args:
            level: Optimizasyon seviyesi (0-3)
                0: Optimizasyon yok
                1: Hafif optimizasyon
                2: Orta düzey optimizasyon
                3: Maksimum optimizasyon
        """
        level = max(0, min(3, level))
        if level == self.optimization_level:
            return
            
        self.optimization_level = level
        logger.info(f"Optimizasyon seviyesi değiştirildi: {level}")
        
        # Frame işleme stratejisini güncelle
        if level == 0:
            self.set_frame_strategy("none")
        elif level == 1:
            self.set_frame_strategy("low_res_50")
        elif level == 2:
            self.set_frame_strategy("low_res_25")
        elif level == 3:
            self.set_frame_strategy("skip_2")
    
    def update_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Performans metriklerini güncelle
        
        Args:
            metrics: Metrik sözlüğü
        """
        self.metrics.update(metrics)
        
        # Otomatik optimizasyon aktifse, performansa göre ayarla
        if self.auto_optimize:
            self._adjust_optimization_based_on_performance()
    
    def _adjust_optimization_based_on_performance(self) -> None:
        """Performans metriklerine göre optimizasyon seviyesini otomatik ayarla"""
        fps = self.metrics.get('fps', 0)
        drop_rate = self.metrics.get('frame_drop_rate', 0)
        inference_time = self.metrics.get('model_inference_time', 0)
        
        # FPS düşükse optimizasyon seviyesini artır
        if fps < 15 and drop_rate > 10:
            new_level = min(3, self.optimization_level + 1)
            if new_level != self.optimization_level:
                logger.info(f"Performans düşük (FPS: {fps:.1f}, Drop: {drop_rate:.1f}%). "
                          f"Optimizasyon seviyesi artırılıyor: {new_level}")
                self.set_optimization_level(new_level)
                
                # Optimizasyon önerisi ekle
                if new_level == 3:
                    self.optimization_suggestions.append(
                        "Model quantization önerilir. 'optimize_model()' çağrısıyla modeli optimize edin."
                    )
        
        # Performans iyiyse ve yüksek optimizasyon varsa, azalt
        elif fps > 25 and drop_rate < 2 and self.optimization_level > 0:
            new_level = max(0, self.optimization_level - 1)
            logger.info(f"Performans iyi (FPS: {fps:.1f}, Drop: {drop_rate:.1f}%). "
                      f"Optimizasyon seviyesi azaltılıyor: {new_level}")
            self.set_optimization_level(new_level)
    
    def get_optimization_suggestions(self) -> List[str]:
        """
        Optimizasyon önerilerini al
        
        Returns:
            Öneri listesi
        """
        # Önerileri temizle ve döndür
        suggestions = self.optimization_suggestions.copy()
        self.optimization_suggestions = []
        return suggestions
    
    def start_monitoring(self, interval: float = 5.0) -> bool:
        """
        Performans izlemeyi başlat
        
        Args:
            interval: İzleme aralığı (saniye)
            
        Returns:
            Başarı durumu
        """
        if self.monitor_active:
            return False
            
        self.monitor_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Performans izleme başlatıldı (interval: {interval:.1f}s)")
        return True
    
    def stop_monitoring(self) -> None:
        """Performans izlemeyi durdur"""
        self.monitor_active = False
        logger.info("Performans izleme durduruldu")
    
    def _monitoring_loop(self, interval: float) -> None:
        """
        Performans izleme döngüsü
        
        Args:
            interval: İzleme aralığı (saniye)
        """
        while self.monitor_active:
            try:
                # Sistem metriklerini topla
                self._collect_system_metrics()
                
                # Optimize edilmemiş model varsa öneri ekle
                if (
                    self.model_quantizer and 
                    not os.path.exists(os.path.join(self.model_quantizer.output_dir, 
                                                  f"{self.model_quantizer.model_stem}_int8.onnx"))
                ):
                    self.optimization_suggestions.append(
                        "Model henüz optimize edilmemiş. 'optimize_model()' çağrısıyla modeli optimize edin."
                    )
                
                # Otomatik optimizasyon ayarı
                if self.auto_optimize:
                    self._adjust_optimization_based_on_performance()
                    
            except Exception as e:
                logger.error(f"Performans izleme hatası: {e}")
                
            # Bekleme
            time.sleep(interval)
    
    def _collect_system_metrics(self) -> None:
        """Sistem performans metriklerini topla"""
        try:
            # Psutil mevcutsa bellek kullanımını al
            import psutil
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics['memory_usage'] = memory_mb
            
            # Yüksek bellek kullanımı uyarısı
            if memory_mb > 500:  # 500 MB üzerinde uyarı ver
                logger.warning(f"Yüksek bellek kullanımı: {memory_mb:.1f} MB")
                self.optimization_suggestions.append(
                    f"Yüksek bellek kullanımı ({memory_mb:.1f} MB). "
                    f"Model quantization ve veri buffer boyutunun azaltılması önerilir."
                )
        except:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Tüm optimizasyon istatistiklerini al
        
        Returns:
            İstatistik sözlüğü
        """
        stats = {
            'optimization_level': self.optimization_level,
            'auto_optimize': self.auto_optimize,
            'performance_metrics': self.metrics,
            'frame_processing': self.frame_optimizer.get_stats() if self.frame_optimizer else {},
            'model_backends': self.model_manager.get_available_backends() if self.model_manager else [],
            'preferred_backend': self.model_manager.preferred_backend if self.model_manager else None
        }
        
        return stats 