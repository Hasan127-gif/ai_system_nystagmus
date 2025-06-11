"""
Göz İzleme Sistemi - Frame Optimizasyonu

Bu modül, frame önbelleğe alma ve işleme stratejilerini optimize ederek
düşük güçlü cihazlarda bile yüksek performans sağlamayı amaçlar.
"""

import numpy as np
import cv2
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Deque
from collections import deque
from abc import ABC, abstractmethod

logger = logging.getLogger('eye_tracker.optimization.frame')

class FrameBuffer:
    """Frame önbelleğe alma ve yönetim sınıfı"""
    
    def __init__(self, buffer_size: int = 5, preprocessing_fn: Optional[Callable] = None):
        """
        Args:
            buffer_size: Önbellek boyutu (frame sayısı)
            preprocessing_fn: Her frame için uygulanacak ön işleme fonksiyonu
        """
        self.buffer_size = max(1, buffer_size)
        self.preprocessing_fn = preprocessing_fn
        self.frame_buffer = deque(maxlen=self.buffer_size)
        self.timestamps = deque(maxlen=self.buffer_size)
        self.processed_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        
        # Performans metrikleri
        self.processing_times = deque(maxlen=100)
        self.dropped_frames = 0
        self.total_frames = 0
    
    def add_frame(self, frame: np.ndarray, timestamp: float = None) -> bool:
        """
        Frame'i önbelleğe ekle ve işle
        
        Args:
            frame: Eklenecek görüntü
            timestamp: Frame zaman damgası (None: otomatik)
            
        Returns:
            Ekleme başarısı
        """
        if frame is None:
            return False
        
        self.total_frames += 1
        
        # Zaman damgası yoksa oluştur
        if timestamp is None:
            timestamp = time.time()
        
        # Buffer dolu mu?
        if len(self.frame_buffer) >= self.buffer_size:
            self.dropped_frames += 1
        
        # Önişleme uygula
        processed_frame = frame
        processing_start = time.time()
        
        if self.preprocessing_fn is not None:
            try:
                processed_frame = self.preprocessing_fn(frame)
            except Exception as e:
                logger.error(f"Frame önişleme hatası: {e}")
                processed_frame = frame
        
        processing_time = (time.time() - processing_start) * 1000
        self.processing_times.append(processing_time)
        
        # Thread-safe ekleme
        with self.buffer_lock:
            self.frame_buffer.append(frame)
            self.timestamps.append(timestamp)
            self.processed_buffer.append(processed_frame)
        
        return True
    
    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        En son frame'i al
        
        Returns:
            (frame, timestamp) çifti veya (None, None)
        """
        with self.buffer_lock:
            if not self.frame_buffer:
                return None, None
            return self.frame_buffer[-1], self.timestamps[-1]
    
    def get_latest_processed(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        """
        En son işlenmiş frame'i al
        
        Returns:
            (processed_frame, timestamp) çifti veya (None, None)
        """
        with self.buffer_lock:
            if not self.processed_buffer:
                return None, None
            return self.processed_buffer[-1], self.timestamps[-1]
    
    def get_frame_batch(self, batch_size: int = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Frame'leri toplu olarak al
        
        Args:
            batch_size: İstenilen batch boyutu (None: tüm buffer)
            
        Returns:
            ([frame1, frame2, ...], [timestamp1, timestamp2, ...]) çifti
        """
        with self.buffer_lock:
            if not self.frame_buffer:
                return [], []
            
            if batch_size is None or batch_size >= len(self.frame_buffer):
                return list(self.frame_buffer), list(self.timestamps)
            
            # Son n frame
            frames = list(self.frame_buffer)[-batch_size:]
            times = list(self.timestamps)[-batch_size:]
            return frames, times
    
    def clear(self) -> None:
        """Önbelleği temizle"""
        with self.buffer_lock:
            self.frame_buffer.clear()
            self.timestamps.clear()
            self.processed_buffer.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Performans istatistiklerini al
        
        Returns:
            İstatistik sözlüğü
        """
        avg_processing_time = 0
        if self.processing_times:
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        drop_rate = 0
        if self.total_frames > 0:
            drop_rate = (self.dropped_frames / self.total_frames) * 100
            
        return {
            "buffer_size": self.buffer_size,
            "current_frames": len(self.frame_buffer),
            "avg_processing_time_ms": avg_processing_time,
            "dropped_frames": self.dropped_frames,
            "total_frames": self.total_frames,
            "drop_rate_percent": drop_rate
        }

class FrameProcessingStrategy(ABC):
    """Frame işleme stratejisi temel sınıfı"""
    
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame'i işle
        
        Args:
            frame: İşlenecek görüntü
            
        Returns:
            İşlenmiş görüntü
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """
        Strateji adını al
        
        Returns:
            Strateji adı
        """
        pass

class LowResolutionStrategy(FrameProcessingStrategy):
    """Düşük çözünürlüklü işleme stratejisi"""
    
    def __init__(self, scale_factor: float = 0.5):
        """
        Args:
            scale_factor: Ölçek faktörü (0.1-1.0 arası)
        """
        self.scale_factor = max(0.1, min(1.0, scale_factor))
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame'i işle - düşük çözünürlüğe ölçekle
        
        Args:
            frame: İşlenecek görüntü
            
        Returns:
            İşlenmiş görüntü
        """
        if frame is None:
            return None
            
        # Boyutu hesapla
        h, w = frame.shape[:2]
        new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
        
        # Küçült
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    def get_name(self) -> str:
        """Strateji adını al"""
        return f"LowRes_{int(self.scale_factor*100)}"

class FrameSkippingStrategy(FrameProcessingStrategy):
    """Frame atlama stratejisi"""
    
    def __init__(self, process_every_n: int = 2):
        """
        Args:
            process_every_n: Her n frame'den birini işle
        """
        self.process_every_n = max(1, process_every_n)
        self.counter = 0
        self.last_processed = None
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame'i işle - her n frame'den birini seç
        
        Args:
            frame: İşlenecek görüntü
            
        Returns:
            İşlenmiş görüntü veya son işlenen görüntü
        """
        if frame is None:
            return self.last_processed
        
        self.counter += 1
        if self.counter >= self.process_every_n:
            self.counter = 0
            self.last_processed = frame.copy()
            
        return self.last_processed
    
    def get_name(self) -> str:
        """Strateji adını al"""
        return f"Skip_{self.process_every_n}"

class ROIProcessingStrategy(FrameProcessingStrategy):
    """Sadece ilgilenilen bölgeyi (ROI) işle"""
    
    def __init__(self, roi_func: Callable[[np.ndarray], Tuple[int, int, int, int]]):
        """
        Args:
            roi_func: ROI belirleyen fonksiyon (frame -> x, y, w, h)
        """
        self.roi_func = roi_func
        self.last_roi = None  # Son geçerli ROI
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame'i işle - sadece ROI kısmını döndür
        
        Args:
            frame: İşlenecek görüntü
            
        Returns:
            ROI kısmını içeren görüntü
        """
        if frame is None:
            return None
        
        try:
            # ROI'yi hesapla
            x, y, w, h = self.roi_func(frame)
            if x >= 0 and y >= 0 and w > 0 and h > 0:
                self.last_roi = (x, y, w, h)
        except Exception as e:
            logger.debug(f"ROI hesaplama hatası: {e}")
            
        # Geçerli bir ROI varsa kes
        if self.last_roi:
            x, y, w, h = self.last_roi
            return frame[y:y+h, x:x+w]
            
        return frame
    
    def get_name(self) -> str:
        """Strateji adını al"""
        return "ROI"

class FrameProcessingOptimizer:
    """Frame işleme optimizasyon yöneticisi"""
    
    def __init__(self, buffer_size: int = 5):
        """
        Args:
            buffer_size: Frame önbellek boyutu
        """
        self.buffer = FrameBuffer(buffer_size=buffer_size)
        self.strategies = {}
        self.active_strategy = None
        
        # Varsayılan stratejileri kaydet
        self.register_strategy("none", FrameProcessingStrategy())
        self.register_strategy("low_res_50", LowResolutionStrategy(0.5))
        self.register_strategy("low_res_25", LowResolutionStrategy(0.25))
        self.register_strategy("skip_2", FrameSkippingStrategy(2))
        self.register_strategy("skip_3", FrameSkippingStrategy(3))
    
    def register_strategy(self, name: str, strategy: FrameProcessingStrategy) -> None:
        """
        Yeni strateji kaydet
        
        Args:
            name: Strateji adı
            strategy: Strateji nesnesi
        """
        self.strategies[name] = strategy
        if self.active_strategy is None:
            self.active_strategy = name
    
    def set_active_strategy(self, name: str) -> bool:
        """
        Aktif stratejiyi değiştir
        
        Args:
            name: Strateji adı
            
        Returns:
            Başarı durumu
        """
        if name in self.strategies:
            self.active_strategy = name
            logger.info(f"Frame işleme stratejisi değiştirildi: {name}")
            return True
        return False
    
    def process_frame(self, frame: np.ndarray, timestamp: float = None) -> np.ndarray:
        """
        Frame'i işle ve önbelleğe ekle
        
        Args:
            frame: İşlenecek görüntü
            timestamp: Frame zaman damgası
            
        Returns:
            İşlenmiş görüntü
        """
        if frame is None:
            return None
        
        # Aktif stratejiyi kullan
        if self.active_strategy and self.active_strategy in self.strategies:
            processed = self.strategies[self.active_strategy].process(frame)
        else:
            processed = frame
        
        # Önbelleğe ekle
        self.buffer.add_frame(processed, timestamp)
        
        return processed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Performans istatistiklerini al
        
        Returns:
            İstatistik sözlüğü
        """
        stats = self.buffer.get_stats()
        stats["active_strategy"] = self.active_strategy
        stats["available_strategies"] = list(self.strategies.keys())
        
        return stats 