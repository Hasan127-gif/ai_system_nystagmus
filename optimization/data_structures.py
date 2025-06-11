"""
Göz İzleme Sistemi - Optimize Veri Yapıları ve Hafif Model Bağımlılıkları

Bu modül, göz izleme sisteminde performans iyileştirmeleri için:
1. Gerçek zamanlı analiz için optimize edilmiş veri yapıları
2. Daha hafif model kütüphane bağımlılıkları 
sunar.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Callable, Iterable
from collections import deque
import threading
import os

logger = logging.getLogger('eye_tracker.optimization.data')

class EfficientEyeTrackingData:
    """Göz izleme verisi için optimize edilmiş veri yapısı"""
    
    def __init__(self, max_history: int = 300, sampling_rate: float = 30):
        """
        Args:
            max_history: Saklanacak maksimum örnek sayısı
            sampling_rate: Örnekleme hızı (Hz)
        """
        # İzleme verileri için sabit boyutlu numpy dizileri 
        # (Liste/deque'den daha az bellek ve daha hızlı)
        self.max_history = max_history
        self.sampling_rate = sampling_rate
        
        # Veri dizileri (numpy sabit boyutlu diziler - daha verimli)
        self.timestamps = np.zeros(max_history, dtype=np.float64)
        self.gaze_x = np.zeros(max_history, dtype=np.float32)
        self.gaze_y = np.zeros(max_history, dtype=np.float32)
        self.pupil_size = np.zeros(max_history, dtype=np.float32)
        self.confidence = np.zeros(max_history, dtype=np.float32)
        
        # Ek özellikler
        self.velocity = np.zeros(max_history, dtype=np.float32)  # Hız
        self.acceleration = np.zeros(max_history, dtype=np.float32)  # İvme
        
        # Veri indeksleri
        self.current_index = 0
        self.buffer_full = False
        
        # Thread güvenliği
        self._lock = threading.Lock()
    
    def add_sample(self, 
                  timestamp: float, 
                  gaze_x: float, 
                  gaze_y: float,
                  pupil_size: Optional[float] = None,
                  confidence: Optional[float] = None) -> None:
        """
        Yeni bir göz izleme örneği ekle
        
        Args:
            timestamp: Zaman damgası (saniye)
            gaze_x: X koordinatı
            gaze_y: Y koordinatı
            pupil_size: Göz bebeği boyutu (None: veri yok)
            confidence: Güven değeri (0-1 arası, None: veri yok)
        """
        with self._lock:
            # Verileri ekle
            self.timestamps[self.current_index] = timestamp
            self.gaze_x[self.current_index] = gaze_x
            self.gaze_y[self.current_index] = gaze_y
            
            # Opsiyonel değerler
            if pupil_size is not None:
                self.pupil_size[self.current_index] = pupil_size
            else:
                self.pupil_size[self.current_index] = np.nan
                
            if confidence is not None:
                self.confidence[self.current_index] = confidence
            else:
                self.confidence[self.current_index] = np.nan
            
            # Hız ve ivme hesapla (en az 2 örnek için)
            if self.buffer_full or self.current_index > 0:
                prev_idx = (self.current_index - 1) % self.max_history
                dt = self.timestamps[self.current_index] - self.timestamps[prev_idx]
                
                if dt > 0:
                    # Öklid mesafesi
                    dx = self.gaze_x[self.current_index] - self.gaze_x[prev_idx]
                    dy = self.gaze_y[self.current_index] - self.gaze_y[prev_idx]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Hız (piksel/s)
                    velocity = distance / dt
                    self.velocity[self.current_index] = velocity
                    
                    # İvme (en az 3 örnek için)
                    if self.buffer_full or self.current_index > 1:
                        prev_vel_idx = (self.current_index - 2) % self.max_history
                        prev_vel = self.velocity[prev_idx]
                        accel = (velocity - prev_vel) / dt
                        self.acceleration[self.current_index] = accel
            
            # İndeksi güncelle
            self.current_index = (self.current_index + 1) % self.max_history
            
            if self.current_index == 0:
                self.buffer_full = True
    
    def get_window(self, window_size: Optional[int] = None, seconds: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Belirli bir zaman penceresi için verileri al
        
        Args:
            window_size: Pencere büyüklüğü (örnek sayısı)
            seconds: Saniye cinsinden pencere boyutu (window_size'a göre öncelikli)
            
        Returns:
            Veri penceresi: {özellik: değerler}
        """
        with self._lock:
            if self.current_index == 0 and not self.buffer_full:
                return {  # Hiç veri yoksa boş sözlük döndür
                    'timestamps': np.array([]),
                    'gaze_x': np.array([]),
                    'gaze_y': np.array([]),
                    'pupil_size': np.array([]),
                    'confidence': np.array([]),
                    'velocity': np.array([]),
                    'acceleration': np.array([])
                }
            
            # Pencere boyutunu hesapla
            if seconds is not None:
                # Saniye cinsinden pencere
                sample_count = min(int(seconds * self.sampling_rate), self.max_history)
            elif window_size is not None:
                # Örnek sayısı cinsinden pencere
                sample_count = min(window_size, self.max_history)
            else:
                # Tüm geçmiş
                sample_count = self.max_history if self.buffer_full else self.current_index
            
            # Başlangıç indeksini hesapla
            if not self.buffer_full:
                start_idx = max(0, self.current_index - sample_count)
                indices = np.arange(start_idx, self.current_index)
            else:
                # Buffer doluysa, dairesel şekilde indeksle
                start_idx = (self.current_index - sample_count) % self.max_history
                if start_idx < self.current_index:
                    indices = np.arange(start_idx, self.current_index)
                else:
                    part1 = np.arange(start_idx, self.max_history)
                    part2 = np.arange(0, self.current_index)
                    indices = np.concatenate([part1, part2])
            
            # Verileri döndür
            return {
                'timestamps': self.timestamps[indices],
                'gaze_x': self.gaze_x[indices],
                'gaze_y': self.gaze_y[indices],
                'pupil_size': self.pupil_size[indices],
                'confidence': self.confidence[indices],
                'velocity': self.velocity[indices],
                'acceleration': self.acceleration[indices]
            }
    
    def compute_statistics(self, window_seconds: float = 1.0) -> Dict[str, float]:
        """
        Belirli bir pencere için istatistikleri hesapla
        
        Args:
            window_seconds: Pencere boyutu (saniye)
            
        Returns:
            İstatistik sözlüğü
        """
        # Veri penceresini al
        window = self.get_window(seconds=window_seconds)
        
        # Veri yoksa boş sözlük döndür
        if len(window['timestamps']) == 0:
            return {}
        
        # Temel istatistikler
        mean_velocity = np.nanmean(window['velocity'])
        std_velocity = np.nanstd(window['velocity'])
        max_velocity = np.nanmax(window['velocity'])
        
        # Güven filtresi (güven değeri yüksek örneklerle çalış)
        confident_samples = window['confidence'] >= 0.5
        confident_gaze_x = window['gaze_x'][confident_samples] if np.any(confident_samples) else window['gaze_x']
        confident_gaze_y = window['gaze_y'][confident_samples] if np.any(confident_samples) else window['gaze_y']
        
        # Dispersiyon (dağılım - göz sabitlemesi için önemli)
        dispersion_x = np.nanmax(confident_gaze_x) - np.nanmin(confident_gaze_x)
        dispersion_y = np.nanmax(confident_gaze_y) - np.nanmin(confident_gaze_y)
        dispersion = np.sqrt(dispersion_x**2 + dispersion_y**2)
        
        return {
            'samples': len(window['timestamps']),
            'duration': window['timestamps'][-1] - window['timestamps'][0],
            'mean_velocity': mean_velocity,
            'std_velocity': std_velocity,
            'max_velocity': max_velocity,
            'dispersion': dispersion,
            'mean_pupil_size': np.nanmean(window['pupil_size']),
            'std_pupil_size': np.nanstd(window['pupil_size']),
            'mean_confidence': np.nanmean(window['confidence'])
        }
    
    def clear(self) -> None:
        """Tüm verileri temizle"""
        with self._lock:
            self.current_index = 0
            self.buffer_full = False

class LightweightModelManager:
    """Hafif model kütüphane bağımlılıklarını yöneten sınıf"""
    
    def __init__(self):
        # Kullanılabilen işleyiciler
        self.available_backends = []
        self.preferred_backend = None
        
        # Kullanılabilir backend'leri tespit et
        self._detect_backends()
    
    def _detect_backends(self) -> None:
        """Kurulu backend'leri tespit et"""
        # ONNX Runtime
        try:
            import onnxruntime
            version = onnxruntime.__version__
            providers = onnxruntime.get_available_providers()
            
            self.available_backends.append({
                'name': 'onnxruntime',
                'version': version,
                'providers': providers,
                'has_gpu': 'CUDAExecutionProvider' in providers,
                'module': onnxruntime
            })
            
            if not self.preferred_backend:
                self.preferred_backend = 'onnxruntime'
                
            logger.info(f"ONNX Runtime bulundu: v{version}, Sağlayıcılar: {', '.join(providers)}")
        except ImportError:
            logger.debug("ONNX Runtime bulunamadı")
        
        # OpenCV DNN
        try:
            import cv2
            version = cv2.__version__
            
            # OpenCV'nin CUDA desteği var mı kontrol et
            has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 if hasattr(cv2, 'cuda') else False
            
            self.available_backends.append({
                'name': 'opencv',
                'version': version,
                'has_gpu': has_cuda,
                'module': cv2
            })
            
            if not self.preferred_backend:
                self.preferred_backend = 'opencv'
                
            logger.info(f"OpenCV bulundu: v{version}, CUDA: {'Var' if has_cuda else 'Yok'}")
        except ImportError:
            logger.debug("OpenCV bulunamadı")
        
        # TensorFlow Lite
        try:
            import tflite_runtime.interpreter as tflite
            version = "Bilinmiyor"  # TFLite runtime'da sürüm bilgisi kolayca alınamıyor
            
            self.available_backends.append({
                'name': 'tflite',
                'version': version,
                'has_gpu': False,  # Varsayılan olarak GPU desteği bilinmiyor
                'module': tflite
            })
            
            if not self.preferred_backend:
                self.preferred_backend = 'tflite'
                
            logger.info(f"TensorFlow Lite bulundu")
        except ImportError:
            try:
                import tensorflow as tf
                version = tf.__version__
                has_gpu = tf.test.is_gpu_available() if hasattr(tf.test, 'is_gpu_available') else False
                
                self.available_backends.append({
                    'name': 'tensorflow',
                    'version': version,
                    'has_gpu': has_gpu,
                    'module': tf
                })
                
                if not self.preferred_backend:
                    self.preferred_backend = 'tensorflow'
                    
                logger.info(f"TensorFlow bulundu: v{version}, GPU: {'Var' if has_gpu else 'Yok'}")
            except ImportError:
                logger.debug("TensorFlow/TFLite bulunamadı")
    
    def get_available_backends(self) -> List[Dict[str, Any]]:
        """
        Kullanılabilir tüm modelleri ve özelliklerini al
        
        Returns:
            Backend listesi
        """
        return self.available_backends
    
    def set_preferred_backend(self, backend_name: str) -> bool:
        """
        Tercih edilen backend'i ayarla
        
        Args:
            backend_name: Backend adı
            
        Returns:
            Başarı durumu
        """
        for backend in self.available_backends:
            if backend['name'] == backend_name:
                self.preferred_backend = backend_name
                logger.info(f"Tercih edilen model backend: {backend_name}")
                return True
        
        logger.warning(f"Backend bulunamadı: {backend_name}")
        return False
    
    def get_model_loader(self, model_path: str, use_gpu: bool = True) -> Any:
        """
        Belirtilen model için uygun bir model yükleyici oluştur
        
        Args:
            model_path: Model dosya yolu
            use_gpu: GPU kullanılsın mı
            
        Returns:
            Model yükleyici nesnesi veya None
        """
        # Model uzantısını kontrol et
        ext = os.path.splitext(model_path)[1].lower()
        
        if self.preferred_backend == 'onnxruntime' and ext == '.onnx':
            # ONNX model için ONNX Runtime kullan
            for backend in self.available_backends:
                if backend['name'] == 'onnxruntime':
                    from models.model_loader import ONNXModelLoader
                    return ONNXModelLoader(model_path, use_gpu=use_gpu)
        
        elif self.preferred_backend == 'opencv':
            # OpenCV DNN kullanan bir yükleyici
            for backend in self.available_backends:
                if backend['name'] == 'opencv':
                    return self._create_opencv_model_loader(model_path, use_gpu)
        
        elif self.preferred_backend == 'tflite' and ext == '.tflite':
            # TensorFlow Lite kullanan bir yükleyici
            for backend in self.available_backends:
                if backend['name'] == 'tflite':
                    return self._create_tflite_model_loader(model_path, use_gpu)
        
        # Varsayılan: İlk bulduğumuz backend'i kullan
        for backend in self.available_backends:
            if backend['name'] == 'onnxruntime' and ext == '.onnx':
                from models.model_loader import ONNXModelLoader
                return ONNXModelLoader(model_path, use_gpu=use_gpu)
            elif backend['name'] == 'opencv':
                return self._create_opencv_model_loader(model_path, use_gpu)
            elif backend['name'] == 'tflite' and ext == '.tflite':
                return self._create_tflite_model_loader(model_path, use_gpu)
        
        # Hiçbir backend bulunamadı
        logger.error(f"Model için uygun backend bulunamadı: {model_path}")
        return None
    
    def _create_opencv_model_loader(self, model_path: str, use_gpu: bool) -> Any:
        """OpenCV tabanlı model yükleyici oluştur"""
        # Bu kısım uygulamaya göre genişletilebilir
        logger.info(f"OpenCV DNN model yükleyici oluşturuluyor: {model_path}")
        return None
    
    def _create_tflite_model_loader(self, model_path: str, use_gpu: bool) -> Any:
        """TFLite tabanlı model yükleyici oluştur"""
        # Bu kısım uygulamaya göre genişletilebilir
        logger.info(f"TensorFlow Lite model yükleyici oluşturuluyor: {model_path}")
        return None 