"""
Göz Takip ve Hareket Analizi Modülü
-----------------------------------
Bu modül, göz hareketlerini takip etmek ve nistagmus tespiti yapmak için
gerekli işlevleri ve yardımcı sınıfları sağlar.
"""

import cv2
import numpy as np
import logging
import time
import traceback
from typing import Dict, List, Tuple, Union, Optional
from scipy.signal import find_peaks, butter, filtfilt
from collections import deque

# MediaPipe'ı kontrol et
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe bulunamadı. Yüklemek için: pip install mediapipe")

# Logging
logger = logging.getLogger('eye_tracking')

class EyeTracker:
    """
    Göz takip ve analiz işlevleri sağlayan ana sınıf.
    MediaPipe Face Mesh kullanarak gözleri tespit eder,
    göz hareketlerini izler ve nistagmus analizi yapar.
    """
    
    def __init__(self, config=None):
        """
        Args:
            config (dict, optional): Yapılandırma parametreleri
        """
        self.config = config or {}
        
        # Varsayılan parametreler
        self.default_config = {
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "max_num_faces": 1,
            "refine_landmarks": True,
            "smoothing_window": 5,
            "sample_rate": 30.0,  # FPS
            "bandpass_freq_low": 0.5,
            "bandpass_freq_high": 20.0,
            "pixel_to_degree_ratio": 0.05,  # derece/piksel
            "spv_threshold": 20.0,  # derece/saniye
            "nistagmus_freq_min": 2.0,  # Hz
            "nistagmus_freq_max": 10.0,  # Hz
            "history_size": 90,  # 30fps'de 3 saniye
            "horizontal_weight": 1.0,
            "vertical_weight": 1.0,
            "rotation_weight": 0.5,
        }
        
        # Eksik yapılandırma değerlerini varsayılanlarla doldur
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        # MediaPipe'ı başlat (varsa)
        self._setup_face_mesh()
        
        # Göz izleme veri yapıları
        self.left_eye_positions_x = deque(maxlen=self.config["history_size"])
        self.left_eye_positions_y = deque(maxlen=self.config["history_size"])
        self.right_eye_positions_x = deque(maxlen=self.config["history_size"])
        self.right_eye_positions_y = deque(maxlen=self.config["history_size"])
        self.timestamps = deque(maxlen=self.config["history_size"])
        
        # Son hesaplanan nistagmus olasılığı
        self.nistagmus_probability = 0.0
        self.last_analysis_results = {}
        self.frame_count = 0
    
    def _setup_face_mesh(self):
        """MediaPipe Face Mesh dedektörünü kurma"""
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=self.config["max_num_faces"],
                    refine_landmarks=self.config["refine_landmarks"],
                    min_detection_confidence=self.config["min_detection_confidence"],
                    min_tracking_confidence=self.config["min_tracking_confidence"]
                )
                logger.info("MediaPipe Face Mesh başarıyla başlatıldı")
                self.is_initialized = True
            except Exception as e:
                logger.error(f"MediaPipe başlatma hatası: {str(e)}")
                self.is_initialized = False
        else:
            logger.warning("MediaPipe bulunamadı, alternatif yöntem kullanılacak")
            self.is_initialized = False
    
    def process_frame(self, frame):
        """
        Verilen kareyi işleme alır ve göz takibi yapar
        
        Args:
            frame: İşlenecek video karesi
            
        Returns:
            dict: İşleme sonuçları
        """
        if frame is None:
            return {"success": False, "error": "Kare bulunamadı"}
            
        if not self.is_initialized and MEDIAPIPE_AVAILABLE:
            self._setup_face_mesh()
        
        if not self.is_initialized:
            return {"success": False, "error": "MediaPipe başlatılamadı"}
            
        try:
            # BGR -> RGB dönüşümü
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yüz işaretlerini tespit et
            results = self.face_mesh.process(rgb_frame)
            
            # Kare sayacını arttır
            self.frame_count += 1
            
            # Şu anki zamanı kaydet
            current_time = time.time()
            
            # Yüz tespit edildi mi?
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Göz bölgelerini çıkar
                left_eye, right_eye = self._extract_eye_landmarks(face_landmarks, frame.shape)
                
                # Göz pozisyonlarını kaydet
                self.left_eye_positions_x.append(left_eye[0])
                self.left_eye_positions_y.append(left_eye[1])
                self.right_eye_positions_x.append(right_eye[0])
                self.right_eye_positions_y.append(right_eye[1])
                self.timestamps.append(current_time)
                
                # Her 10 karede bir analiz yap (performans için)
                if self.frame_count % 10 == 0 and len(self.timestamps) >= 30:
                    self._analyze_eye_movements()
                
                return {
                    "success": True,
                    "left_eye": left_eye,
                    "right_eye": right_eye,
                    "nistagmus_probability": self.nistagmus_probability,
                    "analysis_results": self.last_analysis_results if self.frame_count % 10 == 0 else {},
                    "face_detected": True
                }
            else:
                return {"success": True, "face_detected": False}
                
        except Exception as e:
            logger.error(f"Kare işleme hatası: {str(e)}")
            traceback.print_exc()
            return {"success": False, "error": str(e)}
    
    def _extract_eye_landmarks(self, face_landmarks, frame_shape):
        """
        Yüz işaretlerinden göz/iris koordinatlarını çıkarır.
        
        Args:
            face_landmarks: MediaPipe yüz işaretleri
            frame_shape: Görüntü boyutları
            
        Returns:
            tuple: (sol_iris_koordinatları, sağ_iris_koordinatları)
        """
        h, w = frame_shape[:2]
        
        # MediaPipe iris indeksleri
        # Sol iris: 468, Sağ iris: 473
        try:
            # Left iris
            left_iris_x = int(face_landmarks.landmark[468].x * w)
            left_iris_y = int(face_landmarks.landmark[468].y * h)
            
            # Right iris
            right_iris_x = int(face_landmarks.landmark[473].x * w)
            right_iris_y = int(face_landmarks.landmark[473].y * h)
            
            return (left_iris_x, left_iris_y), (right_iris_x, right_iris_y)
            
        except (IndexError, AttributeError):
            # Sol göz köşeleri (landmarks 33, 133)
            left_corner_1_x = int(face_landmarks.landmark[33].x * w)
            left_corner_1_y = int(face_landmarks.landmark[33].y * h)
            left_corner_2_x = int(face_landmarks.landmark[133].x * w)
            left_corner_2_y = int(face_landmarks.landmark[133].y * h)
            
            # Sol göz merkezi (yaklaşık)
            left_eye_x = (left_corner_1_x + left_corner_2_x) // 2
            left_eye_y = (left_corner_1_y + left_corner_2_y) // 2
            
            # Sağ göz köşeleri (landmarks 362, 263)
            right_corner_1_x = int(face_landmarks.landmark[362].x * w)
            right_corner_1_y = int(face_landmarks.landmark[362].y * h)
            right_corner_2_x = int(face_landmarks.landmark[263].x * w)
            right_corner_2_y = int(face_landmarks.landmark[263].y * h)
            
            # Sağ göz merkezi (yaklaşık)
            right_eye_x = (right_corner_1_x + right_corner_2_x) // 2
            right_eye_y = (right_corner_1_y + right_corner_2_y) // 2
            
            return (left_eye_x, left_eye_y), (right_eye_x, right_eye_y)
    
    def _smooth_signal(self, signal_data, window_size=5):
        """Verilen sinyali hareketli ortalama ile düzleştirir."""
        if len(signal_data) < window_size:
            return np.array(signal_data)
            
        if window_size <= 1:
            return np.array(signal_data)
            
        signal_arr = np.array(signal_data)
        smoothed = np.convolve(signal_arr, np.ones(window_size)/window_size, mode='valid')
        
        # Konvolüsyon nedeniyle kaybolan elemanları tekrar ekle
        padding = np.full(window_size-1, smoothed[0])
        return np.concatenate((padding, smoothed))
    
    def _bandpass_filter(self, data, lowcut, highcut, fs, order=4):
        """Bant geçiren filtre uygular."""
        nyq = 0.5 * fs
        low = max(0.001, lowcut / nyq)  # 0'dan büyük olmalı
        high = min(0.999, highcut / nyq)  # 1'den küçük olmalı
        
        # Yetersiz veri durumunu kontrol et
        if len(data) <= order * 2:
            return data
            
        try:
            b, a = butter(order, [low, high], btype='band')
            y = filtfilt(b, a, data)
            return y
        except Exception as e:
            logger.error(f"Filtreleme hatası: {str(e)}")
            return data
    
    def _spectral_analysis(self, data, fs):
        """Veri üzerinde spektral analiz yapar."""
        # Yetersiz veri kontrolü
        if len(data) < 10:
            return np.array([]), np.array([])
            
        # FFT kullanarak spektrum hesapla
        n = len(data)
        freqs = np.fft.rfftfreq(n, d=1/fs)
        fft_vals = np.abs(np.fft.rfft(data))
        
        # Gücü normalize et
        psd = fft_vals**2 / n
        
        return freqs, psd
    
    def _find_dominant_frequency(self, freqs, psd):
        """En baskın frekansı bulur."""
        # Yetersiz veri kontrolü
        if len(freqs) == 0 or len(psd) == 0:
            return 0, 0
            
        # DC bileşenini çıkart (0 Hz)
        if len(freqs) > 1:
            mask = freqs > 0.1  # 0.1 Hz'den büyük frekansları al
            freqs_filtered = freqs[mask]
            psd_filtered = psd[mask]
            
            if len(freqs_filtered) == 0:
                return 0, 0
        else:
            freqs_filtered = freqs
            psd_filtered = psd
        
        # En güçlü frekans
        max_idx = np.argmax(psd_filtered)
        
        if max_idx < len(freqs_filtered):
            return freqs_filtered[max_idx], psd_filtered[max_idx]
        else:
            return 0, 0
    
    def compute_nystagmus_frequency(self, y_positions, frame_rate):
        """
        Göz merkezi y koordinatlarının zaman serisinden nistagmus frekansını (Hz) hesaplar.
        İyileştirilmiş FFT tabanlı hesaplama.
        
        Args:
            y_positions: [y1, y2, ..., yN] şeklinde pozisyon listesi
            frame_rate: Videonun kare hızı (FPS)
        
        Returns:
            float: Dominant nistagmus frekansı (Hz)
        """
        if len(y_positions) < 2:
            return 0.0  # Yeterli veri yoksa frekans 0 kabul edilir
        
        y = np.array(y_positions, dtype=float)
        
        # DC bileşeni (ortalama konum) çıkarılır
        y = y - np.mean(y)
        
        # Eğer tüm değerler aynıysa (varyans 0), frekans hesaplanamaz
        if np.std(y) < 1e-6:
            return 0.0
            
        # Hızlı Fourier Dönüşümü hesaplanır
        fft_vals = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), d=1.0/frame_rate)
        
        # Sıfır frekansı (DC) dışında en güçlü bileşeni bul
        # (mutlak değer karesi enerjiye denktir, ama tepe bulmak için abs yeterli)
        power = np.abs(fft_vals)
        power[0] = 0  # DC bileşeni göz ardı et
        
        # Sadece pozitif frekansları dikkate al (negatif frekanslar simetrik)
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_freqs) == 0 or len(positive_power) == 0:
            return 0.0
            
        # Nistagmus için tipik frekans aralığında filtrele (0.5-15 Hz)
        freq_mask = (positive_freqs >= 0.5) & (positive_freqs <= 15.0)
        filtered_freqs = positive_freqs[freq_mask]
        filtered_power = positive_power[freq_mask]
        
        if len(filtered_freqs) == 0 or len(filtered_power) == 0:
            # Filtreli aralıkta veri yoksa, tüm pozitif frekansları kullan
            filtered_freqs = positive_freqs[1:]  # 0 Hz hariç
            filtered_power = positive_power[1:]
            
        if len(filtered_freqs) == 0:
            return 0.0
            
        # En güçlü frekans bileşenini bul
        dominant_idx = np.argmax(filtered_power)
        dominant_freq = abs(filtered_freqs[dominant_idx])
        
        return dominant_freq
    
    def _calculate_spv(self, positions, timestamps):
        """Yavaş Faz Hızını (SPV) hesaplar."""
        if len(positions) < 2 or len(timestamps) < 2:
            return np.array([])
            
        # Pozisyonları dereceye dönüştür
        positions_deg = np.array(positions) * self.config["pixel_to_degree_ratio"]
        
        # Zaman aralıklarını hesapla
        time_diffs = np.diff(timestamps)
        
        # Sıfır zaman farklarını kontrol et
        time_diffs[time_diffs == 0] = 1e-6  # Çok küçük değerler ekle
        
        # Hızı hesapla (derece/saniye)
        velocities = np.diff(positions_deg) / time_diffs
        
        return velocities
    
    def _analyze_eye_movements(self):
        """Toplanan göz hareketi verilerini analiz eder."""
        try:
            # Yeterli veri var mı kontrol et
            if len(self.timestamps) < 30:  # En az 1 saniyelik veri (30fps)
                return
                
            # Veriden zaman serisini oluştur
            timestamps_arr = np.array(self.timestamps)
            time_diffs = np.diff(timestamps_arr)
            if len(time_diffs) == 0 or np.mean(time_diffs) == 0:
                return
                
            # Ortalama örnekleme hızını hesapla
            sample_rate = 1.0 / np.mean(time_diffs)
            
            # Göz pozisyonlarını düzgünleştir
            left_x_smooth = self._smooth_signal(self.left_eye_positions_x, self.config["smoothing_window"])
            left_y_smooth = self._smooth_signal(self.left_eye_positions_y, self.config["smoothing_window"])
            right_x_smooth = self._smooth_signal(self.right_eye_positions_x, self.config["smoothing_window"])
            right_y_smooth = self._smooth_signal(self.right_eye_positions_y, self.config["smoothing_window"])
            
            # Bant geçiren filtreleme
            left_x_filtered = self._bandpass_filter(
                left_x_smooth, 
                self.config["bandpass_freq_low"],
                self.config["bandpass_freq_high"],
                sample_rate
            )
            
            right_x_filtered = self._bandpass_filter(
                right_x_smooth, 
                self.config["bandpass_freq_low"],
                self.config["bandpass_freq_high"],
                sample_rate
            )
            
            # Spektral analiz
            left_freqs, left_psd = self._spectral_analysis(left_x_filtered, sample_rate)
            right_freqs, right_psd = self._spectral_analysis(right_x_filtered, sample_rate)
            
            # YENİ FFT tabanlı frekans hesaplama
            left_x_freq = self.compute_nystagmus_frequency(list(self.left_eye_positions_x), sample_rate)
            left_y_freq = self.compute_nystagmus_frequency(list(self.left_eye_positions_y), sample_rate)
            right_x_freq = self.compute_nystagmus_frequency(list(self.right_eye_positions_x), sample_rate)
            right_y_freq = self.compute_nystagmus_frequency(list(self.right_eye_positions_y), sample_rate)
            
            # Çok eksenliler arası ortalama frekans (x ve y eksenleri)
            left_dominant_freq = max(left_x_freq, left_y_freq)  # En baskın eksendeki frekans
            right_dominant_freq = max(right_x_freq, right_y_freq)  # En baskın eksendeki frekans
            
            # Güç hesaplama (eski uyumluluğu için)
            _, left_psd_max = self._find_dominant_frequency(left_freqs, left_psd)
            _, right_psd_max = self._find_dominant_frequency(right_freqs, right_psd)
            left_dominant_power = left_psd_max
            right_dominant_power = right_psd_max
            
            # Hız analizi (SPV - Slow Phase Velocity)
            left_spv = self._calculate_spv(list(self.left_eye_positions_x), list(self.timestamps))
            right_spv = self._calculate_spv(list(self.right_eye_positions_x), list(self.timestamps))
            
            # Nistagmus olasılığını değerlendir
            # 1. Dominant frekans nistagmus aralığında mı? (tipik olarak 2-10 Hz)
            # 2. SPV yeterince yüksek mi?
            left_freq_match = (self.config["nistagmus_freq_min"] <= left_dominant_freq <= 
                              self.config["nistagmus_freq_max"]) and left_dominant_power > 0.5
                              
            right_freq_match = (self.config["nistagmus_freq_min"] <= right_dominant_freq <= 
                               self.config["nistagmus_freq_max"]) and right_dominant_power > 0.5
            
            # SPV değerlerini kontrol et
            left_spv_high = len(left_spv) > 0 and np.max(np.abs(left_spv)) > self.config["spv_threshold"]
            right_spv_high = len(right_spv) > 0 and np.max(np.abs(right_spv)) > self.config["spv_threshold"]
            
            # Güven seviyesi hesapla
            confidence_factors = [
                left_freq_match, right_freq_match,
                left_spv_high, right_spv_high
            ]
            
            confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
            
            # Nistagmus tipi belirleme (yatay, dikey, döndürme)
            horizontal_amplitude = np.std(left_x_filtered) + np.std(right_x_filtered) \
                                   if len(left_x_filtered) > 0 and len(right_x_filtered) > 0 else 0
                                   
            vertical_amplitude = np.std(self._smooth_signal(np.array(list(self.left_eye_positions_y)), 
                                                          self.config["smoothing_window"])) + \
                                np.std(self._smooth_signal(np.array(list(self.right_eye_positions_y)), 
                                                         self.config["smoothing_window"]))
            
            # Toplam genlik
            total_amplitude = horizontal_amplitude + vertical_amplitude + 1e-6  # Sıfıra bölmeyi önle
            
            # Hareket tipleri
            type_confidence = {
                "horizontal": horizontal_amplitude / total_amplitude,
                "vertical": vertical_amplitude / total_amplitude,
                "rotatory": 0.0  # Rotatory için ek analiz gerekiyor
            }
            
            # Sonuçları kaydet
            self.nistagmus_probability = confidence
            self.last_analysis_results = {
                "confidence": float(confidence),
                "left_eye": {
                    "dominant_frequency": float(left_dominant_freq),
                    "dominant_power": float(left_dominant_power),
                    "max_spv": float(np.max(np.abs(left_spv))) if len(left_spv) > 0 else 0.0,
                    # Yeni FFT tabanlı frekans analizi sonuçları
                    "x_frequency": float(left_x_freq),
                    "y_frequency": float(left_y_freq),
                },
                "right_eye": {
                    "dominant_frequency": float(right_dominant_freq),
                    "dominant_power": float(right_dominant_power),
                    "max_spv": float(np.max(np.abs(right_spv))) if len(right_spv) > 0 else 0.0,
                    # Yeni FFT tabanlı frekans analizi sonuçları
                    "x_frequency": float(right_x_freq),
                    "y_frequency": float(right_y_freq),
                },
                "movement_types": {
                    "horizontal": float(type_confidence["horizontal"]),
                    "vertical": float(type_confidence["vertical"]),
                    "rotatory": float(type_confidence["rotatory"]),
                },
                "frequency_analysis": {
                    "method": "improved_fft",
                    "sample_rate": float(sample_rate),
                    "total_samples": len(self.timestamps),
                    "analysis_duration": float(self.timestamps[-1] - self.timestamps[0]) if len(self.timestamps) > 1 else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Göz hareketi analiz hatası: {str(e)}")
            traceback.print_exc()
    
    def get_detection_results(self):
        """
        En son nistagmus tespit sonuçlarını döndürür
        
        Returns:
            dict: Tespit sonuçları
        """
        return {
            "probability": self.nistagmus_probability,
            "details": self.last_analysis_results,
            "has_nistagmus": self.nistagmus_probability > 0.5,
            "recommendation": self._generate_recommendation(
                self.nistagmus_probability > 0.5, 
                self.nistagmus_probability
            )
        }
    
    def _generate_recommendation(self, has_nistagmus, confidence):
        """Analiz sonuçlarına göre bir öneri oluşturur."""
        if has_nistagmus:
            if confidence > 0.8:
                return "Yüksek güvenle nistagmus tespit edilmiştir. Bir göz doktoruna başvurulması önerilir."
            else:
                return "Nistagmus belirtileri gözlemlenmektedir, ancak kesin teşhis için uzman muayenesi gereklidir."
        else:
            if confidence > 0.7:
                return "Nistagmus belirtisi tespit edilmemiştir."
            else:
                return "Net bir nistagmus belirtisi gözlemlenmemiştir, ancak daha kaliteli görüntüler veya uzman değerlendirmesi önerilir."
    
    def reset(self):
        """Veri yapılarını sıfırlar"""
        self.left_eye_positions_x.clear()
        self.left_eye_positions_y.clear()
        self.right_eye_positions_x.clear()
        self.right_eye_positions_y.clear()
        self.timestamps.clear()
        self.nistagmus_probability = 0.0
        self.last_analysis_results = {}
        self.frame_count = 0
    
    def release(self):
        """MediaPipe kaynaklarını serbest bırakır"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
            
# Göz verilerini kaydetmek için yardımcı fonksiyon
def save_eye_tracking_data(output_file, left_eyes, right_eyes, timestamps):
    """
    Göz takip verilerini CSV dosyasına kaydetme
    
    Args:
        output_file (str): Çıktı dosya yolu
        left_eyes (list): Sol göz koordinatları listesi [(x1,y1), (x2,y2), ...]
        right_eyes (list): Sağ göz koordinatları listesi [(x1,y1), (x2,y2), ...]
        timestamps (list): Zaman damgaları listesi
    """
    import csv
    
    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Başlık
            writer.writerow(['timestamp', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y'])
            
            # Veriler
            for i in range(min(len(timestamps), len(left_eyes), len(right_eyes))):
                writer.writerow([
                    timestamps[i],
                    left_eyes[i][0],
                    left_eyes[i][1],
                    right_eyes[i][0],
                    right_eyes[i][1]
                ])
                
        logger.info(f"Göz takip verileri kaydedildi: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Veri kaydetme hatası: {str(e)}")
        return False

def draw_eye_tracking_results(frame, left_eye=None, right_eye=None, 
                             nistagmus_prob=None, face_detected=True):
    """
    Göz takip sonuçlarını görüntü üzerine çizer
    
    Args:
        frame: İşlenecek video karesi
        left_eye: Sol göz koordinatları (x, y)
        right_eye: Sağ göz koordinatları (x, y)
        nistagmus_prob: Nistagmus olasılığı (0-1)
        face_detected: Yüz tespit edildi mi?
    
    Returns:
        frame: İşaretlenmiş video karesi
    """
    height, width = frame.shape[:2]
    
    # Varsayılan yazı formatı
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 2
    text_color = (255, 255, 255)  # Beyaz
    
    # Yüz tespit edilmediyse uyarı göster
    if not face_detected:
        cv2.putText(frame, "Yüz tespit edilemedi", 
                   (50, height - 50), font, font_scale, (0, 0, 255), font_thickness)
        return frame
    
    # Göz koordinatlarını çiz
    if left_eye is not None:
        cv2.circle(frame, left_eye, 5, (0, 255, 0), -1)  # Yeşil
        cv2.putText(frame, f"Sol göz: ({left_eye[0]}, {left_eye[1]})", 
                   (10, 30), font, font_scale, text_color, font_thickness)
    
    if right_eye is not None:
        cv2.circle(frame, right_eye, 5, (255, 0, 0), -1)  # Mavi
        cv2.putText(frame, f"Sağ göz: ({right_eye[0]}, {right_eye[1]})", 
                   (10, 60), font, font_scale, text_color, font_thickness)
    
    # Nistagmus olasılığını göster
    if nistagmus_prob is not None:
        # Renk: Düşük olasılık (yeşil) -> Yüksek olasılık (kırmızı)
        prob_color = (
            0, 
            int(255 * (1 - nistagmus_prob)),
            int(255 * nistagmus_prob)
        )
        
        cv2.putText(frame, f"Nistagmus olasılığı: {nistagmus_prob:.2f}", 
                   (10, 90), font, font_scale, prob_color, font_thickness)
        
        # Olasılık çubuğu
        bar_width = 200
        bar_height = 20
        filled_width = int(bar_width * nistagmus_prob)
        
        # Çubuk arka planı
        cv2.rectangle(frame, 
                     (10, 100), 
                     (10 + bar_width, 100 + bar_height), 
                     (120, 120, 120), 
                     -1)
        
        # Doldurulmuş kısım
        cv2.rectangle(frame, 
                     (10, 100), 
                     (10 + filled_width, 100 + bar_height), 
                     prob_color, 
                     -1)
        
        # Çubuk çerçevesi
        cv2.rectangle(frame, 
                     (10, 100), 
                     (10 + bar_width, 100 + bar_height), 
                     (200, 200, 200), 
                     2)
    
    return frame

# Test işlevi
def test_eye_tracker(video_source=0, output_file=None):
    """
    EyeTracker sınıfını test eder
    
    Args:
        video_source: Video kaynağı (dosya yolu veya kamera indeksi)
        output_file: Sonuç video dosyası (opsiyonel)
    """
    # Tracker oluştur
    tracker = EyeTracker()
    
    # Kamerayı aç
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Hata: Video kaynağı açılamadı: {video_source}")
        return
    
    # Video yazıcı (opsiyonel)
    video_writer = None
    if output_file:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Göz takibi
        results = tracker.process_frame(frame)
        
        if results["success"]:
            # Sonuçları çiz
            if results["face_detected"]:
                frame = draw_eye_tracking_results(
                    frame,
                    results.get("left_eye"),
                    results.get("right_eye"),
                    results.get("nistagmus_probability"),
                    True
                )
            else:
                frame = draw_eye_tracking_results(
                    frame, face_detected=False
                )
        
        # Görüntüyü göster
        cv2.imshow("Göz Takibi", frame)
        
        # Video yazıcı
        if video_writer is not None:
            video_writer.write(frame)
        
        # ESC tuşu ile çıkış
        if cv2.waitKey(1) == 27:
            break
    
    # Kaynakları serbest bırak
    cap.release()
    if video_writer is not None:
        video_writer.release()
    tracker.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_eye_tracker()

def _prepare_data_arrays(left_positions_x, left_positions_y, right_positions_x, right_positions_y, timestamps):
    """Veri dizilerini numpy array'lerine dönüştürür ve temel kontrolleri yapar"""
    left_x = np.array(left_positions_x)
    left_y = np.array(left_positions_y)
    right_x = np.array(right_positions_x)
    right_y = np.array(right_positions_y)
    times = np.array(timestamps)
    
    if len(times) < 2:
        return None, {"error": "Yeterli zaman verisi yok"}
    
    return {
        "left_x": left_x, "left_y": left_y,
        "right_x": right_x, "right_y": right_y,
        "times": times
    }, None

def _calculate_velocities_and_accelerations(data_arrays):
    """Hız ve ivme hesaplamalarını yapar"""
    left_x, left_y = data_arrays["left_x"], data_arrays["left_y"]
    right_x, right_y = data_arrays["right_x"], data_arrays["right_y"]
    times = data_arrays["times"]
    
    dt = np.diff(times)
    avg_dt = np.mean(dt)
    
    # Hızlar (piksel/saniye)
    left_vel_x = np.diff(left_x) / dt
    left_vel_y = np.diff(left_y) / dt
    right_vel_x = np.diff(right_x) / dt
    right_vel_y = np.diff(right_y) / dt
    
    # İvmeler (piksel/saniye^2)
    if len(left_vel_x) >= 2:
        left_acc_x = np.diff(left_vel_x) / dt[:-1]
        left_acc_y = np.diff(left_vel_y) / dt[:-1]
        right_acc_x = np.diff(right_vel_x) / dt[:-1]
        right_acc_y = np.diff(right_vel_y) / dt[:-1]
    else:
        left_acc_x = np.array([0.0])
        left_acc_y = np.array([0.0])
        right_acc_x = np.array([0.0])
        right_acc_y = np.array([0.0])
    
    return {
        "velocities": {
            "left_vel_x": left_vel_x, "left_vel_y": left_vel_y,
            "right_vel_x": right_vel_x, "right_vel_y": right_vel_y
        },
        "accelerations": {
            "left_acc_x": left_acc_x, "left_acc_y": left_acc_y,
            "right_acc_x": right_acc_x, "right_acc_y": right_acc_y
        },
        "avg_dt": avg_dt
    }

def _calculate_fft_analysis(data_arrays, avg_dt):
    """FFT analizi yapar"""
    def calculate_fft(signal):
        if len(signal) < 4:
            return {"freqs": np.array([]), "power": np.array([])}
        
        signal_detrended = signal - np.mean(signal)
        window = np.hanning(len(signal_detrended))
        windowed_signal = signal_detrended * window
        
        n = len(windowed_signal)
        fft_result = np.fft.rfft(windowed_signal)
        power = np.abs(fft_result)**2 / n
        freqs = np.fft.rfftfreq(n, d=avg_dt)
        
        return {"freqs": freqs, "power": power}
    
    return {
        "left_x": calculate_fft(data_arrays["left_x"]),
        "left_y": calculate_fft(data_arrays["left_y"]),
        "right_x": calculate_fft(data_arrays["right_x"]),
        "right_y": calculate_fft(data_arrays["right_y"])
    }

def _calculate_statistical_features(data_arrays, velocities):
    """İstatistiksel özellikleri hesaplar"""
    def get_position_stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "range": float(np.ptp(arr))
        }
    
    def get_velocity_stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "std": 0.0, "max_abs": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "max_abs": float(np.max(np.abs(arr)))
        }
    
    return {
        "positions": {
            "left_x": get_position_stats(data_arrays["left_x"]),
            "left_y": get_position_stats(data_arrays["left_y"]),
            "right_x": get_position_stats(data_arrays["right_x"]),
            "right_y": get_position_stats(data_arrays["right_y"])
        },
        "velocities": {
            "left_x": get_velocity_stats(velocities["left_vel_x"]),
            "left_y": get_velocity_stats(velocities["left_vel_y"]),
            "right_x": get_velocity_stats(velocities["right_vel_x"]),
            "right_y": get_velocity_stats(velocities["right_vel_y"])
        }
    }

def _calculate_dominant_frequencies(fft_results):
    """Baskın frekansları hesaplar"""
    def find_dominant_freq(freqs, power):
        if len(freqs) == 0 or len(power) == 0:
            return 0.0, 0.0
        max_idx = np.argmax(power)
        if max_idx < len(freqs):
            return float(freqs[max_idx]), float(power[max_idx])
        return 0.0, 0.0
    
    def find_dominant_in_range(freqs, power, min_freq=2.0, max_freq=10.0):
        if len(freqs) == 0 or len(power) == 0:
            return 0.0, 0.0
        
        mask = (freqs >= min_freq) & (freqs <= max_freq)
        filtered_freqs = freqs[mask]
        filtered_power = power[mask]
        
        if len(filtered_freqs) == 0:
            return 0.0, 0.0
        
        max_idx = np.argmax(filtered_power)
        if max_idx < len(filtered_freqs):
            return float(filtered_freqs[max_idx]), float(filtered_power[max_idx])
        return 0.0, 0.0
    
    dominant_freqs = {}
    nistagmus_range = {}
    
    for eye_axis in ["left_x", "left_y", "right_x", "right_y"]:
        if eye_axis in fft_results and len(fft_results[eye_axis]["freqs"]) > 0:
            freqs = fft_results[eye_axis]["freqs"]
            power = fft_results[eye_axis]["power"]
            
            dom_freq, dom_power = find_dominant_freq(freqs, power)
            nyst_freq, nyst_power = find_dominant_in_range(freqs, power)
            
            dominant_freqs[eye_axis] = {"freq": dom_freq, "power": dom_power}
            nistagmus_range[eye_axis] = {"freq": nyst_freq, "power": nyst_power}
    
    return {"dominant": dominant_freqs, "nistagmus_range": nistagmus_range}

def extract_features_from_video(left_positions_x, left_positions_y, 
                               right_positions_x, right_positions_y, 
                               timestamps):
    """
    Video verisinden göz hareketi özelliklerini çıkarır (Refaktör edilmiş)
    """
    try:
        # 1. Veri hazırlığı
        data_arrays, error = _prepare_data_arrays(
            left_positions_x, left_positions_y, 
            right_positions_x, right_positions_y, timestamps
        )
        if error:
            return error
        
        # 2. Hız ve ivme hesaplamaları
        motion_data = _calculate_velocities_and_accelerations(data_arrays)
        
        # 3. FFT analizi
        fft_results = _calculate_fft_analysis(data_arrays, motion_data["avg_dt"])
        
        # 4. İstatistiksel özellikler
        statistical_features = _calculate_statistical_features(
            data_arrays, motion_data["velocities"]
        )
        
        # 5. Temel bilgiler
        sample_rate = 1.0 / motion_data["avg_dt"]
        times = data_arrays["times"]
        
        features = {
            **statistical_features,
            "frequency_info": {
                "sample_rate": float(sample_rate),
                "duration": float(times[-1] - times[0]) if len(times) >= 2 else 0.0,
                "num_samples": len(times),
            },
            "raw_data": {
                "timestamps": times.tolist(),
                "left_x": data_arrays["left_x"].tolist(),
                "left_y": data_arrays["left_y"].tolist(),
                "right_x": data_arrays["right_x"].tolist(),
                "right_y": data_arrays["right_y"].tolist(),
            }
        }
        
        # 6. FFT sonuçlarını ekle (boş değilse)
        if any(len(fft_results[axis]["freqs"]) > 0 for axis in fft_results):
            features["frequency_analysis"] = {
                axis: {
                    "freqs": fft_results[axis]["freqs"].tolist(), 
                    "power": fft_results[axis]["power"].tolist()
                }
                for axis in fft_results if len(fft_results[axis]["freqs"]) > 0
            }
            
            # 7. Baskın frekansları hesapla
            freq_analysis = _calculate_dominant_frequencies(fft_results)
            features["dominant_frequencies"] = freq_analysis["dominant"]
            if freq_analysis["nistagmus_range"]:
                features["dominant_frequencies"]["nistagmus_range"] = freq_analysis["nistagmus_range"]
            
            # 8. İyileştirilmiş FFT analizi
            temp_tracker = EyeTracker()
            improved_fft = {
                "left_x": temp_tracker.compute_nystagmus_frequency(data_arrays["left_x"].tolist(), sample_rate),
                "left_y": temp_tracker.compute_nystagmus_frequency(data_arrays["left_y"].tolist(), sample_rate),
                "right_x": temp_tracker.compute_nystagmus_frequency(data_arrays["right_x"].tolist(), sample_rate),
                "right_y": temp_tracker.compute_nystagmus_frequency(data_arrays["right_y"].tolist(), sample_rate),
            }
            
            features["dominant_frequencies"]["improved_fft"] = {
                **improved_fft,
                "left_dominant": max(improved_fft["left_x"], improved_fft["left_y"]),
                "right_dominant": max(improved_fft["right_x"], improved_fft["right_y"]),
                "overall_dominant": max(improved_fft.values()),
                "method": "enhanced_fft_with_filtering"
            }
        
        return features
        
    except Exception as e:
        logger.error(f"Özellik çıkarma hatası: {str(e)}")
        traceback.print_exc()
        return {"error": f"Özellik çıkarma sırasında hata oluştu: {str(e)}"}

def analyze_nystagmus(features):
    """
    Çıkarılan göz hareketi özelliklerini kullanarak nistagmus analizi yapar
    
    Args:
        features: extract_features_from_video() tarafından üretilen özellikler sözlüğü
        
    Returns:
        dict: Analiz sonuçları
    """
    try:
        # Hata kontrolü
        if "error" in features:
            return {"error": features["error"]}
            
        # Veri noktalarının sayısı
        num_samples = features.get("frequency_info", {}).get("num_samples", 0)
        if num_samples < 3:  # En az 3 veri noktası gereklidir
            return {"error": "Yeterli veri noktası yok", "is_nystagmus": False}
        
        # Temel istatistiksel değerler
        positions = features.get("positions", {})
        velocities = features.get("velocities", {})
        dominant_freqs = features.get("dominant_frequencies", {})
        
        # Nistagmus puanını hesaplamak için kriterler
        
        # 1. Frekans kriteri: 2-10 Hz arasında baskın frekans
        freq_score = 0.0
        if "nistagmus_range" in dominant_freqs:
            # Yatay hareketlerdeki nistagmus frekanslarını kontrol et (daha çok yatay olur)
            nx_freq_left = dominant_freqs["nistagmus_range"]["left_x"]["freq"]
            nx_power_left = dominant_freqs["nistagmus_range"]["left_x"]["power"]
            nx_freq_right = dominant_freqs["nistagmus_range"]["right_x"]["freq"]
            nx_power_right = dominant_freqs["nistagmus_range"]["right_x"]["power"]
            
            # Verinin gücü ve frekansı önemli
            if nx_freq_left >= 2.0 and nx_freq_left <= 10.0 and nx_power_left > 0:
                freq_score += 0.5
            if nx_freq_right >= 2.0 and nx_freq_right <= 10.0 and nx_power_right > 0:
                freq_score += 0.5
                
            # Ağırlıklı ortalama
            freq_score = min(1.0, freq_score)
        
        # 2. Hız kriteri: Yüksek hız değişkenlikleri
        vel_score = 0.0
        if velocities:
            # Sol göz yatay hız değişkenliği
            std_left_x = velocities.get("left_x", {}).get("std", 0)
            # Sağ göz yatay hız değişkenliği
            std_right_x = velocities.get("right_x", {}).get("std", 0)
            
            # Hız değişkenlikleri nistagmusta yüksektir
            vel_threshold = 20.0  # Piksel/saniye
            if std_left_x > vel_threshold:
                vel_score += 0.4
            if std_right_x > vel_threshold:
                vel_score += 0.4
                
            # Maximum hızlar
            max_vel_left_x = velocities.get("left_x", {}).get("max_abs", 0)
            max_vel_right_x = velocities.get("right_x", {}).get("max_abs", 0)
            
            if max_vel_left_x > 2 * vel_threshold:
                vel_score += 0.1
            if max_vel_right_x > 2 * vel_threshold:
                vel_score += 0.1
                
            vel_score = min(1.0, vel_score)
        
        # 3. Hareket aralığı kriteri
        range_score = 0.0
        if positions:
            # Yatay hareket aralığı
            range_left_x = positions.get("left_x", {}).get("range", 0)
            range_right_x = positions.get("right_x", {}).get("range", 0)
            
            # Hareket aralığı (piksel)
            if range_left_x > 15.0:
                range_score += 0.25
            if range_right_x > 15.0:
                range_score += 0.25
                
            # Dikey hareket aralığı (genellikle daha az)
            range_left_y = positions.get("left_y", {}).get("range", 0)
            range_right_y = positions.get("right_y", {}).get("range", 0)
            
            if range_left_y > 10.0:
                range_score += 0.25
            if range_right_y > 10.0:
                range_score += 0.25
                
            range_score = min(1.0, range_score)
        
        # Ağırlıklı toplam puan
        total_score = (freq_score * 0.5) + (vel_score * 0.3) + (range_score * 0.2)
        
        # Nistagmus tespiti için eşik değeri
        threshold = 0.6
        is_nystagmus = total_score >= threshold
        
        # Güven skoru
        confidence = min(1.0, total_score / threshold) if threshold > 0 else 0.0
        
        # Analiz sonuçları
        results = {
            "is_nystagmus": is_nystagmus,
            "confidence": float(confidence),
            "total_score": float(total_score),
            "frequency_score": float(freq_score),
            "velocity_score": float(vel_score),
            "range_score": float(range_score),
            "sample_count": int(num_samples),
            "duration": float(features.get("frequency_info", {}).get("duration", 0.0)),
            "timestamp": time.time()
        }
        
        # Önemli frekans bilgilerini ekle
        if "nistagmus_range" in dominant_freqs:
            results["dominant_horizontal_frequency"] = {
                "left": float(dominant_freqs["nistagmus_range"]["left_x"]["freq"]),
                "right": float(dominant_freqs["nistagmus_range"]["right_x"]["freq"])
            }
        
        return results
        
    except Exception as e:
        logger.error(f"Nistagmus analizi hatası: {str(e)}")
        traceback.print_exc()
        return {"error": f"Analiz sırasında hata oluştu: {str(e)}", "is_nystagmus": False} 