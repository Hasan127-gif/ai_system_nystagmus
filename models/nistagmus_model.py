"""
Nistagmus Tespit Modeli
----------------------
Bu modül, nistagmus tespiti için temel model sınıflarını içerir.
Göz hareketlerinden nistagmus analizi yapar.
"""

import numpy as np
import os
import logging
from scipy import signal
import time
import json
from pathlib import Path

logger = logging.getLogger('nistagmus_model')

class NistagmusModel:
    """Nistagmus tespiti için model sınıfı."""
    
    def __init__(self, model_path=None):
        """
        Args:
            model_path (str, optional): Model dosyasının yolu
        """
        self.model_path = model_path
        self.is_initialized = False
        self.model_info = {
            "name": "Nistagmus Tespit Modeli",
            "version": "1.0.0",
            "type": "statistical",
            "features": ["frequency_analysis", "eye_movement_patterns"],
            "creation_date": time.strftime("%Y-%m-%d")
        }
        
        self._initialize()
    
    def _initialize(self):
        """Model parametrelerini başlat"""
        try:
            # Model parametreleri
            self.freq_threshold = 3.0  # Hz
            self.amplitude_threshold = 5.0  # piksel
            self.regularity_threshold = 0.6
            self.spv_threshold = 20.0  # derece/saniye
            self.pixel_to_degree_ratio = 0.05  # derece/piksel
            
            # Model konfigürasyonu dosyadan yükleme
            if self.model_path and os.path.exists(self.model_path):
                with open(self.model_path, 'r') as f:
                    config = json.load(f)
                    
                    # Parametreleri güncelle
                    for key, value in config.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            
                logger.info(f"Model parametreleri yüklendi: {self.model_path}")
            
            self.is_initialized = True
            logger.info("Nistagmus model başlatıldı")
            
        except Exception as e:
            logger.error(f"Model başlatma hatası: {str(e)}")
            self.is_initialized = False
    
    def predict(self, image_data):
        """
        Görüntüden nistagmus tahmini yapar
        
        Args:
            image_data: Görüntü verisi
            
        Returns:
            dict: Tahmin sonuçları
        """
        if not self.is_initialized:
            return {"error": "Model başlatılmadı"}
        
        # Statik görüntülerden nistagmus tespiti zordur
        # Gerçek bir uygulama için göz takip verilerine ihtiyaç vardır
        # Bu basit bir tanıtım uygulamasıdır
        
        return {
            "prediction": False,
            "confidence": 0.3,
            "horizontal_score": 0.2,
            "vertical_score": 0.1,
            "rotatory_score": 0.0,
            "message": "Tek görüntüden nistagmus tespiti sınırlıdır. Video analizi önerilir."
        }
    
    def analyze_eye_movement(self, eye_positions, timestamps, sample_rate=30):
        """
        Göz hareketi verilerini analiz eder.
        
        Args:
            eye_positions: Göz pozisyonları listesi (x veya y)
            timestamps: Zaman damgaları listesi
            sample_rate: Örnekleme hızı (Hz)
            
        Returns:
            dict: Analiz sonuçları
        """
        if not self.is_initialized:
            return {"error": "Model başlatılmadı"}
            
        if len(eye_positions) < 30:
            return {"error": "Yetersiz veri"}
            
        try:
            # Verileri düzgünleştir
            eye_positions = np.array(eye_positions)
            
            # Trend çıkarma (DC bileşenini kaldır)
            detrended = signal.detrend(eye_positions)
            
            # Spektral analiz
            freqs, psd = signal.welch(detrended, sample_rate, nperseg=min(256, len(detrended)))
            
            # Baskın frekansı bul
            max_idx = np.argmax(psd)
            dominant_freq = freqs[max_idx] if max_idx < len(freqs) else 0
            dominant_power = psd[max_idx] if max_idx < len(psd) else 0
            
            # Nistagmus frekans aralığı: 2-10 Hz
            is_in_range = 2.0 <= dominant_freq <= 10.0
            
            # Hareket genliği
            amplitude = np.std(detrended)
            
            # Yavaş faz hızı (SPV)
            time_diffs = np.diff(timestamps)
            time_diffs[time_diffs == 0] = 1e-6  # Sıfıra bölmeyi önle
            velocities = np.diff(eye_positions) / time_diffs
            max_velocity = np.max(np.abs(velocities)) * self.pixel_to_degree_ratio
            
            # Nistagmus olasılığı
            nistagmus_probability = 0.0
            
            # Frekans, genlik ve hız kriterlerini değerlendir
            if is_in_range:
                freq_score = 1.0 - abs(dominant_freq - 4.0) / 6.0  # 4 Hz ideal
                
                # Genlik skoru
                amp_score = min(1.0, amplitude / self.amplitude_threshold)
                
                # Hız skoru
                vel_score = min(1.0, max_velocity / self.spv_threshold)
                
                # Olasılık hesapla (ağırlıklı ortalama)
                nistagmus_probability = (freq_score * 0.5 + amp_score * 0.25 + vel_score * 0.25)
            
            return {
                "dominant_frequency": float(dominant_freq),
                "dominant_power": float(dominant_power),
                "amplitude": float(amplitude),
                "max_velocity": float(max_velocity),
                "nistagmus_probability": float(nistagmus_probability),
                "is_nistagmus": nistagmus_probability > 0.5
            }
            
        except Exception as e:
            logger.error(f"Göz hareketi analiz hatası: {str(e)}")
            return {"error": str(e)}
    
    def save_model(self, model_path=None):
        """
        Model parametrelerini kaydet
        
        Args:
            model_path (str, optional): Kayıt yolu. None ise self.model_path kullanılır
        """
        if model_path is None:
            model_path = self.model_path
            
        if not model_path:
            logger.error("Model kayıt yolu belirtilmedi")
            return False
            
        try:
            # Model parametrelerini topla
            model_params = {
                "freq_threshold": self.freq_threshold,
                "amplitude_threshold": self.amplitude_threshold,
                "regularity_threshold": self.regularity_threshold,
                "spv_threshold": self.spv_threshold,
                "pixel_to_degree_ratio": self.pixel_to_degree_ratio,
                "model_info": self.model_info
            }
            
            # JSON olarak kaydet
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'w') as f:
                json.dump(model_params, f, indent=2)
                
            logger.info(f"Model parametreleri kaydedildi: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Model kaydetme hatası: {str(e)}")
            return False
    
    def load_weights(self, weights_path):
        """
        Model ağırlıklarını yükle (uyumluluk için)
        
        Args:
            weights_path (str): Ağırlık dosyasının yolu
        """
        try:
            if os.path.exists(weights_path):
                # JSON formatında parametre dosyası olarak kabul et
                with open(weights_path, 'r') as f:
                    params = json.load(f)
                    
                # Parametreleri güncelle
                for key, value in params.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
                logger.info(f"Model ağırlıkları yüklendi: {weights_path}")
                return True
            else:
                logger.warning(f"Ağırlık dosyası bulunamadı: {weights_path}")
                return False
        except Exception as e:
            logger.error(f"Ağırlık yükleme hatası: {str(e)}")
            return False


class NistagmusClassifier:
    """
    Göz hareketi özelliklerine göre nistagmus sınıflandırması yapar.
    """
    
    def __init__(self):
        # Nistagmus türleri ve özellikleri
        self.nistagmus_types = {
            "horizontal": {
                "description": "Yatay Nistagmus",
                "features": ["horizontal eye movement", "left-right oscillation"],
                "common_causes": ["vestibular disorder", "multiple sclerosis"],
                "typical_frequency": "2-5 Hz"
            },
            "vertical": {
                "description": "Dikey Nistagmus",
                "features": ["up-down eye movement", "vertical oscillation"],
                "common_causes": ["brainstem lesion", "cerebellar disease"],
                "typical_frequency": "3-5 Hz"
            },
            "rotatory": {
                "description": "Rotasyonel Nistagmus",
                "features": ["torsional eye movement", "rotational oscillation"],
                "common_causes": ["otolith disorders", "vestibular neuritis"],
                "typical_frequency": "2-4 Hz"
            },
            "pendular": {
                "description": "Pendüler Nistagmus",
                "features": ["equal velocity in both directions", "sinusoidal pattern"],
                "common_causes": ["multiple sclerosis", "visual impairment"],
                "typical_frequency": "3-6 Hz"
            },
            "jerk": {
                "description": "Sarsıntılı Nistagmus",
                "features": ["slow drift + fast correction", "saw-tooth pattern"],
                "common_causes": ["peripheral vestibular disorders", "brainstem lesions"],
                "typical_frequency": "2-5 Hz"
            }
        }
        
        # Sınıflandırma eşikleri
        self.thresholds = {
            "movement_amplitude": 3.0,  # piksel
            "min_confidence": 0.6,
            "horizontal_vertical_ratio": 2.0,  # Yatay/dikey hareket oranı
            "pattern_regularity": 0.7  # Hareket düzenliliği
        }
    
    def classify(self, horizontal_movement, vertical_movement, torsional_movement=None):
        """
        Göz hareketlerini sınıflandırır
        
        Args:
            horizontal_movement (dict): Yatay hareket analizi
            vertical_movement (dict): Dikey hareket analizi
            torsional_movement (dict, optional): Rotasyonel hareket analizi
            
        Returns:
            dict: Sınıflandırma sonucu
        """
        # Nistagmus var mı?
        h_prob = horizontal_movement.get("nistagmus_probability", 0)
        v_prob = vertical_movement.get("nistagmus_probability", 0)
        t_prob = torsional_movement.get("nistagmus_probability", 0) if torsional_movement else 0
        
        # Toplam nistagmus olasılığı
        combined_prob = max(h_prob, v_prob, t_prob)
        
        if combined_prob < self.thresholds["min_confidence"]:
            return {
                "is_nistagmus": False,
                "confidence": combined_prob,
                "message": "Yeterli düzeyde nistagmus belirtisi tespit edilmedi."
            }
        
        # Hareket genliklerini değerlendir
        h_amp = horizontal_movement.get("amplitude", 0)
        v_amp = vertical_movement.get("amplitude", 0)
        t_amp = torsional_movement.get("amplitude", 0) if torsional_movement else 0
        
        # Dominant hareket tipini belirle
        total_amp = h_amp + v_amp + t_amp
        if total_amp == 0:
            total_amp = 1.0  # Sıfıra bölmeyi önle
            
        type_scores = {
            "horizontal": h_amp / total_amp,
            "vertical": v_amp / total_amp,
            "rotatory": t_amp / total_amp
        }
        
        # En yüksek skoru al
        dominant_type = max(type_scores, key=type_scores.get)
        
        # Pendüler mi yoksa sarsıntılı mı?
        is_pendular = self._is_pendular_pattern(horizontal_movement, vertical_movement)
        
        # Alt tip belirle
        subtype = "pendular" if is_pendular else "jerk"
        
        # Sınıflandırma sonucu
        result = {
            "is_nistagmus": True,
            "confidence": combined_prob,
            "dominant_direction": dominant_type,
            "direction_confidence": type_scores[dominant_type],
            "pattern_type": subtype,
            "type_info": self.nistagmus_types.get(dominant_type, {}),
            "movement_details": {
                "horizontal": horizontal_movement,
                "vertical": vertical_movement
            }
        }
        
        # Torsiyonel hareket varsa ekle
        if torsional_movement:
            result["movement_details"]["torsional"] = torsional_movement
        
        return result
    
    def _is_pendular_pattern(self, horizontal_movement, vertical_movement):
        """
        Hareketin pendüler mi yoksa sarsıntılı mı olduğunu belirler
        
        Pendüler nistagmus, her iki yöndeki hareketin benzer hızda olduğu düzgün sinüzoidal bir paterndir.
        Sarsıntılı nistagmus, yavaş bir sürüklenme ve ardından hızlı bir düzeltme içeren testere dişi paterndir.
        
        Bu basit bir tahminden ibarettir, gerçek bir uygulamada daha detaylı analiz gerekir.
        """
        # Basit bir yaklaşım: hız simetrisini kontrol et
        # Gerçek uygulamada hız histogramı ve daha karmaşık analizler kullanılabilir
        return np.random.random() < 0.5  # 50% pendüler, 50% sarsıntılı (örnek)

# Test işlevi
def test_model():
    """Model sınıfını test et"""
    model = NistagmusModel()
    
    # Rastgele göz hareketi verileri oluştur
    # Gerçek bir senaryoda, bu veriler göz takip modülünden gelir
    timestamps = np.linspace(0, 3, 90)  # 30fps, 3 saniye
    
    # Gerçek nistagmus benzeri hareket simülasyonu:
    # Nistagmus genellikle 2-10 Hz arasında, yavaş faz + hızlı faz şeklindedir
    freq = 4.0  # Hz
    
    # Testere dişi sinyal (sawtooth) için
    t = np.linspace(0, 1, 90, endpoint=False)
    sawtooth = signal.sawtooth(2 * np.pi * freq * t)
    
    # Gürültü ekle
    noise = np.random.normal(0, 0.5, 90)
    
    # Nistagmus benzeri göz hareketi (x-yönü)
    eye_positions_x = 10 * sawtooth + noise
    
    # Model tahmini
    results = model.analyze_eye_movement(eye_positions_x, timestamps)
    
    print("\nGöz Hareketi Analiz Sonuçları:")
    print(f"Dominant Frekans: {results.get('dominant_frequency', 0):.2f} Hz")
    print(f"Genlik: {results.get('amplitude', 0):.2f} piksel")
    print(f"Nistagmus Olasılığı: {results.get('nistagmus_probability', 0):.2f}")
    print(f"Nistagmus tespit edildi mi? {'Evet' if results.get('is_nistagmus', False) else 'Hayır'}")
    
    # Sınıflandırıcı test et
    classifier = NistagmusClassifier()
    
    # Sınıflandırma için yapay hareket verileri
    h_movement = {
        "nistagmus_probability": 0.8,
        "amplitude": 10.0,
        "dominant_frequency": 4.0
    }
    
    v_movement = {
        "nistagmus_probability": 0.3,
        "amplitude": 3.0,
        "dominant_frequency": 3.0
    }
    
    classification = classifier.classify(h_movement, v_movement)
    
    print("\nNistagmus Sınıflandırması:")
    print(f"Nistagmus var mı? {'Evet' if classification.get('is_nistagmus', False) else 'Hayır'}")
    print(f"Güven: {classification.get('confidence', 0):.2f}")
    print(f"Dominant Yön: {classification.get('dominant_direction', 'N/A')}")
    print(f"Patern Tipi: {classification.get('pattern_type', 'N/A')}")
    
    if classification.get('is_nistagmus', False):
        type_info = classification.get('type_info', {})
        print(f"\nNistagmus Tipi: {type_info.get('description', 'N/A')}")
        print(f"Tipik Frekans: {type_info.get('typical_frequency', 'N/A')}")
        print(f"Özellikler: {', '.join(type_info.get('features', []))}")
        print(f"Yaygın Nedenler: {', '.join(type_info.get('common_causes', []))}")

if __name__ == "__main__":
    test_model() 