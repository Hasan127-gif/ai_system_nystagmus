#!/usr/bin/env python3
"""
ÖZNİTELİK ÇIKARIMI MODÜLü
========================
Göz hareketi verilerinden nistagmus ve şaşılık özniteliklerini çıkarır.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def extract_movement_features(y_positions: List[float], 
                            x_differences: List[float], 
                            frame_rate: float,
                            calibration_params: Dict = None) -> Dict[str, float]:
    """
    Göz hareketi verilerinden temel öznitelikleri çıkarır.
    
    Args:
        y_positions: Sol gözün dikey pozisyonları (nistagmus için)
        x_differences: Sağ-sol göz yatay farkları (şaşılık için)
        frame_rate: Video kare hızı
        calibration_params: Kalibrasyon parametreleri
        
    Returns:
        Dict: Çıkarılan öznitelikler
    """
    features = {}
    
    try:
        # 1. Nistagmus frekansı (FFT tabanlı)
        features['nystagmus_frequency'] = compute_nystagmus_frequency(y_positions, frame_rate)
        
        # 2. Şaşılık açısı (piksel farkından dereceye)
        features['strabismus_angle'] = pixel_to_prism_diopter(x_differences, calibration_params)
        
        # 3. Hareket büyüklüğü (standart sapma)
        features['movement_amplitude'] = float(np.std(y_positions)) if y_positions else 0.0
        
        # 4. Düzenlilik ölçüsü (varyasyon katsayısı)
        mean_pos = np.mean(y_positions) if y_positions else 0
        features['regularity'] = float(features['movement_amplitude'] / abs(mean_pos)) if mean_pos != 0 else 0.0
        
        # 5. Şaşılık kararlılığı
        features['strabismus_stability'] = float(np.std(x_differences)) if x_differences else 0.0
        
        logger.debug(f"Öznitelikler çıkarıldı: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Öznitelik çıkarma hatası: {str(e)}")
        return {
            'nystagmus_frequency': 0.0,
            'strabismus_angle': 0.0,
            'movement_amplitude': 0.0,
            'regularity': 0.0,
            'strabismus_stability': 0.0
        }

def compute_nystagmus_frequency(y_positions: List[float], frame_rate: float) -> float:
    """
    Mevcut sistemiyle uyumlu nistagmus frekansı hesaplama.
    """
    # Mevcut analyze_utils'deki fonksiyonu kullan
    try:
        from analysis_utils import calculate_nystagmus_frequency_unified
        return calculate_nystagmus_frequency_unified(y_positions, frame_rate)
    except ImportError:
        # Fallback basit FFT
        if len(y_positions) < 2:
            return 0.0
        
        y = np.array(y_positions, dtype=float)
        y = y - np.mean(y)  # DC çıkar
        
        if np.std(y) < 1e-6:
            return 0.0
            
        fft_vals = np.fft.fft(y)
        freqs = np.fft.fftfreq(len(y), d=1.0/frame_rate)
        
        power = np.abs(fft_vals)
        power[0] = 0  # DC sıfırla
        
        positive_freqs = freqs[:len(freqs)//2]
        positive_power = power[:len(power)//2]
        
        if len(positive_freqs) == 0:
            return 0.0
            
        max_idx = np.argmax(positive_power[1:]) + 1  # 0 Hz atla
        return float(positive_freqs[max_idx]) if max_idx < len(positive_freqs) else 0.0

def pixel_to_prism_diopter(x_differences: List[float], 
                          calibration_params: Dict = None) -> float:
    """
    Piksel farkını prizma dioptere dönüştürür.
    
    Args:
        x_differences: Pixel cinsinden yatay göz farkları
        calibration_params: Kalibrasyon parametreleri
        
    Returns:
        float: Şaşılık açısı (derece)
    """
    if not x_differences:
        return 0.0
    
    mean_pixel_diff = float(np.mean(x_differences))
    
    # Kalibrasyon sistemi entegrasyonu
    try:
        from calibration import get_calibrator
        calibrator = get_calibrator()
        return calibrator.pixel_to_degree(mean_pixel_diff)
    except ImportError:
        # Fallback: Varsayılan kalibrasyon
        if calibration_params is None:
            # 1 piksel ≈ 0.1 derece (tipik webcam çözünürlüğü için)
            pixel_to_degree = 0.1
        else:
            pixel_to_degree = calibration_params.get('pixel_to_degree', 0.1)
        
        return abs(mean_pixel_diff * pixel_to_degree)

def extract_features(landmarks_sequence: List[List[List[float]]]) -> List[float]:
    """
    MediaPipe landmark'larından öznitelik çıkarır (test sistemi için).
    
    Args:
        landmarks_sequence: MediaPipe landmark dizisi
        
    Returns:
        List[float]: [nystagmus_freq, movement_amplitude, regularity, strabismus_angle, strabismus_stability]
    """
    try:
        if not landmarks_sequence or len(landmarks_sequence) < 3:
            return [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Sol göz pozisyonları (y ekseninde)
        left_y_positions = []
        # Göz arası farklar (x ekseninde)
        eye_x_differences = []
        
        for frame_landmarks in landmarks_sequence:
            if len(frame_landmarks) >= 473:  # MediaPipe yeterli landmark
                # Sol göz (landmark 468)
                left_eye_y = frame_landmarks[468][1] if len(frame_landmarks[468]) > 1 else 0.5
                left_y_positions.append(left_eye_y)
                
                # Sağ göz (landmark 473) 
                right_eye_x = frame_landmarks[473][0] if len(frame_landmarks[473]) > 0 else 0.5
                left_eye_x = frame_landmarks[468][0] if len(frame_landmarks[468]) > 0 else 0.5
                
                eye_x_differences.append(abs(right_eye_x - left_eye_x))
        
        # Öznitelikleri hesapla
        features_dict = extract_movement_features(
            y_positions=left_y_positions,
            x_differences=eye_x_differences,
            frame_rate=30.0  # Varsayılan FPS
        )
        
        # Test sistemi için liste formatında dön
        return [
            features_dict['nystagmus_frequency'],
            features_dict['movement_amplitude'],
            features_dict['regularity'], 
            features_dict['strabismus_angle'],
            features_dict['strabismus_stability']
        ]
        
    except Exception as e:
        logger.error(f"Landmark öznitelik çıkarma hatası: {e}")
        return [0.0, 0.0, 0.0, 0.0, 0.0]

# Export functions
__all__ = ['extract_features', 'extract_movement_features', 'compute_nystagmus_frequency', 'pixel_to_prism_diopter']

if __name__ == "__main__":
    # Test
    test_landmarks = [[[0.5, 0.5, 0.0] for _ in range(468)] for _ in range(10)]
    features = extract_features(test_landmarks)
    print(f"Test öznitelikleri: {features}") 