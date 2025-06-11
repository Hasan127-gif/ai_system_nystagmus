#!/usr/bin/env python3
"""
NISTAGMUS SINIFLANDIRMA MODELİ
=============================
PyTorch tabanlı basit MLP ile nistagmus ve şaşılık sınıflandırması.
PyTorch yoksa kural tabanlı fallback sistemi.
"""

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

import numpy as np
from typing import Dict, Tuple, Any
import logging
import os

logger = logging.getLogger(__name__)

if PYTORCH_AVAILABLE:
    class SimpleNystagmusNet(nn.Module):
        """Nistagmus ve şaşılık tespiti için basit MLP."""
        
        def __init__(self, input_dim: int = 5):
            super().__init__()
            
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 2)  # [nistagmus_prob, strabismus_prob]
            )
            
        def forward(self, x):
            return torch.sigmoid(self.fc(x))

class NystagmusClassifier:
    """Nistagmus sınıflandırıcı ana sınıfı."""
    
    def __init__(self, model_path: str = "models/nystagmus_model.pth"):
        self.model_path = model_path
        self.model = None
        self.pytorch_available = PYTORCH_AVAILABLE
        
        if self.pytorch_available:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._load_or_create_model()
        else:
            logger.info("PyTorch bulunamadı, kural tabanlı sınıflandırma kullanılacak")
    
    def _load_or_create_model(self):
        """Model yükleme veya oluşturma (sadece PyTorch varsa)."""
        if not self.pytorch_available:
            return
            
        self.model = SimpleNystagmusNet(input_dim=5)
        
        if os.path.exists(self.model_path):
            try:
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Model yüklendi: {self.model_path}")
            except Exception as e:
                logger.warning(f"Model yüklenemedi, varsayılan kullanılıyor: {e}")
                self._initialize_default_weights()
        else:
            logger.info("Model dosyası bulunamadı, varsayılan ağırlıklar kullanılıyor")
            self._initialize_default_weights()
        
        self.model.to(self.device)
        self.model.eval()
    
    def _initialize_default_weights(self):
        """Varsayılan ağırlıklar (klinik kurallara dayalı)."""
        if not self.pytorch_available:
            return
            
        with torch.no_grad():
            # İlk katman: frekans ve açı öznitelikleri önemli
            self.model.fc[0].weight[0:8, 0] = 0.8  # nystagmus_frequency
            self.model.fc[0].weight[8:16, 1] = 0.8  # strabismus_angle
            
            # Bias'ları sıfırla
            self.model.fc[0].bias.zero_()
            self.model.fc[3].bias.zero_()
            self.model.fc[5].bias.zero_()
    
    def predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Özniteliklerden nistagmus ve şaşılık varlığını tahmin eder.
        """
        try:
            if self.pytorch_available and self.model is not None:
                return self._pytorch_predict(features)
            else:
                return self._rule_based_predict(features)
                
        except Exception as e:
            logger.error(f"Tahmin hatası: {str(e)}")
            return {
                'predictions': {'nystagmus_probability': 0.0, 'strabismus_probability': 0.0,
                              'nystagmus_detected': False, 'strabismus_detected': False},
                'classification': {'nystagmus': 'YOK', 'strabismus': 'YOK'},
                'regression': {'nystagmus_frequency': 0.0, 'strabismus_angle': 0.0}
            }
    
    def _pytorch_predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """PyTorch ile tahmin."""
        # Öznitelikleri normalize et
        normalized_features = self._normalize_features(features)
        
        # Tensor'e dönüştür
        input_tensor = torch.FloatTensor(normalized_features).unsqueeze(0).to(self.device)
        
        # Tahmin yap
        with torch.no_grad():
            probabilities = self.model(input_tensor).cpu().numpy()[0]
        
        nystagmus_prob = float(probabilities[0])
        strabismus_prob = float(probabilities[1])
        
        # Kural tabanlı eşik kontrolü (güvenlik için)
        nystagmus_detected = self._rule_based_nystagmus_check(features) or nystagmus_prob > 0.5
        strabismus_detected = self._rule_based_strabismus_check(features) or strabismus_prob > 0.5
        
        return self._format_results(features, nystagmus_prob, strabismus_prob, 
                                  nystagmus_detected, strabismus_detected)
    
    def _rule_based_predict(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Kural tabanlı tahmin (PyTorch fallback)."""
        logger.info("Kural tabanlı sınıflandırma kullanılıyor")
        
        # Kural tabanlı kontroller
        nystagmus_detected = self._rule_based_nystagmus_check(features)
        strabismus_detected = self._rule_based_strabismus_check(features)
        
        # Basit olasılık hesaplama
        freq = features.get('nystagmus_frequency', 0.0)
        angle = features.get('strabismus_angle', 0.0)
        amplitude = features.get('movement_amplitude', 0.0)
        
        # Normalleştirilmiş puanlama (0-1 arası)
        nystagmus_prob = min(1.0, max(0.0, (freq - 1.0) / 10.0 + amplitude / 50.0))
        strabismus_prob = min(1.0, max(0.0, angle / 10.0))
        
        return self._format_results(features, nystagmus_prob, strabismus_prob,
                                  nystagmus_detected, strabismus_detected)
    
    def _format_results(self, features: Dict[str, float], nystagmus_prob: float, 
                       strabismus_prob: float, nystagmus_detected: bool, 
                       strabismus_detected: bool) -> Dict[str, Any]:
        """Sonuçları formatla."""
        return {
            'predictions': {
                'nystagmus_probability': nystagmus_prob,
                'strabismus_probability': strabismus_prob,
                'nystagmus_detected': nystagmus_detected,
                'strabismus_detected': strabismus_detected
            },
            'classification': {
                'nystagmus': 'VAR' if nystagmus_detected else 'YOK',
                'strabismus': 'VAR' if strabismus_detected else 'YOK'
            },
            'regression': {
                'nystagmus_frequency': features.get('nystagmus_frequency', 0.0),
                'strabismus_angle': features.get('strabismus_angle', 0.0)
            }
        }
    
    def _normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """Öznitelikleri normalize eder (PyTorch için)."""
        # Öznitelik normalizasyonu için istatistikler
        feature_stats = {
            'nystagmus_frequency': {'mean': 5.0, 'std': 3.0},
            'strabismus_angle': {'mean': 5.0, 'std': 4.0},
            'movement_amplitude': {'mean': 10.0, 'std': 8.0},
            'regularity': {'mean': 0.5, 'std': 0.3},
            'strabismus_stability': {'mean': 3.0, 'std': 2.0}
        }
        
        normalized = []
        for feature_name in ['nystagmus_frequency', 'strabismus_angle', 
                           'movement_amplitude', 'regularity', 'strabismus_stability']:
            value = features.get(feature_name, 0.0)
            stats = feature_stats[feature_name]
            
            # Z-score normalizasyon
            normalized_value = (value - stats['mean']) / stats['std']
            normalized.append(normalized_value)
        
        return np.array(normalized, dtype=np.float32)
    
    def _rule_based_nystagmus_check(self, features: Dict[str, float]) -> bool:
        """Kural tabanlı nistagmus kontrolü."""
        freq = features.get('nystagmus_frequency', 0.0)
        amplitude = features.get('movement_amplitude', 0.0)
        
        # Tipik nistagmus: 2-15 Hz arası frekans, belirgin hareket
        return freq >= 2.0 and freq <= 15.0 and amplitude >= 5.0
    
    def _rule_based_strabismus_check(self, features: Dict[str, float]) -> bool:
        """Kural tabanlı şaşılık kontrolü."""
        angle = features.get('strabismus_angle', 0.0)
        stability = features.get('strabismus_stability', 0.0)
        
        # Belirgin şaşılık: >2 derece açı, kararlı sapma
        return angle >= 2.0 and stability <= 5.0
    
    def classify(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Alias for predict method - backward compatibility.
        Özniteliklerden nistagmus ve şaşılık varlığını tahmin eder.
        """
        return self.predict(features)
    
    def evaluate(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Another alias for predict method.
        """
        return self.predict(features)

def create_simple_classifier() -> NystagmusClassifier:
    """Basit sınıflandırıcı oluşturur."""
    return NystagmusClassifier() 