#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Analiz Betiği - Nistagmus AI modelini ve tahminlerini analiz eder
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

# Loglama ayarları
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_analyzer')

# Proje kök dizinini belirleme
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

class ModelAnalyzer:
    """Model analiz sınıfı"""
    
    def __init__(self):
        """Model analiz sınıfını başlat"""
        self.model_path = os.path.join(ROOT_DIR, 'ai_system/models/trained/default_model.pkl')
        self.config_path = os.path.join(ROOT_DIR, 'ai_system/models/trained/detector_config.json')
        self.model = None
        self.config = None
        self.features = []
        self.feature_names = []
        
    def load_model(self):
        """Eğitilmiş modeli ve konfigürasyonu yükle"""
        try:
            logger.info(f"Model yükleniyor: {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Konfigürasyon yükleniyor: {self.config_path}")
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            self.feature_names = self.config.get('features', [])
            
            logger.info(f"Model tipi: {self.config.get('model_type')}")
            logger.info(f"Model versiyonu: {self.config.get('model_version')}")
            logger.info(f"Eğitim tarihi: {self.config.get('trained_date')}")
            logger.info(f"Özellik sayısı: {len(self.feature_names)}")
            
            return True
        except Exception as e:
            logger.error(f"Model yükleme hatası: {str(e)}")
            return False
            
    def analyze_model_structure(self):
        """Model yapısını analiz et"""
        if not self.model:
            logger.error("Model yüklenmemiş")
            return
            
        logger.info("Model yapısı analiz ediliyor...")
        
        # Model tipi kontrolü
        model_type = type(self.model).__name__
        logger.info(f"Model türü: {model_type}")
        
        # RandomForest modeli ise ağaç sayısı ve derinliği göster
        if hasattr(self.model, 'n_estimators'):
            logger.info(f"Ağaç sayısı: {self.model.n_estimators}")
            
        if hasattr(self.model, 'max_depth'):
            logger.info(f"Maksimum derinlik: {self.model.max_depth}")
            
        # Sınıf dengesi
        if hasattr(self.model, 'classes_'):
            logger.info(f"Sınıflar: {self.model.classes_}")
            
        # Özellik önemleri
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            logger.info("Özellik önemleri:")
            
            feature_importance = {}
            for i, importance in enumerate(importances):
                feature_name = self.feature_names[i] if i < len(self.feature_names) else f"Özellik_{i}"
                feature_importance[feature_name] = importance
                logger.info(f"  - {feature_name}: {importance:.4f}")
                
            # En önemli 10 özelliği göster
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("En önemli 10 özellik:")
            for name, importance in sorted_features[:10]:
                logger.info(f"  - {name}: {importance:.4f}")
                
    def generate_test_samples(self, num_samples=10):
        """Test için örnek özellik vektörleri oluştur"""
        if not self.feature_names:
            logger.error("Özellik isimleri yüklenmemiş")
            return None
            
        logger.info(f"{num_samples} test örneği oluşturuluyor...")
        
        # Rastgele örnekler oluştur
        samples = []
        
        # Normal örnek (tüm değerler düşük)
        normal_sample = np.zeros(len(self.feature_names))
        samples.append(("Normal (sıfır değerler)", normal_sample))
        
        # Nistagmus örneği (yüksek hız ve frekans değerleri)
        nystagmus_sample = np.zeros(len(self.feature_names))
        
        # Frekans değerlerini ayarla (2-15 Hz arası tipik nistagmus)
        freq_indices = [i for i, name in enumerate(self.feature_names) 
                        if 'freq' in name.lower() or 'dominant' in name.lower()]
        for idx in freq_indices:
            nystagmus_sample[idx] = 8.0  # 8 Hz tipik nistagmus frekansı
            
        # Hız değerlerini ayarla (yüksek hızlar)
        vel_indices = [i for i, name in enumerate(self.feature_names) 
                       if 'velocity' in name.lower() or 'speed' in name.lower()]
        for idx in vel_indices:
            nystagmus_sample[idx] = 40.0  # 40 derece/saniye hız
            
        # Skor değerlerini ayarla
        score_indices = [i for i, name in enumerate(self.feature_names) 
                         if 'score' in name.lower()]
        for idx in score_indices:
            nystagmus_sample[idx] = 0.8  # Yüksek skor
            
        samples.append(("Nistagmus (tipik değerler)", nystagmus_sample))
        
        # Farklı varyasyonlar oluştur
        for i in range(num_samples - 2):
            # Rastgele değerlerle örnek oluştur
            sample = np.random.random(len(self.feature_names))
            
            # Bazı özellikleri nistagmus benzeri yap
            if i % 2 == 0:  # Çift indeksli örnekler nistagmus benzeri
                for idx in freq_indices:
                    sample[idx] = 3.0 + np.random.random() * 10  # 3-13 Hz
                for idx in vel_indices:
                    sample[idx] = 15.0 + np.random.random() * 35  # 15-50 derece/saniye
                
                samples.append((f"Nistagmus varyasyon {i}", sample))
            else:
                samples.append((f"Normal varyasyon {i}", sample))
                
        return samples
                
    def predict_samples(self, samples):
        """Örnekler için tahmin yap"""
        if not self.model or not samples:
            logger.error("Model veya örnekler yüklenmemiş")
            return
            
        logger.info("Örnek tahminleri yapılıyor...")
        
        for name, sample in samples:
            # Tahmin yap
            try:
                pred_class = self.model.predict([sample])[0]
                pred_proba = self.model.predict_proba([sample])[0]
                
                # Sonuçları yazdır
                logger.info(f"Örnek: {name}")
                logger.info(f"  - Tahmin: {pred_class} (0=Normal, 1=Nistagmus)")
                logger.info(f"  - Olasılık: Normal={pred_proba[0]:.4f}, Nistagmus={pred_proba[1]:.4f}")
                
                # Karar ağaçlarının kararlarını kontrol et (RandomForest ise)
                if hasattr(self.model, 'estimators_'):
                    tree_decisions = [tree.predict([sample])[0] for tree in self.model.estimators_]
                    normal_votes = tree_decisions.count(0)
                    nystagmus_votes = tree_decisions.count(1)
                    logger.info(f"  - Ağaç oyları: Normal={normal_votes}, Nistagmus={nystagmus_votes}")
            except Exception as e:
                logger.error(f"Tahmin hatası: {str(e)}")
                
    def analyze_model_data(self):
        """Model eğitim verilerini analiz et"""
        if not self.model:
            logger.error("Model yüklenmemiş")
            return
            
        logger.info("Model eğitim verileri analiz ediliyor...")
        
        # RandomForest için her ağacın kullandığı özellikler ve kararları kontrol et
        if hasattr(self.model, 'estimators_'):
            # Ağaç sayısı
            n_trees = len(self.model.estimators_)
            logger.info(f"Model {n_trees} ağaçtan oluşuyor")
            
            # İlk birkaç ağacı analiz et
            for i in range(min(3, n_trees)):
                tree = self.model.estimators_[i]
                
                # Ağaç özelliklerini yazdır
                if hasattr(tree, 'tree_'):
                    tree_struct = tree.tree_
                    logger.info(f"Ağaç {i+1}:")
                    logger.info(f"  - Düğüm sayısı: {tree_struct.node_count}")
                    logger.info(f"  - Maksimum derinlik: {tree_struct.max_depth}")
                    logger.info(f"  - Sınıf dağılımı: {tree_struct.value.squeeze()}")
            
            logger.info("Sonuç: Model yapısı incelendi")
        
    def run(self):
        """Model analizini çalıştır"""
        # Modeli yükle
        if not self.load_model():
            logger.error("Model yüklenemedi, analiz tamamlanamadı.")
            return
            
        # Model yapısını analiz et
        self.analyze_model_structure()
        
        # Örnek tahminler
        samples = self.generate_test_samples(6)
        if samples:
            self.predict_samples(samples)
            
        # Model eğitim verilerini analiz et
        self.analyze_model_data()
        
        logger.info("Model analizi tamamlandı.")
        
if __name__ == "__main__":
    analyzer = ModelAnalyzer()
    analyzer.run() 