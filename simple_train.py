#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basit Nistagmus Tespit Modeli Eğitim Betiği
"""

import os
import sys
import cv2
import logging
import numpy as np
import pickle
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# Loglama ayarları
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('nistagmus_model_trainer')

# Proje kök dizinini belirleme
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from ai_system.detector import NistagmusDetector

class SimpleModelTrainer:
    """Basit Nistagmus tespit modeli eğitim sınıfı"""
    
    def __init__(self):
        """Eğitim sınıfını başlat"""
        self.detector = NistagmusDetector()
        self.model = None
        self.model_path = os.path.join(ROOT_DIR, 'ai_system/models/trained/default_model.pkl')
        self.config_path = os.path.join(ROOT_DIR, 'ai_system/models/trained/detector_config.json')
        self.features = []
        self.labels = []
        self.training_dir = os.path.join(ROOT_DIR, 'ai_system/data/training')
        self.validation_dir = os.path.join(ROOT_DIR, 'ai_system/data/validation')
        
        # Eğitim parametreleri
        self.config = {
            'model_type': 'RandomForest',
            'model_version': '1.0.0',
            'trained_date': datetime.now().isoformat(),
            'features': [],
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'thresholds': {
                'nistagmus_confidence': 0.65,
                'normal_confidence': 0.60
            },
            'metrics': {}
        }
        
    def extract_features_from_video(self, video_path, label):
        """Video dosyasından nistagmus özelliklerini çıkar"""
        logger.info(f"Video işleniyor: {video_path}")
        
        try:
            results = self.detector.analyze_video(video_path)
            
            if results is None or "error" in results:
                logger.warning(f"Video analiz hatası: {results.get('error', 'Bilinmeyen hata')}")
                return
                
            # Çeşitli göz hareketi özelliklerini topla
            features = []
            
            # 2023-05-12: API yapısına göre güncellendi - API sonuçlarında eye_analysis yerine data içinde özellikler var
            
            # Data bölümü kontrolü
            if "data" in results and "eye_movement_stats" in results["data"]:
                eye_stats = results["data"]["eye_movement_stats"]
                
                # Sol göz özellikleri
                if "left_eye" in eye_stats:
                    left_eye = eye_stats["left_eye"]
                    features.extend([
                        left_eye.get("position_std_x", 0),
                        left_eye.get("position_std_y", 0),
                        left_eye.get("velocity_std_x", 0),
                        left_eye.get("velocity_std_y", 0),
                        left_eye.get("max_velocity_x", 0),
                        left_eye.get("max_velocity_y", 0)
                    ])
                
                # Sağ göz özellikleri
                if "right_eye" in eye_stats:
                    right_eye = eye_stats["right_eye"]
                    features.extend([
                        right_eye.get("position_std_x", 0),
                        right_eye.get("position_std_y", 0),
                        right_eye.get("velocity_std_x", 0),
                        right_eye.get("velocity_std_y", 0),
                        right_eye.get("max_velocity_x", 0),
                        right_eye.get("max_velocity_y", 0)
                    ])
            
            # Frekans analizi özellikleri
            if "frequency_analysis" in results:
                freq = results["frequency_analysis"]
                features.extend([
                    freq.get("left_eye_dominant_frequency", 0),
                    freq.get("right_eye_dominant_frequency", 0),
                    freq.get("left_nistagmus_frequency_ratio", 0),
                    freq.get("right_nistagmus_frequency_ratio", 0)
                ])
            
            # Hareket istatistikleri
            if "movement_stats" in results:
                move_stats = results["movement_stats"]
                features.extend([
                    move_stats.get("left_eye_velocity_std", 0),
                    move_stats.get("right_eye_velocity_std", 0),
                    move_stats.get("left_fast_phases", 0),
                    move_stats.get("right_fast_phases", 0),
                    move_stats.get("left_direction_ratio", 0),
                    move_stats.get("right_direction_ratio", 0)
                ])
            
            # Nistagmus skoru
            features.extend([
                results.get("confidence", 0),
                results.get("total_score", 0),
                results.get("rhythmicity_score", 0),
                results.get("jerkiness_score", 0),
                results.get("direction_consistency_score", 0),
                1.0 if results.get("nystagmus_detected", False) else 0.0
            ])
            
            # Temel göz analizi
            if "basic_analysis" in results:
                basic = results["basic_analysis"]
                features.extend([
                    basic.get("eye_distance", 0),
                    basic.get("horizontal_difference", 0),
                    basic.get("vertical_difference", 0),
                    basic.get("alignment_score", 0),
                    basic.get("initial_strabismus_estimate", 0)
                ])
            
            # İşleme bilgileri
            if "processing" in results:
                proc = results["processing"]
                features.append(proc.get("frames_processed", 0) / 300.0)  # Normalize et
            
            # Yeterli özellik varsa listeye ekle
            if len(features) > 5:  # En az 5 özellik olmalı
                self.features.append(features)
                self.labels.append(1 if label == "nistagmus" else 0)
                
                # Feature isimlerini kaydet
                if not self.config['features']:
                    self.config['features'] = [
                        # Sol göz
                        "left_position_std_x", "left_position_std_y", 
                        "left_velocity_std_x", "left_velocity_std_y",
                        "left_max_velocity_x", "left_max_velocity_y",
                        # Sağ göz
                        "right_position_std_x", "right_position_std_y", 
                        "right_velocity_std_x", "right_velocity_std_y",
                        "right_max_velocity_x", "right_max_velocity_y",
                        # Frekans
                        "left_dominant_freq", "right_dominant_freq",
                        "left_freq_ratio", "right_freq_ratio",
                        # Hareket
                        "left_velocity_std", "right_velocity_std",
                        "left_fast_phases", "right_fast_phases",
                        "left_direction_ratio", "right_direction_ratio",
                        # Nistagmus skorları
                        "confidence", "total_score", 
                        "rhythmicity_score", "jerkiness_score", 
                        "direction_consistency_score", "is_nystagmus",
                        # Temel göz
                        "eye_distance", "horizontal_difference", 
                        "vertical_difference", "alignment_score", 
                        "strabismus_estimate",
                        # İşleme
                        "frames_processed_ratio"
                    ]
                
                logger.info(f"Video özellikleri çıkarıldı: {len(features)} özellik")
            else:
                # Doğrudan video sonuçlarını özellik olarak kullan (fallback)
                flat_features = self._flatten_dict(results)
                
                if len(flat_features) > 5:
                    self.features.append(flat_features)
                    self.labels.append(1 if label == "nistagmus" else 0)
                    logger.info(f"Video özellikleri (düzleştirilmiş) çıkarıldı: {len(flat_features)} özellik")
                else:
                    logger.warning("Video özellik çıkarımı başarısız.")
        
        except Exception as e:
            logger.error(f"Video işleme hatası: {str(e)}")
            
    def _flatten_dict(self, d, parent_key='', sep='_'):
        """Sözlüğü düzleştirme - iç içe yapıları düz liste haline getirir"""
        items = []
        
        if not isinstance(d, dict):
            return [d]  # Sözlük değilse, değeri liste olarak döndür
            
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                # İç içe sözlük
                items.extend(self._flatten_dict(v, new_key, sep=sep))
            elif isinstance(v, (list, tuple)) and len(v) > 0 and not isinstance(v[0], (int, float, str, bool)):
                # İç içe liste (ancak basit veri tipleri değilse)
                for i, item in enumerate(v):
                    items.extend(self._flatten_dict(item, f"{new_key}{sep}{i}", sep=sep))
            elif isinstance(v, (bool, int, float, str)):
                # Sayısal değer veya string ise
                items.append(float(v) if isinstance(v, (int, float)) else (1.0 if v is True else 0.0 if v is False else 0.0))
                
        return items
    
    def prepare_data(self):
        """Eğitim ve doğrulama verileri hazırla"""
        logger.info("Veri hazırlama başladı")
        
        # Eğitim verilerini topla
        normal_dir = os.path.join(self.training_dir, "normal")
        nistagmus_dir = os.path.join(self.training_dir, "nistagmus")
        
        # Normal videolar
        for video_file in os.listdir(normal_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov', '.MP4', '.MOV')):
                video_path = os.path.join(normal_dir, video_file)
                self.extract_features_from_video(video_path, "normal")
        
        # Nistagmus videolar
        for video_file in os.listdir(nistagmus_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov', '.MP4', '.MOV')):
                video_path = os.path.join(nistagmus_dir, video_file)
                self.extract_features_from_video(video_path, "nistagmus")
        
        logger.info(f"Toplam {len(self.features)} örnek toplandı: {sum(self.labels)} nistagmus, {len(self.labels) - sum(self.labels)} normal")
        
        # Doğrulama verileri için test yapma (daha sonra ayrı bir değerlendirme için kullanılabilir)
        # Bu kısım şu an için atlanıyor
        
    def train_model(self):
        """Modeli eğit"""
        logger.info("Model eğitimi başladı")
        
        if len(self.features) < 10:
            logger.error(f"Yetersiz veri sayısı: {len(self.features)}. En az 10 örnek gerekli.")
            return False
            
        # Veriyi eğitim ve test setlerine ayır
        X = np.array(self.features)
        y = np.array(self.labels)
        
        # Sınıf dengesizliği varsa SMOTE uygula
        if len(self.features) >= 10 and sum(self.labels) >= 3 and (len(self.labels) - sum(self.labels)) >= 3:
            logger.info("SMOTE uygulanıyor")
            try:
                smote = SMOTE(random_state=42)
                X, y = smote.fit_resample(X, y)
                logger.info(f"SMOTE sonrası örnek sayısı: {len(y)}")
            except Exception as e:
                logger.warning(f"SMOTE uygulanamadı: {str(e)}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Random Forest modeli oluştur ve eğit
        self.model = RandomForestClassifier(
            n_estimators=self.config['params']['n_estimators'],
            max_depth=self.config['params']['max_depth'],
            random_state=self.config['params']['random_state']
        )
        
        self.model.fit(X_train, y_train)
        
        # Modeli değerlendir
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Metrikleri kaydet
        self.config['metrics'] = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
        
        logger.info(f"Model eğitimi tamamlandı. Doğruluk: {accuracy:.2f}")
        logger.info(f"Performans metrikleri: {self.config['metrics']}")
        
        return True
        
    def save_model(self):
        """Eğitilmiş modeli kaydet"""
        logger.info(f"Model kaydediliyor: {self.model_path}")
        
        # Model dizinini oluştur
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Modeli kaydet
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # Yapılandırmayı kaydet
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
            
        logger.info("Model ve yapılandırma başarıyla kaydedildi.")
        
        return True
        
    def run(self):
        """Tam eğitim sürecini çalıştır"""
        logger.info("Nistagmus model eğitimi başlıyor")
        start_time = time.time()
        
        self.prepare_data()
        
        if self.train_model():
            self.save_model()
            
        elapsed_time = time.time() - start_time
        logger.info(f"Eğitim süreci tamamlandı. Toplam süre: {elapsed_time:.2f} saniye")
        
        
if __name__ == "__main__":
    trainer = SimpleModelTrainer()
    trainer.run() 