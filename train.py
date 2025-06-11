"""
Nistagmus Tespit Modeli Eğitim Modülü
-------------------------------------
Bu modül, nistagmus tespiti için modeli eğitmek üzere kullanılır.
Video veya görüntü verilerinden nistagmus özelliklerini çıkarır ve bir model eğitir.
İyileştirilmiş: Veri dengeleme ve çapraz doğrulama eklendi.
"""

import os
import sys
import cv2
import numpy as np
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import time
import traceback
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Veri Dengeleme için SMOTE
from imblearn.over_sampling import SMOTE
# Derin öğrenme entegrasyonu için
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

# ai_system/detector.py'yi import etmek için ana dizine path ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai_system.detector import NistagmusDetector

# Logging yapılandırması
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_training')

# Sabit dizin tanımları
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "ai_system", "data")
TRAINING_DIR = os.path.join(DATA_DIR, "training")
VALIDATION_DIR = os.path.join(DATA_DIR, "validation")
MODEL_DIR = os.path.join(ROOT_DIR, "ai_system", "models", "trained")

# Veri dizinlerini oluştur
def setup_data_directories():
    """Veri dizinlerini oluşturur."""
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    os.makedirs(os.path.join(TRAINING_DIR, "nistagmus"), exist_ok=True)
    os.makedirs(os.path.join(TRAINING_DIR, "normal"), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "nistagmus"), exist_ok=True)
    os.makedirs(os.path.join(VALIDATION_DIR, "normal"), exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    logger.info(f"Veri dizinleri oluşturuldu:")
    logger.info(f"  - Eğitim: {TRAINING_DIR}")
    logger.info(f"  - Validasyon: {VALIDATION_DIR}")
    logger.info(f"  - Model: {MODEL_DIR}")

def list_training_data():
    """Eğitim verilerini listeler."""
    nistagmus_videos = glob.glob(os.path.join(TRAINING_DIR, "nistagmus", "*.mp4"))
    nistagmus_videos += glob.glob(os.path.join(TRAINING_DIR, "nistagmus", "*.avi"))
    nistagmus_videos += glob.glob(os.path.join(TRAINING_DIR, "nistagmus", "*.mov"))
    
    normal_videos = glob.glob(os.path.join(TRAINING_DIR, "normal", "*.mp4"))
    normal_videos += glob.glob(os.path.join(TRAINING_DIR, "normal", "*.avi"))
    normal_videos += glob.glob(os.path.join(TRAINING_DIR, "normal", "*.mov"))
    
    logger.info(f"Nistagmus eğitim videoları: {len(nistagmus_videos)}")
    logger.info(f"Normal eğitim videoları: {len(normal_videos)}")
    
    for video in nistagmus_videos:
        logger.info(f"  - Nistagmus: {os.path.basename(video)}")
    
    for video in normal_videos:
        logger.info(f"  - Normal: {os.path.basename(video)}")
    
    return {
        "nistagmus": nistagmus_videos,
        "normal": normal_videos
    }

def extract_features_from_video(video_path, detector):
    """Video dosyasından göz hareketi özellikleri çıkarır."""
    features = []
    
    # Video sınıfını belirle (dosya yolundan)
    is_nistagmus = "nistagmus" in os.path.normpath(video_path).lower()
    label = 1 if is_nistagmus else 0
    
    logger.info(f"Video işleniyor: {os.path.basename(video_path)} (Etiket: {'Nistagmus' if is_nistagmus else 'Normal'})")
    
    # Video'yu aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Video açılamadı: {video_path}")
        return []
    
    # Video özelliklerini al
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Varsayılan FPS
    
    # Analiz için veri yapıları
    left_positions_x = []
    left_positions_y = []
    right_positions_x = []
    right_positions_y = []
    timestamps = []
    
    # Kare sayacı
    frame_idx = 0
    start_time = time.time()
    
    # Yüz tespit edilen kare sayısı
    face_detected_frames = 0
    
    try:
        # Kareler üzerinde analiz yap
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Her 2. kareyi işle (işlem yükünü azalt)
            if frame_idx % 2 == 0:
                # RGB'ye dönüştür
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Işık normalizasyonu (CLAHE)
                lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
                
                # Yüz işaretlerini tespit et
                results = detector.face_mesh.process(enhanced_rgb)
                
                # İşaretleri çıkar
                if results.multi_face_landmarks:
                    face_detected_frames += 1
                    left_iris, right_iris = detector._extract_eye_landmarks(
                        results.multi_face_landmarks[0], frame.shape
                    )
                    
                    # Göz konumlarını kaydet
                    left_positions_x.append(left_iris[0])
                    left_positions_y.append(left_iris[1])
                    right_positions_x.append(right_iris[0])
                    right_positions_y.append(right_iris[1])
                    
                    # Zaman damgası ekle
                    timestamps.append(frame_idx / fps)
            
            frame_idx += 1
            
            # İlerlemeyi göster
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                logger.info(f"  İşlenen kare: {frame_idx}/{frame_count} "
                            f"({frame_idx/frame_count*100:.1f}%) - "
                            f"Geçen süre: {elapsed:.2f}s - "
                            f"Yüz tespit oranı: {face_detected_frames/(frame_idx//2+1)*100:.1f}%")
    
    except Exception as e:
        logger.error(f"Video analiz hatası: {str(e)}")
        traceback.print_exc()
    
    cap.release()
    
    # Yeterli veri var mı?
    if len(timestamps) < 30:
        logger.warning(f"Yetersiz veri: {len(timestamps)} kare. En az 30 kare gerekli.")
        return []
    
    try:
        # Göz hareketi analizi - tam segment analizi
        full_video_analysis_results = detector._analyze_eye_movements(
            np.array(left_positions_x),
            np.array(left_positions_y),
            np.array(right_positions_x),
            np.array(right_positions_y),
            np.array(timestamps),
            fps
        )
        
        # Hata kontrolü
        if isinstance(full_video_analysis_results, dict) and "error" in full_video_analysis_results:
            logger.error(f"Analiz hatası: {full_video_analysis_results['error']}")
            return []
        
        # Tam video özellikleri
        all_features = []
        
        # Görüş penceresi yaklaşımı - Daha fazla özellik üretmek için videoyu segmentlere böl
        window_sizes = [30, 45, 60]  # 1, 1.5 ve 2 saniyelik pencereler
        
        # Her segment boyutu için pencere özelliklerini ekstra
        for window_size in window_sizes:
            if len(timestamps) < window_size:
                continue
                
            # Kayan pencereler oluştur
            for start_idx in range(0, len(timestamps) - window_size + 1, window_size // 2):
                segment_left_x = left_positions_x[start_idx:start_idx + window_size]
                segment_left_y = left_positions_y[start_idx:start_idx + window_size]
                segment_right_x = right_positions_x[start_idx:start_idx + window_size]
                segment_right_y = right_positions_y[start_idx:start_idx + window_size]
                segment_timestamps = timestamps[start_idx:start_idx + window_size]
                
                # Segmenti analiz et
                segment_analysis_results = detector._analyze_eye_movements(
                    np.array(segment_left_x),
                    np.array(segment_left_y),
                    np.array(segment_right_x),
                    np.array(segment_right_y),
                    np.array(segment_timestamps),
                    fps
                )
                
                # Analiz başarılı ise özellik vektörü oluştur
                if isinstance(segment_analysis_results, dict) and "details" in segment_analysis_results:
                    segment_features = _create_feature_vector(
                        segment_analysis_results, 
                        label,
                        video_path,
                        f"segment_{start_idx}_{start_idx+window_size}"
                    )
                    all_features.extend(segment_features)
        
        # Tam video analizi için özellik vektörü
        if isinstance(full_video_analysis_results, dict) and "details" in full_video_analysis_results:
            full_video_features = _create_feature_vector(
                full_video_analysis_results, 
                label, 
                video_path,
                "full_video"
            )
            all_features.extend(full_video_features)
        
        return all_features
        
    except Exception as e:
        logger.error(f"Özellik çıkarma hatası: {str(e)}")
        traceback.print_exc()
        return []

def _create_feature_vector(analysis_results, label, video_path, segment_info=""):
    """Analiz sonuçlarından özellik vektörü oluşturur."""
    if not isinstance(analysis_results, dict) or "details" not in analysis_results:
        return []
        
    details = analysis_results["details"]
    confidence = analysis_results.get("confidence", 0)
    has_nistagmus = analysis_results.get("has_nistagmus", False)
    
    # Sol göz özellikleri
    left_eye = details.get("left_eye", {})
    left_freq = left_eye.get("dominant_frequency", 0)
    left_power = left_eye.get("dominant_power", 0)
    left_spv = left_eye.get("max_spv", 0)
    
    # Sağ göz özellikleri
    right_eye = details.get("right_eye", {})
    right_freq = right_eye.get("dominant_frequency", 0)
    right_power = right_eye.get("dominant_power", 0)
    right_spv = right_eye.get("max_spv", 0)
    
    # Hareket tipi
    horizontal_score = details.get("horizontal", 0)
    vertical_score = details.get("vertical", 0)
    rotatory_score = details.get("rotatory", 0)
    
    # Şaşılık özellikleri
    strabismus = details.get("strabismus", {})
    strabismus_detected = strabismus.get("detected", False)
    strabismus_angle = strabismus.get("angle", 0)
    strabismus_h_angle = strabismus.get("horizontal_angle", 0)
    strabismus_v_angle = strabismus.get("vertical_angle", 0)
    
    # İleri Özellikler
    avg_spv = (left_spv + right_spv) / 2 if left_spv is not None and right_spv is not None else 0
    freq_diff = abs(left_freq - right_freq) if left_freq is not None and right_freq is not None else 0
    power_ratio = left_power / (right_power + 1e-6) if left_power is not None and right_power is not None else 0
    
    # Özellik vektörü
    feature_vector = {
        "left_freq": float(left_freq),
        "left_power": float(left_power),
        "left_spv": float(left_spv),
        "right_freq": float(right_freq),
        "right_power": float(right_power),
        "right_spv": float(right_spv),
        "horizontal_score": float(horizontal_score),
        "vertical_score": float(vertical_score),
        "rotatory_score": float(rotatory_score),
        "confidence": float(confidence),
        "strabismus_detected": bool(strabismus_detected),
        "strabismus_angle": float(strabismus_angle),
        "strabismus_h_angle": float(strabismus_h_angle),
        "strabismus_v_angle": float(strabismus_v_angle),
        "avg_spv": float(avg_spv),
        "freq_diff": float(freq_diff),
        "power_ratio": float(power_ratio),
        "has_nistagmus": bool(has_nistagmus),
        "label": label,
        "video_path": video_path,
        "segment_info": segment_info
    }
    
    return [feature_vector]

# LSTM tabanlı özellik çıkarıcı (Tensorflow/Keras)
class LSTMFeatureExtractor(BaseEstimator, TransformerMixin):
    """LSTM tabanlı özellik çıkarıcı transformatör"""
    
    def __init__(self, input_shape=(90, 4), hidden_units=64):
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.model = None
    
    def fit(self, X, y=None):
        """LSTM modelini oluştur ve eğit"""
        # LSTM modeli
        inputs = tf.keras.Input(shape=self.input_shape)
        lstm = tf.keras.layers.LSTM(self.hidden_units, return_sequences=False)(inputs)
        outputs = tf.keras.layers.Dense(32, activation='relu')(lstm)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Eğitim verisinden mini-batch oluştur (tekli özellik vektörlerini zaman serilerine dönüştür)
        if X.shape[1] >= self.input_shape[0] * self.input_shape[1]:
            # Veriyi yeniden şekillendir
            X_seq = X.values.reshape(-1, self.input_shape[0], self.input_shape[1])
            
            # Autoencoder gibi eğit
            self.model.compile(optimizer='adam', loss='mse')
            self.model.fit(X_seq, X_seq, epochs=10, batch_size=32, verbose=0)
        
        return self
    
    def transform(self, X):
        """Özellikleri dönüştür (LSTM çıktıları)"""
        if self.model is None:
            logger.warning("LSTM modeli eğitilmedi, ham özellikleri kullanıyoruz")
            return X
        
        try:
            # Veriyi yeniden şekillendir
            X_seq = X.values.reshape(-1, self.input_shape[0], self.input_shape[1])
            # LSTM özelliklerini çıkar
            lstm_features = self.model.predict(X_seq)
            return lstm_features
        except Exception as e:
            logger.error(f"LSTM dönüşümü hatası: {str(e)}")
            return X

def train_model(features_df):
    """
    Özellikleri kullanarak sınıflandırma modeli eğit
    (İyileştirilmiş: SMOTE, k-fold ve ensemble desteği)
    
    Args:
        features_df: Özellikleri içeren DataFrame
        
    Returns:
        dict: Eğitilmiş model sonuçları
    """
    logger.info("Model eğitimi başlatılıyor...")
    
    # Verileri ön işle
    logger.info(f"Toplam örnek sayısı: {len(features_df)}")
    
    if 'label' not in features_df.columns:
        logger.error("Veri setinde 'label' sütunu bulunamadı!")
        return {"success": False, "error": "Label sütunu bulunamadı"}
    
    # Etiket dağılımını görüntüle
    label_counts = features_df['label'].value_counts()
    logger.info(f"Etiket dağılımı (dengeleme öncesi): {label_counts.to_dict()}")
    
    # Eksik değerleri doldur
    features_df = features_df.fillna(0)
    
    # Özellikleri ve etiketleri ayır
    X = features_df.drop(['label', 'source'], axis=1, errors='ignore')
    y = features_df['label']
    
    # Veri dengesizliğini kontrol et ve SMOTE uygula
    if len(label_counts) > 1 and min(label_counts) < len(features_df) * 0.3:
        logger.info("Veri dengesizliği tespit edildi, SMOTE uygulanıyor...")
        try:
            # SMOTE ile dengeleme
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logger.info(f"SMOTE sonrası örnek sayısı: {len(X_resampled)}")
            
            # Yeni etiket dağılımını görüntüle
            logger.info(f"Etiket dağılımı (SMOTE sonrası): {pd.Series(y_resampled).value_counts().to_dict()}")
            
            X, y = X_resampled, y_resampled
        except Exception as e:
            logger.error(f"SMOTE hatası: {str(e)}")
            logger.info("SMOTE uygulanamadı, orijinal veri kullanılıyor")
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    logger.info(f"Eğitim seti: {X_train.shape}, Test seti: {X_test.shape}")
    
    # Eğitim pipeline'ları
    pipelines = {
        "rf": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        "gb": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=42))
        ]),
        "lstm_rf": Pipeline([
            ('scaler', StandardScaler()),
            ('lstm_features', LSTMFeatureExtractor()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    }
    
    # Hiperparametre aramalarını yapılandır
    param_grids = {
        "rf": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        "gb": {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__learning_rate': [0.01, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 10]
        },
        "lstm_rf": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 20]
        }
    }
    
    # En iyi model ve sonuçlar
    best_model = None
    best_score = -1
    best_results = {}
    
    # K-fold çapraz doğrulama
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Her model tipini dene
    for name, pipeline in pipelines.items():
        logger.info(f"{name} model eğitimi başlatılıyor...")
        try:
            # GridSearch ile hiperparametre optimizasyonu
            grid_search = GridSearchCV(
                pipeline, 
                param_grids[name],
                cv=kfold,
                scoring='f1',
                n_jobs=-1  # Tüm CPU'ları kullan
            )
            grid_search.fit(X_train, y_train)
            
            # En iyi modeli al
            best_pipeline = grid_search.best_estimator_
            
            # Test seti üzerinde değerlendir
            y_pred = best_pipeline.predict(X_test)
            
            # Metrikler
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            logger.info(f"{name} model performansı:")
            logger.info(f"  Doğruluk: {accuracy:.4f}")
            logger.info(f"  Kesinlik: {precision:.4f}")
            logger.info(f"  Duyarlılık: {recall:.4f}")
            logger.info(f"  F1 Skoru: {f1:.4f}")
            logger.info(f"  En iyi parametreler: {grid_search.best_params_}")
            
            # Sonuçları kaydet
            model_results = {
                "model_type": name,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "confusion_matrix": conf_matrix.tolist(),
                "best_params": grid_search.best_params_,
                "grid_scores": grid_search.cv_results_,
                "feature_importances": None
            }
            
            # Özellik önemini hesapla (varsa)
            if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
                feature_importance = best_pipeline.named_steps['classifier'].feature_importances_
                feature_names = X.columns
                importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
                importance_df = importance_df.sort_values('importance', ascending=False)
                model_results["feature_importances"] = importance_df.to_dict(orient='records')
            
            # En iyi modeli güncelle
            if f1 > best_score:
                best_score = f1
                best_model = best_pipeline
                best_results = model_results
                logger.info(f"Yeni en iyi model: {name} (F1: {f1:.4f})")
        
        except Exception as e:
            logger.error(f"{name} model eğitim hatası: {str(e)}")
            traceback.print_exc()
    
    # En iyi modeli kaydet
    if best_model is not None:
        logger.info(f"En iyi model kaydediliyor (F1: {best_score:.4f})...")
        
        # Model dosyasını kaydet
        model_path = os.path.join(MODEL_DIR, "default_model.pkl")
        joblib.dump(best_model, model_path)
        
        # Sonuçları JSON olarak kaydet
        results_path = os.path.join(MODEL_DIR, "model_results.json")
        with open(results_path, 'w') as f:
            json.dump(best_results, f, indent=2)
        
        logger.info(f"Model kaydedildi: {model_path}")
        logger.info(f"Sonuçlar kaydedildi: {results_path}")
        
        # Ensemble model (en iyi 2 modeli birleştir)
        # NOT: Bu, üretim uygulaması için basitleştirilmiş bir örnek
        try:
            # İkinci en iyi modeli al
            pipelines.pop(best_results["model_type"])  # En iyi modeli çıkar
            second_best_name = list(pipelines.keys())[0]
            second_best_pipeline = GridSearchCV(
                pipelines[second_best_name], 
                param_grids[second_best_name],
                cv=3,
                scoring='f1'
            ).fit(X_train, y_train).best_estimator_
            
            # Basit voting ensemble oluştur
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(
                estimators=[
                    ('best', best_model),
                    ('second', second_best_pipeline)
                ],
                voting='soft'  # Olasılıkları kullan
            )
            
            # Ensemble'ı eğit
            ensemble.fit(X_train, y_train)
            
            # Performansını değerlendir
            y_pred_ensemble = ensemble.predict(X_test)
            f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
            
            logger.info(f"Ensemble model performansı (F1): {f1_ensemble:.4f}")
            
            # Eğer ensemble daha iyiyse onu kaydet
            if f1_ensemble > best_score:
                joblib.dump(ensemble, os.path.join(MODEL_DIR, "ensemble_model.pkl"))
                logger.info(f"Ensemble model kaydedildi (F1: {f1_ensemble:.4f})")
                
                # Default model olarak da kaydet
                joblib.dump(ensemble, model_path)
                logger.info("Ensemble model, varsayılan model olarak ayarlandı")
        except Exception as e:
            logger.error(f"Ensemble oluşturma hatası: {str(e)}")
    
    return {
            "success": True,
        "model_path": model_path,
            "results": best_results
        }
    else:
        logger.error("Eğitilebilir model bulunamadı!")
        return {
            "success": False,
            "error": "Eğitilebilir model bulunamadı"
    }

def update_model_config(model_result):
    """Model sonuçlarına göre detector yapılandırmasını günceller."""
    if not model_result:
        logger.error("Model sonucu bulunamadı, yapılandırma güncellenemiyor")
        return
    
    # Meta verisini oku
    with open(model_result["meta_path"], 'r') as f:
        meta = json.load(f)
    
    # Önemli özellikleri çıkar
    importance = meta["feature_importance"]
    
    # Model performansına göre parametreleri ayarla
    config_updates = {}
    
    # Örnek: SPV eşiği için özellik önemine bak
    if "left_spv" in importance and importance["left_spv"] > 0.2:
        # Yüksek öneme sahipse SPV eşiğini düşür
        config_updates["spv_threshold"] = 15.0
    
    # Mediapipe güven eşikleri
    config_updates["min_detection_confidence"] = 0.5
    config_updates["min_tracking_confidence"] = 0.5
    
    # Yapılandırma dosyasını güncelle
    config_path = os.path.join(MODEL_DIR, "detector_config.json")
    
    # Mevcut yapılandırmayı oku (varsa)
    existing_config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
    
    # Yapılandırmayı güncelle
    updated_config = {**existing_config, **config_updates}
    
    # Yapılandırmayı kaydet
    with open(config_path, 'w') as f:
        json.dump(updated_config, f, indent=2)
    
    logger.info(f"Detector yapılandırması güncellendi: {config_path}")
    logger.info(f"Yeni yapılandırma: {updated_config}")

def set_default_model():
    """En son eğitilmiş modeli varsayılan model olarak ayarlar."""
    # En son modeli bul
    model_files = glob.glob(os.path.join(MODEL_DIR, "nistagmus_model_*.pkl"))
    if not model_files:
        logger.error("Kullanılabilir model bulunamadı")
        return
    
    # En son oluşturulana göre sırala
    latest_model = max(model_files, key=os.path.getctime)
    model_name = os.path.basename(latest_model)
    
    # Varsayılan model sembolik bağlantısı
    default_model_path = os.path.join(MODEL_DIR, "default_model.pkl")
    
    # Eski bağlantıyı kaldır (varsa)
    if os.path.exists(default_model_path):
        os.remove(default_model_path)
    
    # Sembolik bağlantı yerine kopyala (platform bağımsızlığı için)
    import shutil
    shutil.copy2(latest_model, default_model_path)
    
    logger.info(f"Varsayılan model ayarlandı: {model_name}")
    logger.info(f"Varsayılan model yolu: {default_model_path}")

def main():
    """Ana eğitim işlemi."""
    logger.info("Nistagmus Tespit Modeli Eğitimi Başlıyor")
    
    # Veri dizinlerini oluştur
    setup_data_directories()
    
    # Eğitim verilerini listele
    training_data = list_training_data()
    
    # Veri yoksa uyarı ver
    if len(training_data["nistagmus"]) == 0 and len(training_data["normal"]) == 0:
        logger.error("Eğitim verisi bulunamadı. Lütfen ai_system/data/training klasörüne videolar ekleyin.")
        logger.info("Örnek dizin yapısı:")
        logger.info("  - ai_system/data/training/nistagmus/video1.mp4")
        logger.info("  - ai_system/data/training/normal/video1.mp4")
        return
    
    # Detector nesnesi oluştur
    detector = NistagmusDetector()
    
    # Tüm videolardan özellikler çıkar
    all_features = []
    
    # Nistagmus videoları
    for video_path in training_data["nistagmus"]:
        features = extract_features_from_video(video_path, detector)
        all_features.extend(features)
    
    # Normal videolar
    for video_path in training_data["normal"]:
        features = extract_features_from_video(video_path, detector)
        all_features.extend(features)
    
    # Özellikleri dataframe'e dönüştür
    if not all_features:
        logger.error("Hiçbir videodan özellik çıkarılamadı.")
        return
    
    features_df = pd.DataFrame(all_features)
    
    # Dataframe'i kaydet
    features_path = os.path.join(MODEL_DIR, f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    features_df.to_csv(features_path, index=False)
    logger.info(f"Özellik vektörleri kaydedildi: {features_path}")
    
    # Modeli eğit
    model_result = train_model(features_df)
    
    if model_result:
        # Detector yapılandırmasını güncelle
        update_model_config(model_result)
        
        # Varsayılan modeli ayarla
        set_default_model()
    
    logger.info("Nistagmus Tespit Modeli Eğitimi Tamamlandı.")

if __name__ == "__main__":
    main() 