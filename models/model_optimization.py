"""
Model Optimizasyonu ve Kuantizasyon Modülü
------------------------------------------
Bu modül, eğitilmiş modelleri optimize etmek ve daha küçük boyutlara indirgemek için 
çeşitli teknikler sunar. Böylece modeller daha hızlı çalışır ve daha az bellek kullanır.
"""

import os
import numpy as np
import pandas as pd
import logging
import joblib
import json
from pathlib import Path
import time
import sys

# Model tipleri için gerekli kütüphaneler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import tensorflow as tf

# Optimize edilmiş modeller için klasör
MODEL_DIR = os.path.join(Path(__file__).parent, "trained")
OPTIMIZED_DIR = os.path.join(Path(__file__).parent, "optimized")

# Logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('model_optimization')

def create_dirs():
    """Gerekli dizinleri oluştur"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OPTIMIZED_DIR, exist_ok=True)

def get_latest_model():
    """En son eğitilmiş modeli bul"""
    model_path = os.path.join(MODEL_DIR, "default_model.pkl")
    if os.path.exists(model_path):
        return model_path
    
    # Belirtilen model yoksa dizindeki en son modeli bul
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]
    if not models:
        return None
    
    # En son değiştirilen modeli seç
    latest_model = max(models, key=lambda x: os.path.getmtime(os.path.join(MODEL_DIR, x)))
    return os.path.join(MODEL_DIR, latest_model)

def optimize_random_forest(model, n_features, output_path=None):
    """
    RandomForest modelini optimize eder:
    1. Ağaç sayısını azaltma
    2. Özellik seçimi
    3. Bellek ayak izini azaltma
    
    Args:
        model: RandomForestClassifier modeli
        n_features: Giriş özelliklerinin sayısı
        output_path: Kaydedilecek model yolu (opsiyonel)
        
    Returns:
        Optimize edilmiş model
    """
    logger.info("RandomForest modeli optimize ediliyor...")
    start_time = time.time()
    
    # Orijinal model boyutunu ölç
    orig_size = sys.getsizeof(model)
    
    # 1. Önemli özellikleri tespit et
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    # En önemli özellikleri seç (toplam önemin %90'ını oluşturan özellikler)
    importance_threshold = 0.9 * np.sum(feature_importances)
    cumulative_importance = 0.0
    important_indices = []
    
    for i in indices:
        cumulative_importance += feature_importances[i]
        important_indices.append(i)
        if cumulative_importance >= importance_threshold:
            break
    
    # 2. Önemli özelliklere göre yeni bir model oluştur
    n_estimators_opt = min(100, model.n_estimators)  # Ağaç sayısını azalt
    
    optimized_model = RandomForestClassifier(
        n_estimators=n_estimators_opt,
        max_depth=model.max_depth,
        random_state=42,
        n_jobs=-1,
        warm_start=False  # Bellek ayak izini azaltmak için
    )
    
    # 3. Feature selector matris oluştur (1 = seçilen özellik, 0 = kullanılmayan özellik)
    feature_mask = np.zeros(n_features, dtype=bool)
    feature_mask[important_indices] = True
    
    # 4. Meta verileri kaydet
    meta = {
        "original_size_bytes": orig_size,
        "original_n_estimators": model.n_estimators,
        "optimized_n_estimators": n_estimators_opt,
        "original_n_features": n_features,
        "optimized_n_features": len(important_indices),
        "feature_mask": feature_mask.tolist(),
        "feature_importances": {i: float(feature_importances[i]) for i in range(len(feature_importances))},
        "selected_features": important_indices
    }
    
    # Modeli kaydet
    if output_path:
        joblib.dump(optimized_model, output_path)
        
        # Meta verileri kaydet
        meta_path = output_path.replace(".pkl", "_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Model boyutunu ölç
        new_size = os.path.getsize(output_path)
        compression_ratio = orig_size / new_size if new_size > 0 else 0
        logger.info(f"Model optimize edildi: {orig_size/1024:.1f} KB -> {new_size/1024:.1f} KB ({compression_ratio:.1f}x)")
    
    elapsed = time.time() - start_time
    logger.info(f"RandomForest optimizasyonu tamamlandı ({elapsed:.2f} saniye)")
    
    return optimized_model, meta

def optimize_gradient_boosting(model, n_features, output_path=None):
    """GradientBoosting modelini optimize eder"""
    logger.info("GradientBoosting modeli optimize ediliyor...")
    start_time = time.time()
    
    # Orijinal model boyutunu ölç
    orig_size = sys.getsizeof(model)
    
    # Önemli özellikleri tespit et
    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]
    
    # En önemli özellikleri seç (toplam önemin %90'ını oluşturan özellikler)
    importance_threshold = 0.9 * np.sum(feature_importances)
    cumulative_importance = 0.0
    important_indices = []
    
    for i in indices:
        cumulative_importance += feature_importances[i]
        important_indices.append(i)
        if cumulative_importance >= importance_threshold:
            break
    
    # Feature mask oluştur
    feature_mask = np.zeros(n_features, dtype=bool)
    feature_mask[important_indices] = True
    
    # Kuantize modeli oluştur (derinlik azalt)
    optimized_model = GradientBoostingClassifier(
        n_estimators=min(100, model.n_estimators),
        learning_rate=model.learning_rate,
        max_depth=min(3, model.max_depth if model.max_depth else 3),
        random_state=42
    )
    
    # Meta verileri kaydet
    meta = {
        "original_size_bytes": orig_size,
        "original_n_estimators": model.n_estimators,
        "optimized_n_estimators": min(100, model.n_estimators),
        "original_max_depth": model.max_depth,
        "optimized_max_depth": min(3, model.max_depth if model.max_depth else 3),
        "original_n_features": n_features,
        "optimized_n_features": len(important_indices),
        "feature_mask": feature_mask.tolist(),
        "selected_features": important_indices
    }
    
    # Modeli kaydet
    if output_path:
        joblib.dump(optimized_model, output_path)
        
        # Meta verileri kaydet
        meta_path = output_path.replace(".pkl", "_meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Model boyutunu ölç
        new_size = os.path.getsize(output_path)
        compression_ratio = orig_size / new_size if new_size > 0 else 0
        logger.info(f"Model optimize edildi: {orig_size/1024:.1f} KB -> {new_size/1024:.1f} KB ({compression_ratio:.1f}x)")
    
    elapsed = time.time() - start_time
    logger.info(f"GradientBoosting optimizasyonu tamamlandı ({elapsed:.2f} saniye)")
    
    return optimized_model, meta

def quantize_sklearn_model(model_path, output_path=None):
    """
    Scikit-learn modelini kuantize eder:
    1. Model ağırlıklarını daha düşük hassasiyete indirir (32-bit -> 16-bit)
    2. Optimize edilmiş versiyonu kaydeder
    """
    logger.info(f"Modeli kuantize ediyor: {model_path}")
    
    if output_path is None:
        output_path = os.path.join(OPTIMIZED_DIR, os.path.basename(model_path))
    
    try:
        # Modeli yükle
        model = joblib.load(model_path)
        
        # Model tipi kontrolü
        if isinstance(model, RandomForestClassifier):
            # RandomForest modeli için özel optimizasyon
            return optimize_random_forest(model, model.n_features_in_, output_path)
        
        elif isinstance(model, GradientBoostingClassifier):
            # GradientBoosting modeli için özel optimizasyon  
            return optimize_gradient_boosting(model, model.n_features_in_, output_path)
        
        else:
            # Genel model optimizasyonu
            logger.info(f"Desteklenmeyen model tipi: {type(model)}, genel optimizasyon yapılıyor")
            orig_size = os.path.getsize(model_path)
            
            # Model parametrelerini dönüştür (32-bit -> 16-bit)
            if hasattr(model, 'estimators_'):
                for i, estimator in enumerate(model.estimators_):
                    if hasattr(estimator, 'tree_'):
                        # Ağaç yapısındaki değerleri dönüştür
                        if hasattr(estimator.tree_, 'value'):
                            estimator.tree_.value = estimator.tree_.value.astype(np.float16)
                        if hasattr(estimator.tree_, 'threshold'):
                            estimator.tree_.threshold = estimator.tree_.threshold.astype(np.float16)
            
            # Modeli kaydet
            joblib.dump(model, output_path, compress=3)
            
            # Boyut karşılaştırması
            new_size = os.path.getsize(output_path)
            compression_ratio = orig_size / new_size if new_size > 0 else 0
            logger.info(f"Model kuantize edildi: {orig_size/1024:.1f} KB -> {new_size/1024:.1f} KB ({compression_ratio:.1f}x)")
            
            return model, {
                "original_size_bytes": orig_size,
                "quantized_size_bytes": new_size,
                "compression_ratio": compression_ratio
            }
    
    except Exception as e:
        logger.error(f"Model kuantizasyon hatası: {str(e)}")
        return None, None

def quantize_tensorflow_model(model_path, output_path=None):
    """
    TensorFlow modelini kuantize eder (TFLite dönüşümü)
    """
    logger.info(f"TensorFlow modelini kuantize ediyor: {model_path}")
    
    if output_path is None:
        output_path = os.path.join(OPTIMIZED_DIR, os.path.basename(model_path).split('.')[0] + '.tflite')
    
    try:
        # Keras modelini yükle
        model = tf.keras.models.load_model(model_path)
        
        # Model boyutunu ölç
        orig_size = os.path.getsize(model_path)
        
        # TFLite dönüştürücü
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Kuantizasyon ayarları
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Kuantize edilmiş model
        tflite_model = converter.convert()
        
        # Modeli kaydet
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        # Boyut karşılaştırması
        new_size = os.path.getsize(output_path)
        compression_ratio = orig_size / new_size if new_size > 0 else 0
        logger.info(f"TensorFlow modeli kuantize edildi: {orig_size/1024:.1f} KB -> {new_size/1024:.1f} KB ({compression_ratio:.1f}x)")
        
        return True, {
            "original_size_bytes": orig_size,
            "quantized_size_bytes": new_size,
            "compression_ratio": compression_ratio
        }
    
    except Exception as e:
        logger.error(f"TensorFlow model kuantizasyon hatası: {str(e)}")
        return False, None

def optimize_ensemble_model(model_path, output_path=None):
    """
    Ensemble modeli optimize eder
    """
    logger.info(f"Ensemble modeli optimize ediyor: {model_path}")
    
    if output_path is None:
        output_path = os.path.join(OPTIMIZED_DIR, os.path.basename(model_path))
    
    try:
        # Modeli yükle
        model = joblib.load(model_path)
        
        # Ensemble modeli mi kontrol et
        if hasattr(model, 'estimators'):
            logger.info(f"Ensemble model tespit edildi, {len(model.estimators)} tahmin edici var")
            
            # Her bir tahmin ediciyi ayrı ayrı optimize et
            for i, (name, estimator) in enumerate(model.estimators):
                logger.info(f"Tahmin edici optimize ediliyor: {name}")
                
                # Model tipine göre optimizasyon seçimi
                if isinstance(estimator, RandomForestClassifier):
                    optimized_estimator, _ = optimize_random_forest(estimator, estimator.n_features_in_, None)
                elif isinstance(estimator, GradientBoostingClassifier):
                    optimized_estimator, _ = optimize_gradient_boosting(estimator, estimator.n_features_in_, None)
                else:
                    logger.warning(f"Desteklenmeyen tahmin edici tipi: {type(estimator)}")
                    optimized_estimator = estimator
                
                # Optimize edilmiş tahminci ile güncelle
                model.estimators[i] = (name, optimized_estimator)
            
            # Optimizasyonu ölç
            orig_size = os.path.getsize(model_path)
            
            # Modeli kaydet
            joblib.dump(model, output_path, compress=3)
            
            # Boyut karşılaştırması
            new_size = os.path.getsize(output_path)
            compression_ratio = orig_size / new_size if new_size > 0 else 0
            logger.info(f"Ensemble model optimize edildi: {orig_size/1024:.1f} KB -> {new_size/1024:.1f} KB ({compression_ratio:.1f}x)")
            
            return model, {
                "original_size_bytes": orig_size,
                "optimized_size_bytes": new_size,
                "compression_ratio": compression_ratio
            }
        
        else:
            logger.warning(f"Bu bir ensemble modeli değil: {type(model)}")
            return None, None
    
    except Exception as e:
        logger.error(f"Ensemble model optimizasyon hatası: {str(e)}")
        return None, None

def main():
    """Ana işlev - en son modeli bul ve optimize et"""
    create_dirs()
    
    # En son modeli bul
    model_path = get_latest_model()
    if not model_path:
        logger.error("Optimize edilecek model bulunamadı")
        return
    
    # Model tipini belirle ve optimize et
    logger.info(f"Model optimize ediliyor: {model_path}")
    
    try:
        # Modeli yükle
        model = joblib.load(model_path)
        
        # Model tipine göre optimizasyon
        if isinstance(model, RandomForestClassifier):
            output_path = os.path.join(OPTIMIZED_DIR, "optimized_rf_model.pkl")
            optimize_random_forest(model, model.n_features_in_, output_path)
        
        elif isinstance(model, GradientBoostingClassifier):
            output_path = os.path.join(OPTIMIZED_DIR, "optimized_gb_model.pkl")
            optimize_gradient_boosting(model, model.n_features_in_, output_path)
            
        elif hasattr(model, 'estimators'):
            output_path = os.path.join(OPTIMIZED_DIR, "optimized_ensemble_model.pkl")
            optimize_ensemble_model(model_path, output_path)
            
        else:
            output_path = os.path.join(OPTIMIZED_DIR, "optimized_model.pkl")
            quantize_sklearn_model(model_path, output_path)
            
        logger.info(f"Model optimizasyonu tamamlandı: {output_path}")
        
    except Exception as e:
        logger.error(f"Model optimizasyon hatası: {str(e)}")

if __name__ == "__main__":
    main() 