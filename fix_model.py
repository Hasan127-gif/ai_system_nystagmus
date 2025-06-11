import os
import sys
import json
import logging
import numpy as np
import joblib
import traceback

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_fixer")

def fix_model():
    """
    Nistagmus modeli yapılandırmasını düzeltir
    """
    try:
        # Model dizini
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
        default_model_path = os.path.join(model_dir, "default_model.pkl")
        config_path = os.path.join(model_dir, "detector_config.json")
        backup_config_path = os.path.join(model_dir, "detector_config.json.backup")
        
        # Konfigürasyon dosyası varsa, yedek oluştur
        if os.path.exists(config_path):
            try:
                import shutil
                shutil.copy2(config_path, backup_config_path)
                logger.info(f"Yapılandırma dosyası yedeklendi: {backup_config_path}")
            except Exception as e:
                logger.error(f"Yapılandırma yedekleme hatası: {str(e)}")
                return False
        
        # Model kontrolü
        if not os.path.exists(default_model_path):
            logger.error("Model dosyası bulunamadı!")
            return False
            
        # Modeli yükle
        model = joblib.load(default_model_path)
        logger.info(f"Model yüklendi: {type(model).__name__}")
        
        # Konfigürasyon dosyasını yükle veya oluştur
        config = {}
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Mevcut yapılandırma yüklendi: {config_path}")
            except Exception as e:
                logger.error(f"Yapılandırma dosyası okuma hatası: {str(e)}")
                return False
        else:
            logger.warning(f"Yapılandırma dosyası bulunamadı, yeni oluşturulacak: {config_path}")
            # Temel yapılandırma
            config = {
                "model_type": "RandomForest" if "RandomForest" in str(type(model)) else "Unknown",
                "model_version": "1.0.0",
                "trained_date": "2025-05-12T12:00:00.000000",
                "thresholds": {},
                "metrics": {}
            }
        
        # Yapılandırmayı güncelle
        
        # 1. Nistagmus eşik değerini düşür (daha hassas tespit için)
        if "thresholds" not in config:
            config["thresholds"] = {}
            
        # Eski eşik değerini yazdır
        old_threshold = config.get("thresholds", {}).get("nistagmus_confidence", 0.65)
        logger.info(f"Eski nistagmus eşik değeri: {old_threshold}")
        
        # Eşik değerini 0.25'e düşür
        config["thresholds"]["nistagmus_confidence"] = 0.25
        
        # Genel nistagmus_threshold değeri de ekle (detector.py bu değeri kullanıyor)
        config["nistagmus_threshold"] = 0.25
        logger.info(f"Yeni nistagmus eşik değeri: {config['thresholds']['nistagmus_confidence']}")
        
        # 2. Ek yapılandırma iyileştirmeleri
        config["feature_importance_boost"] = {
            "left_fast_phases": 1.5,  # Fast fazlara daha fazla ağırlık ver
            "right_fast_phases": 1.5,
            "left_max_velocity": 1.3,  # Hıza daha fazla ağırlık ver
            "right_max_velocity": 1.3
        }
        
        # 3. Hassasiyet artırıcı parametreler
        config["sensitivity_boost"] = True
        config["detection_mode"] = "sensitive"  # normal, sensitive, specific
        
        # 4. Test tahminleri
        logger.info("İyileştirilmiş yapılandırma ile test tahminleri yapılıyor...")
        
        # Nistagmus örneği için test
        nistagmus_features = [0] * 19
        # Fast fazlar - en önemli özellik
        nistagmus_features[5] = 8  # Çok sayıda hızlı faz
        # Hakim frekans - nistagmus için tipik 2-8 Hz
        nistagmus_features[6] = 4.0  # Tipik frekans
        # Maksimum hız
        nistagmus_features[4] = 25  # Yüksek hız
        
        try:
            pred_proba = model.predict_proba([nistagmus_features])[0]
            logger.info(f"Olasılık: {pred_proba}")
            
            # Eski ve yeni eşik değeri ile sınıflandırma
            old_class = 1 if pred_proba[1] >= old_threshold else 0
            new_class = 1 if pred_proba[1] >= config["thresholds"]["nistagmus_confidence"] else 0
            
            logger.info(f"Eski eşik ile sınıflandırma: {old_class} (Nistagmus: {'Evet' if old_class==1 else 'Hayır'})")
            logger.info(f"Yeni eşik ile sınıflandırma: {new_class} (Nistagmus: {'Evet' if new_class==1 else 'Hayır'})")
            
        except Exception as e:
            logger.error(f"Test tahmini hatası: {str(e)}")
        
        # Yapılandırmayı kaydet
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Yapılandırma dosyası güncellendi: {config_path}")
        except Exception as e:
            logger.error(f"Yapılandırma kaydetme hatası: {str(e)}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Model düzeltme hatası: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Nistagmus model düzeltme işlemi başlatılıyor...")
    success = fix_model()
    if success:
        logger.info("Model düzeltme işlemi tamamlandı.")
    else:
        logger.error("Model düzeltme işlemi başarısız oldu.") 