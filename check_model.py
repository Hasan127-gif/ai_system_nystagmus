import os
import sys
import json
import logging
import numpy as np
import joblib
import traceback

# Log yapılandırması
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("model_checker")

def check_model():
    """
    Nistagmus modelini inceler ve ayrıntılarını gösterir
    """
    try:
        # Model dizini
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "trained")
        default_model_path = os.path.join(model_dir, "default_model.pkl")
        config_path = os.path.join(model_dir, "detector_config.json")
        
        # Model ve dizin kontrolü
        logger.info(f"Model dizini: {model_dir}")
        logger.info(f"Model dosyası mevcut mu: {os.path.exists(default_model_path)}")
        
        if not os.path.exists(default_model_path):
            logger.error("Model dosyası bulunamadı!")
            return False
            
        # Modeli yükle
        model = joblib.load(default_model_path)
        
        # Model türünü kontrol et
        logger.info(f"Model türü: {type(model).__name__}")
        
        # Model ayrıntıları
        model_details = {}
        
        # Scikit-learn modeli mi?
        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
            logger.info("Bu bir scikit-learn modeli.")
            
            # RandomForest veya benzeri bir model mi?
            if hasattr(model, 'estimators_'):
                logger.info(f"Ensemble model, {len(model.estimators_)} estimator içeriyor")
                
                # Özellik önemleri var mı?
                if hasattr(model, 'feature_importances_'):
                    logger.info("Özellik önemleri:")
                    importances = model.feature_importances_
                    for i, importance in enumerate(importances):
                        logger.info(f"Özellik {i+1}: {importance:.4f}")
                        
                    # En önemli 5 özellik
                    top_indices = np.argsort(importances)[::-1][:5]
                    logger.info(f"En önemli 5 özellik indeksleri: {top_indices}")
                
            # Sınıf bilgisi
            if hasattr(model, 'classes_'):
                logger.info(f"Sınıflar: {model.classes_}")
                
            # Test amaçlı örnek tahmin yapma
            dummy_features = [0] * 19  # Modelin beklediği özellik sayısı
            
            # İlk özellik - sol göz pozisyon standardizasyonu
            dummy_features[0] = 2.5  # Yüksek değişkenlik
            
            # Fast fazlar - en önemli özellik
            dummy_features[5] = 10  # Çok sayıda hızlı faz
            
            # Hakim frekans - nistagmus için tipik 2-8 Hz
            dummy_features[6] = 4.5  # Nistagmus için tipik frekans
            
            # Maksimum hız
            dummy_features[4] = 30  # Yüksek hız
            
            # Test tahmini
            try:
                logger.info("Örnek tahmin testi yapılıyor...")
                pred = model.predict([dummy_features])
                pred_proba = model.predict_proba([dummy_features])
                
                logger.info(f"Tahmin: {pred}")
                logger.info(f"Olasılık: {pred_proba}")
                
                # Eşik değeri kontrolü
                if pred[0] == 0 and pred_proba[0][1] > 0.3:
                    logger.warning(f"Eşik değeri sorunlu olabilir. Nistagmus olasılığı {pred_proba[0][1]:.2f} olmasına rağmen sınıf 0 tahmin edildi")
                
            except Exception as e:
                logger.error(f"Tahmin testi hatası: {str(e)}")
                traceback.print_exc()
            
            # Düşük değerlerde (normal durum) tahmin testi
            normal_features = [0.1] * 19
            try:
                normal_pred = model.predict([normal_features])
                normal_proba = model.predict_proba([normal_features])
                logger.info(f"Normal durum tahmini: {normal_pred}")
                logger.info(f"Normal durum olasılık: {normal_proba}")
            except Exception as e:
                logger.error(f"Normal tahmin testi hatası: {str(e)}")
        
        # Konfigürasyon dosyasını kontrol et
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                logger.info(f"Yapılandırma yüklendi: {config_path}")
                logger.info(f"Nistagmus eşik değeri: {config.get('nistagmus_threshold', 'Belirlenmemiş')}")
                logger.info(f"Diğer yapılandırmalar: {json.dumps(config, indent=2)}")
            except Exception as e:
                logger.error(f"Yapılandırma dosyası okuma hatası: {str(e)}")
        else:
            logger.warning(f"Yapılandırma dosyası bulunamadı: {config_path}")
        
        return True
    except Exception as e:
        logger.error(f"Model inceleme hatası: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("Nistagmus modeli incelemesi başlatılıyor...")
    success = check_model()
    if success:
        logger.info("Model incelemesi tamamlandı.")
    else:
        logger.error("Model incelemesi başarısız oldu.") 