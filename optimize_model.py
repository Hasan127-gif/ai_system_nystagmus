#!/usr/bin/env python
"""
Nistagmus AI Modeli Optimizasyon Aracı
--------------------------------------
Bu script, eğitilmiş Nistagmus AI modellerini optimize eder ve kuantize eder.

Kullanım:
  python optimize_model.py [model_yolu] [çıktı_yolu]

Argümanlar:
  model_yolu  - Optimize edilecek modelin yolu (opsiyonel)
  çıktı_yolu  - Optimize edilmiş modelin kaydedileceği yol (opsiyonel)

Örnekler:
  python optimize_model.py                                  # En son modeli optimize et
  python optimize_model.py models/trained/my_model.pkl      # Belirli bir modeli optimize et
  python optimize_model.py models/trained/my_model.pkl models/optimized/optimized_model.pkl  # Çıktı yolunu belirt
"""

import os
import sys
import logging
import argparse
from models.model_optimization import (
    create_dirs, get_latest_model, quantize_sklearn_model, 
    quantize_tensorflow_model, optimize_ensemble_model
)

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('optimize_model')

def parse_args():
    """Komut satırı argümanlarını ayrıştır"""
    parser = argparse.ArgumentParser(
        description="Nistagmus AI Modeli Optimizasyon Aracı"
    )
    parser.add_argument(
        "model_path", nargs="?", default=None,
        help="Optimize edilecek modelin yolu (belirtilmezse en son model kullanılır)"
    )
    parser.add_argument(
        "output_path", nargs="?", default=None,
        help="Optimize edilmiş modelin kaydedileceği yol (belirtilmezse otomatik oluşturulur)"
    )
    parser.add_argument(
        "--tensorflow", action="store_true",
        help="Model bir TensorFlow modeli ise bu bayrağı ekleyin"
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Model bir ensemble modeli ise bu bayrağı ekleyin"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Ayrıntılı çıktı göster"
    )
    return parser.parse_args()

def main():
    """Ana işlev"""
    args = parse_args()
    
    # Dizinleri oluştur
    create_dirs()
    
    # Model yolunu belirle
    model_path = args.model_path
    if not model_path:
        model_path = get_latest_model()
        if not model_path:
            logger.error("Optimize edilecek model bulunamadı")
            return 1
    
    # Model var mı kontrol et
    if not os.path.exists(model_path):
        logger.error(f"Model bulunamadı: {model_path}")
        return 1
    
    logger.info(f"Model optimize ediliyor: {model_path}")
    
    # Çıktı yolunu belirle
    output_path = args.output_path
    if not output_path:
        # Çıktı yolunu otomatik oluştur
        model_filename = os.path.basename(model_path)
        if args.tensorflow:
            output_path = os.path.join("models/optimized", model_filename.split('.')[0] + "_optimized.tflite")
        else:
            output_path = os.path.join("models/optimized", "optimized_" + model_filename)
    
    # Model tipine göre optimize et
    try:
        if args.tensorflow:
            success, meta = quantize_tensorflow_model(model_path, output_path)
        elif args.ensemble:
            model, meta = optimize_ensemble_model(model_path, output_path)
            success = model is not None
        else:
            model, meta = quantize_sklearn_model(model_path, output_path)
            success = model is not None
        
        if success:
            logger.info(f"Model başarıyla optimize edildi: {output_path}")
            
            # Meta verileri göster
            if args.verbose and meta:
                logger.info("Optimizasyon detayları:")
                for key, value in meta.items():
                    logger.info(f"  {key}: {value}")
            
            return 0
        else:
            logger.error("Model optimizasyonu başarısız")
            return 1
    
    except Exception as e:
        logger.error(f"Model optimizasyon hatası: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 