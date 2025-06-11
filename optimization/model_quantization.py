"""
Göz İzleme Sistemi - Model Quantization Desteği

Bu modül, ONNX modellerinin quantize edilmesi ve optimize edilmesi için
gerekli fonksiyonları sağlar. Düşük güçlü cihazlarda performans artışı sağlar.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

logger = logging.getLogger('eye_tracker.optimization.quantization')

class ModelQuantizer:
    """ONNX model quantization ve optimizasyonu"""
    
    def __init__(self, original_model_path: str, output_dir: str = "optimized_models"):
        """
        Args:
            original_model_path: Orijinal ONNX model dosya yolu
            output_dir: Optimize edilmiş modellerin kaydedileceği dizin
        """
        self.original_model_path = original_model_path
        self.output_dir = output_dir
        
        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Model adını ayıkla
        self.model_name = os.path.basename(original_model_path)
        self.model_stem = os.path.splitext(self.model_name)[0]
    
    def quantize_to_int8(self, calibration_data_reader=None) -> str:
        """
        Modeli int8 formatına dönüştür (statik quantization)
        
        Args:
            calibration_data_reader: Kalibrasyon verisi okuyucu (None: varsayılan)
            
        Returns:
            Optimize edilmiş model dosya yolu
        """
        try:
            import onnx
            from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
            
            logger.info(f"Orijinal model yükleniyor: {self.original_model_path}")
            
            # Çıkış dosya yolu
            output_model_path = os.path.join(self.output_dir, f"{self.model_stem}_int8.onnx")
            
            # Quantization gerçekleştir
            if calibration_data_reader:
                # Kalibrasyon verisiyle
                quantize_static(
                    model_input=self.original_model_path,
                    model_output=output_model_path,
                    calibration_data_reader=calibration_data_reader,
                    quant_format=QuantType.QInt8
                )
            else:
                # Varsayılan kalibrasyon ile
                quantize_static(
                    model_input=self.original_model_path,
                    model_output=output_model_path,
                    quant_format=QuantType.QInt8
                )
            
            logger.info(f"Model başarıyla INT8 formatında quantize edildi: {output_model_path}")
            
            # Dosya boyutlarını karşılaştır
            orig_size = os.path.getsize(self.original_model_path) / (1024 * 1024)
            quant_size = os.path.getsize(output_model_path) / (1024 * 1024)
            reduction = (1 - quant_size / orig_size) * 100
            
            logger.info(f"Orijinal model boyutu: {orig_size:.2f} MB")
            logger.info(f"INT8 model boyutu: {quant_size:.2f} MB (-%{reduction:.1f})")
            
            return output_model_path
            
        except ImportError as e:
            logger.error(f"Gerekli kütüphaneler yüklü değil: {e}")
            logger.error("Quantization için: pip install onnx onnxruntime-tools")
            return self.original_model_path
            
        except Exception as e:
            logger.error(f"INT8 quantization hatası: {e}")
            return self.original_model_path
    
    def convert_to_fp16(self) -> str:
        """
        Modeli fp16 formatına dönüştür
        
        Returns:
            Optimize edilmiş model dosya yolu
        """
        try:
            import onnx
            from onnxconverter_common import float16
            
            logger.info(f"Orijinal model FP16 için yükleniyor: {self.original_model_path}")
            
            # Çıkış dosya yolu
            output_model_path = os.path.join(self.output_dir, f"{self.model_stem}_fp16.onnx")
            
            # Modeli yükle
            model = onnx.load(self.original_model_path)
            
            # FP16'ya dönüştür
            model_fp16 = float16.convert_float_to_float16(model)
            
            # Kaydet
            onnx.save(model_fp16, output_model_path)
            
            logger.info(f"Model başarıyla FP16 formatına dönüştürüldü: {output_model_path}")
            
            # Dosya boyutlarını karşılaştır
            orig_size = os.path.getsize(self.original_model_path) / (1024 * 1024)
            fp16_size = os.path.getsize(output_model_path) / (1024 * 1024)
            reduction = (1 - fp16_size / orig_size) * 100
            
            logger.info(f"Orijinal model boyutu: {orig_size:.2f} MB")
            logger.info(f"FP16 model boyutu: {fp16_size:.2f} MB (-%{reduction:.1f})")
            
            return output_model_path
            
        except ImportError as e:
            logger.error(f"Gerekli kütüphaneler yüklü değil: {e}")
            logger.error("FP16 dönüşümü için: pip install onnx onnxconverter-common")
            return self.original_model_path
            
        except Exception as e:
            logger.error(f"FP16 dönüşüm hatası: {e}")
            return self.original_model_path
    
    def optimize_for_inference(self, optimization_level: int = 99) -> str:
        """
        Modeli çıkarım için optimize et (boyut ve hız)
        
        Args:
            optimization_level: Optimizasyon seviyesi (0-99, yüksek = daha agresif)
            
        Returns:
            Optimize edilmiş model dosya yolu
        """
        try:
            import onnx
            from onnxruntime.transformers import optimizer
            
            logger.info(f"Orijinal model optimize ediliyor: {self.original_model_path}")
            
            # Çıkış dosya yolu
            output_model_path = os.path.join(self.output_dir, f"{self.model_stem}_optimized.onnx")
            
            # Model optimizer
            model_optimizer = optimizer.get_optimizer('onnxruntime')
            
            # Optimize et
            optimized_model = model_optimizer.optimize_model(
                input=self.original_model_path,
                model_type='eye_tracker',
                num_heads=12,
                hidden_size=768,
                optimization_level=optimization_level
            )
            
            # Kaydet
            optimized_model.save_model_to_file(output_model_path)
            
            logger.info(f"Model başarıyla optimize edildi: {output_model_path}")
            
            # Dosya boyutlarını karşılaştır
            orig_size = os.path.getsize(self.original_model_path) / (1024 * 1024)
            opt_size = os.path.getsize(output_model_path) / (1024 * 1024)
            reduction = (1 - opt_size / orig_size) * 100
            
            logger.info(f"Orijinal model boyutu: {orig_size:.2f} MB")
            logger.info(f"Optimize model boyutu: {opt_size:.2f} MB (-%{reduction:.1f})")
            
            return output_model_path
            
        except ImportError as e:
            logger.error(f"Gerekli kütüphaneler yüklü değil: {e}")
            logger.error("Optimizasyon için: pip install onnx onnxruntime-transformers")
            return self.original_model_path
            
        except Exception as e:
            logger.error(f"Model optimizasyon hatası: {e}")
            return self.original_model_path

# Kalibrasyon veri sınıfı örneği
class EyeTrackingCalibrationData:
    """ONNX model kalibrasyonu için göz izleme veri okuyucu"""
    
    def __init__(self, sample_images: List[np.ndarray], batch_size: int = 8):
        """
        Args:
            sample_images: Kalibrasyon için örnek görüntüler
            batch_size: Batch büyüklüğü
        """
        self.sample_images = sample_images
        self.batch_size = batch_size
        self.index = 0
    
    def get_next(self) -> Dict[str, np.ndarray]:
        """
        Bir sonraki kalibrasyon batch'ini al
        
        Returns:
            Girdi-çıktı dictionary {input_name: girdi_verileri}
        """
        if self.index >= len(self.sample_images):
            return None
        
        # Batch oluştur
        end_idx = min(self.index + self.batch_size, len(self.sample_images))
        batch_data = self.sample_images[self.index:end_idx]
        self.index = end_idx
        
        # Doğru formata dönüştür
        return {"input": np.array(batch_data, dtype=np.float32)}
    
    def rewind(self) -> None:
        """Veri okuyucuyu başa sar"""
        self.index = 0 