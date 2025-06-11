"""
Göz İzleme Sistemi - Performans Optimizasyonu Örnekleri

Bu modül, performans optimizasyon bileşenlerinin nasıl kullanılacağını gösteren
örnek kodlar içerir.
"""

import numpy as np
import cv2
import time
import os
import logging

# Optimizasyon modüllerini içe aktar
from optimization.performance_manager import PerformanceOptimizationManager
from optimization.model_quantization import ModelQuantizer
from optimization.frame_optimization import FrameProcessingOptimizer, LowResolutionStrategy

# Loglama konfigürasyonu
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def model_quantization_example():
    """Model quantization örneği"""
    print("\n=== Model Quantization Örneği ===")
    
    # Test için bir model yolu (örnek amaçlı)
    model_path = "models/sample_model.onnx"
    
    # Gerçek dosya yoksa, örnek veremiyoruz
    if not os.path.exists(model_path):
        print(f"Örnek model dosyası bulunamadı: {model_path}")
        print("Bu örnek, mevcut bir ONNX model dosyası gerektirir.")
        return
    
    # Model quantizer oluştur
    quantizer = ModelQuantizer(model_path, output_dir="optimized_models")
    
    # INT8 quantization uygula
    print("INT8 quantization uygulanıyor...")
    int8_model = quantizer.quantize_to_int8()
    print(f"INT8 model: {int8_model}")
    
    # FP16 dönüşümü uygula
    print("\nFP16 dönüşümü uygulanıyor...")
    fp16_model = quantizer.convert_to_fp16()
    print(f"FP16 model: {fp16_model}")
    
    # Çıkarım optimizasyonu uygula
    print("\nGenel optimizasyon uygulanıyor...")
    opt_model = quantizer.optimize_for_inference()
    print(f"Optimize model: {opt_model}")

def frame_optimization_example():
    """Frame işleme optimizasyon örneği"""
    print("\n=== Frame İşleme Optimizasyon Örneği ===")
    
    # Kamera yakalama nesnesini başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    # Frame optimizer oluştur
    optimizer = FrameProcessingOptimizer(buffer_size=5)
    
    print("Kullanılabilir stratejiler:")
    for strategy in optimizer.strategies.keys():
        print(f"  - {strategy}")
    
    # Farklı stratejileri test et
    strategies = ["none", "low_res_50", "low_res_25", "skip_2"]
    frames_to_process = 30
    
    for strategy in strategies:
        optimizer.set_active_strategy(strategy)
        print(f"\nStrateji: {strategy}")
        
        processing_times = []
        for _ in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Frame'i işle ve süreyi ölç
            start_time = time.time()
            processed = optimizer.process_frame(frame)
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            # İşlenmiş frame boyutunu göster
            h, w = processed.shape[:2] if processed is not None else (0, 0)
            print(f"  Frame boyutu: {w}x{h}, İşleme süresi: {processing_time:.2f} ms")
        
        # Ortalama işleme süresini hesapla
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"  Ortalama işleme süresi: {avg_time:.2f} ms")
    
    # Kaynakları serbest bırak
    cap.release()

def performance_manager_example():
    """Tam performans yönetimi örneği"""
    print("\n=== Performans Yönetimi Örneği ===")
    
    # Optimizer yöneticisi oluştur
    config = {
        'optimization_level': 1,  # Başlangıç seviyesi
        'auto_optimize': True,
        'frame_buffer_size': 5,
        'data_history_size': 300,
        'sampling_rate': 30
    }
    
    optimizer = PerformanceOptimizationManager(config)
    
    # Optimize edilmiş model al (varsayılan olarak bellek durumuna göre)
    model_path = "models/sample_model.onnx"
    if os.path.exists(model_path):
        print(f"Model optimizasyonu: {model_path}")
        optimized_model = optimizer.optimize_model(model_path)
        print(f"Optimize edilmiş model: {optimized_model}")
    
    # Kamera yakala
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı!")
        return
    
    # Performans izlemeyi başlat
    optimizer.start_monitoring(interval=2.0)
    
    print("\nFarklı optimizasyon seviyeleri test ediliyor...")
    
    # Tüm optimizasyon seviyelerini test et
    for level in range(4):  # 0-3 arası
        optimizer.set_optimization_level(level)
        print(f"\nOptimizasyon seviyesi: {level}")
        
        # Birkaç frame işle
        frames = 0
        start_time = time.time()
        
        while frames < 50:  # 50 frame işle
            ret, frame = cap.read()
            if not ret:
                break
            
            # Frame'i optimize et
            processed = optimizer.process_frame(frame)
            
            # Artifisyel göz izleme verileri ekle
            timestamp = time.time()
            gaze_x = np.random.uniform(0, frame.shape[1])
            gaze_y = np.random.uniform(0, frame.shape[0])
            pupil_size = np.random.uniform(2.0, 5.0)
            confidence = np.random.uniform(0.7, 1.0)
            
            optimizer.add_tracking_sample(
                timestamp=timestamp,
                gaze_x=gaze_x,
                gaze_y=gaze_y,
                pupil_size=pupil_size,
                confidence=confidence
            )
            
            # İstatistikleri hesapla
            if frames % 10 == 0:
                stats = optimizer.get_tracking_stats(window_seconds=0.5)
                print(f"  İzleme istatistikleri: Ortalama hız={stats.get('mean_velocity', 0):.2f}, "
                     f"Dispersiyon={stats.get('dispersion', 0):.2f}")
            
            frames += 1
        
        # FPS hesapla
        elapsed = time.time() - start_time
        fps = frames / elapsed
        
        # Metrikleri güncelle
        optimizer.update_performance_metrics({
            'fps': fps,
            'frame_drop_rate': 0.0,
            'model_inference_time': 10.0  # Örnek değer
        })
        
        print(f"  Performans: {fps:.1f} FPS")
        
        # Optimizasyon önerileri
        suggestions = optimizer.get_optimization_suggestions()
        if suggestions:
            print("  Optimizasyon önerileri:")
            for suggestion in suggestions:
                print(f"    - {suggestion}")
    
    # Performans izlemeyi durdur
    optimizer.stop_monitoring()
    
    # İstatistikleri göster
    print("\nOptimizasyon istatistikleri:")
    stats = optimizer.get_stats()
    print(f"  Optimizasyon seviyesi: {stats['optimization_level']}")
    print(f"  Otomatik optimizasyon: {'Açık' if stats['auto_optimize'] else 'Kapalı'}")
    print(f"  Tercih edilen backend: {stats['preferred_backend']}")
    print(f"  Performans metrikleri: {stats['performance_metrics']}")
    
    # Kaynakları serbest bırak
    cap.release()

if __name__ == "__main__":
    print("Göz İzleme Sistemi - Performans Optimizasyon Örnekleri")
    print("=" * 60)
    
    # Seçili örnekleri çalıştır
    try:
        model_quantization_example()
    except Exception as e:
        print(f"Model quantization örneği hatası: {e}")
    
    try:
        frame_optimization_example()
    except Exception as e:
        print(f"Frame optimizasyon örneği hatası: {e}")
    
    try:
        performance_manager_example()
    except Exception as e:
        print(f"Performans yönetimi örneği hatası: {e}")
    
    print("\nTüm testler tamamlandı!") 