#!/usr/bin/env python3
"""
TAM SİSTEM ENTEGRASYON TESTİ
============================
Kalibrasyon + ML + Validasyon entegrasyon testi.
"""

import logging
import time
from typing import Dict, Any

logger = logging.getLogger(__name__)

def test_complete_system():
    """Tam sistem entegrasyon testini çalıştır."""
    print("🚀 TAM SİSTEM ENTEGRASYON TESTİ")
    print("=" * 50)
    
    test_results = {
        "calibration": False,
        "ml_integration": False,
        "validation": False,
        "overall_success": False
    }
    
    # 1. KALİBRASYON TESTİ
    print("\n🎯 1. KALİBRASYON SİSTEMİ TESTİ...")
    try:
        from calibration import get_calibrator
        
        calibrator = get_calibrator()
        status = calibrator.get_calibration_status()
        
        # Test kalibrasyon
        test_points = [
            {"screen_pos": (100, 100), "pixel_pos": (150, 120), "angle_deg": 10.0},
            {"screen_pos": (400, 300), "pixel_pos": (450, 320), "angle_deg": 5.0},
            {"screen_pos": (700, 500), "pixel_pos": (750, 520), "angle_deg": 8.0}
        ]
        
        success = calibrator.calibrate_from_points(test_points)
        
        if success:
            print("   ✅ Kalibrasyon sistemi çalışıyor")
            print(f"   📏 Piksel/Derece oranı: {calibrator.calibration_data['pixel_to_degree_ratio']:.4f}")
            test_results["calibration"] = True
        else:
            print("   ❌ Kalibrasyon başarısız")
            
    except Exception as e:
        print(f"   ❌ Kalibrasyon hatası: {e}")
    
    # 2. ML ENTEGRASYON TESTİ
    print("\n🧠 2. ML ENTEGRASYON TESTİ...")
    try:
        from features import extract_movement_features
        from model import create_simple_classifier
        
        # Test verileri
        y_positions = [100.0, 102.0, 98.0, 101.0, 99.0] * 10
        x_differences = [5.0, 4.8, 5.2, 4.9, 5.1] * 10
        
        # Öznitelik çıkarımı
        features = extract_movement_features(y_positions, x_differences, 30.0)
        print(f"   📊 Öznitelikler: {len(features)} adet")
        
        # ML sınıflandırma
        classifier = create_simple_classifier()
        ml_results = classifier.predict(features)
        
        print(f"   🤖 Nistagmus tahmini: {ml_results['predictions']['nystagmus_detected']}")
        print(f"   🤖 Şaşılık tahmini: {ml_results['predictions']['strabismus_detected']}")
        
        test_results["ml_integration"] = True
        print("   ✅ ML entegrasyonu çalışıyor")
        
    except Exception as e:
        print(f"   ❌ ML entegrasyon hatası: {e}")
    
    # 3. VALİDASYON TESTİ
    print("\n🔬 3. VALİDASYON SİSTEMİ TESTİ...")
    try:
        from validation import run_quick_validation
        
        # Hızlı validasyon testi (3 video)
        print("   📹 Test videoları oluşturuluyor...")
        validation_results = run_quick_validation(test_count=3, use_ml=True)
        
        if "error" not in validation_results:
            overall = validation_results["overall_assessment"]
            reg_metrics = validation_results["regression_metrics"]
            
            print(f"   📈 MAE Frekans: {reg_metrics['mae_frequency']:.3f} Hz")
            print(f"   📈 MAE Açı: {reg_metrics['mae_angle']:.3f}°")
            print(f"   🎯 Validasyon geçti: {overall['validation_passed']}")
            
            test_results["validation"] = True
            print("   ✅ Validasyon sistemi çalışıyor")
        else:
            print(f"   ❌ Validasyon hatası: {validation_results['error']}")
            
    except Exception as e:
        print(f"   ❌ Validasyon testi hatası: {e}")
    
    # 4. TAM SİSTEM TESTİ
    print("\n🎉 4. TAM SİSTEM ENTEGRASYON TESTİ...")
    try:
        # Test videosu ile tam pipeline testi
        from improved_test_video_generator import MedicalGradeVideoGenerator
        from analysis_utils import analyze_video_with_ml
        
        generator = MedicalGradeVideoGenerator()
        test_video = "complete_system_test.mp4"
        
        # Test videosu oluştur
        success = generator.create_realistic_nystagmus_video(
            test_video, duration=2.0, nystagmus_freq=4.0, strabismus_angle=5.0
        )
        
        if success:
            # Tam sistem analizi
            results = analyze_video_with_ml(test_video, use_ml_classification=True)
            
            if "error" not in results:
                print(f"   📊 Temel analiz: {results['nistagmus_frequency']:.2f} Hz, {results['strabismus_angle']:.2f}°")
                
                if "ml_analysis" in results:
                    ml_data = results["ml_analysis"]
                    print(f"   🤖 ML analiz: Nistagmus={ml_data['predictions']['nystagmus_detected']}, Şaşılık={ml_data['predictions']['strabismus_detected']}")
                
                print("   ✅ Tam sistem entegrasyonu başarılı")
                test_results["overall_success"] = True
            else:
                print(f"   ❌ Sistem analiz hatası: {results['error']}")
        
        # Test dosyasını temizle
        import os
        if os.path.exists(test_video):
            os.remove(test_video)
            
    except Exception as e:
        print(f"   ❌ Tam sistem testi hatası: {e}")
    
    # SONUÇ RAPORU
    print("\n" + "=" * 50)
    print("📋 SİSTEM ENTEGRASYON RAPORU")
    print("=" * 50)
    
    success_count = sum(test_results.values())
    total_tests = len(test_results) - 1  # overall_success hariç
    
    for test_name, success in test_results.items():
        if test_name != "overall_success":
            icon = "✅" if success else "❌"
            print(f"   {icon} {test_name.upper()}: {'BAŞARILI' if success else 'BAŞARISIZ'}")
    
    print(f"\n🎯 GENEL BAŞARI: {success_count}/{total_tests + 1}")
    
    if test_results["overall_success"]:
        print("🎉 TAM SİSTEM ENTEGRASYONU BAŞARILI!")
        print("   • Kalibrasyon sistemi aktif")
        print("   • ML entegrasyonu çalışıyor") 
        print("   • Validasyon sistemi hazır")
        print("   • Göz hareketi analizi tam pipeline")
    else:
        print("⚠️  Bazı bileşenler düzeltilmeli")
    
    return test_results

def main():
    """Ana fonksiyon."""
    logging.basicConfig(level=logging.INFO)
    start_time = time.time()
    
    results = test_complete_system()
    
    duration = time.time() - start_time
    print(f"\n⏱️  Test süresi: {duration:.2f} saniye")
    
    return results

if __name__ == "__main__":
    main() 