#!/usr/bin/env python3
"""
ML ENTEGRASYON TESTİ
====================
Öznitelik çıkarımı ve ML sınıflandırma entegrasyonunu test eder.
"""

import logging
import os
import sys

# Test için gerekli modülleri kontrol et
def test_ml_integration():
    """ML entegrasyon testini çalıştır."""
    print("🧠 ML ENTEGRASYONu TESTİ")
    print("=" * 50)
    
    success = True
    
    # 1. Modül import testleri
    print("\n📦 Modül İmport Testleri...")
    try:
        import features
        print("   ✅ features.py başarıyla import edildi")
    except ImportError as e:
        print(f"   ❌ features.py import hatası: {e}")
        success = False
    
    try:
        import model
        print("   ✅ model.py başarıyla import edildi")
    except ImportError as e:
        print(f"   ❌ model.py import hatası: {e}")
        success = False
    
    try:
        import torch
        print("   ✅ PyTorch mevcut")
    except ImportError:
        print("   ⚠️  PyTorch bulunamadı, basit mode kullanılacak")
    
    # 2. Test verisi oluştur
    print("\n📊 Test Verisi Hazırlama...")
    test_y_positions = [10.0, 15.0, 8.0, 12.0, 18.0, 6.0, 14.0, 11.0] * 5  # Nistagmus benzeri
    test_x_differences = [2.0, 2.5, 1.8, 2.2, 2.4, 1.9, 2.1, 2.3] * 5  # Şaşılık benzeri
    test_fps = 30.0
    
    print(f"   Test verisi hazırlandı: {len(test_y_positions)} veri noktası")
    
    # 3. Öznitelik çıkarımı testi
    print("\n🔍 Öznitelik Çıkarımı Testi...")
    try:
        from features import extract_movement_features
        
        features_result = extract_movement_features(
            y_positions=test_y_positions,
            x_differences=test_x_differences,
            frame_rate=test_fps
        )
        
        print("   ✅ Öznitelik çıkarımı başarılı:")
        for key, value in features_result.items():
            print(f"      {key}: {value:.3f}")
            
    except Exception as e:
        print(f"   ❌ Öznitelik çıkarımı hatası: {e}")
        success = False
    
    # 4. ML sınıflandırma testi (PyTorch varsa)
    print("\n🤖 ML Sınıflandırma Testi...")
    try:
        from model import create_simple_classifier
        
        classifier = create_simple_classifier()
        ml_predictions = classifier.predict(features_result)
        
        print("   ✅ ML sınıflandırma başarılı:")
        print(f"      Nistagmus: {ml_predictions['classification']['nystagmus']} "
              f"(prob: {ml_predictions['predictions']['nystagmus_probability']:.3f})")
        print(f"      Şaşılık: {ml_predictions['classification']['strabismus']} "
              f"(prob: {ml_predictions['predictions']['strabismus_probability']:.3f})")
        print(f"      Regresyon - Frekans: {ml_predictions['regression']['nystagmus_frequency']:.2f} Hz")
        print(f"      Regresyon - Açı: {ml_predictions['regression']['strabismus_angle']:.2f}°")
        
    except Exception as e:
        print(f"   ❌ ML sınıflandırma hatası: {e}")
        success = False
    
    # 5. Gerçek video ile entegrasyon testi (varsa)
    print("\n🎬 Video Entegrasyon Testi...")
    try:
        # Test videosu oluştur
        from improved_test_video_generator import MedicalGradeVideoGenerator
        generator = MedicalGradeVideoGenerator()
        
        test_video_path = "ml_test_video.mp4"
        video_created = generator.create_realistic_nystagmus_video(
            test_video_path, duration=2.0, nystagmus_freq=3.0, strabismus_angle=2.5
        )
        
        if video_created:
            print(f"   ✅ Test videosu oluşturuldu: {test_video_path}")
            
            # ML destekli analiz
            from analysis_utils import analyze_video_with_ml
            results = analyze_video_with_ml(test_video_path, use_ml_classification=True)
            
            if "error" not in results and "ml_analysis" in results:
                ml_data = results["ml_analysis"]
                print("   ✅ ML destekli video analizi başarılı:")
                print(f"      Geleneksel - Frekans: {results['nistagmus_frequency']:.2f} Hz")
                print(f"      ML - Frekans: {ml_data['regression']['nystagmus_frequency']:.2f} Hz")
                print(f"      Geleneksel - Açı: {results['strabismus_angle']:.2f}°")
                print(f"      ML - Açı: {ml_data['regression']['strabismus_angle']:.2f}°")
                print(f"      ML Sınıflandırma: N={ml_data['classification']['nystagmus']}, "
                      f"S={ml_data['classification']['strabismus']}")
            else:
                print(f"   ❌ ML video analizi hatası: {results.get('error', 'Bilinmeyen hata')}")
                success = False
            
            # Temizlik
            if os.path.exists(test_video_path):
                os.remove(test_video_path)
        
    except Exception as e:
        print(f"   ❌ Video entegrasyon hatası: {e}")
        success = False
    
    # Sonuç
    print("\n" + "=" * 50)
    if success:
        print("🎉 ML ENTEGRASYONu BAŞARILI!")
        print("   ✅ Öznitelik çıkarımı çalışıyor")
        print("   ✅ ML sınıflandırma çalışıyor") 
        print("   ✅ Video entegrasyon çalışıyor")
        print("   🚀 Sistem hazır!")
    else:
        print("❌ ML entegrasyonunda sorunlar var!")
        print("   🔧 Lütfen hataları kontrol edin")
    
    return success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_ml_integration() 