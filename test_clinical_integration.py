#!/usr/bin/env python3
"""
Klinik Entegrasyon Test Sistemi
===============================
Bu dosya, klinik karar destek sistemi entegrasyonunu test eder.
"""

import sys
import os
import numpy as np
import cv2
import time
import json
from pathlib import Path

# Test için örnek video oluştur
def create_test_video_with_movement(filename: str, duration: float = 5.0, fps: int = 30):
    """Test için simüle edilmiş göz hareketi içeren video oluşturur."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    total_frames = int(duration * fps)
    
    for frame_idx in range(total_frames):
        # Boş siyah kare
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Zaman damgası
        t = frame_idx / fps
        
        # Simüle edilmiş göz pozisyonları (nistagmus hareketi)
        # Hafif nistagmus: 1.5 Hz frekans
        amplitude = 20
        base_y = 240
        nystagmus_movement = amplitude * np.sin(2 * np.pi * 1.5 * t)
        
        # Sol göz (yeşil)
        left_x = 200
        left_y = int(base_y + nystagmus_movement)
        cv2.circle(frame, (left_x, left_y), 15, (0, 255, 0), -1)
        
        # Sağ göz (mavi) - hafif şaşılık ile
        right_x = 440 + int(5 * np.sin(t))  # Hafif şaşılık
        right_y = int(base_y + nystagmus_movement * 0.9)  # Hafif fark
        cv2.circle(frame, (right_x, right_y), 15, (255, 0, 0), -1)
        
        # Yüz çerçevesi
        cv2.rectangle(frame, (150, 150), (490, 350), (100, 100, 100), 2)
        
        # Frame bilgisi
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Test videosu oluşturuldu: {filename}")

def test_config_module():
    """Config modülünü test eder."""
    print("\n=== KLİNİK KONFİGÜRASYON TESTİ ===")
    
    try:
        import config
        
        # Konfigürasyon doğrulama
        is_valid = config.validate_clinical_config()
        print(f"✓ Konfigürasyon doğrulaması: {'BAŞARILI' if is_valid else 'BAŞARISIZ'}")
        
        # Eşik değerleri test et
        nyst_threshold = config.get_clinical_threshold("nystagmus_freq_hz", "normal_max")
        print(f"✓ Nistagmus normal eşik: {nyst_threshold} Hz")
        
        strab_threshold = config.get_clinical_threshold("strabismus_pd", "normal_max") 
        print(f"✓ Şaşılık normal eşik: {strab_threshold} PD")
        
        # Yaş faktörü test et
        infant_factor = config.get_age_factor(1.5)  # 1.5 yaş
        adult_factor = config.get_age_factor(30)    # 30 yaş
        print(f"✓ Bebek yaş faktörü (1.5 yaş): {infant_factor}")
        print(f"✓ Yetişkin yaş faktörü (30 yaş): {adult_factor}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config testi başarısız: {str(e)}")
        return False

def test_decision_module():
    """Decision modülünü test eder."""
    print("\n=== KLİNİK KARAR SİSTEMİ TESTİ ===")
    
    try:
        from decision import (
            classify_nystagmus_frequency,
            classify_strabismus_angle,
            classify_findings,
            quick_assessment,
            create_structured_report
        )
        
        # Test verileri
        test_cases = [
            {"freq": 0.3, "angle": 0.5, "age": 25, "desc": "Normal"},
            {"freq": 1.2, "angle": 2.0, "age": 25, "desc": "Hafif patolojik"},
            {"freq": 3.5, "angle": 8.0, "age": 25, "desc": "Orta patolojik"},
            {"freq": 6.2, "angle": 15.0, "age": 8, "desc": "Şiddetli (çocuk)"}
        ]
        
        for i, case in enumerate(test_cases):
            print(f"\nTest {i+1}: {case['desc']}")
            
            # Nistagmus değerlendirmesi
            nyst_finding = classify_nystagmus_frequency(case["freq"], case["age"])
            print(f"  Nistagmus: {nyst_finding.severity.value} - {nyst_finding.description}")
            
            # Şaşılık değerlendirmesi  
            strab_finding = classify_strabismus_angle(case["angle"], case["age"])
            print(f"  Şaşılık: {strab_finding.severity.value} - {strab_finding.description}")
            
            # Genel değerlendirme
            overall = classify_findings(
                nystagmus_freq=case["freq"],
                strabismus_angle=case["angle"],
                age=case["age"]
            )
            
            print(f"  Genel: {overall['overall_assessment']['category']}")
            print(f"  Patoloji: {'VAR' if overall['overall_assessment']['has_pathology'] else 'YOK'}")
            
            # Hızlı değerlendirme
            quick = quick_assessment(case["freq"], case["angle"] * 1.75)
            print(f"  Hızlı değerlendirme: {'PATOLOJİK' if quick['any_pathology'] else 'NORMAL'}")
        
        print("\n✓ Decision modülü testi başarılı")
        return True
        
    except Exception as e:
        print(f"✗ Decision testi başarısız: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_analysis_utils_integration():
    """Analysis utils klinik entegrasyonunu test eder."""
    print("\n=== ANALİZ ARAÇLARI KLİNİK ENTEGRASYON TESTİ ===")
    
    try:
        from analysis_utils import analyze_video_file, format_results_for_api, CLINICAL_SUPPORT_AVAILABLE
        
        print(f"Klinik destek mevcut: {'EVET' if CLINICAL_SUPPORT_AVAILABLE else 'HAYIR'}")
        
        # Test videosu oluştur
        test_video = "test_clinical_video.mp4"
        create_test_video_with_movement(test_video, duration=3.0)
        
        # Video analizi (klinik entegrasyonlu)
        print("Video analizi başlatılıyor...")
        results = analyze_video_file(test_video, max_frames=60)
        
        # Sonuçları kontrol et
        if "error" in results:
            print(f"✗ Video analizi başarısız: {results['error']}")
            return False
        
        print(f"✓ Nistagmus frekansı: {results.get('nistagmus_frequency', 0):.2f} Hz")
        print(f"✓ Şaşılık açısı: {results.get('strabismus_angle', 0):.2f}°")
        print(f"✓ Yüz tespit oranı: {results.get('face_detection_rate', 0):.1%}")
        
        # Klinik değerlendirme kontrolü
        if "clinical_evaluation" in results:
            clinical = results["clinical_evaluation"]
            print(f"✓ Klinik değerlendirme mevcut")
            print(f"  - Kategori: {clinical['overall_assessment']['category']}")
            print(f"  - Patoloji: {'VAR' if clinical['overall_assessment']['has_pathology'] else 'YOK'}")
            
            if "clinical_summary" in results:
                print(f"  - Özet uzunluğu: {len(results['clinical_summary'])} karakter")
        
        # API formatına çevirme
        api_format = format_results_for_api(results)
        print(f"✓ API formatı oluşturuldu")
        
        if "clinical_assessment" in api_format:
            print(f"  - Klinik durum: {api_format['clinical_assessment']['overall_status']}")
            print(f"  - Aciliyet: {api_format['clinical_assessment']['urgency_level']}")
        
        # Temizlik
        os.remove(test_video)
        
        return True
        
    except Exception as e:
        print(f"✗ Analysis utils entegrasyon testi başarısız: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_detector_clinical_integration():
    """Detector klinik entegrasyonunu test eder."""
    print("\n=== DETECTOR KLİNİK ENTEGRASYON TESTİ ===")
    
    try:
        from detector import NistagmusDetector
        
        # Detector oluştur
        detector = NistagmusDetector()
        print(f"✓ Detector oluşturuldu (başlatıldı: {detector.is_initialized})")
        
        # Test videosu oluştur
        test_video = "test_detector_clinical.mp4"
        create_test_video_with_movement(test_video, duration=4.0)
        
        # Yaş faktörlü analiz
        patient_ages = [2.0, 15.0, 45.0, 70.0]  # Farklı yaş grupları
        
        for age in patient_ages:
            print(f"\nYaş grubu testi: {age} yaş")
            
            results = detector.analyze_video(test_video, patient_age=age, max_frames=80)
            
            if "error" in results:
                print(f"  ✗ Analiz başarısız: {results['error']}")
                continue
            
            print(f"  ✓ Temel analiz tamamlandı")
            
            # Yaş spesifik klinik değerlendirme kontrolü
            if "clinical_evaluation_with_age" in results:
                age_eval = results["clinical_evaluation_with_age"]
                print(f"  ✓ Yaş spesifik değerlendirme mevcut")
                print(f"    - Kategori: {age_eval['overall_assessment']['category']}")
                print(f"    - Hasta yaşı: {results.get('patient_age', 'Bilinmiyor')}")
            
            # Klinik özet
            summary = detector.get_clinical_summary(results)
            print(f"  ✓ Klinik özet: {len(summary)} karakter")
            
            # Eşik kontrolü
            nyst_freq = results.get("nistagmus_frequency", 0)
            strab_angle = results.get("strabismus_angle", 0)
            threshold_check = detector.check_clinical_thresholds(nyst_freq, strab_angle)
            
            if "error" not in threshold_check:
                print(f"  ✓ Eşik kontrolü: {'PATOLOJİK' if threshold_check.get('any_pathology', False) else 'NORMAL'}")
        
        # İstatistikler
        stats = detector.get_analysis_statistics()
        print(f"\n✓ Detector istatistikleri:")
        print(f"  - Toplam analiz: {stats['total_analyses']}")
        print(f"  - Başarı oranı: {stats['success_rate']:.1%}")
        print(f"  - Patoloji oranı: {stats['pathology_rate']:.1%}")
        
        # Temizlik
        os.remove(test_video)
        
        return True
        
    except Exception as e:
        print(f"✗ Detector klinik entegrasyon testi başarısız: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_historical_analysis_clinical():
    """Historical analysis klinik entegrasyonunu test eder."""
    print("\n=== TARİHSEL ANALİZ KLİNİK ENTEGRASYON TESTİ ===")
    
    try:
        from clinical.historical_analysis import HistoricalDataAnalyzer
        
        # Analyzer oluştur
        analyzer = HistoricalDataAnalyzer()
        print("✓ Historical analyzer oluşturuldu")
        
        # Test videoları oluştur (farklı tarihler simülasyonu)
        test_videos = []
        for i in range(3):
            video_name = f"test_historical_{i+1}_2024-01-{15+i*5:02d}.mp4"
            create_test_video_with_movement(video_name, duration=2.0)
            test_videos.append(video_name)
        
        print(f"✓ {len(test_videos)} test videosu oluşturuldu")
        
        # Tarihsel video analizi
        from detector import NistagmusDetector
        detector = NistagmusDetector()
        
        patient_id = "TEST_PATIENT_001"
        historical_results = analyzer.analyze_historical_videos(
            patient_id=patient_id,
            video_paths=test_videos,
            detector=detector
        )
        
        print(f"✓ Tarihsel analiz tamamlandı")
        print(f"  - Patient ID: {historical_results['patient_id']}")
        print(f"  - Video sayısı: {historical_results['total_videos']}")
        print(f"  - Başarılı analiz: {historical_results['successful_analyses']}")
        
        # Trend analizi
        if "trend_analysis" in historical_results:
            trend = historical_results["trend_analysis"]
            print(f"  - Nistagmus trendi: {trend.get('nystagmus_trend', {}).get('direction', 'Bilinmiyor')}")
            print(f"  - Şaşılık trendi: {trend.get('strabismus_trend', {}).get('direction', 'Bilinmiyor')}")
            print(f"  - Genel değerlendirme: {trend.get('overall_assessment', 'Değerlendirme yok')}")
        
        # Temizlik
        for video in test_videos:
            os.remove(video)
        
        return True
        
    except Exception as e:
        print(f"✗ Historical analysis klinik testi başarısız: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ana test fonksiyonu."""
    print("KLİNİK ENTEGRASYON TEST SİSTEMİ")
    print("=" * 50)
    
    test_results = []
    
    # Testleri çalıştır
    tests = [
        ("Konfigürasyon Modülü", test_config_module),
        ("Karar Destek Sistemi", test_decision_module),
        ("Analiz Araçları Entegrasyonu", test_analysis_utils_integration),
        ("Detector Klinik Entegrasyonu", test_detector_clinical_integration),
        ("Tarihsel Analiz Entegrasyonu", test_historical_analysis_clinical)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            test_results.append((test_name, result))
            status = "✓ BAŞARILI" if result else "✗ BAŞARISIZ"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            test_results.append((test_name, False))
            print(f"\n{test_name}: ✗ HATA - {str(e)}")
    
    # Özet rapor
    print(f"\n{'='*50}")
    print("TEST SONUÇLARI ÖZETİ")
    print(f"{'='*50}")
    
    total_tests = len(test_results)
    successful_tests = sum(1 for _, success in test_results if success)
    
    for test_name, success in test_results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
    
    print(f"\nToplam: {successful_tests}/{total_tests} test başarılı")
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    print(f"Başarı oranı: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("\n🎉 TÜM TESTLER BAŞARILI! Klinik entegrasyon hazır.")
    elif success_rate >= 80:
        print("\n⚠️  Çoğu test başarılı. Bazı iyileştirmeler gerekebilir.")
    else:
        print("\n❌ Ciddi sorunlar var. Entegrasyon gözden geçirilmeli.")
    
    return success_rate

if __name__ == "__main__":
    main() 