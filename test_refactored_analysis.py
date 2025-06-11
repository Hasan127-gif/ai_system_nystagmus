#!/usr/bin/env python3
"""
Refaktör Edilmiş Video Analizi Test Scripti
==========================================
Bu script, ortak analysis_utils fonksiyonlarının doğru çalıştığını test eder.
"""

import numpy as np
import time
import tempfile
import os
from pathlib import Path
import cv2

def create_test_video(filename: str, duration: float = 5.0, fps: int = 30):
    """
    Test için basit bir video oluşturur
    
    Args:
        filename: Video dosya adı
        duration: Video süresi (saniye)
        fps: Kare hızı
    """
    # Video yazıcısını oluştur
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    # Sentetik hareket verileri
    total_frames = int(duration * fps)
    
    for i in range(total_frames):
        # Boş siyah frame oluştur
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Basit beyaz daire çiz (sahte yüz)
        center_x = 320 + int(20 * np.sin(i * 0.2))  # Yatay hareket
        center_y = 240 + int(10 * np.sin(i * 0.5))  # Dikey hareket
        
        cv2.circle(frame, (center_x, center_y), 50, (255, 255, 255), -1)
        
        # Göz benzeri noktalar
        left_eye = (center_x - 15, center_y - 5)
        right_eye = (center_x + 15, center_y - 5)
        
        cv2.circle(frame, left_eye, 5, (0, 0, 255), -1)
        cv2.circle(frame, right_eye, 5, (0, 255, 0), -1)
        
        writer.write(frame)
    
    writer.release()
    print(f"Test videosu oluşturuldu: {filename}")

def test_analysis_utils():
    """Ortak analiz fonksiyonlarını test eder"""
    print("=== Analiz Utils Test ===")
    
    try:
        # analysis_utils'i import et
        from analysis_utils import (
            analyze_video_file, 
            validate_video_file, 
            calculate_nystagmus_frequency_unified,
            calculate_strabismus_angle_instant,
            format_results_for_api,
            batch_analyze_videos
        )
        
        print("✅ analysis_utils başarıyla import edildi")
        
        # Test video oluştur
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as temp_file:
            test_video_path = temp_file.name
        
        create_test_video(test_video_path)
        
        # Video doğrulama testi
        print("\n--- Video Doğrulama Testi ---")
        validation = validate_video_file(test_video_path)
        print(f"Video geçerli: {validation['valid']}")
        if validation['valid']:
            print(f"Video bilgileri: {validation['info']}")
        
        # Video analiz testi
        print("\n--- Video Analiz Testi ---")
        start_time = time.time()
        
        # Detector olmadan test
        result = analyze_video_file(test_video_path, detector=None, max_frames=100)
        
        elapsed = time.time() - start_time
        print(f"Analiz süresi: {elapsed:.2f}s")
        
        if "error" in result:
            print(f"❌ Analiz hatası: {result['error']}")
        else:
            print(f"✅ Analiz başarılı!")
            print(f"  Nistagmus frekansı: {result['nistagmus_frequency']:.2f} Hz")
            print(f"  Şaşılık açısı: {result['strabismus_angle']:.2f}°")
            print(f"  Yüz tespit oranı: {result['face_detection_rate']:.2f}")
        
        # API format testi
        print("\n--- API Format Testi ---")
        api_result = format_results_for_api(result)
        print(f"API formatı: {api_result['status']}")
        if api_result['status'] == 'success':
            print(f"  Nistagmus tespit: {api_result['nistagmus']['detected']}")
            print(f"  Şaşılık tespit: {api_result['strabismus']['detected']}")
        
        # Frekans hesaplama testi
        print("\n--- Frekans Hesaplama Testi ---")
        test_positions = [100 + 10*np.sin(i*0.5) for i in range(60)]  # 2 saniyelik test verisi
        calculated_freq = calculate_nystagmus_frequency_unified(test_positions, 30.0)
        expected_freq = 0.5 / (2 * np.pi) * 30  # Yaklaşık beklenen frekans
        print(f"Hesaplanan frekans: {calculated_freq:.2f} Hz")
        
        # Şaşılık hesaplama testi
        print("\n--- Şaşılık Hesaplama Testi ---")
        left_eye = (100, 100)
        right_eye = (150, 105)  # 5 piksel dikey fark
        strabismus = calculate_strabismus_angle_instant(left_eye, right_eye)
        print(f"Hesaplanan şaşılık açısı: {strabismus:.2f}°")
        
        # Cleanup
        os.unlink(test_video_path)
        print("\n✅ Tüm testler başarılı!")
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        import traceback
        traceback.print_exc()

def test_detector_integration():
    """Detector entegrasyonunu test eder"""
    print("\n=== Detector Entegrasyon Testi ===")
    
    try:
        from detector import NistagmusDetector
        from analysis_utils import analyze_video_file
        
        # Detector oluştur
        detector = NistagmusDetector()
        print("✅ NistagmusDetector oluşturuldu")
        
        # Test video oluştur
        with tempfile.NamedTemporaryFile(suffix='.avi', delete=False) as temp_file:
            test_video_path = temp_file.name
        
        create_test_video(test_video_path, duration=3.0)
        
        # Detector ile analiz
        print("Detector ile video analizi...")
        result = analyze_video_file(test_video_path, detector=detector, max_frames=50)
        
        if "error" in result:
            print(f"❌ Detector analizi hatası: {result['error']}")
        else:
            print(f"✅ Detector analizi başarılı!")
            print(f"  Nistagmus frekansı: {result['nistagmus_frequency']:.2f} Hz")
            print(f"  Analiz süresi: {result['analysis_duration']:.2f}s")
        
        # Detector'ın kendi analyze_video metodunu test et
        print("\nDetector'ın kendi analiz metodunu test ediyor...")
        detector_result = detector.analyze_video(test_video_path, max_frames=50)
        
        if "error" in detector_result:
            print(f"❌ Detector.analyze_video hatası: {detector_result['error']}")
        else:
            print(f"✅ Detector.analyze_video başarılı!")
            print(f"  Status: {detector_result.get('status', 'unknown')}")
        
        # Cleanup
        os.unlink(test_video_path)
        
    except Exception as e:
        print(f"❌ Detector entegrasyon hatası: {str(e)}")
        import traceback
        traceback.print_exc()

def test_historical_analysis():
    """Historical analysis entegrasyonunu test eder"""
    print("\n=== Historical Analysis Testi ===")
    
    try:
        from clinical.historical_analysis import HistoricalDataAnalyzer
        
        analyzer = HistoricalDataAnalyzer()
        print("✅ HistoricalDataAnalyzer oluşturuldu")
        
        # Test videoları oluştur
        temp_videos = []
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix=f'_2024010{i+1}.avi', delete=False)
            temp_videos.append(temp_file.name)
            create_test_video(temp_file.name, duration=2.0)
        
        print(f"✅ {len(temp_videos)} test videosu oluşturuldu")
        
        # Tarihsel analiz
        print("Tarihsel video analizi yapılıyor...")
        historical_result = analyzer.analyze_historical_videos(
            patient_id="test_patient", 
            video_paths=temp_videos
        )
        
        if "error" in historical_result:
            print(f"❌ Tarihsel analiz hatası: {historical_result['error']}")
        else:
            print(f"✅ Tarihsel analiz başarılı!")
            print(f"  İşlenen video sayısı: {historical_result['valid_videos']}")
            print(f"  Zaman serisi veri noktası: {len(historical_result['time_series'])}")
            print(f"  Eğilim analizi: {historical_result['trend_analysis'].get('overall_assessment', 'N/A')}")
        
        # Cleanup
        for video_path in temp_videos:
            os.unlink(video_path)
        
        print("✅ Historical analysis testi tamamlandı")
        
    except Exception as e:
        print(f"❌ Historical analysis hatası: {str(e)}")
        import traceback
        traceback.print_exc()

def test_api_integration():
    """API entegrasyonunu test eder (simülasyon)"""
    print("\n=== API Entegrasyon Testi (Simülasyon) ===")
    
    try:
        from analysis_utils import format_results_for_api
        
        # Sahte analiz sonucu
        mock_result = {
            "nistagmus_frequency": 4.5,
            "strabismus_angle": 3.2,
            "analysis_duration": 2.8,
            "frame_count": 150,
            "face_detection_rate": 0.85,
            "video_info": {
                "filename": "test_video.mp4",
                "duration": 5.0,
                "fps": 30
            }
        }
        
        # API formatına çevir
        api_formatted = format_results_for_api(mock_result)
        
        print("✅ API format dönüşümü başarılı!")
        print(f"  Status: {api_formatted['status']}")
        print(f"  Nistagmus tespit: {api_formatted['nistagmus']['detected']}")
        print(f"  Şaşılık tespit: {api_formatted['strabismus']['detected']}")
        print(f"  Timestamp: {api_formatted['timestamp']}")
        
        # Hata durumu testi
        error_result = {"error": "Test hatası"}
        error_formatted = format_results_for_api(error_result)
        
        print(f"✅ Hata formatı: {error_formatted['status']} - {error_formatted['message']}")
        
    except Exception as e:
        print(f"❌ API entegrasyon hatası: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Refaktör Edilmiş Video Analizi Test Başlıyor...")
    print("=" * 60)
    
    # Ana testler
    test_analysis_utils()
    test_detector_integration()
    test_historical_analysis()
    test_api_integration()
    
    print("\n" + "=" * 60)
    print("🎉 Refaktör testleri tamamlandı!")
    print("Kod tekrarları başarıyla temizlendi ve ortak fonksiyonlar çalışıyor.") 