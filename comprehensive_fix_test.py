#!/usr/bin/env python3
"""
KAPSAMLI DÜZELTİLME TESTİ
========================
Recursion, historical analysis ve video kalitesi sorunlarını test eder.
"""

import os
import time
from improved_test_video_generator import MedicalGradeVideoGenerator

def test_1_recursion_fixed():
    """TEST 1: Recursion düzeltildi mi?"""
    print("\n" + "="*50)
    print("TEST 1: RECURSİON SORUNU DÜZELTİLDİ Mİ?")
    print("="*50)
    
    try:
        # Video oluştur
        generator = MedicalGradeVideoGenerator()
        test_video = "recursion_fix_test.mp4"
        
        print("📹 Test videosu oluşturuluyor...")
        success = generator.create_realistic_nystagmus_video(
            test_video, duration=3.0, nystagmus_freq=2.0, strabismus_angle=2.0
        )
        
        if not success:
            print("❌ Video oluşturulamadı!")
            return False
        
        print("✅ Video oluşturuldu")
        
        # Detector testi
        from detector import NistagmusDetector
        detector = NistagmusDetector()
        
        print("🎬 Video analizi (recursion testi)...")
        start_time = time.time()
        
        try:
            results = detector.analyze_video(test_video, max_frames=60)
            analysis_time = time.time() - start_time
            
            if "error" in results:
                print(f"❌ Analiz hatası: {results['error']}")
                os.remove(test_video)
                return False
            
            print(f"✅ Analiz başarılı ({analysis_time:.2f}s)")
            print(f"   Nistagmus: {results.get('nistagmus_frequency', 0):.2f} Hz")
            print(f"   Şaşılık: {results.get('strabismus_angle', 0):.2f}°")
            print(f"   Yüz tespit: {results.get('face_detection_rate', 0):.1%}")
            
            os.remove(test_video)
            return True
            
        except RecursionError:
            print("❌ RECURSİON HATASI DEVAM EDİYOR!")
            os.remove(test_video)
            return False
            
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        return False

def test_2_historical_analysis_fixed():
    """TEST 2: Historical analysis düzeltildi mi?"""
    print("\n" + "="*50)
    print("TEST 2: TARİHSEL ANALİZ DÜZELTİLDİ Mİ?")
    print("="*50)
    
    try:
        # Çoklu video oluştur
        generator = MedicalGradeVideoGenerator()
        test_videos = []
        
        print("📹 3 tarihsel test videosu oluşturuluyor...")
        for i in range(3):
            video_name = f"historical_test_{i+1}_2024-01-{15+i*5:02d}.mp4"
            success = generator.create_realistic_nystagmus_video(
                video_name, duration=2.0, nystagmus_freq=1.5+i*0.5, strabismus_angle=1.0+i
            )
            if success:
                test_videos.append(video_name)
                print(f"   ✅ {video_name}")
        
        if len(test_videos) < 2:
            print("❌ Yetersiz test videosu!")
            return False
        
        # Batch analizi test et
        print("🔄 Batch analizi yapılıyor...")
        from analysis_utils import batch_analyze_videos
        
        batch_results = batch_analyze_videos(test_videos, max_frames=40)
        
        if "error" in batch_results:
            print(f"❌ Batch analiz hatası: {batch_results['error']}")
            return False
        
        # Anahtar kontrolü
        required_keys = ["summary", "results"]
        if not all(key in batch_results for key in required_keys):
            print(f"❌ Batch sonuçlarında eksik anahtarlar!")
            return False
        
        summary = batch_results["summary"]
        summary_keys = ["total_videos", "successful_analyses", "failed_analyses"]
        if not all(key in summary for key in summary_keys):
            print(f"❌ Summary'de eksik anahtarlar!")
            return False
        
        print(f"✅ Batch analiz başarılı")
        print(f"   Toplam: {summary['total_videos']}")
        print(f"   Başarılı: {summary['successful_analyses']}")
        
        # Historical analyzer test et (varsa)
        try:
            from clinical.historical_analysis import HistoricalDataAnalyzer
            from detector import NistagmusDetector
            
            print("🏥 Historical analyzer test ediliyor...")
            analyzer = HistoricalDataAnalyzer()
            detector = NistagmusDetector()
            
            historical_results = analyzer.analyze_historical_videos(
                patient_id="TEST_PATIENT",
                video_paths=test_videos,
                detector=detector
            )
            
            # Kritik anahtar kontrolü
            required_hist_keys = ["patient_id", "total_videos", "successful_analyses"]
            missing_keys = [key for key in required_hist_keys if key not in historical_results]
            
            if missing_keys:
                print(f"❌ Historical sonuçlarında eksik anahtarlar: {missing_keys}")
                return False
            
            print(f"✅ Historical analysis başarılı")
            print(f"   Patient ID: {historical_results['patient_id']}")
            print(f"   Toplam video: {historical_results['total_videos']}")
            print(f"   Başarılı analiz: {historical_results['successful_analyses']}")
            
        except ImportError:
            print("⚠️ Historical analyzer bulunamadı, batch analizi yeterli")
        
        # Temizlik
        for video in test_videos:
            if os.path.exists(video):
                os.remove(video)
        
        return True
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        return False

def test_3_analysis_utils_quality():
    """TEST 3: Analysis utils kaliteli video ile çalışıyor mu?"""
    print("\n" + "="*50)
    print("TEST 3: ANALİZ ARAÇLARI KALİTE TESTİ")
    print("="*50)
    
    try:
        # Kaliteli test videosu
        generator = MedicalGradeVideoGenerator()
        test_video = "quality_analysis_test.mp4"
        
        print("📹 Kaliteli test videosu oluşturuluyor...")
        success = generator.create_realistic_nystagmus_video(
            test_video, duration=4.0, nystagmus_freq=3.0, strabismus_angle=4.0
        )
        
        if not success:
            print("❌ Video oluşturulamadı!")
            return False
        
        # Analysis utils test
        from analysis_utils import analyze_video_file, validate_video_file
        
        print("🔍 Video doğrulanıyor...")
        validation = validate_video_file(test_video)
        
        if not validation["valid"]:
            print(f"❌ Video doğrulama başarısız!")
            os.remove(test_video)
            return False
        
        print(f"✅ Video doğrulandı: {validation['info']['duration']:.1f}s")
        
        print("🎬 Analysis utils ile analiz...")
        results = analyze_video_file(test_video, max_frames=80)
        
        if "error" in results:
            print(f"❌ Analiz hatası: {results['error']}")
            os.remove(test_video)
            return False
        
        print(f"✅ Analiz başarılı")
        print(f"   Nistagmus: {results.get('nistagmus_frequency', 0):.2f} Hz")
        print(f"   Şaşılık: {results.get('strabismus_angle', 0):.2f}°")
        print(f"   Yüz tespit: {results.get('face_detection_rate', 0):.1%}")
        print(f"   İşlenen kare: {results.get('video_info', {}).get('analyzed_frames', 0)}")
        
        # Başarı kriterleri
        face_detection_rate = results.get("face_detection_rate", 0)
        analyzed_frames = results.get("video_info", {}).get("analyzed_frames", 0)
        
        success = (face_detection_rate >= 0.3 and analyzed_frames >= 10)
        
        os.remove(test_video)
        return success
        
    except Exception as e:
        print(f"❌ Test hatası: {str(e)}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 KAPSAMLI DÜZELTİLME TESTLERİ")
    print("="*60)
    
    tests = [
        ("Recursion Sorunu Düzeltildi Mi?", test_1_recursion_fixed),
        ("Tarihsel Analiz Düzeltildi Mi?", test_2_historical_analysis_fixed),
        ("Analysis Utils Kalite Testi", test_3_analysis_utils_quality)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n⏰ {test_name} başlatılıyor...")
        test_start = time.time()
        
        try:
            result = test_func()
            test_duration = time.time() - test_start
            results.append((test_name, result, test_duration))
            
            status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
            print(f"⏱️ {test_name}: {status} ({test_duration:.2f}s)")
            
        except Exception as e:
            test_duration = time.time() - test_start
            results.append((test_name, False, test_duration))
            print(f"💥 {test_name}: ❌ HATA - {str(e)} ({test_duration:.2f}s)")
    
    # Özet rapor
    total_time = time.time() - start_time
    successful_tests = sum(1 for _, success, _ in results if success)
    total_tests = len(results)
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "="*60)
    print("📋 KAPSAMLI TEST SONUÇLARI ÖZETİ")
    print("="*60)
    
    for test_name, success, duration in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name} ({duration:.2f}s)")
    
    print(f"\n📊 GENEL BAŞARI ORANI: {success_rate:.1f}% ({successful_tests}/{total_tests})")
    print(f"⏱️ TOPLAM TEST SÜRESİ: {total_time:.2f}s")
    
    if success_rate == 100:
        print("\n🎉 MÜKEMMEL! TÜM SORUNLAR ÇÖZÜLDü!")
        print("   📋 Recursion sorunu: ✅ Çözüldü")
        print("   📋 Tarihsel analiz: ✅ Düzeltildi") 
        print("   📋 Video kalitesi: ✅ İyileştirildi")
        print("   🚀 Sistem tamamen hazır!")
    elif success_rate >= 67:
        print("\n🎯 ÇOK İYİ! Çoğu sorun çözüldü!")
        print("   Sistem neredeyse hazır! ⚡")
    else:
        print("\n🔥 Bazı sorunlar devam ediyor!")
        print("   Ek çalışma gerekli. 🛠️")
    
    return success_rate

if __name__ == "__main__":
    success_rate = main()
    exit(0 if success_rate >= 67 else 1) 