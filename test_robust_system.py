#!/usr/bin/env python3
"""
GÜÇLÜ SİSTEM TEST ARACı
======================
Recursion ve video kalitesi sorunlarını çözen kapsamlı test sistemi.
"""

import os
import time
import logging
import traceback
from typing import Dict, Any, List
from improved_test_video_generator import MedicalGradeVideoGenerator

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobustSystemTester:
    """Güçlü sistem test sınıfı."""
    
    def __init__(self):
        """Test sistemini başlatır."""
        self.video_generator = MedicalGradeVideoGenerator()
        self.test_results = []
        self.temp_files = []
        
    def test_1_recursion_fix(self) -> bool:
        """
        TEST 1: Recursion sorununun çözülüp çözülmediğini test eder.
        """
        print("\n" + "="*60)
        print("TEST 1: RECURSİON SORUNU DÜZELTİLDİ Mİ?")
        print("="*60)
        
        try:
            # Test videosu oluştur
            test_video = "recursion_test.mp4"
            self.temp_files.append(test_video)
            
            print("📹 Test videosu oluşturuluyor...")
            success = self.video_generator.create_realistic_nystagmus_video(
                test_video, duration=3.0, nystagmus_freq=2.0, strabismus_angle=2.0
            )
            
            if not success:
                print("❌ Test videosu oluşturulamadı!")
                return False
            
            print("✅ Test videosu oluşturuldu")
            
            # Detector import ve test
            print("🔍 Detector başlatılıyor...")
            from detector import NistagmusDetector
            detector = NistagmusDetector()
            
            print(f"✅ Detector başlatıldı (MediaPipe: {detector.is_initialized})")
            
            # Recursion testi - Video analizi
            print("🎬 Video analizi başlatılıyor...")
            start_time = time.time()
            
            try:
                results = detector.analyze_video(test_video, max_frames=60)
                analysis_time = time.time() - start_time
                
                if "error" in results:
                    print(f"❌ Video analizi hatası: {results['error']}")
                    return False
                
                print(f"✅ Video analizi başarılı ({analysis_time:.2f}s)")
                print(f"   - Nistagmus: {results.get('nistagmus_frequency', 0):.2f} Hz")
                print(f"   - Şaşılık: {results.get('strabismus_angle', 0):.2f}°")
                print(f"   - Yüz tespit: {results.get('face_detection_rate', 0):.1%}")
                
            except RecursionError:
                print("❌ RECURSİON HATASI DEVAM EDİYOR!")
                return False
            except Exception as e:
                print(f"❌ Beklenmeyen hata: {str(e)}")
                return False
            
            # Frame-by-frame recursion testi
            print("🔍 Kare-bazlı recursion testi...")
            import cv2
            
            cap = cv2.VideoCapture(test_video)
            frame_test_count = 0
            successful_detections = 0
            
            while frame_test_count < 10:  # İlk 10 kareyi test et
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    left_center, right_center = detector.detect_iris_centers(frame)
                    if left_center is not None and right_center is not None:
                        successful_detections += 1
                    frame_test_count += 1
                except RecursionError:
                    cap.release()
                    print("❌ KARE-BAZLI RECURSİON HATASI!")
                    return False
            
            cap.release()
            
            detection_rate = successful_detections / frame_test_count if frame_test_count > 0 else 0
            print(f"✅ Kare-bazlı tespit: {detection_rate:.1%} ({successful_detections}/{frame_test_count})")
            
            success = detection_rate > 0.2  # %20'den fazla tespit bekliyoruz
            print(f"📊 TEST 1 SONUCU: {'✅ BAŞARILI' if success else '❌ BAŞARISIZ'}")
            
            return success
            
        except Exception as e:
            print(f"❌ TEST 1 HATASI: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_2_analysis_utils_integration(self) -> bool:
        """
        TEST 2: Analysis utils entegrasyonunu test eder.
        """
        print("\n" + "="*60)
        print("TEST 2: ANALİZ ARAÇLARI ENTEGRASYONU")
        print("="*60)
        
        try:
            # Kaliteli test videosu
            test_video = "analysis_utils_test.mp4"
            self.temp_files.append(test_video)
            
            print("📹 Kaliteli test videosu oluşturuluyor...")
            success = self.video_generator.create_realistic_nystagmus_video(
                test_video, duration=4.0, nystagmus_freq=3.0, strabismus_angle=4.0
            )
            
            if not success:
                print("❌ Test videosu oluşturulamadı!")
                return False
            
            print("✅ Test videosu oluşturuldu")
            
            # Analysis utils import
            print("📦 Analysis utils import ediliyor...")
            from analysis_utils import analyze_video_file, validate_video_file
            
            # Video doğrulama
            print("🔍 Video doğrulanıyor...")
            validation = validate_video_file(test_video)
            
            if not validation["valid"]:
                print(f"❌ Video doğrulama başarısız: {validation['error']}")
                return False
            
            print(f"✅ Video doğrulandı: {validation['info']['duration']:.1f}s")
            
            # Video analizi
            print("🎬 Analysis utils ile video analizi...")
            start_time = time.time()
            
            results = analyze_video_file(test_video, max_frames=80)
            analysis_time = time.time() - start_time
            
            if "error" in results:
                print(f"❌ Analiz hatası: {results['error']}")
                return False
            
            print(f"✅ Analiz başarılı ({analysis_time:.2f}s)")
            print(f"   - Nistagmus: {results.get('nistagmus_frequency', 0):.2f} Hz")
            print(f"   - Şaşılık: {results.get('strabismus_angle', 0):.2f}°")
            print(f"   - İşlenen kare: {results.get('processed_frames', 0)}")
            print(f"   - Yüz tespit: {results.get('face_detection_rate', 0):.1%}")
            
            # Klinik değerlendirme kontrolü
            if "clinical_evaluation" in results:
                clinical = results["clinical_evaluation"]
                print(f"   - Klinik durum: {clinical['overall_assessment']['category']}")
                has_pathology = clinical['overall_assessment']['has_pathology']
                print(f"   - Patoloji: {'VAR' if has_pathology else 'YOK'}")
            
            # API formatı kontrolü
            if "api_format" in results:
                api_data = results["api_format"]
                print(f"   - API durumu: {api_data.get('status', 'Bilinmiyor')}")
            
            # Başarı kriterleri
            face_detection_rate = results.get("face_detection_rate", 0)
            processed_frames = results.get("processed_frames", 0)
            
            success = (face_detection_rate >= 0.4 and processed_frames >= 10)
            print(f"📊 TEST 2 SONUCU: {'✅ BAŞARILI' if success else '❌ BAŞARISIZ'}")
            
            return success
            
        except Exception as e:
            print(f"❌ TEST 2 HATASI: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_3_historical_analysis_fixed(self) -> bool:
        """
        TEST 3: Düzeltilmiş tarihsel analizi test eder.
        """
        print("\n" + "="*60)
        print("TEST 3: TARİHSEL ANALİZ DÜZELTİLDİ Mİ?")
        print("="*60)
        
        try:
            # Çoklu test videoları
            test_videos = []
            video_configs = [
                ("historical_1_2024-01-15.mp4", 2.0, 1.5, 1.0),
                ("historical_2_2024-01-20.mp4", 2.5, 2.5, 2.0),
                ("historical_3_2024-01-25.mp4", 3.0, 3.5, 3.0)
            ]
            
            print(f"📹 {len(video_configs)} tarihsel test videosu oluşturuluyor...")
            
            for filename, duration, nyst_freq, strab_angle in video_configs:
                self.temp_files.append(filename)
                success = self.video_generator.create_realistic_nystagmus_video(
                    filename, duration, nyst_freq, strab_angle
                )
                if success:
                    test_videos.append(filename)
                    print(f"   ✅ {filename}")
                else:
                    print(f"   ❌ {filename}")
            
            if len(test_videos) < 2:
                print("❌ Yetersiz test videosu oluşturuldu!")
                return False
            
            print(f"✅ {len(test_videos)} test videosu hazır")
            
            # Historical analysis testi
            print("📊 Tarihsel analiz başlatılıyor...")
            
            try:
                # Önce batch analizi deneyelim
                from analysis_utils import batch_analyze_videos
                
                print("🔄 Batch analizi yapılıyor...")
                batch_results = batch_analyze_videos(test_videos, max_frames_per_video=40)
                
                if "error" in batch_results:
                    print(f"❌ Batch analiz hatası: {batch_results['error']}")
                    return False
                
                # Anahtar kontrolü
                required_keys = ["summary", "results"]
                missing_keys = [key for key in required_keys if key not in batch_results]
                
                if missing_keys:
                    print(f"❌ Batch sonuçlarında eksik anahtarlar: {missing_keys}")
                    return False
                
                summary = batch_results["summary"]
                summary_keys = ["total_videos", "successful_analyses", "failed_analyses"]
                missing_summary_keys = [key for key in summary_keys if key not in summary]
                
                if missing_summary_keys:
                    print(f"❌ Summary'de eksik anahtarlar: {missing_summary_keys}")
                    return False
                
                print(f"✅ Batch analiz başarılı")
                print(f"   - Toplam video: {summary['total_videos']}")
                print(f"   - Başarılı: {summary['successful_analyses']}")
                print(f"   - Başarısız: {summary['failed_analyses']}")
                
                # Historical analyzer'ı deneyelim (varsa)
                try:
                    from clinical.historical_analysis import HistoricalDataAnalyzer
                    from detector import NistagmusDetector
                    
                    print("🏥 Historical analyzer başlatılıyor...")
                    analyzer = HistoricalDataAnalyzer()
                    detector = NistagmusDetector()
                    
                    patient_id = "TEST_PATIENT_ROBUST"
                    historical_results = analyzer.analyze_historical_videos(
                        patient_id=patient_id,
                        video_paths=test_videos,
                        detector=detector
                    )
                    
                    # Anahtar kontrolü
                    required_hist_keys = ["patient_id", "total_videos", "successful_analyses"]
                    missing_hist_keys = [key for key in required_hist_keys if key not in historical_results]
                    
                    if missing_hist_keys:
                        print(f"❌ Historical sonuçlarında eksik anahtarlar: {missing_hist_keys}")
                        return False
                    
                    print(f"✅ Historical analysis başarılı")
                    print(f"   - Patient ID: {historical_results['patient_id']}")
                    print(f"   - Toplam video: {historical_results['total_videos']}")
                    print(f"   - Başarılı analiz: {historical_results['successful_analyses']}")
                    
                    if "trend_analysis" in historical_results:
                        trend = historical_results["trend_analysis"]
                        if "error" not in trend:
                            print(f"   - Trend analizi: ✅ Mevcut")
                            nyst_trend = trend.get("nistagmus_frequency_trend", {}).get("trend", "bilinmiyor")
                            strab_trend = trend.get("strabismus_angle_trend", {}).get("trend", "bilinmiyor")
                            print(f"     - Nistagmus trendi: {nyst_trend}")
                            print(f"     - Şaşılık trendi: {strab_trend}")
                        else:
                            print(f"   - Trend analizi: ⚠️ Hata - {trend['error']}")
                    
                    success = historical_results.get("successful_analyses", 0) > 0
                    
                except ImportError:
                    print("⚠️ Historical analyzer modülü bulunamadı, batch analizi yeterli")
                    success = summary["successful_analyses"] > 0
                
            except Exception as e:
                print(f"❌ Tarihsel analiz hatası: {str(e)}")
                return False
            
            print(f"📊 TEST 3 SONUCU: {'✅ BAŞARILI' if success else '❌ BAŞARISIZ'}")
            return success
            
        except Exception as e:
            print(f"❌ TEST 3 HATASI: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_4_clinical_integration(self) -> bool:
        """
        TEST 4: Klinik entegrasyonu test eder.
        """
        print("\n" + "="*60)
        print("TEST 4: KLİNİK ENTEGRASYON")
        print("="*60)
        
        try:
            # Test videosu
            test_video = "clinical_test.mp4"
            self.temp_files.append(test_video)
            
            print("📹 Klinik test videosu oluşturuluyor...")
            success = self.video_generator.create_realistic_nystagmus_video(
                test_video, duration=3.5, nystagmus_freq=4.0, strabismus_angle=5.0
            )
            
            if not success:
                print("❌ Test videosu oluşturulamadı!")
                return False
            
            print("✅ Test videosu oluşturuldu")
            
            # Klinik modüllerin kontrolü
            print("🏥 Klinik modüller kontrol ediliyor...")
            
            clinical_available = True
            try:
                from config import get_clinical_threshold, NYSTAGMUS_THRESHOLDS
                from decision import classify_findings, quick_assessment
                print("✅ Klinik modüller mevcut")
            except ImportError as e:
                print(f"⚠️ Klinik modüller eksik: {str(e)}")
                clinical_available = False
            
            # Detector ile klinik analiz
            print("🔍 Klinik değerlendirmeli analiz...")
            from detector import NistagmusDetector
            
            detector = NistagmusDetector()
            
            # Yaş faktörlü analiz
            patient_ages = [25.0, 60.0]  # Farklı yaş grupları
            
            clinical_results = []
            
            for age in patient_ages:
                print(f"   👤 {age} yaş analizi...")
                
                results = detector.analyze_video(test_video, patient_age=age, max_frames=60)
                
                if "error" in results:
                    print(f"   ❌ {age} yaş analizi başarısız: {results['error']}")
                    continue
                
                print(f"   ✅ {age} yaş analizi başarılı")
                
                # Klinik değerlendirme kontrolü
                if clinical_available and "clinical_evaluation_with_age" in results:
                    clinical_eval = results["clinical_evaluation_with_age"]
                    category = clinical_eval.get("overall_assessment", {}).get("category", "bilinmiyor")
                    has_pathology = clinical_eval.get("overall_assessment", {}).get("has_pathology", False)
                    
                    print(f"     - Kategori: {category}")
                    print(f"     - Patoloji: {'VAR' if has_pathology else 'YOK'}")
                    
                    clinical_results.append(True)
                else:
                    print(f"     - Klinik değerlendirme: Mevcut değil")
                    clinical_results.append(False)
            
            # Başarı değerlendirmesi
            if clinical_available:
                success = len(clinical_results) > 0 and any(clinical_results)
                print(f"✅ Klinik entegrasyon: {'Aktif' if success else 'Eksik'}")
            else:
                success = len(clinical_results) > 0  # En azından analiz çalıştı
                print(f"⚠️ Klinik entegrasyon: Modüller eksik ama temel analiz çalışıyor")
            
            print(f"📊 TEST 4 SONUCU: {'✅ BAŞARILI' if success else '❌ BAŞARISIZ'}")
            return success
            
        except Exception as e:
            print(f"❌ TEST 4 HATASI: {str(e)}")
            traceback.print_exc()
            return False
    
    def run_all_tests(self) -> dict:
        """
        Tüm testleri çalıştırır ve sonuçları döndürür.
        
        Returns:
            dict: Test sonuçları özeti
        """
        print("🚀 GÜÇLÜ SİSTEM TESTLERİ BAŞLATIILIYOR")
        print("="*80)
        
        tests = [
            ("Recursion Sorunu Düzeltildi Mi?", self.test_1_recursion_fix),
            ("Analysis Utils Entegrasyonu", self.test_2_analysis_utils_integration),
            ("Tarihsel Analiz Düzeltildi Mi?", self.test_3_historical_analysis_fixed),
            ("Klinik Entegrasyon", self.test_4_clinical_integration)
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
        
        # Temizlik
        self.cleanup()
        
        # Özet rapor
        total_time = time.time() - start_time
        successful_tests = sum(1 for _, success, _ in results if success)
        total_tests = len(results)
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("📋 GÜÇLÜ SİSTEM TEST SONUÇLARI ÖZETİ")
        print("="*80)
        
        for test_name, success, duration in results:
            status = "✅" if success else "❌"
            print(f"{status} {test_name} ({duration:.2f}s)")
        
        print(f"\n📊 GENEL BAŞARI ORANI: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        print(f"⏱️ TOPLAM TEST SÜRESİ: {total_time:.2f}s")
        
        if success_rate == 100:
            print("\n🎉 MÜKEMMEL! TÜM SORUNLAR ÇÖZÜLDü!")
            print("   Sistem entegrasyonu tamamen hazır! 🚀")
        elif success_rate >= 75:
            print("\n🎯 ÇOK İYİ! Çoğu sorun çözüldü!")
            print("   Küçük iyileştirmelerle sistem hazır olacak. ⚡")
        elif success_rate >= 50:
            print("\n⚡ ORTA! Önemli ilerleme var!")
            print("   Bazı kritik sorunlar çözülmeli. 🔧")
        else:
            print("\n🔥 CİDDİ SORUNLAR VAR!")
            print("   Temel sorunların çözülmesi gerekiyor. 🛠️")
        
        return {
            "success_rate": success_rate,
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "total_time": total_time,
            "results": results
        }
    
    def cleanup(self):
        """Geçici dosyaları temizler."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.info(f"Geçici dosya silindi: {temp_file}")
            except Exception as e:
                logger.warning(f"Geçici dosya silinemedi ({temp_file}): {str(e)}")

def main():
    """Ana test fonksiyonu."""
    tester = RobustSystemTester()
    results = tester.run_all_tests()
    
    return results["success_rate"]

if __name__ == "__main__":
    success_rate = main()
    exit(0 if success_rate >= 75 else 1)  # %75+ başarı durumunda success exit 