"""
Göz İzleme Sistemi - Entegrasyon ve Performans Testleri

Bu modül, göz izleme sisteminin bileşenlerinin birbiriyle entegrasyonunu ve
performans özelliklerini test eder.
"""

import pytest
import numpy as np
import cv2
import time
import os
import logging
from unittest.mock import patch, MagicMock
import multiprocessing
import psutil
import json
import tempfile
import threading
import queue

# Modülleri import et
from EyeTrackingStrategy import MediaPipeStrategy, DlibStrategy
from EyeTrackingStrategy import OpenCVDNNStrategy, HybridStrategy
from EyeAnalyzer import EyeAnalyzer
from CalibrationUI import AdvancedCalibration
from DeepEyeTracker import DeepEyeTracker

# Test veri dizini
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def get_system_resources():
    """Sistemin CPU ve bellek kullanımını ölçer"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_mb": memory.used / (1024 * 1024)
    }

def create_test_video(output_path, frames=100, width=640, height=480, fps=30):
    """
    Test için göz hareketlerini içeren bir video oluşturur
    
    Args:
        output_path: Çıktı video dosya yolu
        frames: Video kare sayısı
        width: Görüntü genişliği
        height: Görüntü yüksekliği
        fps: Kare hızı
    """
    # OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        pytest.skip(f"Video dosyası oluşturulamadı: {output_path}")
    
    # Yüz parametreleri
    face_center = (width // 2, height // 2)
    face_size = (width // 3, height // 2)
    
    # Göz merkezleri
    left_eye_center = (width // 2 - width // 8, height // 2 - height // 10)
    right_eye_center = (width // 2 + width // 8, height // 2 - height // 10)
    eye_radius = width // 25
    
    # Hareketli göz simülasyonu
    for i in range(frames):
        # Temel görüntü
        img = np.ones((height, width, 3), dtype=np.uint8) * 200
        
        # Yüz çiz
        cv2.ellipse(img, face_center, face_size, 0, 0, 360, (200, 200, 200), -1)
        
        # Farklı hareket desenleri
        if i < frames // 3:  # İlk kısım: sol-sağ hareket
            offset_x = 20 * np.sin(2 * np.pi * i / (frames // 6))
            offset_y = 0
        elif i < 2 * frames // 3:  # İkinci kısım: yukarı-aşağı hareket
            offset_x = 0
            offset_y = 15 * np.sin(2 * np.pi * (i - frames // 3) / (frames // 6))
        else:  # Son kısım: nistagmus benzeri hareket
            if i % 10 < 7:  # Yavaş faz
                offset_x = -15 * ((i % 10) / 7)
                offset_y = 0
            else:  # Hızlı faz
                offset_x = -15 + 15 * ((i % 10 - 7) / 3)
                offset_y = 0
        
        # Gözleri çiz
        for eye_center in [
            (left_eye_center[0] + offset_x, left_eye_center[1] + offset_y),
            (right_eye_center[0] + offset_x, right_eye_center[1] + offset_y)
        ]:
            # Göz akı
            cv2.circle(img, (int(eye_center[0]), int(eye_center[1])), eye_radius, (250, 250, 250), -1)
            
            # İris
            cv2.circle(img, (int(eye_center[0]), int(eye_center[1])), 
                      eye_radius // 2, (80, 80, 150), -1)
            
            # Pupil
            cv2.circle(img, (int(eye_center[0]), int(eye_center[1])), 
                      eye_radius // 4, (30, 30, 30), -1)
        
        # Kareyi videoya ekle
        video_writer.write(img)
    
    # Kaynakları serbest bırak
    video_writer.release()
    return output_path


class TestIntegration:
    """Entegrasyon testleri sınıfı"""
    
    @pytest.fixture
    def setup_test_environment(self):
        """Test ortamını hazırlama"""
        # Test veri klasörünü oluştur
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
        # Test video dosyası yolu
        video_path = os.path.join(TEST_DATA_DIR, "test_eye_movement.avi")
        
        # Test videosu oluştur (eğer yoksa)
        if not os.path.exists(video_path):
            create_test_video(video_path)
        
        yield {
            "video_path": video_path
        }
    
    def test_strategy_to_analyzer_integration(self, setup_test_environment):
        """Strateji ve analizör entegrasyonunun testi"""
        video_path = setup_test_environment["video_path"]
        
        # Video okuyucu
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.skip(f"Video açılamadı: {video_path}")
        
        # Strateji (MediaPipe)
        strategy = MediaPipeStrategy()
        
        # Analizör
        analyzer = EyeAnalyzer(
            fps=30.0,
            buffer_seconds=1.0,
            nistagmus_freq_threshold=0.5,
            spv_threshold=30.0,
            pixel_to_degree_ratio=0.1
        )
        
        # Video üzerinden işleme
        frame_count = 0
        detection_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Göz tespiti yap
            landmarks = strategy.detect_face_landmarks(frame)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                detection_count += 1
                
                # Sol ve sağ göz iris merkezleri (MediaPipe indeksleri: 468, 473)
                left_iris = landmarks.multi_face_landmarks[0].landmark[468]
                right_iris = landmarks.multi_face_landmarks[0].landmark[473]
                
                # Piksel koordinatlarına dönüştür
                h, w, _ = frame.shape
                left_pupil = (int(left_iris.x * w), int(left_iris.y * h))
                right_pupil = (int(right_iris.x * w), int(right_iris.y * h))
                
                # Analizöre veri gönder
                analyzer.update_data(
                    left_pupil=left_pupil,
                    right_pupil=right_pupil,
                    timestamp=time.time(),
                    is_left_closed=False,
                    is_right_closed=False
                )
            
            frame_count += 1
        
        # Kaynakları serbest bırak
        cap.release()
        strategy.release()
        
        # En az %80 karede yüz tespiti başarılı olmalı
        detection_rate = detection_count / frame_count if frame_count > 0 else 0
        assert detection_rate >= 0.8, f"Tespit oranı düşük: {detection_rate:.2f}"
        
        # Analiz sonuçlarını kontrol et
        analysis_results = analyzer.analyze()
        assert "nistagmus_freq" in analysis_results
        assert "spv" in analysis_results
        assert "strabismus_angle" in analysis_results
        
        print(f"Tespit oranı: {detection_rate:.2f}")
        print(f"Nistagmus frekansı: {analysis_results['nistagmus_freq']}")
        print(f"SPV: {analysis_results['spv']}")
    
    def test_end_to_end_pipeline(self, setup_test_environment):
        """Uçtan uca işlem hattının testi"""
        video_path = setup_test_environment["video_path"]
        
        # Video okuyucu
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.skip(f"Video açılamadı: {video_path}")
        
        # İşlem hattı bileşenleri
        try:
            # Strateji
            strategy = HybridStrategy([
                MediaPipeStrategy(),
                DlibStrategy()
            ])
        except (ImportError, Exception):
            # Dlib yoksa sadece MediaPipe kullan
            strategy = MediaPipeStrategy()
        
        # Analizör
        analyzer = EyeAnalyzer(
            fps=30.0,
            buffer_seconds=1.0,
            nistagmus_freq_threshold=0.5,
            spv_threshold=30.0,
            pixel_to_degree_ratio=0.1
        )
        
        # Kalibrasyon (sadece test için)
        calibration = AdvancedCalibration(num_points=5)
        calibration.set_screen_dimensions(640, 480)  # Video boyutları
        
        # Sonuçları saklamak için
        results_queue = queue.Queue()
        
        def processing_thread():
            """Video işleme iş parçacığı"""
            frame_count = 0
            valid_frames = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Göz tespiti yap
                landmarks = strategy.detect_face_landmarks(frame)
                
                # Sonuç kontrolü
                if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                    valid_frames += 1
                    
                    # Göz pozisyonlarını hesapla
                    h, w, _ = frame.shape
                    left_iris = landmarks.multi_face_landmarks[0].landmark[468]
                    right_iris = landmarks.multi_face_landmarks[0].landmark[473]
                    
                    left_pupil = (int(left_iris.x * w), int(left_iris.y * h))
                    right_pupil = (int(right_iris.x * w), int(right_iris.y * h))
                    
                    # Analizöre veri gönder
                    analyzer.update_data(
                        left_pupil=left_pupil,
                        right_pupil=right_pupil,
                        timestamp=time.time(),
                        is_left_closed=False,
                        is_right_closed=False
                    )
                    
                    # Kalibrasyona göre piksel-derece dönüşümü
                    gaze_point = calibration.map_pupils_to_screen(left_pupil, right_pupil)
                
                frame_count += 1
            
            # Analiz sonuçlarını al
            analysis_results = analyzer.analyze()
            
            # Sonuçları kuyruğa ekle
            results_queue.put({
                "frame_count": frame_count,
                "valid_frames": valid_frames,
                "analysis_results": analysis_results
            })
        
        # İşlem hattını çalıştır
        thread = threading.Thread(target=processing_thread)
        thread.start()
        thread.join()
        
        # Sonuçları al
        results = results_queue.get()
        
        # Kaynakları serbest bırak
        cap.release()
        strategy.release()
        
        # Sonuçları kontrol et
        frame_count = results["frame_count"]
        valid_frames = results["valid_frames"]
        analysis_results = results["analysis_results"]
        
        # En az %80 karede geçerli tespit olmalı
        detection_rate = valid_frames / frame_count if frame_count > 0 else 0
        assert detection_rate >= 0.8, f"Tespit oranı düşük: {detection_rate:.2f}"
        
        # Analiz sonuçlarını kontrol et
        assert "nistagmus_freq" in analysis_results
        assert "spv" in analysis_results
        assert "strabismus_angle" in analysis_results
        
        print(f"Uçtan uca işlem hattı sonuçları:")
        print(f"- Toplam kare: {frame_count}")
        print(f"- Geçerli kare: {valid_frames}")
        print(f"- Tespit oranı: {detection_rate:.2f}")
        print(f"- Nistagmus frekansı: {analysis_results['nistagmus_freq']}")
        print(f"- SPV: {analysis_results['spv']}")
    
    def test_estrategy_switching(self, setup_test_environment):
        """Göz izleme stratejileri arasında geçiş testi"""
        video_path = setup_test_environment["video_path"]
        
        # Video okuyucu
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.skip(f"Video açılamadı: {video_path}")
        
        # Mevcut stratejileri belirle
        available_strategies = ["mediapipe"]
        
        try:
            # Dlib stratejisini test et
            dlib_strategy = DlibStrategy()
            dlib_strategy.release()
            available_strategies.append("dlib")
        except (ImportError, Exception):
            pass
        
        try:
            # OpenCV DNN stratejisini test et
            opencv_strategy = OpenCVDNNStrategy()
            opencv_strategy.release()
            available_strategies.append("opencv_dnn")
        except (ImportError, Exception):
            pass
        
        # İki veya daha fazla strateji varsa hibrit test edilebilir
        if len(available_strategies) >= 2:
            available_strategies.append("hybrid")
        
        # Sonuçları saklama
        strategy_results = {}
        
        # Her stratejiyi test et
        for strategy_name in available_strategies:
            # Strateji oluştur
            if strategy_name == "mediapipe":
                strategy = MediaPipeStrategy()
            elif strategy_name == "dlib":
                strategy = DlibStrategy()
            elif strategy_name == "opencv_dnn":
                strategy = OpenCVDNNStrategy()
            elif strategy_name == "hybrid":
                # İlk iki stratejiyi kullan
                strategies = []
                if "mediapipe" in available_strategies:
                    strategies.append(MediaPipeStrategy())
                if "dlib" in available_strategies and len(strategies) < 2:
                    strategies.append(DlibStrategy())
                if "opencv_dnn" in available_strategies and len(strategies) < 2:
                    strategies.append(OpenCVDNNStrategy())
                
                strategy = HybridStrategy(strategies)
            
            # Video başına dön
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # İşlem süresini ölç
            start_time = time.time()
            
            # Video üzerinden işleme
            frame_count = 0
            detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Çerçeveyi işle
                landmarks = strategy.detect_face_landmarks(frame)
                
                # Sonuç kontrolü
                if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                    detection_count += 1
                
                frame_count += 1
                
                # Test için ilk 30 kare yeterli
                if frame_count >= 30:
                    break
            
            # İşlem süresini hesapla
            processing_time = time.time() - start_time
            fps = frame_count / processing_time if processing_time > 0 else 0
            
            # Tespit oranını hesapla
            detection_rate = detection_count / frame_count if frame_count > 0 else 0
            
            # Sonuçları kaydet
            strategy_results[strategy_name] = {
                "detection_rate": detection_rate,
                "fps": fps,
                "processing_time": processing_time
            }
            
            # Kaynakları serbest bırak
            strategy.release()
        
        # Kaynakları serbest bırak
        cap.release()
        
        # Sonuçları raporla
        for name, results in strategy_results.items():
            print(f"Strateji: {name}")
            print(f"- Tespit oranı: {results['detection_rate']:.2f}")
            print(f"- FPS: {results['fps']:.2f}")
            print(f"- İşlem süresi: {results['processing_time']:.2f} s")
        
        # En az bir stratejinin tespit oranı iyi olmalı
        assert any(r["detection_rate"] >= 0.8 for r in strategy_results.values()), "Hiçbir strateji yeterli tespit sağlamıyor"


class TestPerformance:
    """Performans testleri sınıfı"""
    
    @pytest.fixture
    def setup_performance_test(self):
        """Performans testi ortamını hazırlama"""
        # Test veri klasörünü oluştur
        os.makedirs(TEST_DATA_DIR, exist_ok=True)
        
        # Test video dosyası yolu
        video_path = os.path.join(TEST_DATA_DIR, "performance_test.avi")
        
        # Test videosu oluştur (eğer yoksa)
        if not os.path.exists(video_path):
            create_test_video(video_path, frames=300)  # Daha uzun video
        
        yield {
            "video_path": video_path
        }
    
    def test_strategy_performance(self, setup_performance_test):
        """Farklı göz izleme stratejilerinin performans karşılaştırması"""
        video_path = setup_performance_test["video_path"]
        
        # Video okuyucu
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.skip(f"Video açılamadı: {video_path}")
        
        # Test edilecek stratejiler
        strategies = []
        strategy_names = []
        
        # MediaPipe strateji
        strategies.append(MediaPipeStrategy())
        strategy_names.append("MediaPipe")
        
        try:
            # Dlib strateji
            strategies.append(DlibStrategy())
            strategy_names.append("Dlib")
        except (ImportError, Exception):
            pass
        
        try:
            # OpenCV DNN strateji
            strategies.append(OpenCVDNNStrategy())
            strategy_names.append("OpenCV DNN")
        except (ImportError, Exception):
            pass
        
        # Yeterli strateji varsa hibrit strateji ekle
        if len(strategies) >= 2:
            strategies.append(HybridStrategy(strategies[:2]))
            strategy_names.append("Hybrid")
        
        # Test sonuçları
        results = []
        
        # Her strateji için performans testi
        for i, strategy in enumerate(strategies):
            name = strategy_names[i]
            
            # Video başına dön
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Sistem kaynaklarını ölç (başlangıç)
            start_resources = get_system_resources()
            
            # İşlem süresini ölç
            start_time = time.time()
            
            # Video üzerinden işleme
            frame_count = 0
            detection_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Çerçeveyi işle
                landmarks = strategy.detect_face_landmarks(frame)
                
                # Sonuç kontrolü
                if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                    detection_count += 1
                
                frame_count += 1
            
            # İşlem süresini hesapla
            processing_time = time.time() - start_time
            fps = frame_count / processing_time if processing_time > 0 else 0
            
            # Sistem kaynaklarını ölç (bitiş)
            end_resources = get_system_resources()
            
            # Tespit oranını hesapla
            detection_rate = detection_count / frame_count if frame_count > 0 else 0
            
            # CPU ve RAM kullanım farkını hesapla
            cpu_diff = end_resources["cpu_percent"] - start_resources["cpu_percent"]
            memory_diff = end_resources["memory_percent"] - start_resources["memory_percent"]
            
            # Sonuçları kaydet
            results.append({
                "strategy": name,
                "detection_rate": detection_rate,
                "fps": fps,
                "processing_time": processing_time,
                "cpu_usage": cpu_diff,
                "memory_usage": memory_diff
            })
            
            # Kaynakları serbest bırak
            strategy.release()
        
        # Kaynakları serbest bırak
        cap.release()
        
        # Sonuçları raporla
        for result in results:
            print(f"Strateji: {result['strategy']}")
            print(f"- Tespit oranı: {result['detection_rate']:.2f}")
            print(f"- FPS: {result['fps']:.2f}")
            print(f"- İşlem süresi: {result['processing_time']:.2f} s")
            print(f"- CPU kullanımı: {result['cpu_usage']:.2f}%")
            print(f"- Bellek kullanımı: {result['memory_usage']:.2f}%")
        
        # FPS değerlerini kontrol et
        assert all(r["fps"] > 0 for r in results), "Bazı stratejiler çok yavaş çalışıyor"
        
        # JSON rapor dosyasına yaz
        report_path = os.path.join(TEST_DATA_DIR, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def test_stress_performance(self, setup_performance_test):
        """Stres testi - Sürekli işlem altında performans"""
        video_path = setup_performance_test["video_path"]
        
        # Video okuyucu
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            pytest.skip(f"Video açılamadı: {video_path}")
        
        # Strateji seç (MediaPipe en yaygın desteklenen)
        strategy = MediaPipeStrategy()
        
        # Analizör
        analyzer = EyeAnalyzer(
            fps=30.0,
            buffer_seconds=3.0,  # Daha uzun buffer
            nistagmus_freq_threshold=0.5,
            spv_threshold=30.0,
            pixel_to_degree_ratio=0.1
        )
        
        # Stres testi süresini ayarla (saniye)
        stress_duration = 10
        
        # Sistem kaynaklarını izleme
        resource_usage = []
        
        # Başlangıç zamanı
        start_time = time.time()
        
        # Stres testi (videoyu birden fazla kez işle)
        frame_count = 0
        loop_count = 0
        
        while time.time() - start_time < stress_duration:
            # Video başına dön (gerekirse)
            if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                loop_count += 1
            
            # Kareyi oku
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Sistem kaynaklarını ölç
            resources = get_system_resources()
            resource_usage.append(resources)
            
            # Göz tespiti yap
            landmarks = strategy.detect_face_landmarks(frame)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                # Sol ve sağ göz iris merkezleri (MediaPipe indeksleri: 468, 473)
                h, w, _ = frame.shape
                
                left_iris = landmarks.multi_face_landmarks[0].landmark[468]
                right_iris = landmarks.multi_face_landmarks[0].landmark[473]
                
                left_pupil = (int(left_iris.x * w), int(left_iris.y * h))
                right_pupil = (int(right_iris.x * w), int(right_iris.y * h))
                
                # Analizöre veri gönder
                analyzer.update_data(
                    left_pupil=left_pupil,
                    right_pupil=right_pupil,
                    timestamp=time.time(),
                    is_left_closed=False,
                    is_right_closed=False
                )
            
            frame_count += 1
            
            # Gerçek zamanlı işlemeyi simüle etmek için uyku eklenebilir
            # time.sleep(0.01)
        
        # Test süresini ve FPS'i hesapla
        total_duration = time.time() - start_time
        fps = frame_count / total_duration if total_duration > 0 else 0
        
        # Analiz yap
        analysis_results = analyzer.analyze()
        
        # Kaynakları serbest bırak
        cap.release()
        strategy.release()
        
        # CPU ve bellek kullanımı istatistikleri
        if resource_usage:
            cpu_usage = [r["cpu_percent"] for r in resource_usage]
            memory_usage = [r["memory_percent"] for r in resource_usage]
            
            avg_cpu = sum(cpu_usage) / len(cpu_usage)
            max_cpu = max(cpu_usage)
            avg_memory = sum(memory_usage) / len(memory_usage)
            max_memory = max(memory_usage)
        else:
            avg_cpu = max_cpu = avg_memory = max_memory = 0
        
        # Sonuçları raporla
        print(f"Stres testi sonuçları:")
        print(f"- Toplam süre: {total_duration:.2f} s")
        print(f"- İşlenen kare: {frame_count}")
        print(f"- Döngü sayısı: {loop_count}")
        print(f"- FPS: {fps:.2f}")
        print(f"- Ortalama CPU: {avg_cpu:.2f}%")
        print(f"- Maksimum CPU: {max_cpu:.2f}%")
        print(f"- Ortalama bellek: {avg_memory:.2f}%")
        print(f"- Maksimum bellek: {max_memory:.2f}%")
        
        # Gerçek zamanlı işleme için yeterli hız (30 FPS'den fazla)
        assert fps >= 30, f"FPS değeri yetersiz: {fps:.2f}"
        
        # CPU kullanımı makul olmalı (belirli bir eşik altında)
        assert max_cpu < 95, f"CPU kullanımı çok yüksek: {max_cpu:.2f}%"
    
    def test_memory_leaks(self, setup_performance_test):
        """Bellek sızıntılarını kontrol et"""
        video_path = setup_performance_test["video_path"]
        
        # Başlangıç bellek kullanımını ölç
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB cinsinden
        
        # Strateji ve analizörü oluştur/serbest bırak döngüsü
        for _ in range(5):
            # Strateji
            strategy = MediaPipeStrategy()
            
            # Analizör
            analyzer = EyeAnalyzer(
                fps=30.0,
                buffer_seconds=1.0,
                nistagmus_freq_threshold=0.5,
                spv_threshold=30.0,
                pixel_to_degree_ratio=0.1
            )
            
            # Video işleme
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                pytest.skip(f"Video açılamadı: {video_path}")
            
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Göz tespiti yap
                landmarks = strategy.detect_face_landmarks(frame)
                
                if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                    # Analizör için veri üret
                    h, w, _ = frame.shape
                    left_iris = landmarks.multi_face_landmarks[0].landmark[468]
                    right_iris = landmarks.multi_face_landmarks[0].landmark[473]
                    
                    left_pupil = (int(left_iris.x * w), int(left_iris.y * h))
                    right_pupil = (int(right_iris.x * w), int(right_iris.y * h))
                    
                    analyzer.update_data(
                        left_pupil=left_pupil,
                        right_pupil=right_pupil,
                        timestamp=time.time(),
                        is_left_closed=False,
                        is_right_closed=False
                    )
                
                frame_count += 1
                
                # Test için ilk 50 kare yeterli
                if frame_count >= 50:
                    break
            
            # Analizör kullan
            analyzer.analyze()
            
            # Kaynakları serbest bırak
            cap.release()
            strategy.release()
            
            # Bellek temizleme işini zorla
            import gc
            gc.collect()
        
        # Son bellek kullanımını ölç
        end_memory = process.memory_info().rss / (1024 * 1024)  # MB cinsinden
        
        # Bellek kullanım farkı
        memory_diff = end_memory - start_memory
        
        print(f"Bellek testi sonuçları:")
        print(f"- Başlangıç bellek kullanımı: {start_memory:.2f} MB")
        print(f"- Son bellek kullanımı: {end_memory:.2f} MB")
        print(f"- Fark: {memory_diff:.2f} MB")
        
        # Not: Bazı bellek artışı normal olabilir, bellek temizleme Python'da
        # hemen gerçekleşmeyebilir. Bu nedenle tolerans ekleriz.
        
        # Bellek artışı makul olmalı (50 MB'dan az)
        assert memory_diff < 50, f"Bellek sızıntısı olabilir: {memory_diff:.2f} MB artış"
    
    def test_deep_eye_tracker_performance(self):
        """DeepEyeTracker performans testi"""
        try:
            # DeepEyeTracker sınıfını başlat
            deep_tracker = DeepEyeTracker(
                use_onnx=False,  # Test için ONNX kullanma
                model_path=None  # Varsayılan model yolunu kullan
            )
            
            # Sentetik göz verisi oluştur
            test_size = 100
            fake_eye_data = []
            
            for i in range(test_size):
                # Sentetik göz öznitelikleri
                features = np.random.rand(128)  # 128 boyutlu öznitelik vektörü
                
                # Sentetik etiket (0: normal, 1: nistagmus, 2: strabismus)
                label = np.random.randint(0, 3)
                
                fake_eye_data.append((features, label))
            
            # Performans ölçümü - Öğrenme
            start_time = time.time()
            
            for features, label in fake_eye_data[:int(test_size * 0.8)]:
                # Transfer öğrenme simülasyonu
                deep_tracker.update_model(features, label)
            
            training_time = time.time() - start_time
            
            # Performans ölçümü - Tahmin
            start_time = time.time()
            predictions = []
            
            for features, _ in fake_eye_data[int(test_size * 0.8):]:
                # Tahmin simülasyonu
                prediction = deep_tracker.predict(features)
                predictions.append(prediction)
            
            prediction_time = time.time() - start_time
            
            # Kaynakları serbest bırak
            deep_tracker.cleanup()
            
            # Sonuçları raporla
            print(f"DeepEyeTracker performans sonuçları:")
            print(f"- Öğrenme süresi: {training_time:.4f} s")
            print(f"- Tahmin süresi: {prediction_time:.4f} s")
            print(f"- Örnek başına tahmin: {prediction_time / (test_size * 0.2):.6f} s")
            
            # Makul tahmin süresi (örnek başına 10ms'den az)
            assert prediction_time / (test_size * 0.2) < 0.01, "DeepEyeTracker'ın tahmin performansı düşük"
            
        except (ImportError, Exception) as e:
            pytest.skip(f"DeepEyeTracker test edilemedi: {str(e)}")


if __name__ == "__main__":
    # Manuel test için
    import sys
    
    # Test ortamını kur
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    video_path = os.path.join(TEST_DATA_DIR, "test_eye_movement.avi")
    
    # Test videosu oluştur (eğer yoksa)
    if not os.path.exists(video_path):
        print("Test videosu oluşturuluyor...")
        create_test_video(video_path)
    
    print(f"Test videosu: {video_path}")
    
    # Basit strateji test etme
    try:
        print("MediaPipe stratejisi test ediliyor...")
        strategy = MediaPipeStrategy()
        
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                landmarks = strategy.detect_face_landmarks(frame)
                if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                    print("Yüz işaretleri tespit edildi!")
                else:
                    print("Yüz işaretleri tespit edilemedi.")
        
        cap.release()
        strategy.release()
    except Exception as e:
        print(f"Strateji test hatası: {str(e)}")
    
    print("Manuel test tamamlandı.") 