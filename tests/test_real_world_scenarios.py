"""
Göz İzleme Sistemi - Gerçek Dünya Senaryoları Testleri

Bu modül, çeşitli gerçek dünya koşullarında (farklı aydınlatma, gözlük,
farklı göz tipleri, kırpma paternleri vb.) göz izleme stratejilerinin 
performansını test eder.
"""

import pytest
import numpy as np
import cv2
import os
import logging
from unittest.mock import patch, MagicMock
import json
import tempfile

# Modülleri import et
from EyeTrackingStrategy import MediaPipeStrategy, DlibStrategy
from EyeTrackingStrategy import OpenCVDNNStrategy, HybridStrategy
from EyeAnalyzer import EyeAnalyzer

# Test görüntüleri dizini
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "test_images")

# Test verilerini yükle
def load_test_image(filename):
    """Test görüntüsünü yükle"""
    filepath = os.path.join(TEST_IMAGES_DIR, filename)
    if not os.path.exists(filepath):
        pytest.skip(f"Test görüntüsü bulunamadı: {filepath}")
    
    img = cv2.imread(filepath)
    if img is None:
        pytest.skip(f"Görüntü yüklenemedi: {filepath}")
    
    return img

# Test verileri hazırlama yardımcıları
def create_synthetic_image(width=640, height=480, lighting_level=1.0, 
                         with_glasses=False, eye_color="dark",
                         gaze_direction="center"):
    """
    Sentetik test görüntüsü oluştur.
    
    Args:
        width: Görüntü genişliği
        height: Görüntü yüksekliği
        lighting_level: Aydınlatma seviyesi (0.1-2.0)
        with_glasses: Gözlük içersin mi
        eye_color: Göz rengi ("dark", "light", "blue", "green")
        gaze_direction: Bakış yönü ("center", "left", "right", "up", "down")
    
    Returns:
        Sentetik test görüntüsü
    """
    # Temel bir test görüntüsü oluştur
    img = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Yüz bölgesi çiz (basit oval)
    face_center = (width // 2, height // 2)
    face_size = (width // 3, height // 2)
    cv2.ellipse(img, face_center, face_size, 0, 0, 360, (200, 200, 200), -1)
    
    # Göz pozisyonlarını belirle
    left_eye_center = (width // 2 - width // 8, height // 2 - height // 10)
    right_eye_center = (width // 2 + width // 8, height // 2 - height // 10)
    eye_radius = width // 25
    
    # Bakış yönüne göre göz bebeği pozisyonunu ayarla
    pupil_offset_x = 0
    pupil_offset_y = 0
    
    if gaze_direction == "left":
        pupil_offset_x = -eye_radius // 2
    elif gaze_direction == "right":
        pupil_offset_x = eye_radius // 2
    elif gaze_direction == "up":
        pupil_offset_y = -eye_radius // 3
    elif gaze_direction == "down":
        pupil_offset_y = eye_radius // 3
    
    # Göz rengine göre iris ve pupil rengi belirle
    if eye_color == "dark":
        iris_color = (50, 50, 50)
        pupil_color = (20, 20, 20)
    elif eye_color == "light":
        iris_color = (150, 150, 170)
        pupil_color = (30, 30, 30)
    elif eye_color == "blue":
        iris_color = (150, 100, 50)
        pupil_color = (30, 30, 30)
    elif eye_color == "green":
        iris_color = (100, 150, 100)
        pupil_color = (30, 30, 30)
    
    # Gözleri çiz
    for eye_center in [left_eye_center, right_eye_center]:
        # Göz akı
        cv2.circle(img, eye_center, eye_radius, (250, 250, 250), -1)
        
        # İris
        cv2.circle(img, (eye_center[0] + pupil_offset_x, eye_center[1] + pupil_offset_y), 
                  eye_radius // 2, iris_color, -1)
        
        # Pupil
        cv2.circle(img, (eye_center[0] + pupil_offset_x, eye_center[1] + pupil_offset_y), 
                  eye_radius // 4, pupil_color, -1)
    
    # Gözlük ekleme
    if with_glasses:
        # Sol gözlük camı
        cv2.ellipse(img, left_eye_center, (eye_radius*1.5, eye_radius*1.3), 0, 0, 360, (100, 100, 100), 2)
        # Sağ gözlük camı
        cv2.ellipse(img, right_eye_center, (eye_radius*1.5, eye_radius*1.3), 0, 0, 360, (100, 100, 100), 2)
        # Gözlük çerçevesi ortası
        cv2.line(img, 
                (int(left_eye_center[0] + eye_radius*1.2), left_eye_center[1]),
                (int(right_eye_center[0] - eye_radius*1.2), right_eye_center[1]),
                (100, 100, 100), 2)
        
        # Gözlük yansıması (zorluk faktörü)
        reflection_pos = (left_eye_center[0] - eye_radius // 3, left_eye_center[1] - eye_radius // 3)
        cv2.circle(img, reflection_pos, eye_radius // 6, (250, 250, 250), -1)
        reflection_pos = (right_eye_center[0] - eye_radius // 3, right_eye_center[1] - eye_radius // 3)
        cv2.circle(img, reflection_pos, eye_radius // 6, (250, 250, 250), -1)
    
    # Aydınlatma ayarı
    img = cv2.convertScaleAbs(img, alpha=lighting_level, beta=0)
    
    return img

def simulate_eye_movement(frames=30, pattern_type="saccade", fps=30):
    """
    Göz hareketi simülasyonu yaratır.
    
    Args:
        frames: Kare sayısı
        pattern_type: Hareket türü ("saccade", "smooth_pursuit", "fixation", "nystagmus")
        fps: FPS değeri
    
    Returns:
        Sağ ve sol göz pozisyonları dizisi
    """
    width, height = 640, 480
    center_x, center_y = width // 2, height // 2
    
    left_positions = []
    right_positions = []
    
    if pattern_type == "fixation":
        # Sabit bakış (küçük titreşimler ile)
        for i in range(frames):
            jitter_x = np.random.normal(0, 2)
            jitter_y = np.random.normal(0, 2)
            left_x = center_x - 80 + jitter_x
            left_y = center_y - 20 + jitter_y
            right_x = center_x + 80 + jitter_x
            right_y = center_y - 20 + jitter_y
            
            left_positions.append((int(left_x), int(left_y)))
            right_positions.append((int(right_x), int(right_y)))
    
    elif pattern_type == "saccade":
        # Sakkadik göz hareketi (hızlı konum değişimleri)
        positions = [
            (center_x - 100, center_y - 20),  # Sol
            (center_x, center_y - 20),        # Merkez
            (center_x + 100, center_y - 20),  # Sağ
            (center_x, center_y - 80),        # Yukarı
            (center_x, center_y + 40)         # Aşağı
        ]
        
        current_pos = 0
        frames_per_pos = frames // len(positions)
        
        for i in range(frames):
            pos_idx = min(i // frames_per_pos, len(positions) - 1)
            target_x, target_y = positions[pos_idx]
            
            # Sakkad hareketi - hedef noktaya aniden sıçrama
            left_x = target_x - 80
            left_y = target_y
            right_x = target_x + 80
            right_y = target_y
            
            left_positions.append((int(left_x), int(left_y)))
            right_positions.append((int(right_x), int(right_y)))
    
    elif pattern_type == "smooth_pursuit":
        # Düzgün takip hareketi (yavaş, sürekli hareket)
        for i in range(frames):
            # Dairesel hareket
            angle = 2 * np.pi * i / frames
            radius = 50
            offset_x = radius * np.cos(angle)
            offset_y = radius * np.sin(angle)
            
            left_x = center_x - 80 + offset_x
            left_y = center_y - 20 + offset_y
            right_x = center_x + 80 + offset_x
            right_y = center_y - 20 + offset_y
            
            left_positions.append((int(left_x), int(left_y)))
            right_positions.append((int(right_x), int(right_y)))
    
    elif pattern_type == "nystagmus":
        # Nistagmus hareketi (testere dişi deseni - hızlı faz, yavaş faz)
        cycle_frames = 10  # Her nistagmus döngüsü için kare sayısı
        cycles = frames // cycle_frames
        
        for cycle in range(cycles):
            for i in range(cycle_frames):
                # Yavaş faz (linear hareket)
                if i < cycle_frames * 0.7:
                    progress = i / (cycle_frames * 0.7)
                    offset_x = -30 * progress  # Yavaşça sola kayma
                # Hızlı faz (hızlı reset)
                else:
                    progress = (i - cycle_frames * 0.7) / (cycle_frames * 0.3)
                    offset_x = -30 + 30 * progress  # Hızla sağa dönüş
                
                left_x = center_x - 80 + offset_x
                left_y = center_y - 20
                right_x = center_x + 80 + offset_x
                right_y = center_y - 20
                
                left_positions.append((int(left_x), int(left_y)))
                right_positions.append((int(right_x), int(right_y)))
        
        # Kalan kareleri doldur
        remaining = frames - len(left_positions)
        for i in range(remaining):
            left_positions.append(left_positions[-1])
            right_positions.append(right_positions[-1])
    
    return left_positions, right_positions


class TestRealWorldScenarios:
    """Gerçek dünya senaryoları için göz izleme test sınıfı."""
    
    @pytest.fixture
    def strategies(self):
        """Test için tüm göz izleme stratejilerini oluştur"""
        mediapipe_strategy = MediaPipeStrategy()
        
        try:
            dlib_strategy = DlibStrategy()
        except (ImportError, Exception):
            dlib_strategy = None
            
        try:
            opencv_strategy = OpenCVDNNStrategy()
        except (ImportError, Exception):
            opencv_strategy = None
        
        strategies = {
            "mediapipe": mediapipe_strategy
        }
        
        if dlib_strategy:
            strategies["dlib"] = dlib_strategy
            
        if opencv_strategy:
            strategies["opencv"] = opencv_strategy
            
        if dlib_strategy and mediapipe_strategy:
            strategies["hybrid"] = HybridStrategy([mediapipe_strategy, dlib_strategy])
        
        yield strategies
        
        # Temizleme
        for strategy in strategies.values():
            strategy.release()
    
    @pytest.fixture
    def analyzer(self):
        """Test için göz analizörü örneği"""
        return EyeAnalyzer(
            fps=30.0,
            buffer_seconds=1.0,
            nistagmus_freq_threshold=0.5,
            spv_threshold=30.0,
            pixel_to_degree_ratio=0.1
        )
        
    # Test Senaryoları
    
    @pytest.mark.parametrize("lighting_level", [0.3, 0.7, 1.0, 1.5])
    def test_lighting_conditions(self, strategies, lighting_level):
        """Farklı aydınlatma koşullarında göz izleme performansı testi"""
        # Farklı aydınlatma koşullarında test görüntüleri oluştur
        test_image = create_synthetic_image(lighting_level=lighting_level)
        
        results = {}
        
        # Her strateji ile test et
        for name, strategy in strategies.items():
            # Yüz işaretlerini tespit et
            landmarks = strategy.detect_face_landmarks(test_image)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                results[name] = True
            else:
                results[name] = False
        
        # En azından bir stratejinin çalışması gerekir
        assert any(results.values()), f"Tüm stratejiler başarısız - Aydınlatma seviyesi: {lighting_level}"
        
        # Sonuçları raporla
        for name, success in results.items():
            print(f"Strateji: {name}, Başarı: {success}, Aydınlatma seviyesi: {lighting_level}")
    
    @pytest.mark.parametrize("with_glasses", [True, False])
    def test_glasses_handling(self, strategies, with_glasses):
        """Gözlük takma/takmama durumuna göre izleme performansı testi"""
        # Gözlüklü veya gözlüksüz test görüntüsü oluştur
        test_image = create_synthetic_image(with_glasses=with_glasses)
        
        results = {}
        
        # Her strateji ile test et
        for name, strategy in strategies.items():
            # Yüz işaretlerini tespit et
            landmarks = strategy.detect_face_landmarks(test_image)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                results[name] = True
            else:
                results[name] = False
        
        # En azından bir stratejinin çalışması gerekir
        assert any(results.values()), f"Tüm stratejiler başarısız - Gözlük: {with_glasses}"
        
        # Sonuçları raporla
        for name, success in results.items():
            print(f"Strateji: {name}, Başarı: {success}, Gözlük: {with_glasses}")
    
    @pytest.mark.parametrize("eye_color", ["dark", "light", "blue", "green"])
    def test_eye_color_handling(self, strategies, eye_color):
        """Farklı göz renklerine göre izleme performansı testi"""
        # Farklı göz renklerine sahip test görüntüsü oluştur
        test_image = create_synthetic_image(eye_color=eye_color)
        
        results = {}
        
        # Her strateji ile test et
        for name, strategy in strategies.items():
            # Yüz işaretlerini tespit et
            landmarks = strategy.detect_face_landmarks(test_image)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                results[name] = True
            else:
                results[name] = False
        
        # En azından bir stratejinin çalışması gerekir
        assert any(results.values()), f"Tüm stratejiler başarısız - Göz rengi: {eye_color}"
        
        # Sonuçları raporla
        for name, success in results.items():
            print(f"Strateji: {name}, Başarı: {success}, Göz rengi: {eye_color}")
    
    @pytest.mark.parametrize("gaze_direction", ["center", "left", "right", "up", "down"])
    def test_gaze_direction(self, strategies, gaze_direction):
        """Farklı bakış yönlerine göre izleme performansı testi"""
        # Farklı bakış yönlerine sahip test görüntüsü oluştur
        test_image = create_synthetic_image(gaze_direction=gaze_direction)
        
        results = {}
        directions = {}
        
        # Her strateji ile test et
        for name, strategy in strategies.items():
            # Yüz işaretlerini tespit et
            landmarks = strategy.detect_face_landmarks(test_image)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                results[name] = True
                
                # Bakış yönünü değerlendir (sol göz için - 468 indeksi)
                if len(landmarks.multi_face_landmarks[0].landmark) > 468:
                    iris = landmarks.multi_face_landmarks[0].landmark[468]
                    # X koordinatı ile bakış yönünü kabaca değerlendir
                    if iris.x < 0.45:
                        detected_dir = "left"
                    elif iris.x > 0.55:
                        detected_dir = "right"
                    else:
                        if iris.y < 0.45:
                            detected_dir = "up"
                        elif iris.y > 0.55:
                            detected_dir = "down"
                        else:
                            detected_dir = "center"
                    
                    directions[name] = detected_dir
            else:
                results[name] = False
        
        # En azından bir stratejinin çalışması gerekir
        assert any(results.values()), f"Tüm stratejiler başarısız - Bakış yönü: {gaze_direction}"
        
        # Sonuçları raporla
        for name, success in results.items():
            detected = directions.get(name, "algılanamadı")
            print(f"Strateji: {name}, Başarı: {success}, Hedef Yön: {gaze_direction}, Algılanan: {detected}")
    
    @pytest.mark.parametrize("pattern_type", ["fixation", "saccade", "smooth_pursuit", "nystagmus"])
    def test_eye_movement_patterns(self, strategies, analyzer, pattern_type):
        """Farklı göz hareketi paternlerinin analizi testi"""
        # Göz hareketi simülasyonu oluştur
        frames = 30
        left_positions, right_positions = simulate_eye_movement(frames=frames, pattern_type=pattern_type)
        
        # Her kare için analiz yap
        now = 0
        time_step = 1 / 30.0  # 30 FPS
        
        for i in range(frames):
            # Analiz için veri gönder
            analyzer.update_data(
                left_pupil=left_positions[i],
                right_pupil=right_positions[i],
                timestamp=now,
                is_left_closed=False,
                is_right_closed=False
            )
            now += time_step
        
        # Analiz yap
        results = analyzer.analyze()
        
        # Sonuçları değerlendir
        if pattern_type == "nystagmus":
            # Nistagmus algılanmalı
            print(f"Nistagmus Frekansı: {results['nistagmus_freq']}, Algılandı: {results['is_nistagmus_detected']}")
            assert results["is_nistagmus_detected"], "Nistagmus algılanamadı"
        elif pattern_type == "saccade":
            # Hızlı SPV değeri olmalı
            print(f"SPV: {results['spv']}")
            assert results["spv"] > 10.0, "Sakkadik hareket için SPV değeri çok düşük"
        elif pattern_type == "fixation":
            # Düşük SPV değeri olmalı
            print(f"SPV: {results['spv']}")
            assert results["spv"] < 15.0, "Fiksasyon için SPV değeri çok yüksek"
            
    @pytest.mark.parametrize("noise_level", [0.0, 0.1, 0.2, 0.3])
    def test_noise_robustness(self, strategies, noise_level):
        """Farklı gürültü seviyelerinde performans testi"""
        # Temel test görüntüsü oluştur
        test_image = create_synthetic_image()
        
        # Gürültü ekle
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 255, test_image.shape).astype(np.uint8)
            noisy_image = cv2.add(test_image, noise)
        else:
            noisy_image = test_image
        
        results = {}
        
        # Her strateji ile test et
        for name, strategy in strategies.items():
            # Yüz işaretlerini tespit et
            landmarks = strategy.detect_face_landmarks(noisy_image)
            
            # Sonuç kontrolü
            if landmarks and hasattr(landmarks, 'multi_face_landmarks') and landmarks.multi_face_landmarks:
                results[name] = True
            else:
                results[name] = False
        
        # Sonuçları raporla
        success_rate = sum(results.values()) / len(results) if results else 0
        print(f"Gürültü seviyesi: {noise_level}, Başarı oranı: {success_rate:.2f}")
        
        # Düşük gürültü seviyelerinde tüm stratejiler çalışmalı
        if noise_level <= 0.1:
            assert success_rate > 0.5, f"Düşük gürültü seviyesinde ({noise_level}) başarı oranı çok düşük"
    
    def test_strategy_combination(self, strategies):
        """Farklı stratejilerin kombinasyonlarının testi"""
        # Sadece MediaPipe ve Dlib stratejileri varsa hibrit test et
        if "mediapipe" in strategies and "dlib" in strategies and "hybrid" in strategies:
            # Zor bir test senaryosu oluştur
            test_image = create_synthetic_image(
                lighting_level=0.7,
                with_glasses=True,
                eye_color="light"
            )
            
            # Gürültü ekle
            noise = np.random.normal(0, 0.1 * 255, test_image.shape).astype(np.uint8)
            test_image = cv2.add(test_image, noise)
            
            # Her strateji ile ayrı ayrı ve hibrit olarak test et
            mediapipe_result = strategies["mediapipe"].detect_face_landmarks(test_image)
            dlib_result = strategies["dlib"].detect_face_landmarks(test_image)
            hybrid_result = strategies["hybrid"].detect_face_landmarks(test_image)
            
            # Sonuçları kontrol et
            mediapipe_success = (mediapipe_result is not None and 
                               hasattr(mediapipe_result, 'multi_face_landmarks') and 
                               mediapipe_result.multi_face_landmarks)
            
            dlib_success = (dlib_result is not None and 
                          hasattr(dlib_result, 'multi_face_landmarks') and 
                          dlib_result.multi_face_landmarks)
            
            hybrid_success = (hybrid_result is not None and 
                            hasattr(hybrid_result, 'multi_face_landmarks') and 
                            hybrid_result.multi_face_landmarks)
            
            # Raporla
            print(f"MediaPipe başarısı: {mediapipe_success}")
            print(f"Dlib başarısı: {dlib_success}")
            print(f"Hibrit başarısı: {hybrid_success}")
            
            # Hibrit başarısı en az diğerlerinden biri kadar iyi olmalı
            assert hybrid_success >= (mediapipe_success or dlib_success), "Hibrit strateji performansı düşük"
        else:
            pytest.skip("MediaPipe ve Dlib stratejileri mevcut değil, hibrit test atlanıyor")


if __name__ == "__main__":
    # Pytest çalıştırıldığında çalışmaz
    # Manuel test etmek için örnek:
    import matplotlib.pyplot as plt
    
    # Sentetik görüntü oluştur ve göster
    img = create_synthetic_image(with_glasses=True, lighting_level=0.8, eye_color="blue", gaze_direction="left")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    # Göz hareketi simülasyonu yap ve göster
    left_pos, right_pos = simulate_eye_movement(frames=100, pattern_type="nystagmus")
    plt.figure(figsize=(10, 6))
    plt.plot([p[0] for p in left_pos], label="Sol X")
    plt.plot([p[1] for p in left_pos], label="Sol Y")
    plt.legend()
    plt.title("Nistagmus Göz Hareketi Simülasyonu")
    plt.show() 