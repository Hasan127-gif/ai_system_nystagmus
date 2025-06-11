#!/usr/bin/env python3
"""
MediaPipe Iris Merkezi Tespit Test Scripti
==========================================
Güncellenmiş detector.py dosyasındaki detect_iris_centers fonksiyonunu test eder.
"""

import cv2
import numpy as np
from detector import NistagmusDetector
import time

def test_iris_detection():
    """Test iris detection with webcam"""
    # Detector'ı başlat
    print("NistagmusDetector başlatılıyor...")
    detector = NistagmusDetector()
    print(f"MediaPipe durumu: {detector.is_initialized}")
    
    # Webcam'i başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam açılamadı!")
        return
    
    print("Webcam başlatıldı. Test için kameraya bakın. 'q' tuşuna basarak çıkın.")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Her 5 karede bir test et (performans için)
        if frame_count % 5 == 0:
            # Yeni detect_iris_centers fonksiyonunu test et
            start_time = time.time()
            left_center, right_center = detector.detect_iris_centers(frame)
            detection_time = time.time() - start_time
            
            # Sonuçları ekranda göster
            if left_center is not None and right_center is not None:
                # İris merkezlerini çiz
                cv2.circle(frame, left_center, 5, (0, 255, 0), -1)  # Yeşil nokta - Sol iris
                cv2.circle(frame, right_center, 5, (0, 255, 0), -1)  # Yeşil nokta - Sağ iris
                
                # Koordinatları yazdır
                cv2.putText(frame, f"Sol: {left_center}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Sag: {right_center}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Tespit suresi: {detection_time*1000:.1f}ms", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Mevcut _extract_eye_landmarks ile karşılaştır
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.face_mesh.process(rgb_frame)
                if results.multi_face_landmarks:
                    old_left, old_right = detector._extract_eye_landmarks(
                        results.multi_face_landmarks[0], frame.shape
                    )
                    cv2.circle(frame, (int(old_left[0]), int(old_left[1])), 3, (255, 0, 0), -1)  # Mavi nokta - Eski yöntem
                    cv2.circle(frame, (int(old_right[0]), int(old_right[1])), 3, (255, 0, 0), -1)  # Mavi nokta - Eski yöntem
                    
                    cv2.putText(frame, "Yesil: Yeni fonksiyon", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, "Mavi: Eski fonksiyon", (10, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            else:
                cv2.putText(frame, "Yuz tespit edilemedi", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Görüntüyü göster
        cv2.imshow('MediaPipe Iris Tespit Testi', frame)
        
        # 'q' tuşuna basıldığında çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()
    print("Test tamamlandı!")

def test_with_sample_image():
    """Test with a static image if webcam is not available"""
    print("Statik görüntü testi...")
    detector = NistagmusDetector()
    
    # Boş bir görüntü oluştur (yüz yok)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:] = (100, 100, 100)  # Gri arka plan
    
    result = detector.detect_iris_centers(test_image)
    print(f"Boş görüntü sonucu: {result}")
    
    # Analiz fonksiyonunu da test et
    try:
        analysis_result = detector.analyze_image(test_image)
        print(f"Analiz sonucu: {analysis_result}")
    except Exception as e:
        print(f"Analiz hatası: {e}")

if __name__ == "__main__":
    print("MediaPipe Iris Tespit Test Scripti")
    print("=" * 40)
    
    choice = input("Test seçin:\n1. Webcam testi\n2. Statik görüntü testi\nSeçim (1/2): ")
    
    if choice == "1":
        test_iris_detection()
    elif choice == "2":
        test_with_sample_image()
    else:
        print("Geçersiz seçim!") 