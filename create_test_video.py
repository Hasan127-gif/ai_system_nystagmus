#!/usr/bin/env python3
"""
Test video oluşturucu
===================
"""

import cv2
import numpy as np

def create_test_video():
    """Test videosu oluşturur."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_clinical_video.mp4', fourcc, 30.0, (640, 480))

    print('Test videosu oluşturuluyor...')
    for i in range(150):  # 5 saniye video
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Basit yüz simülasyonu
        cv2.circle(frame, (320, 240), 80, (200, 150, 100), -1)  # Yüz
        
        # Göz hareketi simülasyonu (nistagmus)
        left_eye_x = 300 + int(10 * np.sin(i * 0.2))  # Nistagmus simülasyonu
        right_eye_x = 340 + int(8 * np.sin(i * 0.2))
        
        cv2.circle(frame, (left_eye_x, 220), 8, (255, 255, 255), -1)  # Sol göz
        cv2.circle(frame, (right_eye_x, 220), 8, (255, 255, 255), -1)  # Sağ göz
        
        # Iris detayları
        cv2.circle(frame, (left_eye_x, 220), 3, (0, 0, 0), -1)  # Sol iris
        cv2.circle(frame, (right_eye_x, 220), 3, (0, 0, 0), -1)  # Sağ iris
        
        out.write(frame)

    out.release()
    print('Test videosu oluşturuldu: test_clinical_video.mp4')

if __name__ == "__main__":
    create_test_video() 