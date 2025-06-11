#!/usr/bin/env python3
"""
GELİŞTİRİLMİŞ TEST VİDEO ÜRETİCİSİ
===================================
MediaPipe'ın güvenilir şekilde tespit edebileceği gerçekçi test videoları oluşturur.
"""

import cv2
import numpy as np
import os
import time
import logging

logger = logging.getLogger(__name__)

class MedicalGradeVideoGenerator:
    """Tıbbi kalitede test videoları oluşturan sınıf."""
    
    def __init__(self):
        """Video üreticisini başlatır."""
        self.width = 640
        self.height = 480
        self.fps = 30
        
    def create_realistic_nystagmus_video(self, output_path: str, duration: float = 5.0, 
                                       nystagmus_freq: float = 2.0, 
                                       strabismus_angle: float = 3.0) -> bool:
        """
        Gerçekçi nistagmus ve şaşılık hareketi olan test videosu oluşturur.
        
        Args:
            output_path: Çıktı video dosyası yolu
            duration: Video süresi (saniye)
            nystagmus_freq: Nistagmus frekansı (Hz)
            strabismus_angle: Şaşılık açısı (derece)
            
        Returns:
            bool: Başarı durumu
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            
            total_frames = int(duration * self.fps)
            
            logger.info(f"Tıbbi kalitede test videosu oluşturuluyor: {output_path}")
            
            for frame_num in range(total_frames):
                frame = self._create_realistic_frame(frame_num, nystagmus_freq, strabismus_angle)
                out.write(frame)
            
            out.release()
            
            # Video doğrulama
            if self._validate_created_video(output_path):
                logger.info(f"Test videosu başarıyla oluşturuldu: {output_path}")
                return True
            else:
                logger.error(f"Oluşturulan video doğrulanamadı: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Video oluşturma hatası: {str(e)}")
            return False
    
    def _create_realistic_frame(self, frame_num: int, nystagmus_freq: float, strabismus_angle: float) -> np.ndarray:
        """
        Gerçekçi yüz ve göz hareketleri olan tek bir kare oluşturur.
        """
        # Zaman faktörü
        t = frame_num / self.fps
        
        # Arka plan
        frame = np.ones((self.height, self.width, 3), dtype=np.uint8) * 120
        
        # Yüz merkezi ve boyutları
        face_center_x = self.width // 2
        face_center_y = self.height // 2
        face_width = 160
        face_height = 200
        
        # Yüz elipsi (cilt tonu)
        cv2.ellipse(frame, (face_center_x, face_center_y), 
                   (face_width//2, face_height//2), 
                   0, 0, 360, (220, 180, 150), -1)
        
        # Göz pozisyonları hesaplama
        left_base_x = face_center_x - 45
        left_base_y = face_center_y - 30
        right_base_x = face_center_x + 45
        right_base_y = face_center_y - 30
        
        # Nistagmus hareketi simülasyonu
        nystagmus_primary = 15 * np.sin(2 * np.pi * nystagmus_freq * t)
        nystagmus_noise = 4 * np.random.uniform(-1, 1)
        
        left_movement_y = nystagmus_primary + nystagmus_noise
        left_movement_x = 3 * np.sin(2 * np.pi * 0.4 * t)
        
        # Şaşılık simülasyonu
        strabismus_offset_y = strabismus_angle + 1 * np.sin(2 * np.pi * 0.2 * t)
        
        right_movement_y = nystagmus_primary + strabismus_offset_y + nystagmus_noise
        right_movement_x = left_movement_x + 2
        
        # Final göz pozisyonları
        left_eye_pos = (
            int(left_base_x + left_movement_x),
            int(left_base_y + left_movement_y)
        )
        
        right_eye_pos = (
            int(right_base_x + right_movement_x),
            int(right_base_y + right_movement_y)
        )
        
        # Göz çizimi
        self._draw_realistic_eye(frame, left_eye_pos, is_left=True)
        self._draw_realistic_eye(frame, right_eye_pos, is_left=False)
        
        # Yüz detayları
        self._add_facial_features(frame, face_center_x, face_center_y)
        
        return frame
    
    def _draw_realistic_eye(self, frame: np.ndarray, eye_pos: tuple, is_left: bool = True):
        """Gerçekçi göz çizer."""
        # Göz çukuru (beyaz)
        cv2.ellipse(frame, eye_pos, (30, 20), 0, 0, 360, (250, 250, 250), -1)
        
        # İris (kahverengi)
        iris_color = (139, 69, 19) if is_left else (120, 85, 45)
        cv2.circle(frame, eye_pos, 15, iris_color, -1)
        
        # Pupil
        cv2.circle(frame, eye_pos, 7, (20, 20, 20), -1)
        
        # Pupil parlaması
        highlight_pos = (eye_pos[0] - 2, eye_pos[1] - 2)
        cv2.circle(frame, highlight_pos, 2, (255, 255, 255), -1)
    
    def _add_facial_features(self, frame: np.ndarray, center_x: int, center_y: int):
        """Yüz detayları ekler."""
        # Burun
        nose_tip = (center_x, center_y + 25)
        cv2.ellipse(frame, nose_tip, (10, 18), 0, 0, 360, (200, 160, 130), -1)
        
        # Ağız
        mouth_center = (center_x, center_y + 65)
        cv2.ellipse(frame, mouth_center, (25, 8), 0, 0, 360, (150, 80, 80), -1)
        
        # Kaşlar
        left_eyebrow = (center_x - 45, center_y - 50)
        right_eyebrow = (center_x + 45, center_y - 50)
        
        cv2.ellipse(frame, left_eyebrow, (25, 6), -10, 0, 360, (80, 60, 40), -1)
        cv2.ellipse(frame, right_eyebrow, (25, 6), 10, 0, 360, (80, 60, 40), -1)
    
    def _validate_created_video(self, video_path: str) -> bool:
        """Oluşturulan videoyu doğrular."""
        try:
            if not os.path.exists(video_path):
                return False
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            cap.release()
            
            return frame_count > 10 and fps > 0
            
        except Exception:
            return False

def main():
    """Test amaçlı ana fonksiyon."""
    logging.basicConfig(level=logging.INFO)
    
    generator = MedicalGradeVideoGenerator()
    
    # Tekli test videosu
    print("Test videosu oluşturuluyor...")
    success = generator.create_realistic_nystagmus_video(
        "sample_medical_test.mp4", 
        duration=3.0, 
        nystagmus_freq=2.5, 
        strabismus_angle=3.0
    )
    
    if success:
        print("✅ Test videosu başarıyla oluşturuldu!")
    else:
        print("❌ Test videosu oluşturulamadı!")

if __name__ == "__main__":
    main() 