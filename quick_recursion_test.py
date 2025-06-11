#!/usr/bin/env python3
"""Hızlı recursion testi"""

from detector import NistagmusDetector
import numpy as np

print('🔍 Recursion test başlıyor...')
detector = NistagmusDetector()
print(f'✅ Detector başlatıldı: {detector.is_initialized}')

# Test karesi oluştur
test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120

try:
    left, right = detector.detect_iris_centers(test_frame)
    print(f'✅ Iris tespit çalıştı (recursion yok!): left={left}, right={right}')
except RecursionError:
    print('❌ RECURSION HATASI DEVAM EDİYOR!')
except Exception as e:
    print(f'⚠️ Başka hata: {e}') 