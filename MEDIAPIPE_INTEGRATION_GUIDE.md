# MediaPipe Göz Takibi Entegrasyonu Rehberi

## Genel Bakış

Bu rehber, mevcut `detector.py` modülündeki göz tespiti fonksiyonunun MediaPipe Face Mesh kullanacak şekilde nasıl iyileştirildiğini ve nasıl kullanılacağını açıklar.

## Yapılan İyileştirmeler

### 1. Yeni `detect_iris_centers` Fonksiyonu

`detector.py` dosyasına yeni bir fonksiyon eklendi:

```python
def detect_iris_centers(self, frame):
    """
    Verilen BGR karede sol ve sağ iris merkezlerinin piksel koordinatlarını döndürür.
    
    Args:
        frame: BGR formatında görüntü karesi
        
    Returns:
        tuple: (left_center, right_center) piksel koordinatları veya (None, None) 
    """
```

### 2. İyileştirilmiş `_extract_eye_landmarks` Fonksiyonu

Mevcut `_extract_eye_landmarks` fonksiyonu daha doğru iris merkezi tespiti için güncellenmiştir:

- MediaPipe'ın doğrudan iris merkezi landmark'larını kullanır (468 ve 473)
- Daha temiz ve optimize edilmiş kod yapısı
- Daha yüksek doğruluk oranı

## Kurulum Gereksinimleri

Aşağıdaki paketlerin kurulu olması gerekir:

```bash
pip install mediapipe
pip install PyWavelets
```

## Kullanım Örnekleri

### Basit Iris Merkezi Tespiti

```python
from detector import NistagmusDetector
import cv2

# Detector'ı başlat
detector = NistagmusDetector()

# Kameradan kare al veya görüntü yükle
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Iris merkezlerini tespit et
left_center, right_center = detector.detect_iris_centers(frame)

if left_center is not None and right_center is not None:
    print(f"Sol iris merkezi: {left_center}")
    print(f"Sağ iris merkezi: {right_center}")
    
    # Iris merkezlerini görüntü üzerinde göster
    cv2.circle(frame, left_center, 5, (0, 255, 0), -1)
    cv2.circle(frame, right_center, 5, (0, 255, 0), -1)
    cv2.imshow('Iris Tespiti', frame)
    cv2.waitKey(0)
else:
    print("Yüz veya iris tespit edilemedi")
```

### Mevcut Kodları Güncelleme

Eğer daha önce `_extract_eye_landmarks` kullanıyorsanız, şimdi `detect_iris_centers` kullanabilirsiniz:

**Eski kod:**
```python
# RGB'ye dönüştür
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = detector.face_mesh.process(rgb_frame)

if results.multi_face_landmarks:
    left_iris, right_iris = detector._extract_eye_landmarks(
        results.multi_face_landmarks[0], frame.shape
    )
```

**Yeni kod:**
```python
# Daha basit ve direkt kullanım
left_iris, right_iris = detector.detect_iris_centers(frame)
```

## Entegrasyon Adımları

### 1. `eye_tracking.py` için
```python
# Mevcut process_frame fonksiyonunda:
def process_frame(self, frame):
    # ... mevcut kod ...
    
    # Eski yöntem yerine:
    # left_eye, right_eye = self._extract_eye_landmarks(face_landmarks, frame.shape)
    
    # Yeni yöntemi kullan:
    from detector import NistagmusDetector
    if not hasattr(self, 'detector'):
        self.detector = NistagmusDetector()
    
    left_eye, right_eye = self.detector.detect_iris_centers(frame)
    
    if left_eye is not None and right_eye is not None:
        # Göz pozisyonlarını kaydet
        self.left_eye_positions_x.append(left_eye[0])
        # ... devamı
```

### 2. API dosyalarında kullanım
```python
# api/nistagmus_ai_api.py içinde:
@app.post("/analyze-frame")
async def analyze_frame(request: AnalyzeFrameRequest):
    try:
        # Frame'i decode et
        frame = decode_base64_image(request.frame_data)
        
        # Iris merkezlerini tespit et
        left_center, right_center = nistagmus_detector.detect_iris_centers(frame)
        
        if left_center is not None and right_center is not None:
            # Analiz sonuçlarını döndür
            return {
                "success": True,
                "left_iris": left_center,
                "right_iris": right_center,
                # ... diğer analiz sonuçları
            }
    except Exception as e:
        return {"success": False, "error": str(e)}
```

## Performans İyileştirmeleri

### 1. Model Yeniden Yaratılmaması
`detect_iris_centers` fonksiyonu, her çağrıda MediaPipe modelini yeniden yaratmaz. Model, sınıf başlatılırken bir kez oluşturulur ve tekrar kullanılır.

### 2. Daha Az İşlem
Doğrudan iris landmark'larını kullandığı için daha az hesaplama gerektirir.

### 3. Hata Toleransı
Hata durumlarında uygun şekilde `None` değerleri döndürür.

## Test Etme

Test scripti çalıştırarak entegrasyonu test edebilirsiniz:

```bash
python3 test_iris_detection.py
```

Bu script:
- Webcam testi (gerçek zamanlı iris tespiti)
- Statik görüntü testi
- Eski ve yeni yöntem karşılaştırması

seçeneklerini sunar.

## Dikkat Edilmesi Gerekenler

1. **MediaPipe Sürümü**: En az MediaPipe 0.8.0 sürümü gerekir
2. **RGB/BGR Dönüşümü**: `detect_iris_centers` BGR formatında görüntü alır ve otomatik olarak RGB'ye dönüştürür
3. **Koordinat Sistemi**: Döndürülen koordinatlar piksel cinsinden `(x, y)` formatındadır
4. **Hata Durumları**: Yüz tespit edilemezse `(None, None)` döndürülür

## Özellikler

✅ **Yüksek Doğruluk**: MediaPipe'ın gelişmiş iris tespiti
✅ **Gerçek Zamanlı Performans**: Optimize edilmiş işlem süresi  
✅ **Kolay Entegrasyon**: Minimal kod değişikliği gerekir
✅ **Hata Toleransı**: Robust hata yönetimi
✅ **Backward Compatibility**: Eski fonksiyonlar hala çalışır

## Sorun Giderme

### MediaPipe Kurulum Hatası
```bash
pip install --upgrade mediapipe
```

### Model Yükleme Hatası
```python
# Config dosyasında MediaPipe ayarlarını kontrol edin
config = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "refine_landmarks": True
}
detector = NistagmusDetector(config)
```

### Performans Optimizasyonu
```python
# Daha az sıklıkla işlem yapmak için:
if frame_count % 5 == 0:  # Her 5 karede bir
    left_center, right_center = detector.detect_iris_centers(frame)
```

Bu güncellemelerle birlikte, göz takibindeki hata payı azalacak ve özellikle küçük göz hareketlerinin bile tutarlı şekilde yakalandığını göreceksiniz. 