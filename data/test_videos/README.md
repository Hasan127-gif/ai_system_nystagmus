# Test Videoları

Bu klasör nistagmus ve strabismus tespiti için test videolarını içerir.

## Klasör Yapısı

### `nystagmus/`
Nistagmus (göz titremesi) örnekleri içeren test videoları:
- `nystagmus_test_1.mp4` - Nistagmus test videosu 1 (1.4 MB)
- `nystagmus_test_2.mp4` - Nistagmus test videosu 2 (19.6 MB)

### `strabismus/`
Strabismus (şaşılık) örnekleri içeren test videoları:
- `strabismus_test_1.mp4` - Strabismus test videosu 1 (300 KB)
- `strabismus_test_2.mp4` - Strabismus test videosu 2 (1.2 MB)

## Kullanım

Bu videolar sistem testleri, model eğitimi ve validasyon için kullanılabilir:

```python
from detector import NistagmusDetector

detector = NistagmusDetector()

# Nistagmus testi
result = detector.analyze_video("data/test_videos/nystagmus/nystagmus_test_1.mp4")
print(f"Nistagmus tespit edildi: {result['nystagmus_detected']}")

# Strabismus testi
result = detector.analyze_video("data/test_videos/strabismus/strabismus_test_1.mp4")
print(f"Strabismus tespit edildi: {result['strabismus_detected']}")
```

## Not

Bu videolar Git LFS (Large File Storage) ile takip edilmektedir. Klonlama sırasında `git lfs pull` komutunu çalıştırmayı unutmayın.

```bash
git clone https://github.com/Hasan127-gif/ai_system_nystagmus.git
cd ai_system_nystagmus
git lfs pull
``` 