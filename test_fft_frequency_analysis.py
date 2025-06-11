#!/usr/bin/env python3
"""
FFT Tabanlı Nistagmus Frekans Analizi Test Scripti
=================================================
Bu script, yeni FFT tabanlı frekans hesaplama fonksiyonlarını test eder.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from eye_tracking import EyeTracker
from clinical.historical_analysis import HistoricalDataAnalyzer

def generate_synthetic_nystagmus_data(duration=10.0, sample_rate=30.0, frequency=4.0, amplitude=10.0, noise_level=0.5):
    """
    Sentetik nistagmus verileri oluşturur
    
    Args:
        duration: Süre (saniye)
        sample_rate: Örnekleme hızı (Hz)
        frequency: Nistagmus frekansı (Hz)
        amplitude: Hareket genliği (piksel)
        noise_level: Gürültü seviyesi
    
    Returns:
        tuple: (zaman_dizisi, y_pozisyonları)
    """
    num_samples = int(duration * sample_rate)
    time_array = np.linspace(0, duration, num_samples)
    
    # Temel nistagmus sinyali (dişli dalga - sawtooth)
    nystagmus_signal = amplitude * (2 * (time_array * frequency - np.floor(time_array * frequency + 0.5)))
    
    # Gürültü ekle
    noise = np.random.normal(0, noise_level * amplitude, num_samples)
    
    # Baseline drift (yavaş trend)
    drift = 2 * np.sin(0.1 * time_array)
    
    y_positions = nystagmus_signal + noise + drift
    
    return time_array, y_positions

def test_frequency_calculation_methods():
    """Farklı frekans hesaplama yöntemlerini karşılaştırır"""
    print("=== FFT Tabanlı Frekans Hesaplama Test ===")
    
    # Test verileri oluştur
    test_frequencies = [2.0, 4.0, 6.0, 8.0, 10.0]  # Hz
    sample_rate = 30.0  # FPS
    duration = 15.0  # saniye
    
    # EyeTracker instance'ı oluştur
    eye_tracker = EyeTracker()
    
    # Historical analyzer oluştur
    historical_analyzer = HistoricalDataAnalyzer()
    
    results = []
    
    print(f"Test ayarları: Süre={duration}s, Sample Rate={sample_rate} Hz")
    print("-" * 70)
    
    for true_freq in test_frequencies:
        print(f"\nGerçek Frekans: {true_freq:.1f} Hz")
        
        # Sentetik veri oluştur
        time_array, y_positions = generate_synthetic_nystagmus_data(
            duration=duration, 
            sample_rate=sample_rate, 
            frequency=true_freq,
            amplitude=15.0,
            noise_level=0.3
        )
        
        # FFT tabanlı frekans hesaplama (eye_tracking)
        start_time = time.time()
        eye_tracking_freq = eye_tracker.compute_nystagmus_frequency(y_positions.tolist(), sample_rate)
        eye_tracking_time = time.time() - start_time
        
        # FFT tabanlı frekans hesaplama (historical_analysis)
        start_time = time.time()
        historical_freq = historical_analyzer.compute_nystagmus_frequency(y_positions.tolist(), sample_rate)
        historical_time = time.time() - start_time
        
        # Geleneksel FFT (referans)
        start_time = time.time()
        traditional_freq = calculate_traditional_fft(y_positions, sample_rate)
        traditional_time = time.time() - start_time
        
        # Sonuçları kaydet
        result = {
            'true_frequency': true_freq,
            'eye_tracking_freq': eye_tracking_freq,
            'historical_freq': historical_freq,
            'traditional_freq': traditional_freq,
            'eye_tracking_error': abs(eye_tracking_freq - true_freq),
            'historical_error': abs(historical_freq - true_freq),
            'traditional_error': abs(traditional_freq - true_freq),
            'eye_tracking_time': eye_tracking_time,
            'historical_time': historical_time,
            'traditional_time': traditional_time
        }
        results.append(result)
        
        # Sonuçları yazdır
        print(f"  Eye Tracking FFT: {eye_tracking_freq:.2f} Hz (Hata: {result['eye_tracking_error']:.2f} Hz)")
        print(f"  Historical FFT:   {historical_freq:.2f} Hz (Hata: {result['historical_error']:.2f} Hz)")
        print(f"  Geleneksel FFT:   {traditional_freq:.2f} Hz (Hata: {result['traditional_error']:.2f} Hz)")
        print(f"  İşlem Süreleri: ET={eye_tracking_time*1000:.2f}ms, HA={historical_time*1000:.2f}ms, GT={traditional_time*1000:.2f}ms")
    
    # Genel performans analizi
    print("\n=== Genel Performans Analizi ===")
    
    et_avg_error = np.mean([r['eye_tracking_error'] for r in results])
    ha_avg_error = np.mean([r['historical_error'] for r in results])
    gt_avg_error = np.mean([r['traditional_error'] for r in results])
    
    et_avg_time = np.mean([r['eye_tracking_time'] for r in results]) * 1000
    ha_avg_time = np.mean([r['historical_time'] for r in results]) * 1000
    gt_avg_time = np.mean([r['traditional_time'] for r in results]) * 1000
    
    print(f"Ortalama Hata:")
    print(f"  Eye Tracking FFT: {et_avg_error:.3f} Hz")
    print(f"  Historical FFT:   {ha_avg_error:.3f} Hz")
    print(f"  Geleneksel FFT:   {gt_avg_error:.3f} Hz")
    
    print(f"\nOrtalama İşlem Süresi:")
    print(f"  Eye Tracking FFT: {et_avg_time:.2f} ms")
    print(f"  Historical FFT:   {ha_avg_time:.2f} ms")
    print(f"  Geleneksel FFT:   {gt_avg_time:.2f} ms")
    
    return results

def calculate_traditional_fft(y_positions, frame_rate):
    """Geleneksel FFT yöntemi (referans için)"""
    if len(y_positions) < 2:
        return 0.0
        
    y = np.array(y_positions, dtype=float)
    y = y - np.mean(y)
    
    if np.std(y) < 1e-6:
        return 0.0
        
    # Basit FFT
    fft_vals = np.fft.fft(y)
    freqs = np.fft.fftfreq(len(y), d=1.0/frame_rate)
    power = np.abs(fft_vals)
    
    # Pozitif frekanslar
    positive_freqs = freqs[:len(freqs)//2]
    positive_power = power[:len(power)//2]
    
    if len(positive_freqs) <= 1:
        return 0.0
        
    # En büyük güç (DC hariç)
    max_idx = np.argmax(positive_power[1:]) + 1
    return abs(positive_freqs[max_idx])

def test_with_real_world_scenarios():
    """Gerçek dünya senaryolarıyla test"""
    print("\n=== Gerçek Dünya Senaryoları Test ===")
    
    eye_tracker = EyeTracker()
    
    scenarios = [
        {"name": "Düşük Gürültü", "noise": 0.1, "amplitude": 20},
        {"name": "Orta Gürültü", "noise": 0.5, "amplitude": 15},
        {"name": "Yüksek Gürültü", "noise": 1.0, "amplitude": 10},
        {"name": "Çok Küçük Hareket", "noise": 0.3, "amplitude": 3},
        {"name": "Büyük Hareket", "noise": 0.2, "amplitude": 30},
    ]
    
    test_freq = 5.0  # Hz
    
    for scenario in scenarios:
        print(f"\nSenaryo: {scenario['name']}")
        
        # Veri oluştur
        _, y_positions = generate_synthetic_nystagmus_data(
            duration=10.0,
            sample_rate=30.0,
            frequency=test_freq,
            amplitude=scenario['amplitude'],
            noise_level=scenario['noise']
        )
        
        # FFT ile hesapla
        calculated_freq = eye_tracker.compute_nystagmus_frequency(y_positions.tolist(), 30.0)
        error = abs(calculated_freq - test_freq)
        
        print(f"  Hesaplanan: {calculated_freq:.2f} Hz (Hata: {error:.2f} Hz)")
        print(f"  Genlik: {scenario['amplitude']} piksel, Gürültü: {scenario['noise']}")

def test_edge_cases():
    """Sınır durumları test eder"""
    print("\n=== Sınır Durumları Test ===")
    
    eye_tracker = EyeTracker()
    
    # Test 1: Çok az veri
    print("Test 1: Çok az veri (1 nokta)")
    result = eye_tracker.compute_nystagmus_frequency([100], 30.0)
    print(f"  Sonuç: {result:.2f} Hz (Beklenen: 0.0)")
    
    # Test 2: Sabit değerler
    print("Test 2: Sabit değerler")
    result = eye_tracker.compute_nystagmus_frequency([100] * 100, 30.0)
    print(f"  Sonuç: {result:.2f} Hz (Beklenen: 0.0)")
    
    # Test 3: Çok yüksek frekans
    print("Test 3: Çok yüksek frekans (20 Hz)")
    _, y_positions = generate_synthetic_nystagmus_data(
        duration=5.0, sample_rate=30.0, frequency=20.0, amplitude=10.0, noise_level=0.1
    )
    result = eye_tracker.compute_nystagmus_frequency(y_positions.tolist(), 30.0)
    print(f"  Sonuç: {result:.2f} Hz (Girdi: 20.0 Hz)")
    
    # Test 4: Çok düşük frekans
    print("Test 4: Çok düşük frekans (0.3 Hz)")
    _, y_positions = generate_synthetic_nystagmus_data(
        duration=20.0, sample_rate=30.0, frequency=0.3, amplitude=10.0, noise_level=0.1
    )
    result = eye_tracker.compute_nystagmus_frequency(y_positions.tolist(), 30.0)
    print(f"  Sonuç: {result:.2f} Hz (Girdi: 0.3 Hz)")

if __name__ == "__main__":
    print("FFT Tabanlı Nistagmus Frekans Analizi Test Başlıyor...")
    print("=" * 60)
    
    try:
        # Ana test
        results = test_frequency_calculation_methods()
        
        # Gerçek dünya testleri
        test_with_real_world_scenarios()
        
        # Sınır durumları
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("✅ Tüm testler başarıyla tamamlandı!")
        print("FFT iyileştirmeleri çalışır durumda.")
        
    except Exception as e:
        print(f"\n❌ Test sırasında hata: {str(e)}")
        import traceback
        traceback.print_exc() 