#!/usr/bin/env python3
"""
STABİL SİSTEM LAUNCHER
=====================
Farklı arayüz alternatifleriyle sistemi başlatır.
"""

import sys
import os

def print_banner():
    """Sistem banner'ı."""
    print("🏥 NİSTAGMUS TESPİT SİSTEMİ")
    print("=" * 30)
    print("Stabil ve çoklu arayüz desteği")
    print()

def check_dependencies():
    """Gerekli modülleri kontrol eder."""
    deps = {
        "tkinter": False,
        "flask": False,
        "core": True
    }
    
    # Tkinter kontrolü
    try:
        import tkinter
        deps["tkinter"] = True
    except ImportError:
        pass
    
    # Flask kontrolü
    try:
        import flask
        deps["flask"] = True
    except ImportError:
        pass
    
    return deps

def run_cli_dashboard():
    """CLI dashboard'u çalıştırır."""
    print("🖥️  CLI Dashboard başlatılıyor...")
    os.system("python dashboard_cli.py --interactive")

def run_web_dashboard():
    """Web dashboard'u çalıştırır."""
    print("🌐 Web Dashboard başlatılıyor...")
    os.system("python web_dashboard.py")

def run_cli_approval():
    """CLI onay sistemini çalıştırır."""
    print("👨‍⚕️ CLI Onay Sistemi başlatılıyor...")
    os.system("python ui_cli.py")

def run_test_system():
    """Sistem testlerini çalıştırır."""
    print("🔧 Sistem testleri çalıştırılıyor...")
    os.system("python test_clinical_system.py")

def run_analysis():
    """Video analizi çalıştırır."""
    print("📹 Video analizi başlatılıyor...")
    video_file = input("Video dosyası yolu (boş bırakırsa test videosu): ").strip()
    
    if not video_file:
        video_file = "test_clinical_video.mp4"
    
    if os.path.exists(video_file):
        os.system(f"python detector.py --video {video_file}")
    else:
        print(f"❌ Video dosyası bulunamadı: {video_file}")

def main():
    """Ana menü."""
    print_banner()
    
    deps = check_dependencies()
    
    print("📋 MEVCUT ÖZELLİKLER:")
    print(f"✅ Core Sistem: {'✓' if deps['core'] else '✗'}")
    print(f"{'✅' if deps['tkinter'] else '❌'} Tkinter GUI: {'Mevcut' if deps['tkinter'] else 'Yok'}")
    print(f"{'✅' if deps['flask'] else '❌'} Web Arayüzü: {'Mevcut' if deps['flask'] else 'Yok (pip install flask)'}")
    print()
    
    while True:
        print("🚀 SİSTEM BAŞLATICI")
        print("-" * 20)
        print("1 - 🖥️  CLI Dashboard")
        print("2 - 🌐 Web Dashboard" + (" (Flask gerekli)" if not deps['flask'] else ""))
        print("3 - 👨‍⚕️ CLI Onay Sistemi")
        print("4 - 📹 Video Analizi")
        print("5 - 🔧 Sistem Testi")
        print("6 - 📊 Hızlı Durum Kontrolü")
        print("0 - 🚪 Çıkış")
        print()
        
        try:
            choice = input("Seçiminiz (0-6): ").strip()
            
            if choice == "1":
                run_cli_dashboard()
            elif choice == "2":
                if deps['flask']:
                    run_web_dashboard()
                else:
                    print("❌ Flask modülü gerekli!")
                    print("Kurulum: pip install flask")
            elif choice == "3":
                run_cli_approval()
            elif choice == "4":
                run_analysis()
            elif choice == "5":
                run_test_system()
            elif choice == "6":
                quick_status_check()
            elif choice == "0":
                print("👋 Sistem kapatılıyor...")
                break
            else:
                print("❌ Geçersiz seçim!")
            
            if choice != "0":
                input("\nDevam etmek için Enter tuşuna basın...")
                print("\n" + "="*50 + "\n")
                
        except KeyboardInterrupt:
            print("\n👋 Sistem kapatılıyor...")
            break
        except Exception as e:
            print(f"❌ Hata: {e}")

def quick_status_check():
    """Hızlı sistem durumu kontrolü."""
    print("\n⚡ HIZLI DURUM KONTROLÜ")
    print("=" * 25)
    
    # Privacy sistemi
    try:
        from privacy import get_privacy_manager
        privacy = get_privacy_manager()
        compliance = privacy.get_privacy_compliance_status()
        print("✅ Gizlilik Sistemi: Aktif")
        print(f"   Şifreleme: {compliance['encryption_status']}")
    except Exception as e:
        print(f"❌ Gizlilik Sistemi: {e}")
    
    # Logger sistemi
    try:
        from logger import get_clinical_logger
        logger = get_clinical_logger()
        print("✅ Kayıt Sistemi: Aktif")
    except Exception as e:
        print(f"❌ Kayıt Sistemi: {e}")
    
    # Kalibrasyon sistemi
    try:
        from calibration import get_calibrator
        calibrator = get_calibrator()
        status = calibrator.get_calibration_status()
        print(f"✅ Kalibrasyon: {'Aktif' if status['calibrated'] else 'Gerekli'}")
        if status['calibrated']:
            print(f"   Oran: {status['pixel_to_degree_ratio']:.4f} derece/piksel")
    except Exception as e:
        print(f"❌ Kalibrasyon: {e}")
    
    # ML sistemi
    try:
        from model import NystagmusClassifier
        classifier = NystagmusClassifier()
        print("✅ ML Sistemi: Aktif")
    except Exception as e:
        print(f"❌ ML Sistemi: {e}")
    
    print("\n📊 DURUM: Sistemler kontrol edildi.")

if __name__ == "__main__":
    main() 