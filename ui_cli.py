#!/usr/bin/env python3
"""
KLİNİK ONAY SİSTEMİ - COMMAND LINE VERSİYONU
============================================
Tkinter olmadan çalışan doktor onay sistemi.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any

class ClinicalApprovalCLI:
    """CLI Klinik onay sistemi."""
    
    def __init__(self):
        self.current_analysis = None
    
    def show_pathology_alert(self, analysis_data: Dict[str, Any]) -> str:
        """Patoloji uyarısı göster."""
        try:
            flags = {
                'nistagmus': analysis_data.get('nystagmus_detected', False),
                'strabismus': analysis_data.get('strabismus_detected', False)
            }
            
            if flags['nistagmus'] or flags['strabismus']:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🚨" * 20)
                print("🚨 PATOLOJİ TESPİT EDİLDİ!")
                print("🚨" * 20)
                print()
                
                pathology_list = []
                if flags['nistagmus']:
                    freq = analysis_data.get('nistagmus_frequency', 0)
                    pathology_list.append(f"• Nistagmus (Frekans: {freq:.1f} Hz)")
                    
                if flags['strabismus']:
                    angle = analysis_data.get('strabismus_angle', 0)
                    pathology_list.append(f"• Şaşılık (Açı: {angle:.1f}°)")
                
                print("📊 TESPİT EDİLEN PATOLOJİLER:")
                for pathology in pathology_list:
                    print(pathology)
                print()
                
                # ML güven skoru
                confidence = analysis_data.get('ml_confidence', 0)
                print(f"🤖 ML Güven Skoru: {confidence:.3f}")
                print()
                
                print("⚠️  Model patoloji tespit etti.")
                print("⚠️  Doktor onayı gerekli.")
                print()
                
                while True:
                    choice = input("Doktor onayı ekranına geçmek istiyor musunuz? (e/h): ").lower().strip()
                    if choice in ['e', 'evet', 'y', 'yes']:
                        return self.doctor_approval_interface(analysis_data)
                    elif choice in ['h', 'hayır', 'n', 'no']:
                        return "pending"
                    else:
                        print("Lütfen 'e' (evet) veya 'h' (hayır) giriniz.")
            
            return "normal"
            
        except Exception as e:
            print(f"❌ Patoloji uyarı hatası: {e}")
            return "error"
    
    def doctor_approval_interface(self, analysis_data: Dict[str, Any]) -> str:
        """Doktor onay arayüzü."""
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("👨‍⚕️ DOKTOR ONAY SİSTEMİ")
            print("=" * 40)
            print()
            
            # Analiz sonuçlarını göster
            print("📊 ANALİZ SONUÇLARI:")
            print("-" * 25)
            print(f"🎯 Nistagmus Frekansı: {analysis_data.get('nistagmus_frequency', 0):.2f} Hz")
            print(f"👁  Şaşılık Açısı: {analysis_data.get('strabismus_angle', 0):.2f}°")
            print(f"🔴 Nistagmus: {'✅ Pozitif' if analysis_data.get('nystagmus_detected') else '❌ Negatif'}")
            print(f"🔴 Şaşılık: {'✅ Pozitif' if analysis_data.get('strabismus_detected') else '❌ Negatif'}")
            print(f"🤖 ML Güven: {analysis_data.get('ml_confidence', 0):.3f}")
            print(f"📹 Video Kalite: {analysis_data.get('video_quality', 'bilinmiyor')}")
            print()
            
            # Doktor ID girişi
            while True:
                doctor_id = input("👨‍⚕️ Doktor ID giriniz: ").strip()
                if doctor_id:
                    break
                print("❌ Doktor ID boş olamaz!")
            
            # Onay durumu seçimi
            print("\n📋 ONAY DURUMU SEÇİNİZ:")
            print("1 - ✅ Onayla")
            print("2 - ❌ Reddet") 
            print("3 - ⏳ Beklemede bırak")
            print()
            
            while True:
                try:
                    choice = int(input("Seçiminiz (1-3): "))
                    if choice == 1:
                        approval_status = "approved"
                        break
                    elif choice == 2:
                        approval_status = "rejected"
                        break
                    elif choice == 3:
                        approval_status = "pending"
                        break
                    else:
                        print("❌ Geçersiz seçim! 1, 2 veya 3 giriniz.")
                except ValueError:
                    print("❌ Geçersiz giriş! Sayı giriniz.")
            
            # Doktor notları
            print(f"\n📝 Doktor notları ({approval_status} için):")
            notes = input("Notlarınızı giriniz (opsiyonel): ").strip()
            
            # Onayı kaydet
            success = self.save_approval(analysis_data, doctor_id, approval_status, notes)
            
            if success:
                print(f"\n✅ Onay başarıyla kaydedildi: {approval_status.upper()}")
                if notes:
                    print(f"📝 Notlar: {notes}")
            else:
                print("\n❌ Onay kaydedilemedi!")
                approval_status = "error"
            
            input("\nDevam etmek için Enter tuşuna basın...")
            return approval_status
            
        except KeyboardInterrupt:
            print("\n\n👋 Onay işlemi iptal edildi.")
            return "cancelled"
        except Exception as e:
            print(f"\n❌ Onay arayüzü hatası: {e}")
            return "error"
    
    def save_approval(self, analysis_data: Dict[str, Any], doctor_id: str, 
                     approval_status: str, notes: str) -> bool:
        """Onayı kaydetme sistemi."""
        try:
            from logger import get_clinical_logger
            
            clinical_logger = get_clinical_logger()
            analysis_id = analysis_data.get('analysis_id', f"cli_{int(datetime.now().timestamp())}")
            
            # Analizi kaydet
            if 'analysis_id' not in analysis_data:
                analysis_data['analysis_id'] = analysis_id
                clinical_logger.log_analysis(analysis_data)
            
            # Doktor onayını kaydet
            clinical_logger.log_doctor_approval(analysis_id, doctor_id, approval_status, notes)
            
            return True
            
        except Exception as e:
            print(f"❌ Kaydetme hatası: {e}")
            return False
    
    def view_analysis_history(self):
        """Analiz geçmişini görüntüle."""
        try:
            from logger import get_clinical_logger
            
            clinical_logger = get_clinical_logger()
            
            os.system('clear' if os.name == 'posix' else 'cls')
            print("📋 ANALİZ GEÇMİŞİ")
            print("=" * 50)
            
            if clinical_logger.analysis_log.exists():
                import csv
                
                with open(clinical_logger.analysis_log, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    analyses = list(reader)[-10:]  # Son 10 analiz
                    
                    if analyses:
                        print(f"Son {len(analyses)} analiz:")
                        print()
                        
                        for i, row in enumerate(analyses, 1):
                            timestamp = row.get('timestamp', '')[:19]
                            patient_id = row.get('patient_id_hash', '')[:8]
                            nystagmus = "✅" if row.get('nystagmus_detected') == 'True' else "❌"
                            strabismus = "✅" if row.get('strabismus_detected') == 'True' else "❌"
                            approval = row.get('doctor_approval', 'pending')
                            
                            print(f"{i:2d}. {timestamp} | Hasta: {patient_id}")
                            print(f"    Nistagmus: {nystagmus} | Şaşılık: {strabismus}")
                            print(f"    Onay: {approval}")
                            print()
                    else:
                        print("❌ Analiz geçmişi bulunamadı.")
            else:
                print("❌ Analiz log dosyası bulunamadı.")
            
            input("Devam etmek için Enter tuşuna basın...")
            
        except Exception as e:
            print(f"❌ Geçmiş görüntüleme hatası: {e}")
            input("Devam etmek için Enter tuşuna basın...")
    
    def generate_clinic_report(self):
        """Klinik rapor oluştur."""
        try:
            from logger import get_clinical_logger
            
            clinical_logger = get_clinical_logger()
            audit_report = clinical_logger.generate_audit_report()
            
            os.system('clear' if os.name == 'posix' else 'cls')
            print("🏥 KLİNİK RAPOR")
            print("=" * 30)
            print(f"📅 Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print()
            
            summary = audit_report["summary"]
            print("📊 GENEL İSTATİSTİKLER:")
            print(f"• Toplam Analiz: {summary['total_analyses']}")
            print(f"• Onaylanmış: {summary['approved_analyses']}")
            print(f"• Bekleyen: {summary['pending_analyses']}")
            print(f"• Reddedilen: {summary['rejected_analyses']}")
            print(f"• Benzersiz Hasta: {summary['unique_patients']}")
            print()
            
            print(f"✅ Uygunluk Durumu: {audit_report['compliance_status'].upper()}")
            
            if audit_report.get('recommendations'):
                print("\n📋 ÖNERİLER:")
                for rec in audit_report['recommendations']:
                    print(f"• {rec}")
            
            print("=" * 30)
            input("Devam etmek için Enter tuşuna basın...")
            
        except Exception as e:
            print(f"❌ Rapor oluşturma hatası: {e}")
            input("Devam etmek için Enter tuşuna basın...")
    
    def main_menu(self):
        """Ana menü."""
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                
                print("🏥 KLİNİK ONAY SİSTEMİ - CLI")
                print("=" * 35)
                print()
                print("📋 MENÜ:")
                print("1 - 🚨 Test Patoloji Uyarısı")
                print("2 - 📋 Analiz Geçmişi")
                print("3 - 📊 Klinik Rapor")
                print("4 - 🚪 Çıkış")
                print()
                
                try:
                    choice = int(input("Seçiminiz (1-4): "))
                    
                    if choice == 1:
                        # Test patoloji uyarısı
                        test_analysis = {
                            "analysis_id": f"test_{int(datetime.now().timestamp())}",
                            "nistagmus_frequency": 4.2,
                            "strabismus_angle": 3.5,
                            "nystagmus_detected": True,
                            "strabismus_detected": True,
                            "ml_confidence": 0.87,
                            "video_quality": "good"
                        }
                        
                        result = self.show_pathology_alert(test_analysis)
                        print(f"\n📋 Sonuç: {result}")
                        input("Devam etmek için Enter tuşuna basın...")
                        
                    elif choice == 2:
                        self.view_analysis_history()
                        
                    elif choice == 3:
                        self.generate_clinic_report()
                        
                    elif choice == 4:
                        print("👋 Sistem kapatılıyor...")
                        break
                        
                    else:
                        print("❌ Geçersiz seçim!")
                        input("Devam etmek için Enter tuşuna basın...")
                        
                except ValueError:
                    print("❌ Geçersiz giriş! Sayı giriniz.")
                    input("Devam etmek için Enter tuşuna basın...")
                except KeyboardInterrupt:
                    print("\n👋 Sistem kapatılıyor...")
                    break
                    
        except Exception as e:
            print(f"❌ Ana menü hatası: {e}")

def show_pathology_alert_cli(analysis_data: Dict[str, Any]) -> str:
    """Standalone patoloji uyarısı fonksiyonu."""
    cli = ClinicalApprovalCLI()
    return cli.show_pathology_alert(analysis_data)

def main():
    """Ana fonksiyon."""
    print("🏥 KLİNİK ONAY SİSTEMİ CLI")
    print("Başlatılıyor...")
    
    cli = ClinicalApprovalCLI()
    cli.main_menu()

if __name__ == "__main__":
    main() 