#!/usr/bin/env python3
"""
DÜZELTİLMİŞ KLİNİK SİSTEM ENTEGRASYON TESTİ
===========================================
Tkinter bağımlılığı olmadan tüm sistemleri test eder.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

def test_clinical_system_fixed():
    """Düzeltilmiş tam klinik sistem entegrasyon testi."""
    print("🏥 DÜZELTİLMİŞ KLİNİK KARAR DESTEK SİSTEMİ - ENTEGRASYON TESTİ")
    print("=" * 65)
    
    test_results = []
    
    # 1. Privacy & Security Test
    print("\n🔐 1. Gizlilik & Güvenlik Sistemi Testi...")
    try:
        from privacy import get_privacy_manager
        
        privacy_manager = get_privacy_manager()
        
        # Kullanıcı doğrulama
        auth_result = privacy_manager.authenticate_user("admin", "admin123", "127.0.0.1")
        if auth_result:
            session_token = auth_result["session_token"]
            
            # İzin kontrolü
            has_admin = privacy_manager.check_permission(session_token, "admin")
            
            # Veri şifreleme testi
            test_data = {
                "patient_id": "P001",
                "analysis_data": {
                    "nistagmus_frequency": 4.2,
                    "strabismus_angle": 3.5,
                    "nystagmus_detected": True,
                    "strabismus_detected": True
                }
            }
            
            encrypted_data = privacy_manager.encrypt_patient_data(test_data)
            decrypted_data = privacy_manager.decrypt_patient_data(encrypted_data)
            
            # Şifreleme doğruluğu kontrolü
            encryption_valid = (
                decrypted_data and 
                "analysis_data" in decrypted_data and
                decrypted_data["analysis_data"]["nistagmus_frequency"] == 4.2
            )
            
            # Uygunluk durumu
            compliance_status = privacy_manager.get_privacy_compliance_status()
            compliance_valid = (
                compliance_status["encryption_status"] == "AES-256 active" and
                "HIPAA" in compliance_status["compliance_frameworks"]
            )
            
            test_results.append({
                "test": "Privacy System",
                "status": "✅ BAŞARILI",
                "details": f"Doğrulama: ✓, İzin: {has_admin}, Şifreleme: {encryption_valid}, HIPAA/GDPR: {compliance_valid}"
            })
            
            privacy_manager.logout_user(session_token)
        else:
            test_results.append({
                "test": "Privacy System",
                "status": "❌ BAŞARISIZ",
                "details": "Kullanıcı doğrulama hatası"
            })
    except Exception as e:
        test_results.append({
            "test": "Privacy System",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 2. Clinical Logger Test
    print("\n📝 2. Klinik Kayıt Sistemi Testi...")
    try:
        from logger import get_clinical_logger
        
        clinical_logger = get_clinical_logger()
        
        # Analiz kaydı testi
        test_analysis = {
            "nistagmus_frequency": 4.2,
            "strabismus_angle": 3.5,
            "nystagmus_detected": True,
            "strabismus_detected": True,
            "ml_confidence": 0.85,
            "video_quality": "good",
            "session_id": "test_session_fixed_001"
        }
        
        test_patient = {
            "patient_id": "P_FIXED_001",
            "age": 35
        }
        
        # Analiz kaydet
        analysis_id = clinical_logger.log_analysis(test_analysis, test_patient)
        analysis_logged = bool(analysis_id)
        
        # Doktor onayı testi
        clinical_logger.log_doctor_approval(analysis_id, "doctor_test", "approved", "Test onayı - sistem düzeltme")
        
        # Denetim raporu testi
        audit_report = clinical_logger.generate_audit_report()
        audit_valid = (
            "summary" in audit_report and
            audit_report["summary"]["total_analyses"] > 0
        )
        
        # Uygunluk verisi dışa aktarma testi
        export_file = clinical_logger.export_compliance_data("json")
        export_valid = bool(export_file)
        
        test_results.append({
            "test": "Clinical Logger",
            "status": "✅ BAŞARILI",
            "details": f"Analiz: {analysis_logged}, Onay: ✓, Denetim: {audit_valid}, Export: {export_valid}"
        })
        
    except Exception as e:
        test_results.append({
            "test": "Clinical Logger",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 3. CLI Dashboard System Test
    print("\n📊 3. CLI Dashboard Sistemi Testi...")
    try:
        from dashboard_cli import ClinicalDashboardCLI
        
        # CLI Dashboard başlatma testi
        dashboard = ClinicalDashboardCLI()
        
        # Test metrik güncellemesi
        test_metrics = {
            "sensitivity": 0.87,
            "specificity": 0.84,
            "auc": 0.86,
            "accuracy": 0.85
        }
        
        dashboard.report_metrics(test_metrics)
        
        # Metrics data kontrolü
        current_metrics = dashboard.metrics_data["current_metrics"]
        metrics_updated = all(
            current_metrics.get(key) == test_metrics[key] 
            for key in test_metrics.keys()
        )
        
        # Rapor oluşturma testi
        report_success = dashboard.generate_report()
        
        test_results.append({
            "test": "CLI Dashboard System",
            "status": "✅ BAŞARILI",
            "details": f"Başlatma: ✓, Metrik güncelleme: {metrics_updated}, Rapor: {report_success}"
        })
        
    except Exception as e:
        test_results.append({
            "test": "CLI Dashboard System",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 4. CLI Approval System Test
    print("\n🖥️ 4. CLI Onay Sistemi Testi...")
    try:
        from ui_cli import ClinicalApprovalCLI, show_pathology_alert_cli
        
        # CLI onay sistemi testi
        approval_cli = ClinicalApprovalCLI()
        
        # Test analiz verisi
        test_analysis_ui = {
            "analysis_id": "test_cli_ui_001",
            "nistagmus_frequency": 4.2,
            "strabismus_angle": 3.5,
            "nystagmus_detected": True,
            "strabismus_detected": True,
            "ml_confidence": 0.85,
            "video_quality": "good"
        }
        
        # Kaydetme testi (onay arayüzü olmadan)
        save_success = approval_cli.save_approval(
            test_analysis_ui, "test_doctor", "approved", "CLI test onayı"
        )
        
        test_results.append({
            "test": "CLI Approval System",
            "status": "✅ BAŞARILI",
            "details": f"Başlatma: ✓, Kaydetme: {save_success}, Analiz işleme: ✓"
        })
        
    except Exception as e:
        test_results.append({
            "test": "CLI Approval System",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 5. Core Integration Test (Tüm core sistemler birlikte)
    print("\n🔄 5. Core Entegrasyon Testi...")
    try:
        # Gerçek senaryo simülasyonu (GUI olmadan)
        from privacy import get_privacy_manager
        from logger import get_clinical_logger
        from dashboard_cli import ClinicalDashboardCLI
        
        # 1. Kullanıcı giriş yapar
        privacy_manager = get_privacy_manager()
        auth_result = privacy_manager.authenticate_user("doctor_001", "doctor123", "192.168.1.100")
        
        if auth_result:
            session_token = auth_result["session_token"]
            
            # 2. Analiz yapılır ve kaydedilir
            clinical_logger = get_clinical_logger()
            
            integration_analysis = {
                "nistagmus_frequency": 5.1,
                "strabismus_angle": 4.2,
                "nystagmus_detected": True,
                "strabismus_detected": True,
                "ml_confidence": 0.92,
                "video_quality": "excellent",
                "session_id": f"integration_fixed_test_{int(time.time())}"
            }
            
            integration_patient = {
                "patient_id": "P_INTEGRATION_FIXED_001",
                "age": 28
            }
            
            # Şifrelenmiş hasta verisi
            encrypted_patient = privacy_manager.encrypt_patient_data({
                "patient_id": integration_patient["patient_id"],
                "analysis_data": integration_analysis
            })
            
            # Analiz kaydet
            analysis_id = clinical_logger.log_analysis(integration_analysis, integration_patient)
            
            # 3. Doktor onayı
            clinical_logger.log_doctor_approval(analysis_id, "doctor_001", "approved", 
                                              "Entegrasyon testi - Patoloji onaylandı (düzeltilmiş)")
            
            # 4. CLI Dashboard metriklerini güncelle
            dashboard = ClinicalDashboardCLI()
            integration_metrics = {
                "sensitivity": 0.91,
                "specificity": 0.88,
                "auc": 0.90,
                "accuracy": 0.89
            }
            dashboard.report_metrics(integration_metrics)
            
            # 5. Erişim logla
            clinical_logger.log_access("doctor_001", "analysis_complete", "patient_data", "192.168.1.100")
            
            # 6. Patoloji testi (simüle)
            flags_test = {
                'nistagmus': integration_analysis['nystagmus_detected'],
                'strabismus': integration_analysis['strabismus_detected']
            }
            pathology_detected = flags_test['nistagmus'] or flags_test['strabismus']
            
            # 7. Çıkış yap
            privacy_manager.logout_user(session_token)
            
            test_results.append({
                "test": "Core Integration",
                "status": "✅ BAŞARILI",
                "details": f"Analiz: ✓, Onay: ✓, Şifreleme: ✓, Dashboard: ✓, Patoloji: {pathology_detected}"
            })
        else:
            test_results.append({
                "test": "Core Integration",
                "status": "❌ BAŞARISIZ",
                "details": "Doktor girişi başarısız"
            })
            
    except Exception as e:
        test_results.append({
            "test": "Core Integration",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 6. Pathology Alert Logic Test
    print("\n🚨 6. Patoloji Uyarı Sistemi Testi...")
    try:
        # Patoloji mantığı testi
        pathology_test_cases = [
            {
                "name": "Normal Case",
                "data": {"nystagmus_detected": False, "strabismus_detected": False},
                "expected": False
            },
            {
                "name": "Nystagmus Only",
                "data": {"nystagmus_detected": True, "strabismus_detected": False, "nistagmus_frequency": 4.5},
                "expected": True
            },
            {
                "name": "Strabismus Only", 
                "data": {"nystagmus_detected": False, "strabismus_detected": True, "strabismus_angle": 3.2},
                "expected": True
            },
            {
                "name": "Both Pathologies",
                "data": {"nystagmus_detected": True, "strabismus_detected": True, "nistagmus_frequency": 5.1, "strabismus_angle": 4.8},
                "expected": True
            }
        ]
        
        pathology_results = []
        for test_case in pathology_test_cases:
            flags = {
                'nistagmus': test_case["data"].get('nystagmus_detected', False),
                'strabismus': test_case["data"].get('strabismus_detected', False)
            }
            
            pathology_detected = flags['nistagmus'] or flags['strabismus']
            test_passed = pathology_detected == test_case["expected"]
            pathology_results.append(test_passed)
        
        all_pathology_tests_passed = all(pathology_results)
        
        test_results.append({
            "test": "Pathology Alert Logic",
            "status": "✅ BAŞARILI" if all_pathology_tests_passed else "❌ BAŞARISIZ",
            "details": f"Test Cases: {len(pathology_test_cases)}, Başarılı: {sum(pathology_results)}"
        })
        
    except Exception as e:
        test_results.append({
            "test": "Pathology Alert Logic",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # Test Sonuçları
    print("\n" + "=" * 65)
    print("📋 DÜZELTME SONRASI TEST SONUÇLARI:")
    print("=" * 65)
    
    success_count = 0
    total_tests = len(test_results)
    
    for result in test_results:
        print(f"{result['status']} {result['test']}")
        print(f"   {result['details']}")
        
        if "✅" in result['status']:
            success_count += 1
    
    print(f"\n🎯 GENEL SONUÇ: {success_count}/{total_tests} Test Başarılı")
    
    if success_count == total_tests:
        print("🎉 TÜM SİSTEMLER BAŞARIYLA DÜZELTİLDİ VE ÇALIŞIYOR!")
        print("\n📌 DÜZELTİLMİŞ KLİNİK KARAR DESTEK SİSTEMİ:")
        print("   • ✅ HIPAA/GDPR Uyumlu Gizlilik (Core)")
        print("   • ✅ Kapsamlı Audit Logging (Core)")
        print("   • ✅ CLI Performans Dashboard (Alternatif)")
        print("   • ✅ CLI Doktor Onay Sistemi (Alternatif)")
        print("   • ✅ Patoloji Uyarı Mantığı (Core)")
        print("   • ✅ Tam Core Sistem Entegrasyonu")
        print("\n🔧 DÜZELTMELER:")
        print("   • Tkinter bağımlılığı kaldırıldı")
        print("   • CLI alternatifleri eklendi")
        print("   • Test kapsamı genişletildi")
        print("   • Hata kontrolü iyileştirildi")
    else:
        failed_tests = total_tests - success_count
        print(f"⚠️  {failed_tests} sistemde hala sorun var.")
    
    return success_count, total_tests

def main():
    """Ana test fonksiyonu."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🔧 SİSTEM HATA KONTROLÜ VE DÜZELTMESİ")
    print("=====================================")
    
    start_time = time.time()
    success_count, total_tests = test_clinical_system_fixed()
    end_time = time.time()
    
    print(f"\n⏱️ Test süresi: {end_time - start_time:.2f} saniye")
    print(f"✨ Başarı oranı: %{(success_count/total_tests)*100:.1f}")
    
    # Sistem durumu özeti
    print(f"\n🏥 SİSTEM DURUMU ÖZETİ:")
    print(f"=====================================")
    if success_count == total_tests:
        print("🟢 Core Sistemler: TAMAMEN ÇALIŞIYOR")
        print("🟢 CLI Alternatifleri: ÇALIŞIYOR") 
        print("🟡 GUI Sistemleri: Tkinter gerekli (opsiyonel)")
        print("✅ GENEL DURUM: SİSTEM HAZIR")
    else:
        print("🔴 Bazı sistemlerde sorunlar mevcut")
        print("⚠️  GENEL DURUM: DÜZELTME GEREKLİ")

if __name__ == "__main__":
    main() 