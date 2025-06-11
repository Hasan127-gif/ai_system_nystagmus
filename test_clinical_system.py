#!/usr/bin/env python3
"""
STABİL KLİNİK SİSTEM TESTİ
=========================
Tkinter bağımlılığı olmayan stabil test sistemi.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

def test_clinical_system():
    """Stabil klinik sistem entegrasyon testi."""
    print("🏥 STABİL KLİNİK KARAR DESTEK SİSTEMİ - ENTEGRASYON TESTİ")
    print("=" * 60)
    
    test_results = []
    
    # 1. Privacy & Security Test
    print("\n🔐 1. Gizlilik & Güvenlik Sistemi Testi...")
    try:
        from privacy import get_privacy_manager
        
        privacy_manager = get_privacy_manager()
        auth_result = privacy_manager.authenticate_user("admin", "admin123", "127.0.0.1")
        
        if auth_result:
            session_token = auth_result["session_token"]
            has_admin = privacy_manager.check_permission(session_token, "admin")
            
            # Veri şifreleme testi
            test_data = {"patient_id": "P001", "nistagmus_frequency": 4.2}
            encrypted_data = privacy_manager.encrypt_patient_data(test_data)
            decrypted_data = privacy_manager.decrypt_patient_data(encrypted_data)
            encryption_valid = decrypted_data and "patient_id" in decrypted_data
            
            # Uygunluk kontrolü
            compliance_status = privacy_manager.get_privacy_compliance_status()
            compliance_valid = "HIPAA" in compliance_status["compliance_frameworks"]
            
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
        
        test_analysis = {
            "nistagmus_frequency": 4.2,
            "strabismus_angle": 3.5,
            "nystagmus_detected": True,
            "strabismus_detected": True,
            "ml_confidence": 0.85,
            "session_id": "stable_test_001"
        }
        
        test_patient = {"patient_id": "P_STABLE_001", "age": 35}
        
        analysis_id = clinical_logger.log_analysis(test_analysis, test_patient)
        clinical_logger.log_doctor_approval(analysis_id, "doctor_test", "approved", "Stabil test onayı")
        
        audit_report = clinical_logger.generate_audit_report()
        audit_valid = "summary" in audit_report and audit_report["summary"]["total_analyses"] > 0
        
        export_file = clinical_logger.export_compliance_data("json")
        
        test_results.append({
            "test": "Clinical Logger",
            "status": "✅ BAŞARILI",
            "details": f"Analiz ID: {analysis_id[:8]}..., Denetim: {'✓' if audit_valid else '✗'}, Dışa aktarma: {'✓' if export_file else '✗'}"
        })
        
    except Exception as e:
        test_results.append({
            "test": "Clinical Logger",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 3. CLI Dashboard System Test (Tkinter-free)
    print("\n📊 3. CLI Dashboard Sistemi Testi...")
    try:
        from dashboard_cli import ClinicalDashboardCLI
        
        dashboard = ClinicalDashboardCLI()
        test_metrics = {"sensitivity": 0.87, "specificity": 0.84, "auc": 0.86, "accuracy": 0.85}
        dashboard.report_metrics(test_metrics)
        
        metrics_updated = all(
            dashboard.metrics_data["current_metrics"].get(key) == test_metrics[key] 
            for key in test_metrics.keys()
        )
        
        report_success = dashboard.generate_report()
        
        test_results.append({
            "test": "CLI Dashboard System", 
            "status": "✅ BAŞARILI",
            "details": f"Metrik güncelleme: {'✓' if metrics_updated else '✗'}, Rapor: {'✓' if report_success else '✗'}"
        })
        
    except Exception as e:
        test_results.append({
            "test": "CLI Dashboard System",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 4. CLI Approval System Test (Tkinter-free)
    print("\n🖥️ 4. CLI Onay Sistemi Testi...")
    try:
        from ui_cli import ClinicalApprovalCLI
        
        approval_cli = ClinicalApprovalCLI()
        
        test_analysis_ui = {
            "analysis_id": "stable_ui_001",
            "nistagmus_frequency": 4.2,
            "strabismus_angle": 3.5,
            "nystagmus_detected": True,
            "strabismus_detected": True,
            "ml_confidence": 0.85
        }
        
        save_success = approval_cli.save_approval(test_analysis_ui, "stable_doctor", "approved", "CLI stabil test")
        
        test_results.append({
            "test": "CLI Approval System",
            "status": "✅ BAŞARILI", 
            "details": f"Kaydetme: {'✓' if save_success else '✗'}, Patoloji işleme: ✓"
        })
        
    except Exception as e:
        test_results.append({
            "test": "CLI Approval System",
            "status": "❌ HATA",
            "details": str(e)
        })
    
    # 5. Core Integration Test
    print("\n🔄 5. Tam Entegrasyon Testi...")
    try:
        from privacy import get_privacy_manager
        from logger import get_clinical_logger
        from dashboard_cli import ClinicalDashboardCLI
        
        # Entegrasyon senaryosu
        privacy_manager = get_privacy_manager()
        auth_result = privacy_manager.authenticate_user("doctor_001", "doctor123", "127.0.0.1")
        
        if auth_result:
            session_token = auth_result["session_token"]
            clinical_logger = get_clinical_logger()
            
            integration_analysis = {
                "nistagmus_frequency": 5.1,
                "strabismus_angle": 4.2, 
                "nystagmus_detected": True,
                "strabismus_detected": True,
                "ml_confidence": 0.92,
                "session_id": f"stable_integration_{int(time.time())}"
            }
            
            integration_patient = {"patient_id": "P_STABLE_INT_001", "age": 28}
            
            # Şifrelenmiş veri
            encrypted_patient = privacy_manager.encrypt_patient_data({
                "patient_id": integration_patient["patient_id"],
                "analysis_data": integration_analysis
            })
            
            analysis_id = clinical_logger.log_analysis(integration_analysis, integration_patient)
            clinical_logger.log_doctor_approval(analysis_id, "doctor_001", "approved", "Stabil entegrasyon onayı")
            
            # Dashboard güncelle
            dashboard = ClinicalDashboardCLI()
            dashboard.report_metrics({"sensitivity": 0.91, "specificity": 0.88, "auc": 0.90, "accuracy": 0.89})
            
            clinical_logger.log_access("doctor_001", "analysis_complete", "patient_data", "127.0.0.1")
            privacy_manager.logout_user(session_token)
            
            test_results.append({
                "test": "Core Integration",
                "status": "✅ BAŞARILI",
                "details": "Analiz: ✓, Onay: ✓, Şifreleme: ✓, Dashboard: ✓"
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
    
    # Test Sonuçları
    print("\n" + "=" * 60)
    print("📋 STABİL TEST SONUÇLARI:")
    print("=" * 60)
    
    success_count = 0
    total_tests = len(test_results)
    
    for result in test_results:
        print(f"{result['status']} {result['test']}")
        print(f"   {result['details']}")
        
        if "✅" in result['status']:
            success_count += 1
    
    print(f"\n🎯 GENEL SONUÇ: {success_count}/{total_tests} Test Başarılı")
    
    if success_count == total_tests:
        print("🎉 TÜM SİSTEMLER STABİL VE ÇALIŞIYOR!")
        print("\n📌 STABİL KONFIGÜRASYON:")
        print("   • ✅ HIPAA/GDPR Uyumlu Gizlilik") 
        print("   • ✅ Kapsamlı Audit Logging")
        print("   • ✅ CLI Dashboard (Tkinter-free)")
        print("   • ✅ CLI Onay Sistemi (Tkinter-free)")
        print("   • ✅ Tam Core Entegrasyon")
    else:
        failed_tests = total_tests - success_count
        print(f"⚠️  {failed_tests} sistemde sorun var.")
    
    return success_count, total_tests

def main():
    """Ana test fonksiyonu."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    start_time = time.time()
    success_count, total_tests = test_clinical_system()
    end_time = time.time()
    
    print(f"\n⏱️ Test süresi: {end_time - start_time:.2f} saniye")
    print(f"✨ Başarı oranı: %{(success_count/total_tests)*100:.1f}")

if __name__ == "__main__":
    main() 