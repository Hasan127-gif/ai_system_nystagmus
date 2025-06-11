#!/usr/bin/env python3
"""
KAPSAMLI SİSTEM TESTİ (A-Z)
==========================
Tüm modülleri derinlemesine test eder ve hataları raporlar.
"""

import sys
import os
import time
import traceback
import json
from datetime import datetime
from typing import Dict, List, Any

class ComprehensiveSystemTester:
    """Kapsamlı sistem test sınıfı."""
    
    def __init__(self):
        self.test_results = []
        self.critical_errors = []
        self.warnings = []
        self.performance_metrics = {}
        
    def log_test(self, component: str, test_name: str, status: str, details: str, error_level: str = "INFO"):
        """Test sonucunu loglar."""
        result = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "test_name": test_name,
            "status": status,
            "details": details,
            "error_level": error_level
        }
        
        self.test_results.append(result)
        
        if error_level == "CRITICAL":
            self.critical_errors.append(result)
        elif error_level == "WARNING":
            self.warnings.append(result)
            
        print(f"{'🔴' if status == 'FAIL' else '🟡' if status == 'WARN' else '🟢'} {component}::{test_name} - {status}")
        if error_level == "CRITICAL":
            print(f"   💥 KRİTİK: {details}")
        elif details:
            print(f"   📋 {details}")
    
    def test_1_imports_and_dependencies(self):
        """1. Import ve bağımlılık testleri."""
        print("\n🔍 1. IMPORT VE BAĞIMLILIK TESTLERİ")
        print("=" * 50)
        
        required_modules = [
            ("cv2", "OpenCV"),
            ("mediapipe", "MediaPipe"),
            ("numpy", "NumPy"),
            ("cryptography", "Cryptography"),
            ("matplotlib", "Matplotlib"),
            ("sklearn", "Scikit-learn"),
            ("torch", "PyTorch")
        ]
        
        for module_name, display_name in required_modules:
            try:
                __import__(module_name)
                self.log_test("Dependencies", f"Import {display_name}", "PASS", f"{display_name} başarıyla yüklendi")
            except ImportError as e:
                self.log_test("Dependencies", f"Import {display_name}", "FAIL", 
                            f"{display_name} yüklenemedi: {e}", "CRITICAL")
        
        # Opsiyonel modüller
        optional_modules = [("tkinter", "Tkinter"), ("flask", "Flask")]
        for module_name, display_name in optional_modules:
            try:
                __import__(module_name)
                self.log_test("Dependencies", f"Optional {display_name}", "PASS", f"{display_name} mevcut")
            except ImportError:
                if module_name == "tkinter":
                    self.log_test("Dependencies", f"Optional {display_name}", "INFO", 
                                f"{display_name} eksik (GUI için gerekli - opsiyonel)", "INFO")
                else:
                    self.log_test("Dependencies", f"Optional {display_name}", "WARN", 
                                f"{display_name} eksik", "WARNING")
    
    def test_2_core_modules(self):
        """2. Core modül testleri."""
        print("\n🔍 2. CORE MODÜL TESTLERİ")
        print("=" * 50)
        
        core_modules = ["detector", "features", "model", "calibration", "validation", 
                       "privacy", "logger", "dashboard_cli", "ui_cli"]
        
        for module_name in core_modules:
            try:
                module = __import__(module_name)
                self.log_test("Core Modules", f"Import {module_name}", "PASS", "Başarıyla yüklendi")
                
                # Modül içindeki temel sınıfları kontrol et
                if module_name == "detector":
                    if hasattr(module, "NystagmusDetector"):
                        self.log_test("Core Modules", f"{module_name} Class Check", "PASS", "NystagmusDetector sınıfı mevcut")
                    else:
                        self.log_test("Core Modules", f"{module_name} Class Check", "FAIL", 
                                    "NystagmusDetector sınıfı bulunamadı", "CRITICAL")
                
                elif module_name == "privacy":
                    if hasattr(module, "PrivacyManager"):
                        self.log_test("Core Modules", f"{module_name} Class Check", "PASS", "PrivacyManager sınıfı mevcut")
                    else:
                        self.log_test("Core Modules", f"{module_name} Class Check", "FAIL", 
                                    "PrivacyManager sınıfı bulunamadı", "CRITICAL")
                        
            except Exception as e:
                self.log_test("Core Modules", f"Import {module_name}", "FAIL", 
                            f"Yüklenemedi: {str(e)}", "CRITICAL")
    
    def test_3_detector_functionality(self):
        """3. Detector fonksiyonalite testleri."""
        print("\n🔍 3. DETECTOR FONKSİYONALİTE TESTLERİ")
        print("=" * 50)
        
        try:
            from detector import NystagmusDetector
            
            # Detector başlatma testi
            try:
                detector = NystagmusDetector()
                self.log_test("Detector", "Initialization", "PASS", "Detector başarıyla başlatıldı")
            except Exception as e:
                self.log_test("Detector", "Initialization", "FAIL", 
                            f"Detector başlatılamadı: {e}", "CRITICAL")
                return
            
            # MediaPipe modeli kontrolü
            try:
                if detector.face_mesh is not None:
                    self.log_test("Detector", "MediaPipe Model", "PASS", "MediaPipe face mesh yüklendi")
                else:
                    self.log_test("Detector", "MediaPipe Model", "FAIL", 
                                "MediaPipe face mesh yüklenemedi", "CRITICAL")
            except Exception as e:
                self.log_test("Detector", "MediaPipe Model", "FAIL", 
                            f"MediaPipe hatası: {e}", "CRITICAL")
            
            # Test videosu varsa analiz testi
            if os.path.exists("test_clinical_video.mp4"):
                try:
                    start_time = time.time()
                    result = detector.analyze_video("test_clinical_video.mp4", max_frames=10)
                    end_time = time.time()
                    
                    if result and "error" not in result:
                        self.log_test("Detector", "Video Analysis", "PASS", 
                                    f"Video analizi başarılı ({end_time-start_time:.2f}s)")
                        self.performance_metrics["video_analysis_time"] = end_time - start_time
                        
                        # Sonuç yapısını kontrol et
                        required_fields = ["processed_frames", "face_detected_frames", "analysis_complete"]
                        for field in required_fields:
                            if field in result:
                                self.log_test("Detector", f"Result Field {field}", "PASS", f"{field} mevcut")
                            else:
                                self.log_test("Detector", f"Result Field {field}", "FAIL", 
                                            f"{field} eksik", "WARNING")
                    else:
                        self.log_test("Detector", "Video Analysis", "FAIL", 
                                    f"Video analizi başarısız: {result.get('error', 'Bilinmeyen hata')}", "CRITICAL")
                except Exception as e:
                    self.log_test("Detector", "Video Analysis", "FAIL", 
                                f"Video analiz hatası: {e}", "CRITICAL")
            else:
                self.log_test("Detector", "Test Video", "WARN", 
                            "Test videosu bulunamadı", "WARNING")
                
        except ImportError as e:
            self.log_test("Detector", "Module Import", "FAIL", 
                        f"Detector modülü yüklenemedi: {e}", "CRITICAL")
    
    def test_4_privacy_security(self):
        """4. Gizlilik ve güvenlik testleri."""
        print("\n🔍 4. GİZLİLİK VE GÜVENLİK TESTLERİ")
        print("=" * 50)
        
        try:
            from privacy import get_privacy_manager
            
            privacy_manager = get_privacy_manager()
            
            # Şifreleme testleri
            try:
                test_data = {"patient_id": "TEST001", "sensitive_data": "test bilgisi"}
                encrypted = privacy_manager.encrypt_patient_data(test_data)
                decrypted = privacy_manager.decrypt_patient_data(encrypted)
                
                if decrypted == test_data:
                    self.log_test("Privacy", "Encryption/Decryption", "PASS", "Şifreleme/çözme başarılı")
                else:
                    self.log_test("Privacy", "Encryption/Decryption", "FAIL", 
                                "Şifreleme/çözme tutarsız", "CRITICAL")
            except Exception as e:
                self.log_test("Privacy", "Encryption/Decryption", "FAIL", 
                            f"Şifreleme hatası: {e}", "CRITICAL")
            
            # Authentication testleri
            try:
                # Geçersiz kullanıcı
                auth_result = privacy_manager.authenticate_user("invalid_user", "wrong_pass", "127.0.0.1")
                if auth_result is None:
                    self.log_test("Privacy", "Invalid Auth", "PASS", "Geçersiz giriş reddedildi")
                else:
                    self.log_test("Privacy", "Invalid Auth", "FAIL", 
                                "Geçersiz giriş kabul edildi", "CRITICAL")
                
                # Geçerli kullanıcı
                auth_result = privacy_manager.authenticate_user("admin", "admin123", "127.0.0.1")
                if auth_result and "session_token" in auth_result:
                    self.log_test("Privacy", "Valid Auth", "PASS", "Geçerli giriş başarılı")
                    
                    # Session validation
                    session_valid = privacy_manager.validate_session(auth_result["session_token"])
                    if session_valid:
                        self.log_test("Privacy", "Session Validation", "PASS", "Session doğrulama başarılı")
                    else:
                        self.log_test("Privacy", "Session Validation", "FAIL", 
                                    "Session doğrulama başarısız", "WARNING")
                    
                    privacy_manager.logout_user(auth_result["session_token"])
                else:
                    self.log_test("Privacy", "Valid Auth", "FAIL", 
                                "Geçerli giriş başarısız", "CRITICAL")
            except Exception as e:
                self.log_test("Privacy", "Authentication", "FAIL", 
                            f"Authentication hatası: {e}", "CRITICAL")
            
            # HIPAA/GDPR compliance kontrolü
            try:
                compliance = privacy_manager.get_privacy_compliance_status()
                if "HIPAA" in compliance.get("compliance_frameworks", []):
                    self.log_test("Privacy", "HIPAA Compliance", "PASS", "HIPAA uyumluluğu aktif")
                else:
                    self.log_test("Privacy", "HIPAA Compliance", "FAIL", 
                                "HIPAA uyumluluğu eksik", "CRITICAL")
            except Exception as e:
                self.log_test("Privacy", "Compliance Check", "FAIL", 
                            f"Compliance kontrol hatası: {e}", "WARNING")
                
        except ImportError as e:
            self.log_test("Privacy", "Module Import", "FAIL", 
                        f"Privacy modülü yüklenemedi: {e}", "CRITICAL")
    
    def test_5_logging_system(self):
        """5. Logging sistem testleri."""
        print("\n🔍 5. LOGGING SİSTEM TESTLERİ")
        print("=" * 50)
        
        try:
            from logger import get_clinical_logger
            
            clinical_logger = get_clinical_logger()
            
            # Log yazma testi
            try:
                test_analysis = {
                    "nistagmus_frequency": 3.5,
                    "strabismus_angle": 2.1,
                    "nystagmus_detected": True,
                    "test_id": "comprehensive_test"
                }
                
                analysis_id = clinical_logger.log_analysis(test_analysis)
                if analysis_id:
                    self.log_test("Logger", "Analysis Logging", "PASS", f"Analiz kaydı oluşturuldu: {analysis_id[:8]}...")
                else:
                    self.log_test("Logger", "Analysis Logging", "FAIL", 
                                "Analiz kaydı oluşturulamadı", "CRITICAL")
            except Exception as e:
                self.log_test("Logger", "Analysis Logging", "FAIL", 
                            f"Analiz kayıt hatası: {e}", "CRITICAL")
            
            # Audit raporu testi
            try:
                audit_report = clinical_logger.generate_audit_report()
                if "summary" in audit_report and "total_analyses" in audit_report["summary"]:
                    self.log_test("Logger", "Audit Report", "PASS", 
                                f"Audit raporu oluşturuldu (Toplam: {audit_report['summary']['total_analyses']})")
                else:
                    self.log_test("Logger", "Audit Report", "FAIL", 
                                "Audit raporu yapısı hatalı", "WARNING")
            except Exception as e:
                self.log_test("Logger", "Audit Report", "FAIL", 
                            f"Audit rapor hatası: {e}", "WARNING")
            
            # Data export testi
            try:
                export_result = clinical_logger.export_compliance_data("json")
                if export_result:
                    self.log_test("Logger", "Data Export", "PASS", "Veri dışa aktarma başarılı")
                else:
                    self.log_test("Logger", "Data Export", "FAIL", 
                                "Veri dışa aktarma başarısız", "WARNING")
            except Exception as e:
                self.log_test("Logger", "Data Export", "FAIL", 
                            f"Veri dışa aktarma hatası: {e}", "WARNING")
                
        except ImportError as e:
            self.log_test("Logger", "Module Import", "FAIL", 
                        f"Logger modülü yüklenemedi: {e}", "CRITICAL")
    
    def test_6_calibration_system(self):
        """6. Kalibrasyon sistem testleri."""
        print("\n🔍 6. KALİBRASYON SİSTEM TESTLERİ")
        print("=" * 50)
        
        try:
            from calibration import get_calibrator
            
            calibrator = get_calibrator()
            
            # Kalibrasyon durumu kontrolü
            try:
                status = calibrator.get_calibration_status()
                if "calibrated" in status:
                    if status["calibrated"]:
                        self.log_test("Calibration", "Status Check", "PASS", 
                                    f"Kalibrasyon aktif (Oran: {status.get('pixel_to_degree_ratio', 'N/A')})")
                    else:
                        self.log_test("Calibration", "Status Check", "WARN", 
                                    "Kalibrasyon gerekli", "WARNING")
                else:
                    self.log_test("Calibration", "Status Check", "FAIL", 
                                "Kalibrasyon durumu okunamadı", "WARNING")
            except Exception as e:
                self.log_test("Calibration", "Status Check", "FAIL", 
                            f"Kalibrasyon durum hatası: {e}", "WARNING")
            
            # Kalibrasyon parametreleri testi
            try:
                if hasattr(calibrator, 'calibration_data') and calibrator.calibration_data:
                    required_params = ['pixel_to_degree_ratio', 'camera_matrix', 'reference_points']
                    for param in required_params:
                        if param in calibrator.calibration_data:
                            self.log_test("Calibration", f"Parameter {param}", "PASS", f"{param} mevcut")
                        else:
                            self.log_test("Calibration", f"Parameter {param}", "WARN", 
                                        f"{param} eksik", "WARNING")
                else:
                    self.log_test("Calibration", "Calibration Data", "WARN", 
                                "Kalibrasyon verisi yok", "WARNING")
            except Exception as e:
                self.log_test("Calibration", "Parameter Check", "FAIL", 
                            f"Parametre kontrol hatası: {e}", "WARNING")
                
        except ImportError as e:
            self.log_test("Calibration", "Module Import", "FAIL", 
                        f"Calibration modülü yüklenemedi: {e}", "CRITICAL")
    
    def test_7_ml_model_system(self):
        """7. ML model sistem testleri."""
        print("\n🔍 7. ML MODEL SİSTEM TESTLERİ")
        print("=" * 50)
        
        try:
            from model import NystagmusClassifier
            
            # Model başlatma
            try:
                classifier = NystagmusClassifier()
                self.log_test("ML Model", "Initialization", "PASS", "ML sınıflandırıcı başlatıldı")
            except Exception as e:
                self.log_test("ML Model", "Initialization", "FAIL", 
                            f"ML sınıflandırıcı başlatılamadı: {e}", "WARNING")
                return
            
            # Test verisi ile classification
            try:
                test_features = {
                    "nystagmus_frequency": 4.2,
                    "movement_amplitude": 0.85,
                    "movement_regularity": 0.72,
                    "strabismus_angle": 3.1,
                    "strabismus_stability": 0.68
                }
                
                result = classifier.classify(test_features)
                # Sonuç yapısını kontrol et - hem predictions hem de doğrudan nystagmus_detected
                if result and ("nystagmus_detected" in result or 
                              ("predictions" in result and "nystagmus_detected" in result["predictions"])):
                    nystagmus_detected = result.get("nystagmus_detected") or result.get("predictions", {}).get("nystagmus_detected", False)
                    self.log_test("ML Model", "Classification", "PASS", 
                                f"Sınıflandırma başarılı (Nistagmus: {nystagmus_detected})")
                else:
                    self.log_test("ML Model", "Classification", "FAIL", 
                                "Sınıflandırma sonucu hatalı", "WARNING")
            except Exception as e:
                self.log_test("ML Model", "Classification", "FAIL", 
                            f"Sınıflandırma hatası: {e}", "WARNING")
            
            # Feature extraction testi
            try:
                from features import extract_features
                
                # Dummy data
                dummy_landmarks = [[0.5, 0.5, 0.0] for _ in range(468)]  # MediaPipe landmark count
                features = extract_features([dummy_landmarks] * 10)  # 10 frame
                
                if features and len(features) == 5:  # 5 feature beklenyor
                    self.log_test("ML Model", "Feature Extraction", "PASS", 
                                f"Feature extraction başarılı ({len(features)} özellik)")
                else:
                    self.log_test("ML Model", "Feature Extraction", "FAIL", 
                                "Feature extraction hatalı", "WARNING")
            except ImportError:
                self.log_test("ML Model", "Feature Module", "FAIL", 
                            "Features modülü yüklenemedi", "WARNING")
            except Exception as e:
                self.log_test("ML Model", "Feature Extraction", "FAIL", 
                            f"Feature extraction hatası: {e}", "WARNING")
                
        except ImportError as e:
            self.log_test("ML Model", "Module Import", "FAIL", 
                        f"Model modülü yüklenemedi: {e}", "WARNING")
    
    def test_8_ui_systems(self):
        """8. UI sistem testleri."""
        print("\n🔍 8. UI SİSTEM TESTLERİ")
        print("=" * 50)
        
        # CLI Dashboard testi
        try:
            from dashboard_cli import ClinicalDashboardCLI
            
            dashboard = ClinicalDashboardCLI()
            
            # Metrik güncelleme testi
            test_metrics = {"sensitivity": 0.85, "specificity": 0.82}
            dashboard.report_metrics(test_metrics)
            
            if dashboard.metrics_data["current_metrics"]["sensitivity"] == 0.85:
                self.log_test("UI Systems", "CLI Dashboard Metrics", "PASS", "CLI dashboard metrik güncelleme başarılı")
            else:
                self.log_test("UI Systems", "CLI Dashboard Metrics", "FAIL", 
                            "CLI dashboard metrik güncelleme başarısız", "WARNING")
                            
        except ImportError as e:
            self.log_test("UI Systems", "CLI Dashboard Import", "FAIL", 
                        f"CLI Dashboard yüklenemedi: {e}", "WARNING")
        except Exception as e:
            self.log_test("UI Systems", "CLI Dashboard", "FAIL", 
                        f"CLI Dashboard hatası: {e}", "WARNING")
        
        # CLI Approval testi
        try:
            from ui_cli import ClinicalApprovalCLI
            
            approval_cli = ClinicalApprovalCLI()
            
            test_analysis = {
                "analysis_id": "ui_test_001",
                "nystagmus_detected": True,
                "nistagmus_frequency": 4.0
            }
            
            # Kaydetme fonksiyonu testi (approval UI olmadan)
            save_result = approval_cli.save_approval(test_analysis, "test_doctor", "approved", "Test onayı")
            
            if save_result:
                self.log_test("UI Systems", "CLI Approval Save", "PASS", "CLI approval kaydetme başarılı")
            else:
                self.log_test("UI Systems", "CLI Approval Save", "FAIL", 
                            "CLI approval kaydetme başarısız", "WARNING")
                            
        except ImportError as e:
            self.log_test("UI Systems", "CLI Approval Import", "FAIL", 
                        f"CLI Approval yüklenemedi: {e}", "WARNING")
        except Exception as e:
            self.log_test("UI Systems", "CLI Approval", "FAIL", 
                        f"CLI Approval hatası: {e}", "WARNING")
        
        # Web Dashboard testi
        try:
            from web_dashboard import WebDashboard
            
            web_dashboard = WebDashboard(port=5001)  # Farklı port
            
            if web_dashboard.app is not None:
                self.log_test("UI Systems", "Web Dashboard Init", "PASS", "Web dashboard başlatıldı")
            else:
                self.log_test("UI Systems", "Web Dashboard Init", "WARN", 
                            "Web dashboard Flask eksik", "WARNING")
                            
        except ImportError as e:
            self.log_test("UI Systems", "Web Dashboard Import", "WARN", 
                        f"Web Dashboard yüklenemedi: {e}", "WARNING")
    
    def test_9_file_system_integrity(self):
        """9. Dosya sistemi bütünlük testleri."""
        print("\n🔍 9. DOSYA SİSTEMİ BÜTÜNLÜK TESTLERİ")
        print("=" * 50)
        
        critical_files = [
            "detector.py", "features.py", "model.py", "calibration.py",
            "validation.py", "privacy.py", "logger.py"
        ]
        
        important_files = [
            "dashboard_cli.py", "ui_cli.py", "web_dashboard.py", "run_system.py"
        ]
        
        # Kritik dosyalar kontrolü
        for file_name in critical_files:
            if os.path.exists(file_name):
                file_size = os.path.getsize(file_name)
                if file_size > 100:  # En az 100 byte
                    self.log_test("File System", f"Critical File {file_name}", "PASS", 
                                f"Dosya mevcut ({file_size} bytes)")
                else:
                    self.log_test("File System", f"Critical File {file_name}", "FAIL", 
                                "Dosya çok küçük", "CRITICAL")
            else:
                self.log_test("File System", f"Critical File {file_name}", "FAIL", 
                            "Dosya bulunamadı", "CRITICAL")
        
        # Önemli dosyalar kontrolü
        for file_name in important_files:
            if os.path.exists(file_name):
                self.log_test("File System", f"Important File {file_name}", "PASS", "Dosya mevcut")
            else:
                self.log_test("File System", f"Important File {file_name}", "WARN", 
                            "Dosya bulunamadı", "WARNING")
        
        # Log dizinleri kontrolü
        log_dirs = ["clinical_logs", "audit_logs"]
        for dir_name in log_dirs:
            if os.path.exists(dir_name):
                self.log_test("File System", f"Log Directory {dir_name}", "PASS", "Dizin mevcut")
            else:
                self.log_test("File System", f"Log Directory {dir_name}", "WARN", 
                            "Dizin yok (otomatik oluşturulacak)", "WARNING")
    
    def test_10_performance_analysis(self):
        """10. Performans analiz testleri."""
        print("\n🔍 10. PERFORMANS ANALİZ TESTLERİ")
        print("=" * 50)
        
        # Memory usage test
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            if memory_mb < 500:  # 500MB altında
                self.log_test("Performance", "Memory Usage", "PASS", f"Bellek kullanımı: {memory_mb:.1f}MB")
            elif memory_mb < 1000:  # 1GB altında
                self.log_test("Performance", "Memory Usage", "WARN", 
                            f"Bellek kullanımı yüksek: {memory_mb:.1f}MB", "WARNING")
            else:
                self.log_test("Performance", "Memory Usage", "FAIL", 
                            f"Bellek kullanımı çok yüksek: {memory_mb:.1f}MB", "WARNING")
                            
            self.performance_metrics["memory_usage_mb"] = memory_mb
            
        except ImportError:
            self.log_test("Performance", "Memory Check", "WARN", 
                        "psutil modülü yok, bellek kontrolü yapılamadı", "WARNING")
        
        # Import speed test
        start_time = time.time()
        try:
            import detector
            import privacy
            import logger
            import_time = time.time() - start_time
            
            if import_time < 2.0:
                self.log_test("Performance", "Import Speed", "PASS", f"Import süresi: {import_time:.2f}s")
            else:
                self.log_test("Performance", "Import Speed", "WARN", 
                            f"Import süresi yavaş: {import_time:.2f}s", "WARNING")
                            
            self.performance_metrics["import_time"] = import_time
            
        except Exception as e:
            self.log_test("Performance", "Import Speed", "FAIL", 
                        f"Import hatası: {e}", "WARNING")
    
    def generate_comprehensive_report(self):
        """Kapsamlı test raporu oluşturur."""
        print("\n" + "="*70)
        print("📊 KAPSAMLI SİSTEM TEST RAPORU")
        print("="*70)
        
        total_tests = len(self.test_results)
        passed_tests = len([t for t in self.test_results if t["status"] == "PASS"])
        failed_tests = len([t for t in self.test_results if t["status"] == "FAIL"])
        warning_tests = len([t for t in self.test_results if t["status"] == "WARN"])
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📈 GENEL İSTATİSTİKLER:")
        print(f"• Toplam Test: {total_tests}")
        print(f"• Başarılı: {passed_tests} (🟢)")
        print(f"• Başarısız: {failed_tests} (🔴)")
        print(f"• Uyarı: {warning_tests} (🟡)")
        print(f"• Başarı Oranı: {success_rate:.1f}%")
        
        # Kritik hatalar
        if self.critical_errors:
            print(f"\n💥 KRİTİK HATALAR ({len(self.critical_errors)}):")
            for error in self.critical_errors:
                print(f"   🔴 {error['component']}::{error['test_name']} - {error['details']}")
        
        # Uyarılar
        if self.warnings:
            print(f"\n⚠️  UYARILAR ({len(self.warnings)}):")
            for warning in self.warnings[:5]:  # İlk 5 uyarı
                print(f"   🟡 {warning['component']}::{warning['test_name']} - {warning['details']}")
            if len(self.warnings) > 5:
                print(f"   ... ve {len(self.warnings)-5} uyarı daha")
        
        # Performans metrikleri
        if self.performance_metrics:
            print(f"\n⚡ PERFORMANS METRİKLERİ:")
            for metric, value in self.performance_metrics.items():
                if isinstance(value, float):
                    print(f"   • {metric}: {value:.2f}")
                else:
                    print(f"   • {metric}: {value}")
        
        # Çözüm önerileri
        self.generate_solution_recommendations()
        
        # Raporu dosyaya kaydet
        self.save_report_to_file()
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warning_tests,
            "success_rate": success_rate,
            "critical_errors": len(self.critical_errors)
        }
    
    def generate_solution_recommendations(self):
        """Çözüm önerileri oluşturur."""
        print(f"\n🔧 ÇÖZÜM ÖNERİLERİ:")
        
        # Kritik hatalar için çözümler
        critical_modules = set()
        for error in self.critical_errors:
            critical_modules.add(error['component'])
        
        if "Dependencies" in critical_modules:
            print("   1. Eksik bağımlılıklar:")
            print("      pip install opencv-python mediapipe torch scikit-learn")
        
        if "Detector" in critical_modules:
            print("   2. Detector sorunları:")
            print("      • MediaPipe kurulumunu kontrol edin")
            print("      • Test videosu test_clinical_video.mp4 ekleyin")
        
        if "Privacy" in critical_modules:
            print("   3. Güvenlik sorunları:")
            print("      • Cryptography modülünü güncelleyin")
            print("      • Şifreleme anahtarlarını yeniden oluşturun")
        
        # Performans iyileştirmeleri
        if "memory_usage_mb" in self.performance_metrics:
            if self.performance_metrics["memory_usage_mb"] > 500:
                print("   4. Performans iyileştirmeleri:")
                print("      • Kullanılmayan modülleri kaldırın")
                print("      • Video işleme frame sayısını azaltın")
        
        # Genel öneriler
        warning_count = len(self.warnings)
        if warning_count > 5:
            print("   5. Genel iyileştirmeler:")
            print("      • Opsiyonel modülleri kurun (Flask, Tkinter)")
            print("      • Log dizinlerini oluşturun")
            print("      • Kalibrasyon yapılandırması tamamlayın")
    
    def save_report_to_file(self):
        """Test raporunu dosyaya kaydeder."""
        try:
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "test_results": self.test_results,
                "critical_errors": self.critical_errors,
                "warnings": self.warnings,
                "performance_metrics": self.performance_metrics
            }
            
            with open("comprehensive_test_report.json", "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Detaylı rapor kaydedildi: comprehensive_test_report.json")
            
        except Exception as e:
            print(f"\n❌ Rapor kaydetme hatası: {e}")
    
    def run_all_tests(self):
        """Tüm testleri çalıştırır."""
        print("🔍 KAPSAMLI SİSTEM TESTİ BAŞLATILIYOR...")
        print("="*70)
        
        start_time = time.time()
        
        # Test sırası - dependency'den başlayarak
        self.test_1_imports_and_dependencies()
        self.test_2_core_modules()
        self.test_3_detector_functionality()
        self.test_4_privacy_security()
        self.test_5_logging_system()
        self.test_6_calibration_system()
        self.test_7_ml_model_system()
        self.test_8_ui_systems()
        self.test_9_file_system_integrity()
        self.test_10_performance_analysis()
        
        end_time = time.time()
        self.performance_metrics["total_test_time"] = end_time - start_time
        
        return self.generate_comprehensive_report()

def main():
    """Ana test fonksiyonu."""
    tester = ComprehensiveSystemTester()
    
    print("🏥 NİSTAGMUS TESPİT SİSTEMİ - KAPSAMLI TEST")
    print("Sistem A'dan Z'ye test ediliyor...")
    print()
    
    results = tester.run_all_tests()
    
    # Özet sonuç
    if results["critical_errors"] == 0:
        if results["success_rate"] >= 90:
            print("\n🎉 SİSTEM DURUMU: MÜKEMMEL")
        elif results["success_rate"] >= 80:
            print("\n✅ SİSTEM DURUMU: İYİ")
        else:
            print("\n⚠️  SİSTEM DURUMU: KABUL EDİLEBİLİR")
    else:
        print("\n🚨 SİSTEM DURUMU: KRİTİK SORUNLAR VAR")
    
    print(f"Test tamamlandı: {results['total_tests']} test, {results['success_rate']:.1f}% başarı")

if __name__ == "__main__":
    main() 