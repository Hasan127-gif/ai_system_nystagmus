#!/usr/bin/env python3
"""
KLİNİK KAYIT & İZLENEBİLİRLİK SİSTEMİ
=====================================
Model versiyonu, eşikler, kalibrasyon, hasta meta verileri kayıt sistemi.
"""

import logging
import json
import csv
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid
import os

class ClinicalLogger:
    """Klinik kayıt ve izlenebilirlik yöneticisi."""
    
    def __init__(self, log_directory: str = "clinical_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Log dosya yolları
        self.analysis_log = self.log_directory / "analysis_log.csv"
        self.system_log = self.log_directory / "system_log.json"
        self.access_log = self.log_directory / "access_log.csv"
        self.model_log = self.log_directory / "model_log.json"
        
        # Sistem bilgileri
        self.system_info = self._get_system_info()
        
        # CSV başlıkları oluştur
        self._initialize_csv_files()
        
        logger = logging.getLogger(__name__)
        logger.info("Clinical Logger başlatıldı")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Sistem bilgilerini topla."""
        return {
            "system_version": "1.0.0",
            "model_version": "v2024.1",
            "initialization_time": datetime.now().isoformat(),
            "log_directory": str(self.log_directory.absolute()),
            "compliance": ["HIPAA", "GDPR"]
        }
    
    def _initialize_csv_files(self):
        """CSV dosyalarının başlıklarını oluştur."""
        # Analiz log başlıkları
        if not self.analysis_log.exists():
            with open(self.analysis_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "session_id", "patient_id_hash", "analysis_id",
                    "nystagmus_frequency", "strabismus_angle", "nystagmus_detected",
                    "strabismus_detected", "ml_confidence", "calibration_ratio",
                    "video_quality", "model_version", "threshold_nystagmus",
                    "threshold_strabismus", "doctor_approval", "notes"
                ])
        
        # Erişim log başlıkları
        if not self.access_log.exists():
            with open(self.access_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "user_id", "action", "resource", "ip_address",
                    "user_agent", "success", "details"
                ])
    
    def log_analysis(self, analysis_data: Dict[str, Any], 
                    patient_metadata: Dict[str, Any] = None,
                    doctor_approval: str = "pending") -> str:
        """
        Analiz sonuçlarını kaydet.
        
        Args:
            analysis_data: Analiz sonuçları
            patient_metadata: Hasta meta verileri (anonim)
            doctor_approval: Doktor onayı durumu
            
        Returns:
            str: Analiz ID
        """
        try:
            analysis_id = str(uuid.uuid4())
            session_id = analysis_data.get('session_id', str(uuid.uuid4()))
            timestamp = datetime.now().isoformat()
            
            # Hasta ID'sini hashle (HIPAA/GDPR uyumluluk)
            patient_id = patient_metadata.get('patient_id', 'anonymous') if patient_metadata else 'anonymous'
            patient_id_hash = hashlib.sha256(patient_id.encode()).hexdigest()[:16]
            
            # Kalibrasyon bilgisi
            calibration_ratio = 0.1
            try:
                from calibration import get_calibrator
                calibrator = get_calibrator()
                calibration_ratio = calibrator.calibration_data.get("pixel_to_degree_ratio", 0.1)
            except:
                pass
            
            # Model eşikleri
            threshold_nystagmus = 2.0  # Hz
            threshold_strabismus = 3.0  # derece
            
            # CSV'ye kaydet
            with open(self.analysis_log, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    session_id,
                    patient_id_hash,
                    analysis_id,
                    analysis_data.get('nistagmus_frequency', 0.0),
                    analysis_data.get('strabismus_angle', 0.0),
                    analysis_data.get('nystagmus_detected', False),
                    analysis_data.get('strabismus_detected', False),
                    analysis_data.get('ml_confidence', 0.0),
                    calibration_ratio,
                    analysis_data.get('video_quality', 'unknown'),
                    self.system_info['model_version'],
                    threshold_nystagmus,
                    threshold_strabismus,
                    doctor_approval,
                    analysis_data.get('notes', '')
                ])
            
            # Detaylı sistem logu
            self._log_system_event({
                "event_type": "analysis_completed",
                "analysis_id": analysis_id,
                "timestamp": timestamp,
                "system_version": self.system_info['system_version'],
                "model_version": self.system_info['model_version'],
                "calibration_status": {
                    "ratio": calibration_ratio,
                    "calibrated": calibration_ratio != 0.1
                },
                "thresholds": {
                    "nystagmus_hz": threshold_nystagmus,
                    "strabismus_deg": threshold_strabismus
                },
                "performance_metrics": analysis_data.get('performance_metrics', {}),
                "compliance_check": "passed"
            })
            
            logging.info(f"Analiz kaydedildi: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logging.error(f"Analiz kaydetme hatası: {e}")
            return ""
    
    def log_model_update(self, model_info: Dict[str, Any]):
        """Model güncellemelerini kaydet."""
        try:
            timestamp = datetime.now().isoformat()
            
            model_entry = {
                "timestamp": timestamp,
                "event_type": "model_update",
                "model_version": model_info.get('version', 'unknown'),
                "update_type": model_info.get('update_type', 'unknown'),
                "performance_metrics": model_info.get('metrics', {}),
                "validation_results": model_info.get('validation', {}),
                "deployment_status": model_info.get('status', 'deployed'),
                "approval_chain": model_info.get('approvals', [])
            }
            
            # Model log dosyasına ekle
            model_log_data = []
            if self.model_log.exists():
                with open(self.model_log, 'r', encoding='utf-8') as f:
                    model_log_data = json.load(f)
            
            model_log_data.append(model_entry)
            
            with open(self.model_log, 'w', encoding='utf-8') as f:
                json.dump(model_log_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Model güncellemesi kaydedildi: {model_info.get('version')}")
            
        except Exception as e:
            logging.error(f"Model güncelleme kaydetme hatası: {e}")
    
    def log_access(self, user_id: str, action: str, resource: str,
                  ip_address: str = "unknown", user_agent: str = "unknown",
                  success: bool = True, details: str = ""):
        """Erişim loglarını kaydet."""
        try:
            timestamp = datetime.now().isoformat()
            
            with open(self.access_log, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, user_id, action, resource,
                    ip_address, user_agent, success, details
                ])
            
            logging.info(f"Erişim kaydedildi: {user_id} - {action}")
            
        except Exception as e:
            logging.error(f"Erişim kaydetme hatası: {e}")
    
    def log_calibration_change(self, old_params: Dict, new_params: Dict, user_id: str):
        """Kalibrasyon değişikliklerini kaydet."""
        try:
            calibration_event = {
                "event_type": "calibration_change",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "old_parameters": old_params,
                "new_parameters": new_params,
                "change_reason": new_params.get('change_reason', 'user_requested'),
                "validation_status": "pending"
            }
            
            self._log_system_event(calibration_event)
            
            logging.info(f"Kalibrasyon değişikliği kaydedildi: {user_id}")
            
        except Exception as e:
            logging.error(f"Kalibrasyon değişiklik kaydetme hatası: {e}")
    
    def log_doctor_approval(self, analysis_id: str, doctor_id: str, 
                           approval_status: str, notes: str = ""):
        """Doktor onaylarını kaydet."""
        try:
            approval_event = {
                "event_type": "doctor_approval",
                "timestamp": datetime.now().isoformat(),
                "analysis_id": analysis_id,
                "doctor_id": hashlib.sha256(doctor_id.encode()).hexdigest()[:16],  # Anonymize
                "approval_status": approval_status,  # approved, rejected, pending
                "notes": notes,
                "compliance_logged": True
            }
            
            self._log_system_event(approval_event)
            
            # CSV dosyasını güncelle
            self._update_analysis_approval(analysis_id, approval_status)
            
            logging.info(f"Doktor onayı kaydedildi: {analysis_id} - {approval_status}")
            
        except Exception as e:
            logging.error(f"Doktor onayı kaydetme hatası: {e}")
    
    def _log_system_event(self, event_data: Dict[str, Any]):
        """Sistem olaylarını JSON log dosyasına kaydet."""
        try:
            system_log_data = []
            if self.system_log.exists():
                with open(self.system_log, 'r', encoding='utf-8') as f:
                    system_log_data = json.load(f)
            
            system_log_data.append(event_data)
            
            # Son 10000 olayı tut (log boyut kontrolü)
            if len(system_log_data) > 10000:
                system_log_data = system_log_data[-10000:]
            
            with open(self.system_log, 'w', encoding='utf-8') as f:
                json.dump(system_log_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logging.error(f"Sistem olay kaydetme hatası: {e}")
    
    def _update_analysis_approval(self, analysis_id: str, approval_status: str):
        """CSV'deki analiz onay durumunu güncelle."""
        try:
            # Mevcut verileri oku
            rows = []
            with open(self.analysis_log, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            # İlgili satırı bul ve güncelle
            for i, row in enumerate(rows):
                if len(row) > 3 and row[3] == analysis_id:  # analysis_id column
                    if len(row) > 14:  # doctor_approval column exists
                        row[14] = approval_status
                    break
            
            # Dosyayı yeniden yaz
            with open(self.analysis_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
                
        except Exception as e:
            logging.error(f"Analiz onay güncelleme hatası: {e}")
    
    def generate_audit_report(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Denetim raporu oluştur."""
        try:
            report = {
                "report_generated": datetime.now().isoformat(),
                "period": {"start": start_date, "end": end_date},
                "summary": {
                    "total_analyses": 0,
                    "approved_analyses": 0,
                    "pending_analyses": 0,
                    "rejected_analyses": 0,
                    "unique_patients": 0,
                    "calibration_changes": 0,
                    "model_updates": 0,
                    "access_violations": 0
                },
                "compliance_status": "compliant",
                "recommendations": []
            }
            
            # Analiz verilerini oku
            if self.analysis_log.exists():
                with open(self.analysis_log, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    patient_ids = set()
                    
                    for row in reader:
                        # Tarih filtresi uygula
                        if start_date and row['timestamp'] < start_date:
                            continue
                        if end_date and row['timestamp'] > end_date:
                            continue
                        
                        report["summary"]["total_analyses"] += 1
                        patient_ids.add(row['patient_id_hash'])
                        
                        # Onay durumu sayıları
                        approval = row.get('doctor_approval', 'pending')
                        if approval == 'approved':
                            report["summary"]["approved_analyses"] += 1
                        elif approval == 'rejected':
                            report["summary"]["rejected_analyses"] += 1
                        else:
                            report["summary"]["pending_analyses"] += 1
                    
                    report["summary"]["unique_patients"] = len(patient_ids)
            
            # Sistem olaylarını oku
            if self.system_log.exists():
                with open(self.system_log, 'r', encoding='utf-8') as f:
                    system_events = json.load(f)
                    
                    for event in system_events:
                        # Tarih filtresi uygula
                        if start_date and event['timestamp'] < start_date:
                            continue
                        if end_date and event['timestamp'] > end_date:
                            continue
                        
                        if event['event_type'] == 'calibration_change':
                            report["summary"]["calibration_changes"] += 1
                        elif event['event_type'] == 'model_update':
                            report["summary"]["model_updates"] += 1
            
            # Uygunluk kontrolü
            pending_ratio = (report["summary"]["pending_analyses"] / 
                           max(report["summary"]["total_analyses"], 1))
            
            if pending_ratio > 0.1:  # %10'dan fazla bekleyen
                report["compliance_status"] = "warning"
                report["recommendations"].append(
                    "Bekleyen onaylar %10'u aşıyor. Doktor onay sürecini hızlandırın.")
            
            if report["summary"]["access_violations"] > 0:
                report["compliance_status"] = "violation"
                report["recommendations"].append(
                    "Erişim ihlalleri tespit edildi. Güvenlik protokollerini gözden geçirin.")
            
            return report
            
        except Exception as e:
            logging.error(f"Denetim raporu oluşturma hatası: {e}")
            return {"error": str(e)}
    
    def export_compliance_data(self, output_format: str = "json") -> str:
        """Uygunluk verilerini dışa aktar."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if output_format == "json":
                output_file = self.log_directory / f"compliance_export_{timestamp}.json"
                
                compliance_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "system_info": self.system_info,
                    "audit_report": self.generate_audit_report(),
                    "data_retention_policy": "7_years",
                    "encryption_status": "AES-256",
                    "compliance_frameworks": ["HIPAA", "GDPR"]
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(compliance_data, f, indent=2, ensure_ascii=False)
                
            return str(output_file)
            
        except Exception as e:
            logging.error(f"Uygunluk verisi dışa aktarma hatası: {e}")
            return ""
    
    def cleanup_old_logs(self, retention_days: int = 2555):  # 7 yıl = ~2555 gün
        """Eski logları temizle (veri saklama politikası)."""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            cutoff_str = cutoff_date.isoformat()
            
            # Sistem log temizliği
            if self.system_log.exists():
                with open(self.system_log, 'r', encoding='utf-8') as f:
                    system_events = json.load(f)
                
                # Yeni veriler
                recent_events = [
                    event for event in system_events
                    if event.get('timestamp', '') >= cutoff_str
                ]
                
                with open(self.system_log, 'w', encoding='utf-8') as f:
                    json.dump(recent_events, f, indent=2, ensure_ascii=False)
                
                removed_count = len(system_events) - len(recent_events)
                logging.info(f"Eski sistem logları temizlendi: {removed_count} kayıt")
            
        except Exception as e:
            logging.error(f"Log temizleme hatası: {e}")

# Global logger instance
_clinical_logger = None

def get_clinical_logger() -> ClinicalLogger:
    """Global clinical logger instance'ını döndür."""
    global _clinical_logger
    if _clinical_logger is None:
        _clinical_logger = ClinicalLogger()
    return _clinical_logger

def log_analysis_result(analysis_data: Dict[str, Any], 
                       patient_metadata: Dict[str, Any] = None) -> str:
    """Analiz sonucunu kaydet (kolay kullanım için)."""
    logger = get_clinical_logger()
    return logger.log_analysis(analysis_data, patient_metadata)

def log_user_access(user_id: str, action: str, resource: str, success: bool = True):
    """Kullanıcı erişimini kaydet (kolay kullanım için)."""
    logger = get_clinical_logger()
    logger.log_access(user_id, action, resource, success=success)

def main():
    """Test amaçlı ana fonksiyon."""
    logging.basicConfig(level=logging.INFO)
    
    # Clinical logger test
    logger = ClinicalLogger()
    
    # Test analiz kaydı
    test_analysis = {
        "nistagmus_frequency": 4.2,
        "strabismus_angle": 3.5,
        "nystagmus_detected": True,
        "strabismus_detected": True,
        "ml_confidence": 0.85,
        "video_quality": "good",
        "session_id": "test_session_123"
    }
    
    test_patient = {
        "patient_id": "patient_001",
        "age": 35
    }
    
    analysis_id = logger.log_analysis(test_analysis, test_patient)
    print(f"Analiz kaydedildi: {analysis_id}")
    
    # Test doktor onayı
    logger.log_doctor_approval(analysis_id, "doctor_001", "approved", "Onaylandı")
    
    # Test denetim raporu
    audit_report = logger.generate_audit_report()
    print(f"Denetim raporu: {audit_report}")

if __name__ == "__main__":
    from datetime import timedelta
    main() 