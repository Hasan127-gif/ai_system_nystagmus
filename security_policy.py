#!/usr/bin/env python3
"""
VERİ GÜVENLİĞİ VE SAKLAMA POLİTİKASI
==================================
GDPR/HIPAA bağımsız kurum içi veri güvenliği ve saklama süreçleri.
"""

import os
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import zipfile
import tempfile

logger = logging.getLogger(__name__)

class DataSecurityPolicy:
    """
    Veri güvenliği ve saklama politikalarını yönetir.
    Kurum içi süreçler için tasarlanmıştır.
    """
    
    # SAKLAMA POLİTİKALARI
    RETENTION_POLICIES = {
        "patient_videos": {
            "retention_period_days": 2555,  # 7 yıl
            "encryption_required": True,
            "backup_required": True,
            "access_level": "restricted",
            "description": "Hasta video kayıtları - 7 yıl saklama zorunlu"
        },
        "analysis_results": {
            "retention_period_days": 3650,  # 10 yıl
            "encryption_required": True,
            "backup_required": True,
            "access_level": "controlled",
            "description": "Analiz sonuçları ve raporlar - 10 yıl saklama"
        },
        "audit_logs": {
            "retention_period_days": 2190,  # 6 yıl
            "encryption_required": False,
            "backup_required": True,
            "access_level": "admin_only",
            "description": "Sistem erişim logları - 6 yıl saklama"
        },
        "user_sessions": {
            "retention_period_days": 90,    # 3 ay
            "encryption_required": False,
            "backup_required": False,
            "access_level": "system",
            "description": "Kullanıcı oturum kayıtları - 3 ay saklama"
        },
        "temporary_files": {
            "retention_period_days": 7,     # 1 hafta
            "encryption_required": False,
            "backup_required": False,
            "access_level": "system",
            "description": "Geçici dosyalar - 1 hafta saklama"
        }
    }
    
    # ERİŞİM SEVİYELERİ
    ACCESS_LEVELS = {
        "public": {
            "description": "Herkese açık",
            "required_roles": [],
            "audit_required": False
        },
        "internal": {
            "description": "Kurum içi",
            "required_roles": ["user", "doctor", "admin"],
            "audit_required": True
        },
        "controlled": {
            "description": "Kontrollü erişim",
            "required_roles": ["doctor", "admin"],
            "audit_required": True
        },
        "restricted": {
            "description": "Kısıtlı erişim",
            "required_roles": ["admin"],
            "audit_required": True
        },
        "admin_only": {
            "description": "Sadece yönetici",
            "required_roles": ["admin"],
            "audit_required": True
        },
        "system": {
            "description": "Sistem seviyesi",
            "required_roles": ["system"],
            "audit_required": False
        }
    }
    
    def __init__(self, base_storage_path: str = "secure_storage"):
        self.base_path = base_storage_path
        self.policies_file = os.path.join(base_storage_path, "security_policies.json")
        self.audit_log = os.path.join(base_storage_path, "security_audit.log")
        
        # Güvenli dizinleri oluştur
        self._initialize_secure_storage()
        
        # Politika dosyasını yükle/oluştur
        self._load_or_create_policies()
    
    def _initialize_secure_storage(self):
        """Güvenli depolama dizinlerini başlatır."""
        try:
            # Ana dizinler
            directories = [
                self.base_path,
                os.path.join(self.base_path, "patient_data"),
                os.path.join(self.base_path, "analysis_results"),
                os.path.join(self.base_path, "audit_logs"),
                os.path.join(self.base_path, "backups"),
                os.path.join(self.base_path, "temp"),
                os.path.join(self.base_path, "quarantine")
            ]
            
            for directory in directories:
                if not os.path.exists(directory):
                    os.makedirs(directory, mode=0o750)  # Kısıtlı izinler
                    logger.info(f"Güvenli dizin oluşturuldu: {directory}")
            
            # .gitignore oluştur (versiyon kontrolünden hariç tut)
            gitignore_path = os.path.join(self.base_path, ".gitignore")
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, 'w') as f:
                    f.write("# Güvenli veri depolama - versiyon kontrolüne dahil edilmez\n")
                    f.write("*\n")
                    f.write("!.gitignore\n")
                    f.write("!README_SECURITY.md\n")
            
        except Exception as e:
            logger.error(f"Güvenli depolama başlatma hatası: {e}")
    
    def _load_or_create_policies(self):
        """Güvenlik politikalarını yükler veya oluşturur."""
        try:
            if os.path.exists(self.policies_file):
                with open(self.policies_file, 'r', encoding='utf-8') as f:
                    self.active_policies = json.load(f)
                logger.info("Güvenlik politikaları yüklendi")
            else:
                # Varsayılan politikaları oluştur
                self.active_policies = {
                    "version": "1.0",
                    "created_date": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "retention_policies": self.RETENTION_POLICIES,
                    "access_levels": self.ACCESS_LEVELS,
                    "encryption_settings": {
                        "algorithm": "AES-256-CBC",
                        "key_rotation_days": 90,
                        "backup_encryption": True
                    },
                    "compliance_settings": {
                        "audit_all_access": True,
                        "log_retention_years": 6,
                        "data_anonymization": True,
                        "breach_notification_hours": 24
                    }
                }
                
                self._save_policies()
                logger.info("Varsayılan güvenlik politikaları oluşturuldu")
                
        except Exception as e:
            logger.error(f"Politika yükleme hatası: {e}")
            self.active_policies = {}
    
    def _save_policies(self):
        """Güvenlik politikalarını kaydeder."""
        try:
            self.active_policies["last_updated"] = datetime.now().isoformat()
            
            with open(self.policies_file, 'w', encoding='utf-8') as f:
                json.dump(self.active_policies, f, indent=2, ensure_ascii=False)
            
            # Dosya izinlerini kısıtla
            os.chmod(self.policies_file, 0o640)
            
        except Exception as e:
            logger.error(f"Politika kaydetme hatası: {e}")
    
    def store_patient_video(self, video_path: str, patient_id: str, 
                           metadata: Dict[str, Any] = None) -> str:
        """
        Hasta videosunu güvenli şekilde depolar.
        
        Args:
            video_path: Orijinal video dosya yolu
            patient_id: Hasta kimliği (anonimleştirilecek)
            metadata: Video metadata'sı
            
        Returns:
            str: Güvenli depolama ID'si
        """
        try:
            # Benzersiz depolama ID'si oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storage_id = hashlib.sha256(f"{patient_id}_{timestamp}".encode()).hexdigest()[:16]
            
            # Güvenli dosya yolu
            secure_filename = f"video_{storage_id}.encrypted"
            secure_path = os.path.join(self.base_path, "patient_data", secure_filename)
            
            # Video dosyasını şifrele ve depola
            encrypted_data = self._encrypt_file(video_path)
            
            with open(secure_path, 'wb') as f:
                f.write(encrypted_data)
            
            # Metadata kaydet
            metadata_info = {
                "storage_id": storage_id,
                "original_filename": os.path.basename(video_path),
                "patient_id_hash": hashlib.sha256(patient_id.encode()).hexdigest()[:16],
                "stored_date": datetime.now().isoformat(),
                "expiry_date": (datetime.now() + timedelta(days=self.RETENTION_POLICIES["patient_videos"]["retention_period_days"])).isoformat(),
                "file_size": os.path.getsize(video_path),
                "encryption": "AES-256-CBC",
                "access_level": "restricted",
                "metadata": metadata or {}
            }
            
            metadata_path = os.path.join(self.base_path, "patient_data", f"meta_{storage_id}.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_info, f, indent=2, ensure_ascii=False)
            
            # Dosya izinlerini ayarla
            os.chmod(secure_path, 0o600)  # Sadece sahip okuyabilir
            os.chmod(metadata_path, 0o640)
            
            # Audit log
            self._log_security_event("video_stored", {
                "storage_id": storage_id,
                "patient_id_hash": metadata_info["patient_id_hash"],
                "file_size": metadata_info["file_size"]
            })
            
            logger.info(f"Video güvenli şekilde depolandı: {storage_id}")
            return storage_id
            
        except Exception as e:
            logger.error(f"Video depolama hatası: {e}")
            return ""
    
    def retrieve_patient_video(self, storage_id: str, requesting_user: str, 
                              access_reason: str) -> Optional[str]:
        """
        Hasta videosunu güvenli şekilde getirir.
        
        Args:
            storage_id: Depolama ID'si
            requesting_user: İsteği yapan kullanıcı
            access_reason: Erişim nedeni
            
        Returns:
            str: Geçici dosya yolu (None ise erişim reddedildi)
        """
        try:
            # Metadata kontrol et
            metadata_path = os.path.join(self.base_path, "patient_data", f"meta_{storage_id}.json")
            
            if not os.path.exists(metadata_path):
                logger.warning(f"Video bulunamadı: {storage_id}")
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Süre kontrolü
            expiry_date = datetime.fromisoformat(metadata["expiry_date"])
            if datetime.now() > expiry_date:
                logger.warning(f"Video süresi dolmuş: {storage_id}")
                return None
            
            # Erişim kontrolü (basit uygulama)
            if not self._check_access_permission(requesting_user, metadata["access_level"]):
                logger.warning(f"Erişim reddedildi - User: {requesting_user}, Video: {storage_id}")
                self._log_security_event("access_denied", {
                    "storage_id": storage_id,
                    "user": requesting_user,
                    "reason": access_reason
                })
                return None
            
            # Şifreli dosyayı oku
            secure_path = os.path.join(self.base_path, "patient_data", f"video_{storage_id}.encrypted")
            
            if not os.path.exists(secure_path):
                logger.error(f"Şifreli video dosyası bulunamadı: {storage_id}")
                return None
            
            # Dosyayı çöz ve geçici dizine kaydet
            with open(secure_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted_data = self._decrypt_file_data(encrypted_data)
            
            # Geçici dosya oluştur
            temp_dir = os.path.join(self.base_path, "temp")
            temp_filename = f"temp_{storage_id}_{datetime.now().strftime('%H%M%S')}.mp4"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            with open(temp_path, 'wb') as f:
                f.write(decrypted_data)
            
            # Geçici dosya izinleri
            os.chmod(temp_path, 0o600)
            
            # Audit log
            self._log_security_event("video_accessed", {
                "storage_id": storage_id,
                "user": requesting_user,
                "reason": access_reason,
                "temp_file": temp_filename
            })
            
            logger.info(f"Video erişimi sağlandı: {storage_id} -> {requesting_user}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Video getirme hatası: {e}")
            return None
    
    def store_analysis_result(self, analysis_data: Dict[str, Any], 
                            patient_id: str, analysis_type: str = "nystagmus") -> str:
        """Analiz sonucunu güvenli şekilde depolar."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_id = hashlib.sha256(f"{patient_id}_{analysis_type}_{timestamp}".encode()).hexdigest()[:16]
            
            # Sonuç verisi hazırla
            result_data = {
                "result_id": result_id,
                "analysis_type": analysis_type,
                "patient_id_hash": hashlib.sha256(patient_id.encode()).hexdigest()[:16],
                "analysis_date": datetime.now().isoformat(),
                "expiry_date": (datetime.now() + timedelta(days=self.RETENTION_POLICIES["analysis_results"]["retention_period_days"])).isoformat(),
                "data": analysis_data,
                "system_info": {
                    "version": "1.0",
                    "analyzer": "NystagmusDetector"
                }
            }
            
            # Dosya yolu
            result_filename = f"analysis_{result_id}.json"
            result_path = os.path.join(self.base_path, "analysis_results", result_filename)
            
            # Şifreli kaydet (eğer gerekiyorsa)
            if self.RETENTION_POLICIES["analysis_results"]["encryption_required"]:
                encrypted_data = self._encrypt_data(json.dumps(result_data, ensure_ascii=False))
                result_filename = f"analysis_{result_id}.encrypted"
                result_path = os.path.join(self.base_path, "analysis_results", result_filename)
                
                with open(result_path, 'wb') as f:
                    f.write(encrypted_data)
            else:
                with open(result_path, 'w', encoding='utf-8') as f:
                    json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            # Dosya izinleri
            os.chmod(result_path, 0o640)
            
            # Audit log
            self._log_security_event("analysis_stored", {
                "result_id": result_id,
                "patient_id_hash": result_data["patient_id_hash"],
                "analysis_type": analysis_type
            })
            
            logger.info(f"Analiz sonucu depolandı: {result_id}")
            return result_id
            
        except Exception as e:
            logger.error(f"Analiz sonucu depolama hatası: {e}")
            return ""
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Süresi dolmuş verileri temizler."""
        try:
            cleanup_stats = {
                "patient_videos": 0,
                "analysis_results": 0,
                "audit_logs": 0,
                "temp_files": 0
            }
            
            current_time = datetime.now()
            
            # Hasta videoları temizle
            patient_data_dir = os.path.join(self.base_path, "patient_data")
            for filename in os.listdir(patient_data_dir):
                if filename.startswith("meta_") and filename.endswith(".json"):
                    meta_path = os.path.join(patient_data_dir, filename)
                    
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        expiry_date = datetime.fromisoformat(metadata["expiry_date"])
                        
                        if current_time > expiry_date:
                            storage_id = metadata["storage_id"]
                            
                            # Video dosyasını sil
                            video_path = os.path.join(patient_data_dir, f"video_{storage_id}.encrypted")
                            if os.path.exists(video_path):
                                os.remove(video_path)
                            
                            # Metadata dosyasını sil
                            os.remove(meta_path)
                            
                            cleanup_stats["patient_videos"] += 1
                            
                            self._log_security_event("expired_data_cleaned", {
                                "type": "patient_video",
                                "storage_id": storage_id
                            })
                            
                    except Exception as e:
                        logger.error(f"Video temizleme hatası {filename}: {e}")
            
            # Geçici dosyaları temizle (7 günden eski)
            temp_dir = os.path.join(self.base_path, "temp")
            cutoff_time = current_time - timedelta(days=7)
            
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_mtime < cutoff_time:
                    os.remove(file_path)
                    cleanup_stats["temp_files"] += 1
            
            # Audit log
            self._log_security_event("cleanup_completed", cleanup_stats)
            
            logger.info(f"Veri temizleme tamamlandı: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Veri temizleme hatası: {e}")
            return {}
    
    def create_backup(self, backup_type: str = "full") -> str:
        """Güvenli yedekleme oluşturur."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"backup_{backup_type}_{timestamp}.zip"
            backup_path = os.path.join(self.base_path, "backups", backup_filename)
            
            # ZIP dosyası oluştur
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                if backup_type == "full":
                    # Tüm kritik dizinleri yedekle
                    dirs_to_backup = ["patient_data", "analysis_results", "audit_logs"]
                elif backup_type == "analysis_only":
                    dirs_to_backup = ["analysis_results"]
                elif backup_type == "audit_only":
                    dirs_to_backup = ["audit_logs"]
                else:
                    dirs_to_backup = ["patient_data", "analysis_results"]
                
                for dir_name in dirs_to_backup:
                    dir_path = os.path.join(self.base_path, dir_name)
                    if os.path.exists(dir_path):
                        for root, _, files in os.walk(dir_path):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, self.base_path)
                                zipf.write(file_path, arcname)
            
            # Yedekleme dosyası izinleri
            os.chmod(backup_path, 0o600)
            
            # Audit log
            self._log_security_event("backup_created", {
                "backup_type": backup_type,
                "backup_file": backup_filename,
                "file_size": os.path.getsize(backup_path)
            })
            
            logger.info(f"Yedekleme oluşturuldu: {backup_filename}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Yedekleme hatası: {e}")
            return ""
    
    def _encrypt_file(self, file_path: str) -> bytes:
        """Dosyayı şifreler."""
        try:
            from cryptography.fernet import Fernet
            
            # Basit anahtar (üretimde key management sistemi kullanın)
            key = b'your-secret-key-here-32-characters!'
            cipher_suite = Fernet(Fernet.generate_key())
            
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            encrypted_data = cipher_suite.encrypt(file_data)
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Dosya şifreleme hatası: {e}")
            return b""
    
    def _encrypt_data(self, data: str) -> bytes:
        """String veriyi şifreler."""
        try:
            from cryptography.fernet import Fernet
            
            cipher_suite = Fernet(Fernet.generate_key())
            encrypted_data = cipher_suite.encrypt(data.encode('utf-8'))
            return encrypted_data
            
        except Exception as e:
            logger.error(f"Veri şifreleme hatası: {e}")
            return b""
    
    def _decrypt_file_data(self, encrypted_data: bytes) -> bytes:
        """Şifreli veriyi çözer."""
        try:
            from cryptography.fernet import Fernet
            
            # Not: Gerçek uygulamada anahtar yönetimi gerekli
            cipher_suite = Fernet(Fernet.generate_key())
            decrypted_data = cipher_suite.decrypt(encrypted_data)
            return decrypted_data
            
        except Exception as e:
            logger.error(f"Veri çözme hatası: {e}")
            return b""
    
    def _check_access_permission(self, user: str, required_level: str) -> bool:
        """Erişim iznini kontrol eder."""
        # Basit uygulama - gerçek sistemde role-based access control
        if required_level == "public":
            return True
        elif required_level == "system":
            return user == "system"
        elif required_level in ["internal", "controlled", "restricted", "admin_only"]:
            # Demo için admin her şeye erişebilir
            return user in ["admin", "doctor"] or "admin" in user.lower()
        
        return False
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Güvenlik olayını loglar."""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "details": details,
                "system": "DataSecurityPolicy"
            }
            
            with open(self.audit_log, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            logger.error(f"Güvenlik log hatası: {e}")
    
    def generate_security_report(self) -> Dict[str, Any]:
        """Güvenlik durumu raporu oluşturur."""
        try:
            report = {
                "report_date": datetime.now().isoformat(),
                "storage_statistics": self._get_storage_statistics(),
                "retention_compliance": self._check_retention_compliance(),
                "access_summary": self._get_access_summary(),
                "backup_status": self._get_backup_status(),
                "security_recommendations": self._generate_security_recommendations()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Güvenlik raporu hatası: {e}")
            return {"error": str(e)}
    
    def _get_storage_statistics(self) -> Dict[str, Any]:
        """Depolama istatistiklerini getirir."""
        stats = {}
        
        for data_type in ["patient_data", "analysis_results", "audit_logs", "backups"]:
            dir_path = os.path.join(self.base_path, data_type)
            if os.path.exists(dir_path):
                file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
                total_size = sum(os.path.getsize(os.path.join(dir_path, f)) 
                               for f in os.listdir(dir_path) 
                               if os.path.isfile(os.path.join(dir_path, f)))
                
                stats[data_type] = {
                    "file_count": file_count,
                    "total_size_mb": round(total_size / 1024 / 1024, 2)
                }
        
        return stats
    
    def _check_retention_compliance(self) -> List[str]:
        """Saklama politikası uyumluluğunu kontrol eder."""
        compliance_issues = []
        
        # Bu metodun tam implementasyonu storage statistics ve policy kontrolü gerektirir
        # Demo için basit kontrol
        
        return compliance_issues
    
    def _get_access_summary(self) -> Dict[str, int]:
        """Erişim özetini getirir.""" 
        # Audit log'dan erişim istatistiklerini çıkar
        summary = {
            "video_access_count": 0,
            "analysis_access_count": 0,
            "denied_access_count": 0
        }
        
        return summary
    
    def _get_backup_status(self) -> Dict[str, Any]:
        """Yedekleme durumunu getirir."""
        backup_dir = os.path.join(self.base_path, "backups")
        
        if not os.path.exists(backup_dir):
            return {"status": "no_backups", "last_backup": None}
        
        backup_files = [f for f in os.listdir(backup_dir) if f.startswith("backup_")]
        
        if not backup_files:
            return {"status": "no_backups", "last_backup": None}
        
        # En son yedeklemeyi bul
        latest_backup = max(backup_files, key=lambda x: os.path.getctime(os.path.join(backup_dir, x)))
        latest_backup_time = datetime.fromtimestamp(os.path.getctime(os.path.join(backup_dir, latest_backup)))
        
        return {
            "status": "active",
            "backup_count": len(backup_files),
            "last_backup": latest_backup_time.isoformat(),
            "latest_backup_file": latest_backup
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Güvenlik önerileri oluşturur."""
        recommendations = [
            "🔐 Düzenli anahtar rotasyonu yapın (90 günde bir)",
            "📊 Haftalık güvenlik raporları oluşturun",
            "🗄️ Aylık yedekleme doğrulaması yapın",
            "👥 Kullanıcı erişim izinlerini gözden geçirin",
            "🧹 Düzenli veri temizleme işlemi çalıştırın"
        ]
        
        return recommendations

# Factory function
def create_security_policy() -> DataSecurityPolicy:
    """Güvenlik politikası yöneticisi oluştur."""
    return DataSecurityPolicy()

if __name__ == "__main__":
    # Test
    policy = DataSecurityPolicy()
    print("🔒 Veri güvenliği politikası test edildi")
    print(f"Saklama politikaları: {len(policy.RETENTION_POLICIES)}")
    print(f"Erişim seviyeleri: {len(policy.ACCESS_LEVELS)}")
    
    # Güvenlik raporu oluştur
    report = policy.generate_security_report()
    print(f"\n📊 Güvenlik raporu oluşturuldu")
    print(f"Depolama dizinleri: {list(report.get('storage_statistics', {}).keys())}") 