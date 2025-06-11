#!/usr/bin/env python3
"""
CANLI ÖĞRENME & SAHADA İZLEME SİSTEMİ
===================================
Her analiz sonrası veri toplama, performans izleme ve otomatik model güncelleme.
"""

import os
import json
import uuid
import time
import shutil
import logging
import schedule
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import sqlite3
import hashlib

# Konfigürasyon
@dataclass
class LearningConfig:
    """Canlı öğrenme konfigürasyonu."""
    data_collection_path: str = "collected_data"
    model_versions_path: str = "model_versions"
    training_data_path: str = "training_datasets"
    logs_path: str = "learning_logs"
    
    # Öğrenme parametreleri
    min_samples_for_retrain: int = 100
    retrain_interval_days: int = 7
    validation_split: float = 0.2
    performance_threshold: float = 0.85
    
    # Veri kalitesi
    min_confidence_threshold: float = 0.7
    max_data_age_days: int = 90
    
    # Güvenlik
    data_encryption: bool = True
    audit_logging: bool = True

@dataclass
class AnalysisRecord:
    """Analiz kaydı veri yapısı."""
    analysis_id: str
    timestamp: datetime
    patient_id: Optional[str]
    video_metadata: Dict[str, Any]
    analysis_results: Dict[str, Any]
    model_version: str
    processing_time_ms: float
    confidence_score: float
    operator_feedback: Optional[Dict[str, Any]] = None
    validation_status: str = "pending"  # pending, validated, rejected
    
class ContinuousLearningSystem:
    """
    Canlı öğrenme ve sahada izleme sistemi.
    """
    
    def __init__(self, config: LearningConfig = None):
        self.config = config or LearningConfig()
        self.logger = self._setup_logging()
        self._setup_directories()
        self._setup_database()
        
        # Monitoring değişkenleri
        self.start_time = time.time()
        self.system_stats = {
            "total_analyses": 0,
            "collected_samples": 0,
            "model_updates": 0,
            "performance_trend": [],
            "last_retrain": None
        }
        
        # Background scheduler
        self.scheduler_thread = None
        self.running = False
        
    def _setup_logging(self) -> logging.Logger:
        """Logging sistemini kur."""
        logger = logging.getLogger("ContinuousLearning")
        logger.setLevel(logging.INFO)
        
        # File handler
        os.makedirs(self.config.logs_path, exist_ok=True)
        fh = logging.FileHandler(f"{self.config.logs_path}/continuous_learning.log")
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def _setup_directories(self):
        """Gerekli dizinleri oluştur."""
        directories = [
            self.config.data_collection_path,
            self.config.model_versions_path,
            self.config.training_data_path,
            self.config.logs_path,
            f"{self.config.data_collection_path}/raw",
            f"{self.config.data_collection_path}/processed",
            f"{self.config.data_collection_path}/validated"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"✅ Dizinler hazırlandı: {len(directories)} dizin")
    
    def _setup_database(self):
        """SQLite veritabanını kur."""
        self.db_path = f"{self.config.logs_path}/learning_database.db"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Analiz kayıtları tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_records (
                    analysis_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    patient_id TEXT,
                    video_metadata TEXT,
                    analysis_results TEXT,
                    model_version TEXT,
                    processing_time_ms REAL,
                    confidence_score REAL,
                    operator_feedback TEXT,
                    validation_status TEXT,
                    created_at TEXT
                )
            """)
            
            # Performans metrikleri tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    model_version TEXT,
                    accuracy REAL,
                    precision_nystagmus REAL,
                    recall_nystagmus REAL,
                    precision_strabismus REAL,
                    recall_strabismus REAL,
                    avg_processing_time REAL,
                    total_samples INTEGER
                )
            """)
            
            # Model versiyonları tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version TEXT PRIMARY KEY,
                    created_at TEXT,
                    training_samples INTEGER,
                    validation_accuracy REAL,
                    deployment_status TEXT,
                    performance_notes TEXT
                )
            """)
            
            conn.commit()
        
        self.logger.info("✅ Veritabanı hazırlandı")
    
    def collect_analysis_data(self, 
                            analysis_id: str,
                            patient_id: Optional[str],
                            video_metadata: Dict[str, Any],
                            analysis_results: Dict[str, Any],
                            model_version: str = "v1.0",
                            processing_time_ms: float = 0.0,
                            operator_feedback: Dict[str, Any] = None) -> bool:
        """
        Analiz sonrası veri toplama.
        
        Args:
            analysis_id: Analiz ID'si
            patient_id: Hasta ID'si (opsiyonel/anonim)
            video_metadata: Video meta verileri
            analysis_results: Analiz sonuçları
            model_version: Kullanılan model versiyonu
            processing_time_ms: İşlem süresi
            operator_feedback: Operatör geri bildirimi
            
        Returns:
            bool: Başarı durumu
        """
        try:
            # Analiz kaydı oluştur
            record = AnalysisRecord(
                analysis_id=analysis_id,
                timestamp=datetime.now(),
                patient_id=patient_id,
                video_metadata=video_metadata,
                analysis_results=analysis_results,
                model_version=model_version,
                processing_time_ms=processing_time_ms,
                confidence_score=analysis_results.get("confidence", 0.0),
                operator_feedback=operator_feedback
            )
            
            # Veri kalitesi kontrolü
            if not self._validate_data_quality(record):
                self.logger.warning(f"⚠️ Düşük kalite veri: {analysis_id}")
                return False
            
            # Raw data kaydet
            raw_file_path = self._save_raw_data(record)
            
            # Veritabanına kaydet
            self._save_to_database(record)
            
            # Stats güncelle
            self.system_stats["total_analyses"] += 1
            self.system_stats["collected_samples"] += 1
            
            self.logger.info(f"📊 Veri toplandı: {analysis_id}")
            
            # Otomatik eğitim kontrolü
            self._check_retrain_trigger()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Veri toplama hatası: {e}")
            return False
    
    def _validate_data_quality(self, record: AnalysisRecord) -> bool:
        """Veri kalitesi kontrolü."""
        # Güven skoru kontrolü
        if record.confidence_score < self.config.min_confidence_threshold:
            return False
        
        # Gerekli alanlar kontrolü
        required_fields = ["nystagmus_frequency", "strabismus_angle"]
        for field in required_fields:
            if field not in record.analysis_results:
                return False
        
        # Video metadata kontrolü
        if not record.video_metadata.get("frame_count", 0) > 0:
            return False
        
        return True
    
    def _save_raw_data(self, record: AnalysisRecord) -> str:
        """Raw veriyi dosyaya kaydet."""
        # Dosya adı oluştur
        timestamp_str = record.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp_str}_{record.analysis_id[:8]}.json"
        filepath = os.path.join(self.config.data_collection_path, "raw", filename)
        
        # Veriyi serileştir
        data = {
            "record": asdict(record),
            "collection_timestamp": datetime.now().isoformat(),
            "data_version": "1.0"
        }
        
        # JSON olarak kaydet
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        return filepath
    
    def _save_to_database(self, record: AnalysisRecord):
        """Veritabanına kaydet."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO analysis_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.analysis_id,
                record.timestamp.isoformat(),
                record.patient_id,
                json.dumps(record.video_metadata),
                json.dumps(record.analysis_results, default=str),
                record.model_version,
                record.processing_time_ms,
                record.confidence_score,
                json.dumps(record.operator_feedback) if record.operator_feedback else None,
                record.validation_status,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def add_operator_feedback(self, 
                            analysis_id: str, 
                            feedback: Dict[str, Any]) -> bool:
        """
        Operatör geri bildirimi ekle.
        
        Args:
            analysis_id: Analiz ID'si
            feedback: Geri bildirim verisi
            
        Returns:
            bool: Başarı durumu
        """
        try:
            # Veritabanını güncelle
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE analysis_records 
                    SET operator_feedback = ?, validation_status = ?
                    WHERE analysis_id = ?
                """, (
                    json.dumps(feedback),
                    feedback.get("validation_status", "validated"),
                    analysis_id
                ))
                conn.commit()
            
            self.logger.info(f"📝 Operatör geri bildirimi eklendi: {analysis_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Geri bildirim hatası: {e}")
            return False
    
    def _check_retrain_trigger(self):
        """Yeniden eğitim tetikleyici kontrolü."""
        # Yeterli örnek kontrolü
        total_samples = self._count_validated_samples()
        
        if total_samples >= self.config.min_samples_for_retrain:
            # Son eğitim tarihi kontrolü
            last_retrain = self.system_stats.get("last_retrain")
            if last_retrain is None:
                should_retrain = True
            else:
                days_since_retrain = (datetime.now() - last_retrain).days
                should_retrain = days_since_retrain >= self.config.retrain_interval_days
            
            if should_retrain:
                self.logger.info(f"🔄 Yeniden eğitim tetiklendi: {total_samples} örnek")
                self._trigger_retraining()
    
    def _count_validated_samples(self) -> int:
        """Doğrulanmış örnek sayısını getir."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM analysis_records 
                WHERE validation_status = 'validated'
                AND confidence_score >= ?
            """, (self.config.min_confidence_threshold,))
            return cursor.fetchone()[0]
    
    def _trigger_retraining(self):
        """Yeniden eğitimi tetikle."""
        try:
            # Eğitim verisini hazırla
            training_data = self._prepare_training_data()
            
            if len(training_data) < self.config.min_samples_for_retrain:
                self.logger.warning("⚠️ Yetersiz eğitim verisi")
                return
            
            # Model versiyonu oluştur
            new_version = self._generate_model_version()
            
            # Eğitim verisini kaydet
            self._save_training_dataset(training_data, new_version)
            
            # Background eğitim başlat (gerçek implementasyon)
            self._start_background_training(new_version, training_data)
            
            # Stats güncelle
            self.system_stats["model_updates"] += 1
            self.system_stats["last_retrain"] = datetime.now()
            
            self.logger.info(f"🚀 Model eğitimi başlatıldı: {new_version}")
            
        except Exception as e:
            self.logger.error(f"❌ Eğitim tetikleme hatası: {e}")
    
    def _prepare_training_data(self) -> List[Dict[str, Any]]:
        """Eğitim verisini hazırla."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT analysis_results, operator_feedback 
                FROM analysis_records 
                WHERE validation_status = 'validated'
                AND confidence_score >= ?
                AND timestamp >= ?
            """, (
                self.config.min_confidence_threshold,
                (datetime.now() - timedelta(days=self.config.max_data_age_days)).isoformat()
            ))
            
            training_data = []
            for row in cursor.fetchall():
                results = json.loads(row[0])
                feedback = json.loads(row[1]) if row[1] else {}
                
                # Training sample oluştur
                sample = {
                    "features": {
                        "nystagmus_frequency": results.get("nystagmus_frequency", 0),
                        "movement_amplitude": results.get("movement_amplitude", 0),
                        "regularity": results.get("regularity", 0),
                        "strabismus_angle": results.get("strabismus_angle", 0),
                        "strabismus_stability": results.get("strabismus_stability", 0)
                    },
                    "labels": {
                        "nystagmus_detected": feedback.get("nystagmus_correct", results.get("nystagmus_detected", False)),
                        "strabismus_detected": feedback.get("strabismus_correct", results.get("strabismus_detected", False))
                    },
                    "weight": feedback.get("confidence", 1.0)
                }
                training_data.append(sample)
            
            return training_data
    
    def _generate_model_version(self) -> str:
        """Yeni model versiyonu oluştur."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        version_hash = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"v{timestamp}_{version_hash}"
    
    def _save_training_dataset(self, training_data: List[Dict[str, Any]], version: str):
        """Eğitim verisini kaydet."""
        dataset_path = os.path.join(self.config.training_data_path, f"dataset_{version}.json")
        
        dataset = {
            "version": version,
            "created_at": datetime.now().isoformat(),
            "total_samples": len(training_data),
            "data": training_data,
            "metadata": {
                "min_confidence": self.config.min_confidence_threshold,
                "max_age_days": self.config.max_data_age_days
            }
        }
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Eğitim verisi kaydedildi: {dataset_path}")
    
    def _start_background_training(self, version: str, training_data: List[Dict[str, Any]]):
        """Background model eğitimi."""
        def train_model():
            try:
                self.logger.info(f"🧠 Model eğitimi başladı: {version}")
                
                # Simulated training (gerçek implementasyon buraya gelecek)
                time.sleep(2)  # Simulated training time
                
                # Model performansını hesapla
                val_accuracy = 0.89 + (len(training_data) / 1000) * 0.05  # Mock accuracy
                val_accuracy = min(val_accuracy, 0.95)
                
                # Model versiyonunu kaydet
                self._save_model_version(version, len(training_data), val_accuracy)
                
                # Performans metrikleri kaydet
                self._save_performance_metrics(version, val_accuracy, training_data)
                
                self.logger.info(f"✅ Model eğitimi tamamlandı: {version} (Accuracy: {val_accuracy:.3f})")
                
                # Model deployment kontrolü
                if val_accuracy >= self.config.performance_threshold:
                    self._deploy_model(version)
                else:
                    self.logger.warning(f"⚠️ Model performansı düşük: {val_accuracy:.3f}")
                
            except Exception as e:
                self.logger.error(f"❌ Model eğitim hatası: {e}")
        
        # Background thread başlat
        training_thread = threading.Thread(target=train_model)
        training_thread.daemon = True
        training_thread.start()
    
    def _save_model_version(self, version: str, training_samples: int, validation_accuracy: float):
        """Model versiyonunu kaydet."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_versions VALUES (?, ?, ?, ?, ?, ?)
            """, (
                version,
                datetime.now().isoformat(),
                training_samples,
                validation_accuracy,
                "training",
                f"Auto-trained with {training_samples} samples"
            ))
            conn.commit()
    
    def _save_performance_metrics(self, version: str, accuracy: float, training_data: List[Dict]):
        """Performans metrikleri kaydet."""
        # Mock metrics calculation
        metrics = {
            "accuracy": accuracy,
            "precision_nystagmus": accuracy + 0.02,
            "recall_nystagmus": accuracy - 0.01,
            "precision_strabismus": accuracy + 0.01,
            "recall_strabismus": accuracy - 0.02,
            "avg_processing_time": 150.0,  # ms
            "total_samples": len(training_data)
        }
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics 
                (timestamp, model_version, accuracy, precision_nystagmus, recall_nystagmus,
                 precision_strabismus, recall_strabismus, avg_processing_time, total_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                version,
                metrics["accuracy"],
                metrics["precision_nystagmus"],
                metrics["recall_nystagmus"],
                metrics["precision_strabismus"],
                metrics["recall_strabismus"],
                metrics["avg_processing_time"],
                metrics["total_samples"]
            ))
            conn.commit()
        
        # Performance trend güncelle
        self.system_stats["performance_trend"].append({
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "accuracy": accuracy
        })
        
        # Son 20 kaydı tut
        if len(self.system_stats["performance_trend"]) > 20:
            self.system_stats["performance_trend"] = self.system_stats["performance_trend"][-20:]
    
    def _deploy_model(self, version: str):
        """Model deployment."""
        try:
            # Model dosyasını production dizinine kopyala
            model_source = os.path.join(self.config.model_versions_path, f"model_{version}.pkl")
            model_target = os.path.join(self.config.model_versions_path, "production_model.pkl")
            
            # Simulated model file
            with open(model_source, 'w') as f:
                json.dump({"version": version, "created_at": datetime.now().isoformat()}, f)
            
            shutil.copy2(model_source, model_target)
            
            # Deployment status güncelle
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE model_versions 
                    SET deployment_status = 'deployed'
                    WHERE version = ?
                """, (version,))
                conn.commit()
            
            self.logger.info(f"🚀 Model deploy edildi: {version}")
            
        except Exception as e:
            self.logger.error(f"❌ Model deployment hatası: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Sistem durumu raporu."""
        uptime = time.time() - self.start_time
        
        # Veritabanı istatistikleri
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Toplam analiz sayısı
            cursor.execute("SELECT COUNT(*) FROM analysis_records")
            total_analyses = cursor.fetchone()[0]
            
            # Son 24 saat analiz sayısı
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM analysis_records WHERE timestamp >= ?", (yesterday,))
            recent_analyses = cursor.fetchone()[0]
            
            # Son model versiyonu
            cursor.execute("SELECT version, validation_accuracy FROM model_versions ORDER BY created_at DESC LIMIT 1")
            latest_model = cursor.fetchone()
        
        return {
            "uptime_seconds": uptime,
            "uptime_formatted": f"{uptime/3600:.1f} saat",
            "system_stats": self.system_stats,
            "database_stats": {
                "total_analyses": total_analyses,
                "recent_analyses_24h": recent_analyses,
                "collected_samples": self.system_stats["collected_samples"]
            },
            "latest_model": {
                "version": latest_model[0] if latest_model else "v1.0",
                "accuracy": latest_model[1] if latest_model else 0.85
            },
            "data_paths": {
                "collection": self.config.data_collection_path,
                "models": self.config.model_versions_path,
                "training": self.config.training_data_path
            },
            "last_check": datetime.now().isoformat()
        }
    
    def generate_learning_report(self, days: int = 30) -> Dict[str, Any]:
        """Öğrenme raporu oluştur."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Performans trend
            cursor.execute("""
                SELECT timestamp, accuracy FROM performance_metrics 
                WHERE timestamp >= ? ORDER BY timestamp
            """, (start_date.isoformat(),))
            performance_trend = cursor.fetchall()
            
            # Model versiyonları
            cursor.execute("""
                SELECT version, created_at, validation_accuracy, deployment_status 
                FROM model_versions 
                WHERE created_at >= ? ORDER BY created_at DESC
            """, (start_date.isoformat(),))
            model_versions = cursor.fetchall()
            
            # Veri toplama istatistikleri
            cursor.execute("""
                SELECT DATE(timestamp) as date, COUNT(*) as count
                FROM analysis_records 
                WHERE timestamp >= ? 
                GROUP BY DATE(timestamp)
            """, (start_date.isoformat(),))
            daily_collections = cursor.fetchall()
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "performance_trend": [
                {"timestamp": t, "accuracy": a} for t, a in performance_trend
            ],
            "model_versions": [
                {
                    "version": v, "created_at": c, 
                    "accuracy": a, "status": s
                } for v, c, a, s in model_versions
            ],
            "data_collection": [
                {"date": d, "samples": c} for d, c in daily_collections
            ],
            "summary": {
                "total_models_trained": len(model_versions),
                "total_samples_collected": sum(c for _, c in daily_collections),
                "avg_daily_collection": sum(c for _, c in daily_collections) / max(len(daily_collections), 1),
                "latest_accuracy": performance_trend[-1][1] if performance_trend else 0.0
            },
            "generated_at": datetime.now().isoformat()
        }
    
    def start_monitoring(self):
        """Monitoring sistemi başlat."""
        if self.running:
            return
        
        self.running = True
        
        # Günlük temizlik işlemi
        schedule.every().day.at("02:00").do(self._daily_cleanup)
        
        # Haftalık performans analizi
        schedule.every().sunday.at("03:00").do(self._weekly_performance_analysis)
        
        # Scheduler thread başlat
        def run_scheduler():
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Her dakika kontrol et
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        self.logger.info("🔍 Monitoring sistemi başlatıldı")
    
    def stop_monitoring(self):
        """Monitoring sistemi durdur."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("🛑 Monitoring sistemi durduruldu")
    
    def _daily_cleanup(self):
        """Günlük temizlik işlemi."""
        try:
            # Eski geçici dosyaları sil
            temp_files = Path(self.config.data_collection_path).glob("temp_*")
            cleaned_count = 0
            
            for temp_file in temp_files:
                if temp_file.stat().st_mtime < time.time() - 86400:  # 24 saat eski
                    temp_file.unlink()
                    cleaned_count += 1
            
            self.logger.info(f"🧹 Günlük temizlik: {cleaned_count} dosya silindi")
            
        except Exception as e:
            self.logger.error(f"❌ Günlük temizlik hatası: {e}")
    
    def _weekly_performance_analysis(self):
        """Haftalık performans analizi."""
        try:
            # Son haftanın performans raporu
            report = self.generate_learning_report(days=7)
            
            # Performans uyarıları
            latest_accuracy = report["summary"]["latest_accuracy"]
            if latest_accuracy < self.config.performance_threshold:
                self.logger.warning(f"⚠️ Performans düşük: {latest_accuracy:.3f}")
            
            # Raporu kaydet
            report_path = f"{self.config.logs_path}/weekly_report_{datetime.now().strftime('%Y%m%d')}.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"📊 Haftalık rapor oluşturuldu: {report_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Haftalık analiz hatası: {e}")

# Factory function
def create_continuous_learning_system(config: LearningConfig = None) -> ContinuousLearningSystem:
    """Canlı öğrenme sistemi oluştur."""
    return ContinuousLearningSystem(config)

# Örnek kullanım
if __name__ == "__main__":
    # Test sistem
    print("🔄 Canlı öğrenme sistemi test ediliyor...")
    
    # Sistem oluştur
    learning_system = create_continuous_learning_system()
    
    # Monitoring başlat
    learning_system.start_monitoring()
    
    # Test verisi ekle
    test_analysis = {
        "analysis_id": str(uuid.uuid4()),
        "patient_id": "TEST_001",
        "video_metadata": {
            "frame_count": 150,
            "fps": 30,
            "duration": 5.0,
            "resolution": "640x480"
        },
        "analysis_results": {
            "nystagmus_detected": True,
            "nystagmus_frequency": 3.2,
            "movement_amplitude": 0.75,
            "regularity": 0.68,
            "strabismus_detected": False,
            "strabismus_angle": 1.1,
            "confidence": 0.87
        },
        "processing_time_ms": 245.0
    }
    
    # Veri topla
    success = learning_system.collect_analysis_data(**test_analysis)
    print(f"✅ Veri toplama: {'Başarılı' if success else 'Başarısız'}")
    
    # Operatör geri bildirimi ekle
    feedback = {
        "nystagmus_correct": True,
        "strabismus_correct": True,
        "confidence": 0.9,
        "validation_status": "validated",
        "notes": "Doğru tespit"
    }
    
    learning_system.add_operator_feedback(test_analysis["analysis_id"], feedback)
    
    # Sistem durumu
    status = learning_system.get_system_status()
    print(f"📊 Sistem durumu: {status['system_stats']['collected_samples']} örnek toplandı")
    
    # Öğrenme raporu
    report = learning_system.generate_learning_report(days=1)
    print(f"📈 Öğrenme raporu: {len(report['data_collection'])} günlük veri")
    
    print("✅ Test tamamlandı") 