from typing import Dict, List, Any, Optional, Callable, Union, Type
import logging
import time
import traceback
import threading
import json
import os
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
import uuid

# Geçici olarak Event, EventType ve EventBus sınıflarını tanımlıyoruz
# İleride bu sınıflar events.event_bus modülünden içe aktarılacak
class EventType(Enum):
    """Olay türleri"""
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SYSTEM = auto()
    USER = auto()


class Event:
    """Temel olay sınıfı"""
    
    def __init__(self, event_type, topic, data, source=""):
        self.event_type = event_type
        self.topic = topic
        self.data = data
        self.source = source
        self.timestamp = time.time()


class EventBus:
    """Basit olay yönetim sistemi"""
    
    def __init__(self):
        self.subscribers = {}
        self._lock = threading.Lock()
    
    def subscribe(self, event_type, callback):
        """Belirli bir olay türüne abone ol"""
        with self._lock:
            if event_type not in self.subscribers:
                self.subscribers[event_type] = []
            self.subscribers[event_type].append(callback)
    
    def publish(self, event):
        """Olay yayınla"""
        with self._lock:
            if event.event_type in self.subscribers:
                for callback in self.subscribers[event.event_type]:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"Event callback error: {str(e)}")
    
    def publish_from_dict(self, event_type, topic, data, source=""):
        """Sözlük verilerinden olay yayınla"""
        event = Event(event_type, topic, data, source)
        self.publish(event)


# Asıl hata yönetim sistemi
logger = logging.getLogger('eye_tracker.error')

class ErrorSeverity(Enum):
    """Hata şiddeti seviyeleri"""
    INFO = auto()       # Bilgilendirme (kullanıcının bilgilendirilmesi gereken)
    WARNING = auto()    # Uyarı (program çalışmaya devam edebilir)
    ERROR = auto()      # Hata (bazı işlevler çalışmayabilir)
    CRITICAL = auto()   # Kritik (program düzgün çalışamaz)
    FATAL = auto()      # Ölümcül (programın kapatılması gerekir)

class ErrorCategory(Enum):
    """Hata kategorileri"""
    HARDWARE = auto()         # Donanım hatası (kamera, sensör vb.)
    CALIBRATION = auto()      # Kalibrasyon hatası
    ANALYSIS = auto()         # Analiz işlemi hatası
    STORAGE = auto()          # Veri depolama hatası
    CONFIGURATION = auto()    # Yapılandırma hatası
    NETWORK = auto()          # Ağ iletişim hatası
    UI = auto()               # Kullanıcı arayüzü hatası
    MODEL = auto()            # Model veya işleme hatası
    SYSTEM = auto()           # Sistem hatası (dosya sistemi, bellek vb.)
    UNKNOWN = auto()          # Sınıflandırılamayan hatalar

@dataclass
class ErrorEvent:
    """Hata olayı veri sınıfı"""
    error_id: str                     # Benzersiz hata kimliği
    timestamp: float                  # Zaman damgası
    severity: ErrorSeverity           # Hata şiddeti
    category: ErrorCategory           # Hata kategorisi
    message: str                      # Hata mesajı
    component: str                    # Hatayı üreten bileşen
    exception: Optional[Exception] = None  # İlgili istisna (varsa)
    traceback: Optional[str] = None        # İstisna izleme (varsa)
    details: Dict[str, Any] = field(default_factory=dict)  # Ek ayrıntılar
    suggestions: List[str] = field(default_factory=list)   # Çözüm önerileri
    handled: bool = False             # Hata işlendi mi
    user_notified: bool = False       # Kullanıcı bilgilendirildi mi
    
    def to_dict(self) -> Dict[str, Any]:
        """Hata olayını sözlüğe dönüştür"""
        data = asdict(self)
        # Enum değerlerini stringe dönüştür
        data["severity"] = self.severity.name
        data["category"] = self.category.name
        # Exception ve traceback bilgisini düzenle
        if self.exception:
            data["exception"] = str(self.exception)
        return data
    
    def to_json(self) -> str:
        """Hata olayını JSON formatına dönüştür"""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorEvent':
        """Sözlükten hata olayı oluştur"""
        # Enum değerlerini dönüştür
        data["severity"] = ErrorSeverity[data["severity"]]
        data["category"] = ErrorCategory[data["category"]]
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ErrorEvent':
        """JSON'dan hata olayı oluştur"""
        data = json.loads(json_str)
        return cls.from_dict(data)

class ErrorManager:
    """
    Merkezi hata yönetim sistemi
    """
    def __init__(self, event_bus: Optional[EventBus] = None, error_log_path: str = "errors.log"):
        """
        Args:
            event_bus: Olay yöneticisi (None: sadece yerel işleme)
            error_log_path: Hata günlük dosyası yolu
        """
        self.event_bus = event_bus
        self.error_log_path = error_log_path
        
        # Hata kaydını tutan liste
        self.error_history: List[ErrorEvent] = []
        self.max_history_size = 1000  # Maksimum kayıt sayısı
        
        # İşleyiciler
        self.error_handlers: Dict[ErrorCategory, List[Callable[[ErrorEvent], None]]] = {}
        for category in ErrorCategory:
            self.error_handlers[category] = []
        
        # Kritik hata sayaçları (kategori ve zaman bazlı)
        self.error_counters: Dict[ErrorCategory, int] = {}
        for category in ErrorCategory:
            self.error_counters[category] = 0
        
        # Son hata zamanlayıcıları (kategori bazlı, aynı hatanın tekrarını önlemek için)
        self.last_error_times: Dict[str, float] = {}
        self.error_cooldown = 5.0  # Saniye cinsinden aynı hatayı tekrar bildirme süresi
        
        # Öneriler veritabanı
        self.suggestion_db = self._load_suggestion_database()
        
        # Thread kilidi
        self._lock = threading.Lock()
        
        logger.info("Hata yöneticisi başlatıldı")
    
    def _load_suggestion_database(self) -> Dict[str, List[str]]:
        """Öneriler veritabanını yükle"""
        # Varsayılan öneriler
        suggestions = {
            # Donanım hataları
            "camera_not_found": [
                "Kamera bağlantısını kontrol edin.",
                "Farklı bir kamera ID'si deneyin.",
                "Kamera sürücülerini güncelleyin."
            ],
            "camera_permission": [
                "Kamera erişim izinlerini kontrol edin.",
                "Programı yönetici olarak çalıştırın.",
                "Başka uygulamaların kamerayı kullanıp kullanmadığını kontrol edin."
            ],
            "low_fps": [
                "Kamera çözünürlüğünü düşürün.",
                "Bilgisayarınızın kaynaklarını kontrol edin.",
                "Diğer uygulamaları kapatın."
            ],
            
            # Kalibrasyon hataları
            "calibration_failed": [
                "Kalibrasyon sırasında başınızı sabit tutun.",
                "Daha iyi aydınlatmalı bir ortamda tekrar deneyin.",
                "Daha fazla kalibrasyon noktası kullanın."
            ],
            "insufficient_calibration_points": [
                "En az 3 kalibrasyon noktası gereklidir.",
                "Daha fazla kalibrasyon noktası ekleyin.",
                "Yeni bir kalibrasyon başlatın."
            ],
            
            # Analiz hataları
            "analysis_error": [
                "Yeniden kalibrasyon yapın.",
                "Analiz parametrelerini varsayılan değerlere sıfırlayın.",
                "Daha iyi aydınlatmalı bir ortamda tekrar deneyin."
            ],
            "signal_processing_error": [
                "Sinyal işleme parametrelerini varsayılan değerlere sıfırlayın.",
                "Daha uzun bir veri tamponu kullanın.",
                "Gürültüyü azaltmak için daha iyi aydınlatmalı bir ortam kullanın."
            ],
            
            # Depolama hataları
            "storage_write_error": [
                "Disk alanınızı kontrol edin.",
                "Program için yazma izinlerini kontrol edin.",
                "Çıktı dizinini değiştirin."
            ],
            "database_error": [
                "Veritabanı bağlantısını kontrol edin.",
                "Veritabanı dosyasının bozulmadığından emin olun.",
                "Veritabanı dosyasının yazılabilir olduğunu kontrol edin."
            ],
            
            # Yapılandırma hataları
            "config_load_error": [
                "Yapılandırma dosyasının geçerli olduğunu kontrol edin.",
                "Varsayılan yapılandırmayı yükleyin.",
                "Yapılandırma dosyasını yeniden oluşturun."
            ],
            "invalid_parameter": [
                "Parametre değerlerini geçerli aralıklara ayarlayın.",
                "Yapılandırmayı varsayılan değerlere sıfırlayın.",
                "Parametre değerlerini kontrol edin."
            ],
            
            # Ağ hataları
            "network_connection_error": [
                "Ağ bağlantınızı kontrol edin.",
                "Güvenlik duvarı ayarlarını kontrol edin.",
                "Sunucu adresini ve port numarasını doğrulayın."
            ],
            "websocket_error": [
                "Web arayüzünü yenileyin.",
                "Sunucuyu yeniden başlatın.",
                "Farklı bir tarayıcı kullanın."
            ],
            
            # UI hataları
            "ui_rendering_error": [
                "Ekran çözünürlüğünüzü kontrol edin.",
                "Grafik sürücülerinizi güncelleyin.",
                "UI ölçekleme ayarlarını değiştirin."
            ],
            
            # Model hataları
            "model_load_error": [
                "Model dosyasının doğru konumda olduğunu kontrol edin.",
                "Model formatının desteklendiğinden emin olun.",
                "GPU kullanımını devre dışı bırakın."
            ],
            "inference_error": [
                "GPU sürücülerini güncelleyin.",
                "CPU moduna geçin.",
                "Daha küçük model kullanın."
            ],
            
            # Sistem hataları
            "memory_error": [
                "Programı yeniden başlatın.",
                "Daha az RAM kullanan bir yapılandırma kullanın.",
                "Diğer uygulamaları kapatın."
            ],
            "file_system_error": [
                "Disk alanınızı kontrol edin.",
                "Dosya erişim izinlerini kontrol edin.",
                "Geçici dosyaları temizleyin."
            ],
            
            # Genel hatalar
            "unknown_error": [
                "Programı yeniden başlatın.",
                "Sisteminizi yeniden başlatın.",
                "Günlük dosyasını inceleyerek daha fazla bilgi alın."
            ]
        }
        
        # Özel öneriler dosyasını yüklemeye çalış
        custom_path = "config/error_suggestions.json"
        if os.path.exists(custom_path):
            try:
                with open(custom_path, 'r', encoding='utf-8') as f:
                    custom_suggestions = json.load(f)
                    # Varsayılan önerileri özelleştirilmiş olanlarla birleştir
                    suggestions.update(custom_suggestions)
                    logger.info(f"Özelleştirilmiş hata önerileri yüklendi: {custom_path}")
            except Exception as e:
                logger.error(f"Özelleştirilmiş hata önerileri yüklenemedi: {str(e)}")
        
        return suggestions
    
    def _find_suggestions(self, error_category: ErrorCategory, error_message: str) -> List[str]:
        """
        Belirtilen kategori ve mesaj için uygun öneriler bul
        
        Args:
            error_category: Hata kategorisi
            error_message: Hata mesajı
            
        Returns:
            Öneri listesi
        """
        suggestions = []
        
        # Mesajda anahtar kelimeleri ara
        error_message_lower = error_message.lower()
        
        for key, suggestions_list in self.suggestion_db.items():
            # Anahtar kelimeleri ara
            if key.lower() in error_message_lower:
                suggestions.extend(suggestions_list)
                
        # Belirli bir anahtar kelime bulunamazsa kategori bazlı varsayılan öneriler
        if not suggestions:
            category_name = error_category.name.lower()
            for key, suggestions_list in self.suggestion_db.items():
                if category_name in key.lower():
                    suggestions.extend(suggestions_list)
        
        # Yine bulunamazsa genel öneriler
        if not suggestions:
            suggestions = self.suggestion_db.get("unknown_error", [
                "Programı yeniden başlatın.",
                "Sisteminizi yeniden başlatın.",
                "Logları inceleyerek daha fazla bilgi alın."
            ])
        
        # Tekrar eden önerileri kaldır
        return list(dict.fromkeys(suggestions))
    
    def report_error(self, 
                    severity: ErrorSeverity,
                    category: ErrorCategory,
                    message: str,
                    component: str,
                    exception: Optional[Exception] = None,
                    details: Optional[Dict[str, Any]] = None,
                    error_code: Optional[str] = None) -> str:
        """
        Hata bildir
        
        Args:
            severity: Hata şiddeti
            category: Hata kategorisi
            message: Hata mesajı
            component: Hatayı bildiren bileşen
            exception: İlgili istisna (varsa)
            details: Ek ayrıntılar
            error_code: Hata kodu (öneriler için kullanılır)
            
        Returns:
            Hata ID'si
        """
        with self._lock:
            # Hata tekrarını kontrol et
            error_key = f"{category.name}:{message}"
            current_time = time.time()
            
            if error_key in self.last_error_times:
                last_time = self.last_error_times[error_key]
                if current_time - last_time < self.error_cooldown:
                    # Aynı hata çok kısa süre içinde tekrar bildirildi, atla
                    # Sadece log kaydı yap
                    logger.debug(f"Hata tekrarı engellendi: {error_key}")
                    
                    # Önceki hata ID'sini döndür
                    for error in reversed(self.error_history):
                        if error.category == category and error.message == message:
                            return error.error_id
                    
                    # Bulunamazsa yeni bir hata olarak işle (bu olmamalı)
            
            # Hata zamanını güncelle
            self.last_error_times[error_key] = current_time
            
            # Hata sayacını güncelle
            self.error_counters[category] += 1
            
            # Benzersiz hata ID'si oluştur
            error_id = str(uuid.uuid4())
            
            # İstisna izi oluştur
            tb_str = None
            if exception:
                tb_str = ''.join(traceback.format_exception(None, exception, exception.__traceback__))
            
            # Öneriler oluştur
            search_key = error_code if error_code else message
            suggestions = self._find_suggestions(category, search_key)
            
            # Hata detaylarını hazırla
            if details is None:
                details = {}
            
            # Hata olayı oluştur
            error_event = ErrorEvent(
                error_id=error_id,
                timestamp=current_time,
                severity=severity,
                category=category,
                message=message,
                component=component,
                exception=exception,
                traceback=tb_str,
                details=details,
                suggestions=suggestions
            )
            
            # Hata geçmişine ekle
            self.error_history.append(error_event)
            
            # Geçmiş boyutunu kontrol et
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
            
            # Hata günlüğüne kaydet
            self._log_error(error_event)
            
            # Event bus üzerinden hata olayı yayınla (varsa)
            if self.event_bus:
                # Hata kategorisine göre olay tipini belirle
                event_type = EventType.ERROR
                
                # EventBus üzerinden hata olayını yayınla
                self.event_bus.publish_from_dict(
                    event_type,
                    f"error_{category.name.lower()}",
                    {
                        "error_id": error_id,
                        "severity": severity.name,
                        "category": category.name,
                        "message": message,
                        "component": component,
                        "timestamp": current_time,
                        "suggestions": suggestions,
                        "details": details
                    },
                    "ErrorManager"
                )
            
            # Hata işleyicilerini çağır
            self._call_error_handlers(error_event)
            
            # Şiddet seviyesine göre loglama
            if severity == ErrorSeverity.INFO:
                logger.info(f"[{category.name}] {message} (ID: {error_id})")
            elif severity == ErrorSeverity.WARNING:
                logger.warning(f"[{category.name}] {message} (ID: {error_id})")
            elif severity == ErrorSeverity.ERROR:
                logger.error(f"[{category.name}] {message} (ID: {error_id})")
            elif severity == ErrorSeverity.CRITICAL:
                logger.critical(f"[{category.name}] {message} (ID: {error_id})")
            elif severity == ErrorSeverity.FATAL:
                logger.critical(f"[FATAL:{category.name}] {message} (ID: {error_id})")
            
            return error_id
    
    def _log_error(self, error_event: ErrorEvent) -> None:
        """
        Hata olayını günlük dosyasına kaydet
        
        Args:
            error_event: Hata olayı
        """
        try:
            # JSON formatına dönüştür
            error_json = error_event.to_json()
            
            # Dosyaya ekle
            with open(self.error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{error_json}\n")
        except Exception as e:
            # Hata günlüğüne yazma başarısız oldu, sadece log kaydı yap
            logger.error(f"Hata günlüğüne yazma başarısız: {str(e)}")
    
    def _call_error_handlers(self, error_event: ErrorEvent) -> None:
        """
        İlgili hata işleyicilerini çağır
        
        Args:
            error_event: Hata olayı
        """
        # Kategori için kayıtlı işleyicileri çağır
        handlers = self.error_handlers.get(error_event.category, [])
        
        for handler in handlers:
            try:
                handler(error_event)
            except Exception as e:
                logger.error(f"Hata işleyicisi çağrılırken hata oluştu: {str(e)}")
    
    def register_error_handler(self, category: ErrorCategory, 
                             handler: Callable[[ErrorEvent], None]) -> None:
        """
        Belirli bir kategori için hata işleyicisi kaydet
        
        Args:
            category: Hata kategorisi
            handler: İşleyici fonksiyon
        """
        with self._lock:
            self.error_handlers[category].append(handler)
    
    def register_global_error_handler(self, 
                                    handler: Callable[[ErrorEvent], None]) -> None:
        """
        Tüm kategoriler için hata işleyicisi kaydet
        
        Args:
            handler: İşleyici fonksiyon
        """
        for category in ErrorCategory:
            self.register_error_handler(category, handler)
    
    def get_recent_errors(self, 
                        count: int = 10, 
                        category: Optional[ErrorCategory] = None,
                        min_severity: Optional[ErrorSeverity] = None) -> List[ErrorEvent]:
        """
        Son hataları getir
        
        Args:
            count: Getirilecek maksimum hata sayısı
            category: Filtrelenecek kategori (None: tüm kategoriler)
            min_severity: Minimum şiddet seviyesi (None: tüm şiddetler)
            
        Returns:
            Hata olayları listesi
        """
        with self._lock:
            filtered_errors = self.error_history
            
            # Kategori filtresi
            if category:
                filtered_errors = [e for e in filtered_errors if e.category == category]
            
            # Şiddet filtresi
            if min_severity:
                filtered_errors = [e for e in filtered_errors if e.severity.value >= min_severity.value]
            
            # Son n hatayı döndür
            return filtered_errors[-count:]
    
    def mark_error_as_handled(self, error_id: str, user_notified: bool = True) -> bool:
        """
        Hatayı işlenmiş olarak işaretle
        
        Args:
            error_id: Hata ID'si
            user_notified: Kullanıcı bilgilendirildi mi
            
        Returns:
            Başarı durumu
        """
        with self._lock:
            for error in self.error_history:
                if error.error_id == error_id:
                    error.handled = True
                    error.user_notified = user_notified
                    return True
            return False
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Hata istatistiklerini al
        
        Returns:
            İstatistikler sözlüğü
        """
        with self._lock:
            stats = {
                "total_errors": len(self.error_history),
                "unhandled_errors": sum(1 for e in self.error_history if not e.handled),
                "by_category": {cat.name: count for cat, count in self.error_counters.items()},
                "by_severity": {}
            }
            
            # Şiddet seviyelerine göre sayılar
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.name] = sum(1 for e in self.error_history 
                                                if e.severity == severity)
            stats["by_severity"] = severity_counts
            
            return stats
    
    def clear_error_history(self) -> None:
        """Hata geçmişini temizle"""
        with self._lock:
            self.error_history.clear()
    
    def get_error_by_id(self, error_id: str) -> Optional[ErrorEvent]:
        """
        ID'ye göre hatayı bul
        
        Args:
            error_id: Hata ID'si
            
        Returns:
            Hata olayı veya None
        """
        with self._lock:
            for error in self.error_history:
                if error.error_id == error_id:
                    return error
            return None

# Hata raporlama için yardımcı fonksiyonlar
def report_hardware_error(error_manager: ErrorManager, message: str, component: str, 
                        exception: Optional[Exception] = None, 
                        severity: ErrorSeverity = ErrorSeverity.ERROR,
                        details: Optional[Dict[str, Any]] = None) -> str:
    """
    Donanım hatası bildir
    
    Args:
        error_manager: Hata yöneticisi
        message: Hata mesajı
        component: Bileşen adı
        exception: İstisna (varsa)
        severity: Hata şiddeti
        details: Ek ayrıntılar
        
    Returns:
        Hata ID'si
    """
    return error_manager.report_error(
        severity=severity,
        category=ErrorCategory.HARDWARE,
        message=message,
        component=component,
        exception=exception,
        details=details
    )

def report_calibration_error(error_manager: ErrorManager, message: str, component: str, 
                           exception: Optional[Exception] = None, 
                           severity: ErrorSeverity = ErrorSeverity.WARNING,
                           details: Optional[Dict[str, Any]] = None) -> str:
    """
    Kalibrasyon hatası bildir
    
    Args:
        error_manager: Hata yöneticisi
        message: Hata mesajı
        component: Bileşen adı
        exception: İstisna (varsa)
        severity: Hata şiddeti
        details: Ek ayrıntılar
        
    Returns:
        Hata ID'si
    """
    return error_manager.report_error(
        severity=severity,
        category=ErrorCategory.CALIBRATION,
        message=message,
        component=component,
        exception=exception,
        details=details
    )

def report_analysis_error(error_manager: ErrorManager, message: str, component: str, 
                        exception: Optional[Exception] = None, 
                        severity: ErrorSeverity = ErrorSeverity.WARNING,
                        details: Optional[Dict[str, Any]] = None) -> str:
    """
    Analiz hatası bildir
    
    Args:
        error_manager: Hata yöneticisi
        message: Hata mesajı
        component: Bileşen adı
        exception: İstisna (varsa)
        severity: Hata şiddeti
        details: Ek ayrıntılar
        
    Returns:
        Hata ID'si
    """
    return error_manager.report_error(
        severity=severity,
        category=ErrorCategory.ANALYSIS,
        message=message,
        component=component,
        exception=exception,
        details=details
    )

def report_storage_error(error_manager: ErrorManager, message: str, component: str, 
                       exception: Optional[Exception] = None, 
                       severity: ErrorSeverity = ErrorSeverity.ERROR,
                       details: Optional[Dict[str, Any]] = None) -> str:
    """
    Depolama hatası bildir
    
    Args:
        error_manager: Hata yöneticisi
        message: Hata mesajı
        component: Bileşen adı
        exception: İstisna (varsa)
        severity: Hata şiddeti
        details: Ek ayrıntılar
        
    Returns:
        Hata ID'si
    """
    return error_manager.report_error(
        severity=severity,
        category=ErrorCategory.STORAGE,
        message=message,
        component=component,
        exception=exception,
        details=details
    )

def report_config_error(error_manager: ErrorManager, message: str, component: str, 
                      exception: Optional[Exception] = None, 
                      severity: ErrorSeverity = ErrorSeverity.ERROR,
                      details: Optional[Dict[str, Any]] = None) -> str:
    """
    Yapılandırma hatası bildir
    
    Args:
        error_manager: Hata yöneticisi
        message: Hata mesajı
        component: Bileşen adı
        exception: İstisna (varsa)
        severity: Hata şiddeti
        details: Ek ayrıntılar
        
    Returns:
        Hata ID'si
    """
    return error_manager.report_error(
        severity=severity,
        category=ErrorCategory.CONFIGURATION,
        message=message,
        component=component,
        exception=exception,
        details=details
    )

def report_ui_error(error_manager: ErrorManager, message: str, component: str, 
                  exception: Optional[Exception] = None, 
                  severity: ErrorSeverity = ErrorSeverity.WARNING,
                  details: Optional[Dict[str, Any]] = None) -> str:
    """
    UI hatası bildir
    
    Args:
        error_manager: Hata yöneticisi
        message: Hata mesajı
        component: Bileşen adı
        exception: İstisna (varsa)
        severity: Hata şiddeti
        details: Ek ayrıntılar
        
    Returns:
        Hata ID'si
    """
    return error_manager.report_error(
        severity=severity,
        category=ErrorCategory.UI,
        message=message,
        component=component,
        exception=exception,
        details=details
    )

# Özelleştirilmiş istisna sınıfları
class EyeTrackerError(Exception):
    """Temel göz izleme hatası"""
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN, 
                component: str = "", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.category = category
        self.component = component
        self.details = details or {}
        super().__init__(message)

class HardwareError(EyeTrackerError):
    """Donanım hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.HARDWARE, component, details)

class CalibrationError(EyeTrackerError):
    """Kalibrasyon hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.CALIBRATION, component, details)

class AnalysisError(EyeTrackerError):
    """Analiz hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.ANALYSIS, component, details)

class StorageError(EyeTrackerError):
    """Depolama hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.STORAGE, component, details)

class ConfigurationError(EyeTrackerError):
    """Yapılandırma hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.CONFIGURATION, component, details)

class NetworkError(EyeTrackerError):
    """Ağ hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.NETWORK, component, details)

class ModelError(EyeTrackerError):
    """Model hatası"""
    def __init__(self, message: str, component: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorCategory.MODEL, component, details) 