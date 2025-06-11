import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Dict, List, Any, Optional, Callable, Tuple
import threading
import logging
import webbrowser
import os
from enum import Enum
import traceback
from PIL import ImageGrab, ImageTk

from .error_manager import ErrorEvent, ErrorSeverity, ErrorCategory, ErrorManager

logger = logging.getLogger('eye_tracker.error.ui')

class ErrorNotificationLevel(Enum):
    """Kullanıcı bildirim seviyesi"""
    SILENT = 0      # Sessiz (sadece log)
    STATUS_BAR = 1  # Durum çubuğu güncellemesi
    TOAST = 2       # Geçici bildirim
    DIALOG = 3      # Dialog penceresi (kullanıcı onayı gerekli)
    BLOCKING = 4    # Engelleyici dialog (devam etmek için çözülmesi gerekli)


class StatusbarComponent:
    """Durum çubuğu hata gösterimi"""
    
    def __init__(self, parent_frame: ttk.Frame):
        """
        Args:
            parent_frame: Üst çerçeve
        """
        self.parent = parent_frame
        
        # Durum çubuğu çerçevesi
        self.frame = ttk.Frame(parent_frame)
        self.frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Sol taraf: son hata mesajı
        self.error_var = tk.StringVar(value="")
        self.error_label = ttk.Label(self.frame, textvariable=self.error_var, 
                                   anchor=tk.W, padding=(5, 2))
        self.error_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Sağ taraf: hata sayaçları
        self.counters_frame = ttk.Frame(self.frame)
        self.counters_frame.pack(side=tk.RIGHT, padx=5)
        
        # Hata sayaçları
        self.counters = {}
        for severity in [ErrorSeverity.ERROR, ErrorSeverity.WARNING, ErrorSeverity.INFO]:
            var = tk.StringVar(value="0")
            label = ttk.Label(self.counters_frame, textvariable=var, 
                            padding=(5, 2), width=8)
            
            # Şiddet seviyesine göre stil
            if severity == ErrorSeverity.ERROR:
                label.configure(foreground="red")
            elif severity == ErrorSeverity.WARNING:
                label.configure(foreground="orange")
            elif severity == ErrorSeverity.INFO:
                label.configure(foreground="blue")
            
            label.pack(side=tk.LEFT)
            self.counters[severity] = var
        
        # Ayırıcı
        ttk.Separator(self.frame, orient=tk.HORIZONTAL).pack(fill=tk.X, side=tk.TOP)
        
        # Temizleme fonksiyonu için zamanlayıcı
        self._clear_timer = None
    
    def show_error(self, error: ErrorEvent, duration_ms: int = 10000) -> None:
        """
        Hatayı durum çubuğunda göster
        
        Args:
            error: Hata olayı
            duration_ms: Gösterim süresi (ms)
        """
        # Şiddet ve kategori
        prefix = f"[{error.severity.name}:{error.category.name}]"
        
        # Hata mesajı
        message = f"{prefix} {error.message}"
        
        # Durum çubuğunda göster
        self.error_var.set(message)
        
        # Label stilini şiddet seviyesine göre ayarla
        if error.severity == ErrorSeverity.ERROR or error.severity == ErrorSeverity.CRITICAL:
            self.error_label.configure(foreground="red")
        elif error.severity == ErrorSeverity.WARNING:
            self.error_label.configure(foreground="orange")
        elif error.severity == ErrorSeverity.INFO:
            self.error_label.configure(foreground="blue")
        else:
            self.error_label.configure(foreground="black")
        
        # Sayaçları güncelle
        for severity in [ErrorSeverity.ERROR, ErrorSeverity.WARNING, ErrorSeverity.INFO]:
            if error.severity == severity:
                count = int(self.counters[severity].get()) + 1
                self.counters[severity].set(str(count))
        
        # Otomatik temizleme zamanlayıcısını iptal et (varsa)
        if self._clear_timer is not None:
            self.parent.after_cancel(self._clear_timer)
        
        # Yeni temizleme zamanlayıcısı
        self._clear_timer = self.parent.after(duration_ms, self._clear_message)
    
    def _clear_message(self) -> None:
        """Hata mesajını temizle"""
        self.error_var.set("")
        self.error_label.configure(foreground="black")
        self._clear_timer = None
    
    def clear_counters(self) -> None:
        """Hata sayaçlarını sıfırla"""
        for var in self.counters.values():
            var.set("0")


class ToastNotification:
    """Geçici ekran bildirimi"""
    
    def __init__(self, parent: tk.Tk):
        """
        Args:
            parent: Ana pencere
        """
        self.parent = parent
        self.notifications = []  # Aktif bildirimleri tut
        self.max_notifications = 3  # Aynı anda gösterilecek maksimum bildirim
    
    def show_notification(self, title: str, message: str, 
                        severity: ErrorSeverity = ErrorSeverity.INFO, 
                        duration_ms: int = 5000) -> None:
        """
        Bildirim göster
        
        Args:
            title: Bildirim başlığı
            message: Bildirim mesajı
            severity: Şiddet seviyesi
            duration_ms: Gösterim süresi (ms)
        """
        # Maksimum bildirim sayısını kontrol et
        if len(self.notifications) >= self.max_notifications:
            # En eski bildirimi kapat
            if self.notifications:
                oldest = self.notifications[0]
                oldest.destroy()
                self.notifications.pop(0)
        
        # Ekran boyutlarını al
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        
        # Bildirim penceresi oluştur
        notification = tk.Toplevel(self.parent)
        notification.title("")
        notification.attributes("-topmost", True)
        notification.overrideredirect(True)  # Başlık çubuğunu gizle
        
        # Pencere boyutu
        width = 300
        height = 100
        
        # Sağ alt köşede konumlandır
        x = screen_width - width - 20
        y = screen_height - height - 40
        
        # Mevcut bildirimlere göre y konumunu ayarla
        y_offset = len(self.notifications) * (height + 10)
        y -= y_offset
        
        notification.geometry(f"{width}x{height}+{x}+{y}")
        
        # Arkaplan rengini şiddete göre ayarla
        bg_color = "#f0f0f0"  # Varsayılan
        fg_color = "#000000"  # Varsayılan
        
        if severity == ErrorSeverity.ERROR or severity == ErrorSeverity.CRITICAL:
            bg_color = "#ffeded"  # Açık kırmızı
            fg_color = "#cc0000"  # Koyu kırmızı
        elif severity == ErrorSeverity.WARNING:
            bg_color = "#fff5e6"  # Açık turuncu
            fg_color = "#cc7000"  # Koyu turuncu
        elif severity == ErrorSeverity.INFO:
            bg_color = "#e6f2ff"  # Açık mavi
            fg_color = "#0066cc"  # Koyu mavi
        
        notification.configure(bg=bg_color)
        
        # İçerik çerçevesi
        frame = ttk.Frame(notification, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = ttk.Label(frame, text=title, font=("TkDefaultFont", 10, "bold"), 
                              foreground=fg_color)
        title_label.pack(anchor=tk.W)
        
        # Mesaj
        message_label = ttk.Label(frame, text=message, wraplength=width-20)
        message_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Kapat düğmesi
        close_btn = ttk.Button(frame, text="X", width=2, 
                             command=lambda: self._close_notification(notification))
        close_btn.place(relx=1.0, rely=0.0, anchor=tk.NE)
        
        # Bildirim listesine ekle
        self.notifications.append(notification)
        
        # Otomatik kapatma zamanlayıcısı
        self.parent.after(duration_ms, lambda: self._close_notification(notification))
    
    def _close_notification(self, notification: tk.Toplevel) -> None:
        """
        Bildirimi kapat
        
        Args:
            notification: Kapatılacak bildirim penceresi
        """
        # Bildirim listesinden kaldır
        if notification in self.notifications:
            self.notifications.remove(notification)
        
        # Pencereyi kapat
        notification.destroy()
        
        # Kalan bildirimleri yeniden konumlandır
        self._reposition_notifications()
    
    def _reposition_notifications(self) -> None:
        """Aktif bildirimleri yeniden konumlandır"""
        screen_width = self.parent.winfo_screenwidth()
        screen_height = self.parent.winfo_screenheight()
        
        width = 300
        height = 100
        
        # Bildirim konumlarını güncelle
        for i, notification in enumerate(self.notifications):
            x = screen_width - width - 20
            y = screen_height - height - 40 - (i * (height + 10))
            notification.geometry(f"+{x}+{y}")


class ErrorDialogManager:
    """Hata dialog penceresi yönetimi"""
    
    def __init__(self, parent: tk.Tk, error_manager: ErrorManager):
        """
        Args:
            parent: Ana pencere
            error_manager: Hata yöneticisi
        """
        self.parent = parent
        self.error_manager = error_manager
        self.active_dialogs = {}  # Aktif dialog pencereleri {error_id: Dialog}
    
    def show_error_dialog(self, error: ErrorEvent, 
                         callback: Optional[Callable[[str, bool], None]] = None) -> None:
        """
        Hata dialog penceresini göster
        
        Args:
            error: Hata olayı
            callback: Kapanış callback fonksiyonu (error_id, handled)
        """
        # Dialog zaten gösteriliyor mu kontrol et
        if error.error_id in self.active_dialogs:
            # Mevcut dialogu öne getir
            dialog = self.active_dialogs[error.error_id]
            dialog.lift()
            return
        
        # Yeni dialog oluştur
        dialog = ErrorDialog(self.parent, error, self.error_manager, 
                           callback=lambda handled: self._on_dialog_closed(error.error_id, handled, callback))
        
        # Aktif dialoglara ekle
        self.active_dialogs[error.error_id] = dialog
    
    def _on_dialog_closed(self, error_id: str, handled: bool, 
                        callback: Optional[Callable[[str, bool], None]]) -> None:
        """
        Dialog kapanışını işle
        
        Args:
            error_id: Hata ID'si
            handled: Hata kullanıcı tarafından işlendi mi
            callback: Dış callback fonksiyonu
        """
        # Aktif dialoglardan kaldır
        if error_id in self.active_dialogs:
            del self.active_dialogs[error_id]
        
        # Callback çağır (varsa)
        if callback:
            callback(error_id, handled)
    
    def close_all_dialogs(self) -> None:
        """Tüm aktif dialogları kapat"""
        # Aktif dialog listesi kopyasını al
        active_errors = list(self.active_dialogs.keys())
        
        # Tüm dialogları kapat
        for error_id in active_errors:
            if error_id in self.active_dialogs:
                dialog = self.active_dialogs[error_id]
                dialog.destroy()
                del self.active_dialogs[error_id]


class ErrorDialog(tk.Toplevel):
    """Hata dialog penceresi"""
    
    def __init__(self, parent: tk.Tk, error: ErrorEvent, error_manager: ErrorManager,
                callback: Optional[Callable[[bool], None]] = None):
        """
        Args:
            parent: Ana pencere
            error: Hata olayı
            error_manager: Hata yöneticisi
            callback: Kapanış callback fonksiyonu (handled)
        """
        super().__init__(parent)
        self.error = error
        self.error_manager = error_manager
        self.callback = callback
        
        # Pencere ayarları
        self.title(f"{error.severity.name}: {error.category.name}")
        self.geometry("500x400")
        self.minsize(400, 300)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Şiddet seviyesine göre simge
        self.iconbitmap("") # Varsayılan simgeyi temizle
        
        # Ana çerçeve
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Hata bilgileri
        info_frame = ttk.LabelFrame(main_frame, text="Hata Bilgileri", padding=10)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Şiddet ve kategori (ikon ve renkli metin)
        severity_frame = ttk.Frame(info_frame)
        severity_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Şiddet seviyesi etiketi
        severity_label = ttk.Label(severity_frame, 
                                 text=f"Şiddet: {error.severity.name}",
                                 font=("TkDefaultFont", 10, "bold"))
        
        # Şiddet seviyesine göre renk
        if error.severity == ErrorSeverity.ERROR or error.severity == ErrorSeverity.CRITICAL:
            severity_label.configure(foreground="red")
        elif error.severity == ErrorSeverity.WARNING:
            severity_label.configure(foreground="orange")
        elif error.severity == ErrorSeverity.INFO:
            severity_label.configure(foreground="blue")
        
        severity_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Kategori etiketi
        ttk.Label(severity_frame, 
                text=f"Kategori: {error.category.name}",
                font=("TkDefaultFont", 10)).pack(side=tk.LEFT)
        
        # Hata mesajı
        message_frame = ttk.Frame(info_frame)
        message_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(message_frame, text="Hata:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(message_frame, text=error.message, wraplength=400).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Bileşen
        comp_frame = ttk.Frame(info_frame)
        comp_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(comp_frame, text="Bileşen:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(comp_frame, text=error.component).pack(side=tk.LEFT)
        
        # Zaman
        time_frame = ttk.Frame(info_frame)
        time_frame.pack(fill=tk.X)
        
        ttk.Label(time_frame, text="Zaman:").pack(side=tk.LEFT, padx=(0, 5))
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(error.timestamp))
        ttk.Label(time_frame, text=time_str).pack(side=tk.LEFT)
        
        # Hata ID'si
        id_frame = ttk.Frame(info_frame)
        id_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(id_frame, text="Hata ID:").pack(side=tk.LEFT, padx=(0, 5))
        id_var = tk.StringVar(value=error.error_id)
        id_entry = ttk.Entry(id_frame, textvariable=id_var, state="readonly", width=40)
        id_entry.pack(side=tk.LEFT)
        
        # Kopyalama düğmesi
        ttk.Button(id_frame, text="Kopyala", width=8, 
                 command=lambda: self._copy_to_clipboard(error.error_id)).pack(side=tk.LEFT, padx=(5, 0))
        
        # Çözüm önerileri
        if error.suggestions:
            suggestions_frame = ttk.LabelFrame(main_frame, text="Çözüm Önerileri", padding=10)
            suggestions_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # Liste görünümü
            suggestions_listbox = tk.Listbox(suggestions_frame, height=5)
            suggestions_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
            
            # Kaydırma çubuğu
            scrollbar = ttk.Scrollbar(suggestions_frame, orient="vertical", 
                                    command=suggestions_listbox.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            suggestions_listbox.configure(yscrollcommand=scrollbar.set)
            
            # Önerileri ekle
            for suggestion in error.suggestions:
                suggestions_listbox.insert(tk.END, suggestion)
        
        # İstisna izleme (varsa)
        if error.traceback:
            traceback_frame = ttk.LabelFrame(main_frame, text="Hata İzleme", padding=10)
            traceback_frame.pack(fill=tk.BOTH, expand=True)
            
            # Metin alanı
            traceback_text = tk.Text(traceback_frame, wrap=tk.WORD, height=5, 
                                   font=("Courier", 9))
            traceback_text.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
            traceback_text.insert(tk.END, error.traceback)
            traceback_text.configure(state="disabled")
            
            # Kaydırma çubuğu
            tb_scrollbar = ttk.Scrollbar(traceback_frame, orient="vertical", 
                                       command=traceback_text.yview)
            tb_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            traceback_text.configure(yscrollcommand=tb_scrollbar.set)
        
        # Düğme çerçevesi
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # İşaretleme düğmeleri
        ttk.Button(button_frame, text="Yardım", 
                 command=self._show_help).pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Rapor", 
                 command=self._report_error).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Günlüğü Görüntüle", 
                 command=self._view_log).pack(side=tk.LEFT)
        
        ttk.Button(button_frame, text="Çözüldü", 
                 command=self._mark_as_handled).pack(side=tk.RIGHT)
        
        ttk.Button(button_frame, text="Kapat", 
                 command=self._on_close).pack(side=tk.RIGHT, padx=5)
    
    def _copy_to_clipboard(self, text: str) -> None:
        """
        Metni panoya kopyala
        
        Args:
            text: Kopyalanacak metin
        """
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
    
    def _show_help(self) -> None:
        """Hata kategorisi için yardım bilgisini göster"""
        # Yardım mesajı
        help_text = "Hata çözümü için bilgiler:\n\n"
        
        if self.error.category == ErrorCategory.HARDWARE:
            help_text += "Donanım hataları genellikle kamera veya sensör ile ilgili sorunlardan kaynaklanır.\n\n"
            help_text += "- Kamera bağlantısını kontrol edin.\n"
            help_text += "- Kamera sürücülerinin güncel olduğundan emin olun.\n"
            help_text += "- Başka bir kamera kullanmayı deneyin.\n"
            help_text += "- Yapılandırma ayarlarını kontrol edin."
        elif self.error.category == ErrorCategory.CALIBRATION:
            help_text += "Kalibrasyon hataları, göz izleme sistemi kalibrasyonu sırasında oluşan sorunlardır.\n\n"
            help_text += "- Daha iyi aydınlatmalı bir ortamda tekrar deneyin.\n"
            help_text += "- Kalibrasyon sırasında başınızı sabit tutun.\n"
            help_text += "- Daha fazla kalibrasyon noktası kullanın.\n"
            help_text += "- Kamera konumunu ayarlayın."
        elif self.error.category == ErrorCategory.ANALYSIS:
            help_text += "Analiz hataları, göz verilerinin işlenmesi sırasında oluşan sorunlardır.\n\n"
            help_text += "- Yeniden kalibrasyon yapın.\n"
            help_text += "- Analiz parametrelerini varsayılan değerlere sıfırlayın.\n"
            help_text += "- Daha uzun bir veri tamponu kullanın.\n"
            help_text += "- Daha iyi aydınlatmalı bir ortamda tekrar deneyin."
        elif self.error.category == ErrorCategory.STORAGE:
            help_text += "Depolama hataları, verilerin kaydedilmesi sırasında oluşan sorunlardır.\n\n"
            help_text += "- Disk alanınızı kontrol edin.\n"
            help_text += "- Çıktı dizininin yazılabilir olduğundan emin olun.\n"
            help_text += "- Dosya izinlerini kontrol edin.\n"
            help_text += "- Farklı bir çıktı dizini deneyin."
        elif self.error.category == ErrorCategory.CONFIGURATION:
            help_text += "Yapılandırma hataları, ayarlar ile ilgili sorunlardan kaynaklanır.\n\n"
            help_text += "- Yapılandırma dosyasının geçerli olduğunu kontrol edin.\n"
            help_text += "- Varsayılan yapılandırmayı yükleyin.\n"
            help_text += "- Yapılandırma parametrelerini kontrol edin.\n"
            help_text += "- Yapılandırma dosyasını yeniden oluşturun."
        else:
            help_text += "Bu hata kategorisi için özel yardım bilgisi bulunmamaktadır.\n\n"
            help_text += "Çözüm önerilerini inceleyiniz veya teknik destek ile iletişime geçiniz."
        
        # Yardım mesajını göster
        messagebox.showinfo("Hata Çözümü Yardım", help_text)
    
    def _report_error(self) -> None:
        """Hata raporlama"""
        # Raporlama mesajı
        report_text = "Hata detayları e-posta olarak göndermek için aşağıdaki bilgileri kullanabilirsiniz:\n\n"
        report_text += f"Hata ID: {self.error.error_id}\n"
        report_text += f"Kategori: {self.error.category.name}\n"
        report_text += f"Şiddet: {self.error.severity.name}\n"
        report_text += f"Mesaj: {self.error.message}\n"
        report_text += f"Bileşen: {self.error.component}\n"
        report_text += f"Zaman: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.error.timestamp))}\n"
        
        # Hata ayrıntıları (varsa)
        if self.error.details:
            report_text += "\nAyrıntılar:\n"
            for key, value in self.error.details.items():
                report_text += f"- {key}: {value}\n"
        
        # E-posta oluştur
        subject = f"Hata Raporu: {self.error.category.name} - {self.error.error_id}"
        body = report_text.replace('\n', '%0D%0A')
        mailto_link = f"mailto:support@example.com?subject={subject}&body={body}"
        
        # E-posta istemcisini aç
        try:
            webbrowser.open(mailto_link)
        except Exception as e:
            messagebox.showerror("Hata", f"E-posta istemcisi açılamadı: {str(e)}")
    
    def _view_log(self) -> None:
        """Günlük dosyasını görüntüle"""
        log_file = "eye_analysis.log"
        
        # Dosya varsa aç
        if os.path.exists(log_file):
            try:
                # Varsayılan metin düzenleyici ile aç
                os.startfile(log_file)
            except Exception:
                try:
                    # Linux sistemlerde
                    import subprocess
                    subprocess.call(["xdg-open", log_file])
                except Exception as e:
                    messagebox.showerror("Hata", f"Günlük dosyası açılamadı: {str(e)}")
        else:
            messagebox.showinfo("Bilgi", f"Günlük dosyası bulunamadı: {log_file}")
    
    def _mark_as_handled(self) -> None:
        """Hatayı çözüldü olarak işaretle"""
        # Hata yöneticisinde işaretle
        self.error_manager.mark_error_as_handled(self.error.error_id, True)
        
        # Kullanıcıya bilgi ver
        messagebox.showinfo("Bilgi", "Hata çözüldü olarak işaretlendi.")
        
        # Dialog penceresini kapat
        self._on_close(handled=True)
    
    def _on_close(self, handled: bool = False) -> None:
        """
        Dialog penceresini kapat
        
        Args:
            handled: Hata işlendi mi
        """
        # Callback çağır (varsa)
        if self.callback:
            self.callback(handled)
        
        # Pencereyi kapat
        self.destroy()


class ErrorNotifier:
    """
    Hata bildirimi yöneticisi
    
    Farklı hata seviyelerine göre kullanıcı arayüzü bildirimleri
    """
    
    def __init__(self, parent: tk.Tk, error_manager: ErrorManager):
        """
        Args:
            parent: Ana pencere
            error_manager: Hata yöneticisi
        """
        self.parent = parent
        self.error_manager = error_manager
        
        # Durum çubuğu
        self.statusbar = None
        
        # Toast bildirimleri
        self.toast = ToastNotification(parent)
        
        # Dialog yöneticisi
        self.dialog_manager = ErrorDialogManager(parent, error_manager)
        
        # Hata bildirim seviyesi ayarları
        self.notification_levels = {
            ErrorSeverity.INFO: ErrorNotificationLevel.STATUS_BAR,
            ErrorSeverity.WARNING: ErrorNotificationLevel.TOAST,
            ErrorSeverity.ERROR: ErrorNotificationLevel.DIALOG,
            ErrorSeverity.CRITICAL: ErrorNotificationLevel.DIALOG,
            ErrorSeverity.FATAL: ErrorNotificationLevel.BLOCKING
        }
        
        # Fatal hatalar için callback
        self.fatal_error_callback = None
    
    def set_statusbar(self, statusbar: StatusbarComponent) -> None:
        """
        Durum çubuğunu ayarla
        
        Args:
            statusbar: Durum çubuğu bileşeni
        """
        self.statusbar = statusbar
    
    def set_notification_level(self, severity: ErrorSeverity, 
                             level: ErrorNotificationLevel) -> None:
        """
        Hata şiddeti için bildirim seviyesini ayarla
        
        Args:
            severity: Hata şiddeti
            level: Bildirim seviyesi
        """
        self.notification_levels[severity] = level
    
    def set_fatal_error_callback(self, callback: Callable[[ErrorEvent], None]) -> None:
        """
        Fatal hata için callback ayarla
        
        Args:
            callback: Callback fonksiyonu
        """
        self.fatal_error_callback = callback
    
    def notify(self, error: ErrorEvent) -> None:
        """
        Hata bildirimi göster
        
        Args:
            error: Hata olayı
        """
        # Bildirim seviyesini belirle
        notification_level = self.notification_levels[error.severity]
        
        # Seviyeye göre bildirim göster
        if notification_level == ErrorNotificationLevel.SILENT:
            # Sessiz mod, sadece loga yaz
            pass
        
        elif notification_level == ErrorNotificationLevel.STATUS_BAR:
            # Durum çubuğunda göster
            if self.statusbar:
                self.statusbar.show_error(error)
        
        elif notification_level == ErrorNotificationLevel.TOAST:
            # Toast bildirimi göster
            title = f"{error.severity.name}: {error.category.name}"
            self.toast.show_notification(title, error.message, error.severity)
            
            # Durum çubuğunda da göster
            if self.statusbar:
                self.statusbar.show_error(error)
        
        elif notification_level == ErrorNotificationLevel.DIALOG:
            # Dialog penceresi göster
            self.dialog_manager.show_error_dialog(error, 
                                               callback=lambda error_id, handled: 
                                               self._on_dialog_closed(error_id, handled))
            
            # Durum çubuğunda da göster
            if self.statusbar:
                self.statusbar.show_error(error)
        
        elif notification_level == ErrorNotificationLevel.BLOCKING:
            # Engelleyici dialog penceresi göster
            self.dialog_manager.show_error_dialog(error, 
                                               callback=lambda error_id, handled: 
                                               self._on_blocking_dialog_closed(error, error_id, handled))
            
            # Durum çubuğunda da göster
            if self.statusbar:
                self.statusbar.show_error(error)
            
            # Fatal hata callback'i (varsa)
            if error.severity == ErrorSeverity.FATAL and self.fatal_error_callback:
                self.fatal_error_callback(error)
    
    def _on_dialog_closed(self, error_id: str, handled: bool) -> None:
        """
        Dialog kapanışını işle
        
        Args:
            error_id: Hata ID'si
            handled: Hata işlendi mi
        """
        # Hata işlendiyse, hata yöneticisinde de işaretle
        if handled:
            self.error_manager.mark_error_as_handled(error_id, True)
    
    def _on_blocking_dialog_closed(self, error: ErrorEvent, error_id: str, handled: bool) -> None:
        """
        Engelleyici dialog kapanışını işle
        
        Args:
            error: Hata olayı
            error_id: Hata ID'si
            handled: Hata işlendi mi
        """
        # Dialog kapanışını işle
        self._on_dialog_closed(error_id, handled)
        
        # Fatal hata ve işlenmediyse uygulamayı kapat
        if error.severity == ErrorSeverity.FATAL and not handled:
            messagebox.showerror("Kritik Hata", 
                               "Kritik bir hata nedeniyle uygulama kapatılacak.")
            self.parent.quit()


# Yardımcı Fonksiyonlar
def format_error_for_display(error: ErrorEvent) -> dict:
    """
    Hata olayını görüntüleme için formatla
    
    Args:
        error: Hata olayı
        
    Returns:
        Formatlanmış hata bilgileri
    """
    return {
        "id": error.error_id,
        "zaman": error.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "seviye": error.severity.name,
        "kategori": error.category.name,
        "mesaj": error.message,
        "detay": error.details or "-",
        "çözüm": error.solution or "-",
        "işlendi": "Evet" if error.handled else "Hayır"
    }

def get_severity_color(severity: ErrorSeverity) -> str:
    """
    Hata şiddeti için renk kodu döndür
    
    Args:
        severity: Hata şiddeti
        
    Returns:
        Renk kodu
    """
    if severity == ErrorSeverity.INFO:
        return "#4caf50"  # Yeşil
    elif severity == ErrorSeverity.WARNING:
        return "#ff9800"  # Turuncu
    elif severity == ErrorSeverity.ERROR:
        return "#f44336"  # Kırmızı
    elif severity == ErrorSeverity.CRITICAL:
        return "#9c27b0"  # Mor
    elif severity == ErrorSeverity.FATAL:
        return "#000000"  # Siyah
    else:
        return "#2196f3"  # Mavi

def get_error_icon(severity: ErrorSeverity, size: int = 16) -> PhotoImage:
    """
    Hata şiddeti için ikon oluştur
    
    Args:
        severity: Hata şiddeti
        size: İkon boyutu
        
    Returns:
        PhotoImage nesnesi
    """
    color = get_severity_color(severity)
    
    # Canvas üzerinde ikon oluştur
    icon_canvas = tk.Canvas(width=size, height=size, highlightthickness=0)
    
    if severity == ErrorSeverity.INFO:
        # Bilgi ikonu (i)
        icon_canvas.create_oval(2, 2, size-2, size-2, outline=color, width=2)
        icon_canvas.create_line(size//2, size//4, size//2, size//2, fill=color, width=2)
        icon_canvas.create_oval(size//2-1, size*3//4, size//2+1, size*3//4+2, fill=color, outline=color)
    
    elif severity == ErrorSeverity.WARNING:
        # Uyarı ikonu (üçgen)
        icon_canvas.create_polygon(size//2, 2, size-2, size-2, 2, size-2, fill="", outline=color, width=2)
        icon_canvas.create_line(size//2, size//3, size//2, size*2//3, fill=color, width=2)
        icon_canvas.create_oval(size//2-1, size*3//4, size//2+1, size*3//4+2, fill=color, outline=color)
    
    elif severity in [ErrorSeverity.ERROR, ErrorSeverity.CRITICAL, ErrorSeverity.FATAL]:
        # Hata ikonu (X)
        icon_canvas.create_oval(2, 2, size-2, size-2, outline=color, width=2)
        icon_canvas.create_line(size//3, size//3, size*2//3, size*2//3, fill=color, width=2)
        icon_canvas.create_line(size*2//3, size//3, size//3, size*2//3, fill=color, width=2)
    
    # Canvas içeriğini fotoğrafa dönüştür
    icon_canvas.update()
    icon_img = ImageGrab.grab(bbox=(
        icon_canvas.winfo_rootx(), 
        icon_canvas.winfo_rooty(),
        icon_canvas.winfo_rootx() + icon_canvas.winfo_width(),
        icon_canvas.winfo_rooty() + icon_canvas.winfo_height()
    ))
    
    return ImageTk.PhotoImage(icon_img)

def truncate_text(text: str, max_length: int = 50) -> str:
    """
    Metni belirli bir uzunlukta kısalt
    
    Args:
        text: Metin
        max_length: Maksimum uzunluk
        
    Returns:
        Kısaltılmış metin
    """
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..." 