import tkinter as tk
from tkinter import ttk
import threading
import time
import random
import os
import sys

# Kendi modüllerimizi import ediyoruz
from error_manager import ErrorManager, ErrorSeverity, ErrorCategory, ErrorEvent
from error_ui_components import ErrorHandler, ErrorLogViewer

class EyeTrackingSimulator:
    """Göz izleme sistemini simüle eder ve çeşitli hatalar üretir"""
    
    def __init__(self):
        self.running = False
        self.error_manager = ErrorManager()
        self.simulation_thread = None
        
        # Hata üretimi için konfigürasyon
        self.error_probability = {
            ErrorSeverity.INFO: 0.6,      # Sık sık bilgi mesajları
            ErrorSeverity.WARNING: 0.3,   # Ara sıra uyarılar
            ErrorSeverity.ERROR: 0.1,     # Nadiren hatalar
            ErrorSeverity.CRITICAL: 0.02  # Çok nadiren kritik hatalar
        }
        
        self.possible_errors = {
            ErrorCategory.CAMERA: [
                "Kamera bağlantısı kesintili",
                "Düşük kamera çözünürlüğü",
                "Kamera ışık seviyesi çok düşük",
                "Kamera bulunamadı",
                "Kamera erişim izni reddedildi"
            ],
            ErrorCategory.TRACKING: [
                "Göz takibi kayboldu",
                "Pupil tespiti başarısız",
                "Göz kırpma algılanamıyor",
                "Çok hızlı göz hareketi",
                "Çok az göz hareketi tespit edildi"
            ],
            ErrorCategory.CALIBRATION: [
                "Kalibrasyon noktaları eksik",
                "Düşük kalibrasyon doğruluğu",
                "Kalibrasyon tamamlanamadı",
                "Kalibrasyon sapması algılandı",
                "Kalibrasyon geçersiz"
            ],
            ErrorCategory.SYSTEM: [
                "Düşük sistem belleği",
                "Yüksek CPU kullanımı",
                "Disk alanı yetersiz",
                "Sistem sıcaklığı yüksek",
                "İşlem gecikmesi algılandı"
            ],
            ErrorCategory.DATA: [
                "Veri kaydedilemedi",
                "Eksik veri noktaları",
                "Veri formatı uyumsuz",
                "Veri bozulması tespit edildi",
                "Analiz tamamlanamadı"
            ]
        }
        
        # Öneriler
        self.suggestions = {
            ErrorCategory.CAMERA: [
                "Kamera bağlantısını kontrol edin",
                "Işık seviyesini artırın",
                "Kamera erişim izinlerini kontrol edin",
                "Kamera sürücülerini güncelleyin",
                "Başka bir USB portuna bağlamayı deneyin"
            ],
            ErrorCategory.TRACKING: [
                "Gözlük kullanıyorsanız çıkarın",
                "Yüzünüzü kameraya daha iyi konumlandırın",
                "Farklı bir göz izleme algoritması deneyin",
                "Işık kaynağının pozisyonunu değiştirin",
                "Daha yüksek kontrastlı ekran ayarları kullanın"
            ],
            ErrorCategory.CALIBRATION: [
                "Kalibrasyonu tekrarlayın",
                "Tüm kalibrasyon noktalarına odaklanın",
                "Daha az baş hareketi ile kalibrasyonu yapın",
                "Kalibrasyon sırasında gözlük kullanmayın",
                "Daha fazla kalibrasyon noktası kullanın"
            ],
            ErrorCategory.SYSTEM: [
                "Diğer uygulamaları kapatın",
                "Sistemin soğumasını bekleyin",
                "Bilgisayarı yeniden başlatın",
                "Göz izleme yazılımını güncelleyin",
                "Donanım gereksinimlerini kontrol edin"
            ],
            ErrorCategory.DATA: [
                "Veritabanını kontrol edin",
                "Dosya izinlerini kontrol edin",
                "Disk alanını boşaltın",
                "Farklı bir veri formatı kullanın",
                "Verileri manuel olarak yedekleyin"
            ]
        }
    
    def start(self):
        """Simülasyonu başlat"""
        if self.simulation_thread is None or not self.simulation_thread.is_alive():
            self.running = True
            self.simulation_thread = threading.Thread(target=self._simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            # Başlangıç bildirimi
            self.error_manager.report_error(
                message="Göz izleme sistemi başlatıldı",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.INFO
            )
    
    def stop(self):
        """Simülasyonu durdur"""
        self.running = False
        if self.simulation_thread and self.simulation_thread.is_alive():
            self.simulation_thread.join(timeout=1.0)
            
            # Durma bildirimi
            self.error_manager.report_error(
                message="Göz izleme sistemi durduruldu",
                category=ErrorCategory.SYSTEM,
                severity=ErrorSeverity.INFO
            )
    
    def _simulation_loop(self):
        """Simülasyon döngüsü - rastgele hatalar üretir"""
        error_count = 0
        
        while self.running:
            # Her döngüde hata üretme olasılığını kontrol et
            if random.random() < 0.3:  # %30 ihtimalle hata üret
                # Hata ciddiyetini belirle
                severity_random = random.random()
                severity = None
                
                # Kümülatif olasılık hesabı ile ciddiyet seviyesini belirle
                cumulative_prob = 0
                for sev, prob in self.error_probability.items():
                    cumulative_prob += prob
                    if severity_random < cumulative_prob:
                        severity = sev
                        break
                
                if severity is None:
                    severity = ErrorSeverity.INFO
                
                # Kategoriyi rastgele seç
                category = random.choice(list(ErrorCategory))
                
                # Mesajı seç
                message = random.choice(self.possible_errors[category])
                
                # Önerileri hazırla
                num_suggestions = random.randint(1, 3)
                suggestions = random.sample(self.suggestions[category], num_suggestions)
                
                # Detayları oluştur
                details = {
                    "timestamp": time.time(),
                    "error_id": error_count,
                    "source": "EyeTrackingSimulator",
                    "location": f"Thread-{threading.current_thread().name}"
                }
                
                # Hatayı raporla
                self.error_manager.report_error(
                    message=message,
                    category=category,
                    severity=severity,
                    details=details,
                    suggestions=suggestions
                )
                
                error_count += 1
            
            # Rastgele gecikme
            time.sleep(random.uniform(1.0, 5.0))

class TestUI:
    """Test arayüzü"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Göz İzleme Hatası Yönetimi Test Uygulaması")
        self.root.geometry("800x600")
        self.root.minsize(800, 600)
        
        # Hata yöneticisi ve simülatörü başlat
        self.error_manager = ErrorManager()
        self.eye_tracking_simulator = EyeTrackingSimulator()
        self.error_handler = ErrorHandler(root)
        
        # Ana çerçeve
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        title_label = ttk.Label(
            main_frame, 
            text="Göz İzleme Hatası Yönetimi Test Uygulaması",
            font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))
        
        # Kontrol paneli
        control_frame = ttk.LabelFrame(main_frame, text="Kontroller", padding=10)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Simülasyon kontrolleri
        simulation_frame = ttk.Frame(control_frame)
        simulation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Başlat/durdur düğmeleri
        start_button = ttk.Button(
            simulation_frame,
            text="Simülasyonu Başlat",
            command=self._start_simulation
        )
        start_button.grid(row=0, column=0, padx=5, pady=5)
        
        stop_button = ttk.Button(
            simulation_frame,
            text="Simülasyonu Durdur",
            command=self._stop_simulation
        )
        stop_button.grid(row=0, column=1, padx=5, pady=5)
        
        # Ayırıcı
        ttk.Separator(simulation_frame, orient=tk.VERTICAL).grid(row=0, column=2, padx=10, pady=5, sticky="ns")
        
        # Manuel hata oluşturma
        ttk.Label(simulation_frame, text="Manuel Hata Oluştur:").grid(row=0, column=3, padx=5, pady=5)
        
        # Ciddiyet seçimi
        self.severity_var = tk.StringVar(value="INFO")
        severity_combo = ttk.Combobox(
            simulation_frame,
            textvariable=self.severity_var,
            values=[s.name for s in ErrorSeverity],
            width=10,
            state="readonly"
        )
        severity_combo.grid(row=0, column=4, padx=5, pady=5)
        
        # Kategori seçimi
        self.category_var = tk.StringVar(value="CAMERA")
        category_combo = ttk.Combobox(
            simulation_frame,
            textvariable=self.category_var,
            values=[c.name for c in ErrorCategory],
            width=10,
            state="readonly"
        )
        category_combo.grid(row=0, column=5, padx=5, pady=5)
        
        create_error_button = ttk.Button(
            simulation_frame,
            text="Hata Oluştur",
            command=self._create_manual_error
        )
        create_error_button.grid(row=0, column=6, padx=5, pady=5)
        
        # Hata günlüğü görüntüleme düğmesi
        view_log_button = ttk.Button(
            simulation_frame,
            text="Hata Günlüğünü Görüntüle",
            command=self._show_error_log
        )
        view_log_button.grid(row=0, column=7, padx=5, pady=5)
        
        # İzleme durumu
        status_frame = ttk.LabelFrame(main_frame, text="İzleme Durumu", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # İzleme alanı
        tracking_canvas = tk.Canvas(status_frame, bg="black")
        tracking_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Göz konum noktasını çiz
        self.eye_position = (400, 300)
        self.eye_target = (400, 300)
        self.eye_point = tracking_canvas.create_oval(
            self.eye_position[0]-5, self.eye_position[1]-5,
            self.eye_position[0]+5, self.eye_position[1]+5,
            fill="red", outline="white", tags="eye_point"
        )
        
        # Hedef noktasını çiz
        self.target_point = tracking_canvas.create_oval(
            self.eye_target[0]-10, self.eye_target[1]-10,
            self.eye_target[0]+10, self.eye_target[1]+10,
            outline="green", width=2, tags="target_point"
        )
        
        # Canvas referansını kaydet
        self.tracking_canvas = tracking_canvas
        
        # Canvas tıklama olayı
        tracking_canvas.bind("<Button-1>", self._set_target_position)
        
        # Animasyon zamanlaması
        self.root.after(50, self._update_eye_position)
        
        # Durum çubuğu
        self.status_var = tk.StringVar(value="Durum: Hazır")
        status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Program kapanırken temizlik işlemleri
        root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _start_simulation(self):
        """Simülasyonu başlat"""
        self.eye_tracking_simulator.start()
        self.status_var.set("Durum: Simülasyon Çalışıyor")
    
    def _stop_simulation(self):
        """Simülasyonu durdur"""
        self.eye_tracking_simulator.stop()
        self.status_var.set("Durum: Simülasyon Durduruldu")
    
    def _create_manual_error(self):
        """Manuel hata oluştur"""
        try:
            # Seçilen ciddiyet ve kategoriyi al
            severity_name = self.severity_var.get()
            category_name = self.category_var.get()
            
            severity = ErrorSeverity[severity_name]
            category = ErrorCategory[category_name]
            
            # Kategori için olası hatalardan birini seç
            message = random.choice(self.eye_tracking_simulator.possible_errors[category])
            
            # Önerileri hazırla
            suggestions = random.sample(
                self.eye_tracking_simulator.suggestions[category], 
                random.randint(1, 3)
            )
            
            # Detayları oluştur
            details = {
                "source": "ManualErrorCreation",
                "user_triggered": True
            }
            
            # Hatayı raporla
            self.error_manager.report_error(
                message=message,
                category=category,
                severity=severity,
                details=details,
                suggestions=suggestions
            )
            
            self.status_var.set(f"Durum: Manuel {severity_name} hatası oluşturuldu")
        except Exception as e:
            self.status_var.set(f"Durum: Hata oluşturma sırasında sorun: {str(e)}")
    
    def _show_error_log(self):
        """Hata günlüğünü görüntüle"""
        ErrorLogViewer(self.root)
    
    def _update_eye_position(self):
        """Göz pozisyonunu güncelle (animasyon için)"""
        # Hedef nokta ile mevcut nokta arasındaki farkı hesapla
        dx = self.eye_target[0] - self.eye_position[0]
        dy = self.eye_target[1] - self.eye_position[1]
        
        # Pozisyonu yumuşak bir şekilde güncelle
        new_x = self.eye_position[0] + dx * 0.1
        new_y = self.eye_position[1] + dy * 0.1
        
        # Rastgele titreşim ekle (gerçekçi göz hareketi)
        jitter_x = random.uniform(-2, 2)
        jitter_y = random.uniform(-2, 2)
        
        new_x += jitter_x
        new_y += jitter_y
        
        # Pozisyonu güncelle
        self.eye_position = (new_x, new_y)
        
        # Canvas üzerindeki noktayı güncelle
        self.tracking_canvas.coords(
            self.eye_point,
            new_x-5, new_y-5, new_x+5, new_y+5
        )
        
        # Tekrar zamanla
        self.root.after(50, self._update_eye_position)
    
    def _set_target_position(self, event):
        """Canvas'a tıklama ile hedef noktayı güncelle"""
        # Yeni hedef konumu
        self.eye_target = (event.x, event.y)
        
        # Hedef noktasını güncelle
        self.tracking_canvas.coords(
            self.target_point,
            event.x-10, event.y-10, event.x+10, event.y+10
        )
    
    def _on_closing(self):
        """Program kapanırken temizlik işlemleri"""
        self.eye_tracking_simulator.stop()
        self.error_handler.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TestUI(root)
    root.mainloop() 