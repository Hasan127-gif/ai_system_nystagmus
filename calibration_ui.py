#!/usr/bin/env python3
"""
KALİBRASYON KULLANICI ARAYÜZÜ
============================
Göz takibi için kamera kalibrasyonu arayüzü.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import threading
import time
from typing import List, Dict, Tuple
import logging

from calibration import get_calibrator

logger = logging.getLogger(__name__)

class CalibrationUI:
    """Kalibrasyon kullanıcı arayüzü."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nistagmus Sistemi - Kamera Kalibrasyonu")
        self.root.geometry("800x600")
        
        self.calibrator = get_calibrator()
        self.camera = None
        self.calibration_points = []
        self.current_point_index = 0
        
        # Kalibrasyon noktaları (ekran pozisyonları ve beklenen açılar)
        self.target_points = [
            {"screen_pos": (100, 100), "angle_deg": 15.0, "label": "Sol Üst"},
            {"screen_pos": (400, 100), "angle_deg": 5.0, "label": "Üst Merkez"},
            {"screen_pos": (700, 100), "angle_deg": 15.0, "label": "Sağ Üst"},
            {"screen_pos": (100, 300), "angle_deg": 10.0, "label": "Sol Merkez"},
            {"screen_pos": (400, 300), "angle_deg": 0.0, "label": "Merkez"},
            {"screen_pos": (700, 300), "angle_deg": 10.0, "label": "Sağ Merkez"},
            {"screen_pos": (100, 500), "angle_deg": 15.0, "label": "Sol Alt"},
            {"screen_pos": (400, 500), "angle_deg": 5.0, "label": "Alt Merkez"},
            {"screen_pos": (700, 500), "angle_deg": 15.0, "label": "Sağ Alt"}
        ]
        
        self.setup_ui()
        self.update_status()
    
    def setup_ui(self):
        """Arayüzü kur."""
        # Ana frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Başlık
        title_label = ttk.Label(main_frame, text="Kamera Kalibrasyonu", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Durum bilgisi
        self.status_frame = ttk.LabelFrame(main_frame, text="Kalibrasyon Durumu", padding="10")
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label = ttk.Label(self.status_frame, text="")
        self.status_label.grid(row=0, column=0, sticky=tk.W)
        
        # Kalibrasyon butonları
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Kalibrasyonu Başlat", 
                                      command=self.start_calibration)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.reset_button = ttk.Button(button_frame, text="Kalibrasyonu Sıfırla", 
                                      command=self.reset_calibration)
        self.reset_button.grid(row=0, column=1, padx=5)
        
        self.test_button = ttk.Button(button_frame, text="Kalibrasyonu Test Et", 
                                     command=self.test_calibration)
        self.test_button.grid(row=0, column=2, padx=5)
        
        # Manuel kalibrasyon
        manual_frame = ttk.LabelFrame(main_frame, text="Manuel Kalibrasyon", padding="10")
        manual_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(manual_frame, text="Piksel/Derece Oranı:").grid(row=0, column=0, sticky=tk.W)
        self.ratio_var = tk.StringVar(value=str(self.calibrator.calibration_data["pixel_to_degree_ratio"]))
        ratio_entry = ttk.Entry(manual_frame, textvariable=self.ratio_var, width=10)
        ratio_entry.grid(row=0, column=1, padx=5)
        
        manual_save_button = ttk.Button(manual_frame, text="Manuel Kaydet", 
                                       command=self.save_manual_calibration)
        manual_save_button.grid(row=0, column=2, padx=5)
        
        # Sonuçlar
        self.results_frame = ttk.LabelFrame(main_frame, text="Kalibrasyon Sonuçları", padding="10")
        self.results_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.results_text = tk.Text(self.results_frame, height=8, width=60)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.results_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.results_text.config(yscrollcommand=scrollbar.set)
    
    def update_status(self):
        """Durum bilgisini güncelle."""
        status = self.calibrator.get_calibration_status()
        
        if status["calibrated"]:
            status_text = f"✅ Kalibre edilmiş\n"
            status_text += f"Piksel/Derece: {status['pixel_to_degree_ratio']:.4f}\n"
            status_text += f"Kalite: {status['quality'].get('ratio_std', 0):.4f} std"
        else:
            status_text = "❌ Kalibre edilmemiş\n"
            status_text += f"Varsayılan oran: {status['pixel_to_degree_ratio']:.4f}"
        
        self.status_label.config(text=status_text)
    
    def start_calibration(self):
        """Kalibrasyon sürecini başlat."""
        try:
            # Kamera aç
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Hata", "Kamera açılamadı!")
                return
            
            self.calibration_points = []
            self.current_point_index = 0
            
            # Kalibrasyon penceresi aç
            self.show_calibration_point()
            
        except Exception as e:
            messagebox.showerror("Hata", f"Kalibrasyon başlatılamadı: {e}")
    
    def show_calibration_point(self):
        """Kalibrasyon noktasını göster."""
        if self.current_point_index >= len(self.target_points):
            # Kalibrasyon tamamlandı
            self.finish_calibration()
            return
        
        point = self.target_points[self.current_point_index]
        
        # Kalibrasyon penceresi
        cal_window = tk.Toplevel(self.root)
        cal_window.title(f"Kalibrasyon - {point['label']}")
        cal_window.geometry("800x600")
        cal_window.configure(bg='black')
        
        # Hedef nokta
        canvas = tk.Canvas(cal_window, width=800, height=600, bg='black')
        canvas.pack()
        
        # Hedef daire
        x, y = point["screen_pos"]
        canvas.create_oval(x-20, y-20, x+20, y+20, fill='red', outline='white', width=2)
        canvas.create_text(x, y-40, text=point['label'], fill='white', font=("Arial", 12))
        
        # Talimatlar
        canvas.create_text(400, 50, text=f"Bu kırmızı noktaya bakın ve SPACE tuşuna basın",
                          fill='white', font=("Arial", 14))
        canvas.create_text(400, 550, text=f"Nokta {self.current_point_index + 1}/{len(self.target_points)}",
                          fill='gray', font=("Arial", 10))
        
        # Klavye olayı
        def on_space(event):
            pixel_pos = self.capture_eye_position()
            if pixel_pos:
                self.calibration_points.append({
                    "screen_pos": point["screen_pos"],
                    "pixel_pos": pixel_pos,
                    "angle_deg": point["angle_deg"]
                })
                
                self.current_point_index += 1
                cal_window.destroy()
                
                # Sonraki nokta
                self.root.after(500, self.show_calibration_point)
            else:
                messagebox.showwarning("Uyarı", "Göz tespit edilemedi, tekrar deneyin")
        
        cal_window.bind('<space>', on_space)
        cal_window.focus_set()
    
    def capture_eye_position(self) -> Tuple[int, int] or None:
        """Mevcut göz pozisyonunu yakala."""
        if not self.camera:
            return None
        
        try:
            # Birkaç kare al ve ortalama al
            positions = []
            for _ in range(5):
                ret, frame = self.camera.read()
                if ret:
                    # Basit göz tespiti (MediaPipe kullanarak)
                    from analysis_utils import detect_iris_centers_basic
                    left_center, right_center = detect_iris_centers_basic(frame)
                    
                    if left_center and right_center:
                        # İki göz merkezinin ortası
                        center_x = (left_center[0] + right_center[0]) // 2
                        center_y = (left_center[1] + right_center[1]) // 2
                        positions.append((center_x, center_y))
                
                time.sleep(0.1)
            
            if positions:
                # Ortalama pozisyon
                avg_x = sum(pos[0] for pos in positions) // len(positions)
                avg_y = sum(pos[1] for pos in positions) // len(positions)
                return (avg_x, avg_y)
            
            return None
            
        except Exception as e:
            logger.error(f"Göz pozisyonu yakalama hatası: {e}")
            return None
    
    def finish_calibration(self):
        """Kalibrasyonu bitir."""
        if self.camera:
            self.camera.release()
        
        if len(self.calibration_points) >= 3:
            success = self.calibrator.calibrate_from_points(self.calibration_points)
            
            if success:
                messagebox.showinfo("Başarılı", "Kalibrasyon başarıyla tamamlandı!")
                self.update_status()
                self.show_results()
            else:
                messagebox.showerror("Hata", "Kalibrasyon başarısız!")
        else:
            messagebox.showerror("Hata", "Yeterli kalibrasyon verisi yok!")
    
    def show_results(self):
        """Kalibrasyon sonuçlarını göster."""
        self.results_text.delete('1.0', tk.END)
        
        status = self.calibrator.get_calibration_status()
        quality = status.get('quality', {})
        
        results = f"Kalibrasyon Sonuçları:\n"
        results += f"==================\n"
        results += f"Piksel/Derece Oranı: {status['pixel_to_degree_ratio']:.4f}\n"
        results += f"Nokta Sayısı: {status['point_count']}\n"
        results += f"Standart Sapma: {quality.get('ratio_std', 0):.4f}\n"
        results += f"Ortalama Oran: {quality.get('ratio_mean', 0):.4f}\n\n"
        
        results += "Kalibrasyon Noktaları:\n"
        for i, point in enumerate(self.calibration_points, 1):
            results += f"{i}. Ekran: {point['screen_pos']}, "
            results += f"Piksel: {point['pixel_pos']}, "
            results += f"Açı: {point['angle_deg']}°\n"
        
        self.results_text.insert('1.0', results)
    
    def reset_calibration(self):
        """Kalibrasyonu sıfırla."""
        result = messagebox.askyesno("Onay", "Kalibrasyon sıfırlansın mı?")
        if result:
            self.calibrator.reset_calibration()
            self.update_status()
            self.results_text.delete('1.0', tk.END)
            messagebox.showinfo("Bilgi", "Kalibrasyon sıfırlandı")
    
    def save_manual_calibration(self):
        """Manuel kalibrasyon kaydet."""
        try:
            ratio = float(self.ratio_var.get())
            
            self.calibrator.calibration_data["pixel_to_degree_ratio"] = ratio
            self.calibrator.calibration_data["calibrated"] = True
            self.calibrator.save_calibration()
            
            self.update_status()
            messagebox.showinfo("Başarılı", "Manuel kalibrasyon kaydedildi")
            
        except ValueError:
            messagebox.showerror("Hata", "Geçersiz oran değeri!")
    
    def test_calibration(self):
        """Kalibrasyonu test et."""
        try:
            # Test videosu ile kalibrasyon doğruluğunu test et
            from improved_test_video_generator import MedicalGradeVideoGenerator
            generator = MedicalGradeVideoGenerator()
            
            test_video = "calibration_test.mp4"
            success = generator.create_realistic_nystagmus_video(
                test_video, duration=2.0, nystagmus_freq=4.0, strabismus_angle=5.0
            )
            
            if success:
                from analysis_utils import analyze_video_file
                results = analyze_video_file(test_video)
                
                if "error" not in results:
                    test_results = f"Kalibrasyon Test Sonuçları:\n"
                    test_results += f"=========================\n"
                    test_results += f"Nistagmus Frekansı: {results['nistagmus_frequency']:.2f} Hz\n"
                    test_results += f"Şaşılık Açısı: {results['strabismus_angle']:.2f}°\n"
                    test_results += f"Analiz Süresi: {results['analysis_duration']:.2f}s\n"
                    test_results += f"Yüz Tespit: {results['face_detection_rate']:.1%}\n"
                    
                    self.results_text.delete('1.0', tk.END)
                    self.results_text.insert('1.0', test_results)
                    
                    messagebox.showinfo("Test Sonucu", "Kalibrasyon test edildi!")
                else:
                    messagebox.showerror("Test Hatası", results["error"])
                
                # Test dosyasını temizle
                import os
                if os.path.exists(test_video):
                    os.remove(test_video)
            
        except Exception as e:
            messagebox.showerror("Test Hatası", f"Test yapılamadı: {e}")
    
    def run(self):
        """Arayüzü çalıştır."""
        self.root.mainloop()

def main():
    """Ana fonksiyon."""
    app = CalibrationUI()
    app.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 