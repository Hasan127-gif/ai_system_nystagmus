#!/usr/bin/env python3
"""
KLİNİK KULLANICI ARAYÜZÜ & ONAY SİSTEMİ
=======================================
Patoloji tespiti durumunda doktor onayı ve uyarı sistemi.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import json
from datetime import datetime
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class ClinicalUI:
    """Klinik kullanıcı arayüzü ve onay sistemi."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nistagmus Klinik Sistem")
        self.root.geometry("900x700")
        
        # Onay kuyruğu
        self.approval_queue = queue.Queue()
        
        # Mevcut analiz verisi
        self.current_analysis = None
        
        # Callbacks
        self.on_approval_callback = None
        
        self.setup_ui()
        self.start_queue_processor()
    
    def setup_ui(self):
        """Kullanıcı arayüzünü kur."""
        # Ana çerçeve
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Başlık
        title_label = ttk.Label(main_frame, text="🏥 Nistagmus Klinik Karar Destek Sistemi", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Sol panel: Analiz sonuçları
        self.create_analysis_panel(main_frame)
        
        # Sağ panel: Onay kontrolleri
        self.create_approval_panel(main_frame)
        
        # Alt panel: Geçmiş ve loglar
        self.create_history_panel(main_frame)
        
        # Grid ağırlıkları
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
    
    def create_analysis_panel(self, parent):
        """Analiz sonuçları paneli."""
        analysis_frame = ttk.LabelFrame(parent, text="📊 Güncel Analiz Sonuçları", padding="10")
        analysis_frame.grid(row=1, column=0, padx=(0, 5), pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Sonuç etiketleri
        self.result_labels = {}
        
        results = [
            ("Nistagmus Frekansı", "nistagmus_frequency", "Hz"),
            ("Şaşılık Açısı", "strabismus_angle", "°"),
            ("Nistagmus Durumu", "nystagmus_detected", ""),
            ("Şaşılık Durumu", "strabismus_detected", ""),
            ("ML Güven Skoru", "ml_confidence", ""),
            ("Video Kalitesi", "video_quality", ""),
            ("Kalibrasyon", "calibration_status", "")
        ]
        
        for i, (name, key, unit) in enumerate(results):
            # Etiket
            label = ttk.Label(analysis_frame, text=f"{name}:", font=("Arial", 10, "bold"))
            label.grid(row=i, column=0, sticky=tk.W, pady=2)
            
            # Değer
            value_label = ttk.Label(analysis_frame, text="---", font=("Arial", 10))
            value_label.grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            
            # Birim
            if unit:
                unit_label = ttk.Label(analysis_frame, text=unit, font=("Arial", 9))
                unit_label.grid(row=i, column=2, sticky=tk.W, padx=(5, 0), pady=2)
            
            self.result_labels[key] = value_label
        
        # Uyarı alanı
        self.warning_frame = tk.Frame(analysis_frame, bg="red", relief=tk.RAISED, bd=2)
        self.warning_label = tk.Label(self.warning_frame, text="", fg="white", bg="red", 
                                     font=("Arial", 11, "bold"), wraplength=250)
        self.warning_label.pack(padx=5, pady=5)
    
    def create_approval_panel(self, parent):
        """Doktor onay paneli."""
        approval_frame = ttk.LabelFrame(parent, text="👨‍⚕️ Doktor Onay Kontrolü", padding="10")
        approval_frame.grid(row=1, column=1, padx=(5, 0), pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Doktor ID girişi
        ttk.Label(approval_frame, text="Doktor ID:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.doctor_id_entry = ttk.Entry(approval_frame, width=20)
        self.doctor_id_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Onay durumu
        ttk.Label(approval_frame, text="Onay Durumu:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.approval_var = tk.StringVar(value="pending")
        
        approval_frame_radio = ttk.Frame(approval_frame)
        approval_frame_radio.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(approval_frame_radio, text="✅ Onayla", 
                       variable=self.approval_var, value="approved").pack(anchor=tk.W)
        ttk.Radiobutton(approval_frame_radio, text="❌ Reddet", 
                       variable=self.approval_var, value="rejected").pack(anchor=tk.W)
        ttk.Radiobutton(approval_frame_radio, text="⏳ Beklemede", 
                       variable=self.approval_var, value="pending").pack(anchor=tk.W)
        
        # Notlar
        ttk.Label(approval_frame, text="Doktor Notları:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky=(tk.W, tk.N), pady=5)
        self.notes_text = tk.Text(approval_frame, height=4, width=25, wrap=tk.WORD)
        self.notes_text.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
        
        # Onay butonları
        button_frame = ttk.Frame(approval_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="💾 Onayı Kaydet", 
                  command=self.save_approval).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Yenile", 
                  command=self.refresh_display).pack(side=tk.LEFT, padx=5)
        
        # Onay geçmişi
        history_label = ttk.Label(approval_frame, text="Onay Geçmişi:", font=("Arial", 10, "bold"))
        history_label.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(10, 5))
        
        self.history_text = tk.Text(approval_frame, height=6, width=40, state=tk.DISABLED)
        history_scrollbar = ttk.Scrollbar(approval_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=history_scrollbar.set)
        
        self.history_text.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        history_scrollbar.grid(row=5, column=2, sticky=(tk.N, tk.S), pady=5)
        
        # Grid ağırlıkları
        approval_frame.columnconfigure(1, weight=1)
    
    def create_history_panel(self, parent):
        """Geçmiş ve log paneli."""
        history_frame = ttk.LabelFrame(parent, text="📋 Analiz Geçmişi", padding="10")
        history_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Treeview oluştur
        columns = ("Tarih", "Hasta ID", "Nistagmus", "Şaşılık", "Onay Durumu", "Doktor")
        self.history_tree = ttk.Treeview(history_frame, columns=columns, show="headings", height=8)
        
        # Başlıklar
        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=120, anchor="center")
        
        # Scrollbar
        tree_scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Pack
        self.history_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Kontrol butonları
        control_frame = ttk.Frame(history_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(control_frame, text="📄 Rapor Oluştur", 
                  command=self.generate_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="💾 Dışa Aktar", 
                  command=self.export_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="🔍 Detayları Göster", 
                  command=self.show_analysis_details).pack(side=tk.LEFT, padx=5)
        
        # Grid ağırlıkları
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        # İlk yükleme
        self.refresh_history()
    
    def show_alert(self, message: str, alert_type: str = "warning"):
        """Patoloji uyarısı göster."""
        try:
            if alert_type == "pathology":
                # Kırmızı uyarı kutusu
                self.warning_frame.grid(row=len(self.result_labels), column=0, columnspan=3, 
                                       pady=10, sticky=(tk.W, tk.E))
                self.warning_label.config(text=message)
                
                # Ses uyarısı (varsa)
                self.root.bell()
                
                # Popup uyarı
                result = messagebox.askquestion(
                    "🚨 PATOLOJİ TESPİT EDİLDİ", 
                    f"{message}\n\nDoktor onayı gerekli. Onay ekranına geçmek istiyor musunuz?",
                    icon="warning"
                )
                
                if result == "yes":
                    # Onay sekmesine odaklan
                    self.doctor_id_entry.focus()
                
            else:
                # Normal uyarı
                messagebox.showwarning("Uyarı", message)
                
            logging.info(f"Uyarı gösterildi: {message}")
            
        except Exception as e:
            logging.error(f"Uyarı gösterme hatası: {e}")
    
    def update_analysis_display(self, analysis_data: Dict[str, Any]):
        """Analiz sonuçlarını ekrana güncelle."""
        try:
            self.current_analysis = analysis_data
            
            # Uyarı çerçevesini gizle
            self.warning_frame.grid_remove()
            
            # Sonuçları güncelle
            for key, label in self.result_labels.items():
                if key in analysis_data:
                    value = analysis_data[key]
                    
                    if key in ["nystagmus_detected", "strabismus_detected"]:
                        # Boolean değerler için renk
                        if value:
                            label.config(text="✅ Pozitif", foreground="red")
                        else:
                            label.config(text="❌ Negatif", foreground="green")
                    elif key == "ml_confidence":
                        # Güven skoru
                        confidence = float(value)
                        label.config(text=f"{confidence:.3f}")
                        if confidence >= 0.8:
                            label.config(foreground="green")
                        elif confidence >= 0.6:
                            label.config(foreground="orange")
                        else:
                            label.config(foreground="red")
                    elif key in ["nistagmus_frequency", "strabismus_angle"]:
                        # Sayısal değerler
                        label.config(text=f"{value:.2f}", foreground="black")
                    else:
                        # Diğer değerler
                        label.config(text=str(value), foreground="black")
                else:
                    label.config(text="---", foreground="gray")
            
            # Patoloji kontrolü
            self.check_pathology(analysis_data)
            
            logging.info("Analiz ekranı güncellendi")
            
        except Exception as e:
            logging.error(f"Analiz ekran güncelleme hatası: {e}")
    
    def check_pathology(self, analysis_data: Dict[str, Any]):
        """Patoloji kontrolü ve uyarı."""
        try:
            flags = {
                'nistagmus': analysis_data.get('nystagmus_detected', False),
                'strabismus': analysis_data.get('strabismus_detected', False)
            }
            
            # Patoloji tespit edildi mi?
            if flags['nistagmus'] or flags['strabismus']:
                pathology_list = []
                if flags['nistagmus']:
                    freq = analysis_data.get('nistagmus_frequency', 0)
                    pathology_list.append(f"Nistagmus (Frekans: {freq:.1f} Hz)")
                if flags['strabismus']:
                    angle = analysis_data.get('strabismus_angle', 0)
                    pathology_list.append(f"Şaşılık (Açı: {angle:.1f}°)")
                
                message = f"Model patoloji tespit etti:\n• " + "\n• ".join(pathology_list) + "\n\nLütfen hekimin onaylayın."
                
                self.show_alert(message, "pathology")
                
                # Analizi onay kuyruğuna ekle
                self.approval_queue.put({
                    "analysis_data": analysis_data,
                    "timestamp": datetime.now().isoformat(),
                    "pathology_detected": pathology_list
                })
            
        except Exception as e:
            logging.error(f"Patoloji kontrol hatası: {e}")
    
    def save_approval(self):
        """Doktor onayını kaydet."""
        try:
            if not self.current_analysis:
                messagebox.showwarning("Uyarı", "Onaylanacak analiz bulunamadı.")
                return
            
            doctor_id = self.doctor_id_entry.get().strip()
            if not doctor_id:
                messagebox.showwarning("Uyarı", "Doktor ID'si gerekli.")
                return
            
            approval_status = self.approval_var.get()
            notes = self.notes_text.get("1.0", tk.END).strip()
            
            # Logger'a kaydet
            try:
                from logger import get_clinical_logger
                clinical_logger = get_clinical_logger()
                
                analysis_id = self.current_analysis.get('analysis_id', 'unknown')
                clinical_logger.log_doctor_approval(analysis_id, doctor_id, approval_status, notes)
                
                # Başarı mesajı
                messagebox.showinfo("Başarılı", f"Onay kaydedildi: {approval_status}")
                
                # Formu temizle
                self.notes_text.delete("1.0", tk.END)
                self.approval_var.set("pending")
                
                # Geçmişi güncelle
                self.refresh_history()
                
                # Callback çağır
                if self.on_approval_callback:
                    self.on_approval_callback(analysis_id, approval_status, notes)
                
            except Exception as e:
                messagebox.showerror("Hata", f"Onay kaydetme hatası: {e}")
            
        except Exception as e:
            logging.error(f"Onay kaydetme hatası: {e}")
            messagebox.showerror("Hata", f"Bir hata oluştu: {e}")
    
    def refresh_display(self):
        """Ekranı yenile."""
        if self.current_analysis:
            self.update_analysis_display(self.current_analysis)
        self.refresh_history()
    
    def refresh_history(self):
        """Analiz geçmişini yenile."""
        try:
            # Mevcut verileri temizle
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Clinical logger'dan verileri yükle
            try:
                from logger import get_clinical_logger
                clinical_logger = get_clinical_logger()
                
                # CSV dosyasından verileri oku
                import csv
                if clinical_logger.analysis_log.exists():
                    with open(clinical_logger.analysis_log, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        
                        # Son 20 analizi göster
                        rows = list(reader)[-20:]
                        
                        for row in rows:
                            timestamp = row.get('timestamp', '')[:16]  # Sadece tarih-saat
                            patient_id = row.get('patient_id_hash', '')[:8]  # Kısa hash
                            nystagmus = "✅" if row.get('nystagmus_detected') == 'True' else "❌"
                            strabismus = "✅" if row.get('strabismus_detected') == 'True' else "❌"
                            approval = row.get('doctor_approval', 'pending')
                            doctor = "---"  # Gizlilik için
                            
                            # Renk kodlaması
                            tags = []
                            if approval == "approved":
                                tags.append("approved")
                            elif approval == "rejected":
                                tags.append("rejected")
                            elif nystagmus == "✅" or strabismus == "✅":
                                tags.append("pathology")
                            
                            self.history_tree.insert("", 0, values=(
                                timestamp, patient_id, nystagmus, strabismus, approval, doctor
                            ), tags=tags)
                
                # Renk etiketleri
                self.history_tree.tag_configure("approved", background="lightgreen")
                self.history_tree.tag_configure("rejected", background="lightcoral")
                self.history_tree.tag_configure("pathology", background="lightyellow")
                
            except Exception as e:
                logging.error(f"Geçmiş yükleme hatası: {e}")
                
        except Exception as e:
            logging.error(f"Geçmiş yenileme hatası: {e}")
    
    def show_analysis_details(self):
        """Seçili analizin detaylarını göster."""
        try:
            selection = self.history_tree.selection()
            if not selection:
                messagebox.showwarning("Uyarı", "Lütfen bir analiz seçin.")
                return
            
            # Seçili satırı al
            item = self.history_tree.item(selection[0])
            values = item['values']
            
            # Detay penceresi
            detail_window = tk.Toplevel(self.root)
            detail_window.title("Analiz Detayları")
            detail_window.geometry("400x300")
            
            # Detay bilgileri
            details_text = tk.Text(detail_window, wrap=tk.WORD, padx=10, pady=10)
            details_text.pack(fill=tk.BOTH, expand=True)
            
            detail_info = f"""
📊 ANALİZ DETAYLARI

🕐 Tarih: {values[0]}
👤 Hasta ID: {values[1]}
🎯 Nistagmus: {values[2]}
👁 Şaşılık: {values[3]}
✅ Onay Durumu: {values[4]}
👨‍⚕️ Doktor: {values[5]}

📋 Bu analiz hakkında daha fazla bilgi için log dosyalarını kontrol edin.
            """
            
            details_text.insert("1.0", detail_info)
            details_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logging.error(f"Detay gösterme hatası: {e}")
            messagebox.showerror("Hata", f"Detay gösterilemiyor: {e}")
    
    def generate_report(self):
        """Analiz raporu oluştur."""
        try:
            from logger import get_clinical_logger
            clinical_logger = get_clinical_logger()
            
            # Denetim raporu oluştur
            audit_report = clinical_logger.generate_audit_report()
            
            # Rapor penceresi
            report_window = tk.Toplevel(self.root)
            report_window.title("Klinik Analiz Raporu")
            report_window.geometry("600x500")
            
            # Rapor metni
            report_text = tk.Text(report_window, wrap=tk.WORD, padx=10, pady=10)
            report_scrollbar = ttk.Scrollbar(report_window, orient=tk.VERTICAL, command=report_text.yview)
            report_text.configure(yscrollcommand=report_scrollbar.set)
            
            report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            report_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Rapor içeriği
            report_content = f"""
🏥 KLİNİK ANALİZ RAPORU
=====================

📅 Rapor Tarihi: {audit_report.get('report_generated', 'Bilinmiyor')}
📊 Özet Bilgiler:

• Toplam Analiz Sayısı: {audit_report['summary']['total_analyses']}
• Onaylanmış Analizler: {audit_report['summary']['approved_analyses']}
• Bekleyen Analizler: {audit_report['summary']['pending_analyses']}
• Reddedilen Analizler: {audit_report['summary']['rejected_analyses']}
• Benzersiz Hasta Sayısı: {audit_report['summary']['unique_patients']}

🔧 Sistem Bilgileri:
• Kalibrasyon Değişiklikleri: {audit_report['summary']['calibration_changes']}
• Model Güncellemeleri: {audit_report['summary']['model_updates']}

✅ Uygunluk Durumu: {audit_report['compliance_status'].upper()}

📋 Öneriler:
"""
            
            for recommendation in audit_report.get('recommendations', []):
                report_content += f"• {recommendation}\n"
            
            if not audit_report.get('recommendations'):
                report_content += "• Sistem normal çalışıyor, özel öneri yok.\n"
            
            report_content += f"""
---
Bu rapor otomatik olarak oluşturulmuştur.
HIPAA/GDPR uyumluluğu sağlanmıştır.
            """
            
            report_text.insert("1.0", report_content)
            report_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logging.error(f"Rapor oluşturma hatası: {e}")
            messagebox.showerror("Hata", f"Rapor oluşturulamadı: {e}")
    
    def export_data(self):
        """Analiz verilerini dışa aktar."""
        try:
            filename = filedialog.asksaveasfilename(
                title="Analiz Verilerini Kaydet",
                defaultextension=".json",
                filetypes=[("JSON dosyası", "*.json"), ("Tüm dosyalar", "*.*")]
            )
            
            if filename:
                from logger import get_clinical_logger
                clinical_logger = get_clinical_logger()
                
                export_file = clinical_logger.export_compliance_data("json")
                
                # Dosyayı kopyala
                import shutil
                shutil.copy2(export_file, filename)
                
                messagebox.showinfo("Başarılı", f"Veriler şu konuma dışa aktarıldı:\n{filename}")
                
        except Exception as e:
            logging.error(f"Veri dışa aktarma hatası: {e}")
            messagebox.showerror("Hata", f"Veri dışa aktarılamadı: {e}")
    
    def start_queue_processor(self):
        """Onay kuyruğu işlemcisini başlat."""
        def process_queue():
            while True:
                try:
                    # Kuyruktan analiz al
                    approval_item = self.approval_queue.get(timeout=1)
                    
                    # Ana thread'e GUI güncellemesi gönder
                    self.root.after(0, self._process_approval_item, approval_item)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logging.error(f"Kuyruk işleme hatası: {e}")
        
        # Arka plan thread'i başlat
        queue_thread = threading.Thread(target=process_queue, daemon=True)
        queue_thread.start()
    
    def _process_approval_item(self, approval_item):
        """Onay öğesini işle (ana thread'de)."""
        try:
            # Analiz verilerini güncelle
            analysis_data = approval_item["analysis_data"]
            self.update_analysis_display(analysis_data)
            
            # Onay geçmişine ekle
            pathology_info = ", ".join(approval_item["pathology_detected"])
            
            self.history_text.config(state=tk.NORMAL)
            self.history_text.insert("1.0", f"[{approval_item['timestamp']}] Patoloji tespit edildi: {pathology_info}\n")
            self.history_text.config(state=tk.DISABLED)
            
        except Exception as e:
            logging.error(f"Onay öğesi işleme hatası: {e}")
    
    def set_approval_callback(self, callback: Callable):
        """Onay callback'i ayarla."""
        self.on_approval_callback = callback
    
    def run(self):
        """UI'ı çalıştır."""
        self.root.mainloop()

def show_pathology_alert(analysis_data: Dict[str, Any]) -> str:
    """Patoloji uyarısı göster (standalone fonksiyon)."""
    flags = {
        'nistagmus': analysis_data.get('nystagmus_detected', False),
        'strabismus': analysis_data.get('strabismus_detected', False)
    }
    
    if flags['nistagmus'] or flags['strabismus']:
        pathology_list = []
        if flags['nistagmus']:
            freq = analysis_data.get('nistagmus_frequency', 0)
            pathology_list.append(f"Nistagmus (Frekans: {freq:.1f} Hz)")
        if flags['strabismus']:
            angle = analysis_data.get('strabismus_angle', 0)
            pathology_list.append(f"Şaşılık (Açı: {angle:.1f}°)")
        
        message = f"Model patoloji tespit etti:\n• " + "\n• ".join(pathology_list) + "\n\nLütfen hekimin onaylayın."
        
        # Basit uyarı penceresi
        root = tk.Tk()
        root.withdraw()  # Ana pencereyi gizle
        
        result = messagebox.askquestion(
            "🚨 PATOLOJİ TESPİT EDİLDİ", 
            message,
            icon="warning"
        )
        
        root.destroy()
        return "approved" if result == "yes" else "pending"
    
    return "normal"

def main():
    """Test amaçlı ana fonksiyon."""
    logging.basicConfig(level=logging.INFO)
    
    # UI test
    ui = ClinicalUI()
    
    # Test analiz verisi
    test_analysis = {
        "analysis_id": "test_123",
        "nistagmus_frequency": 4.2,
        "strabismus_angle": 3.5,
        "nystagmus_detected": True,
        "strabismus_detected": True,
        "ml_confidence": 0.85,
        "video_quality": "good",
        "calibration_status": "calibrated"
    }
    
    # Test callback
    def test_callback(analysis_id, approval_status, notes):
        print(f"Onay alındı: {analysis_id} - {approval_status}")
    
    ui.set_approval_callback(test_callback)
    
    # Test verisini güncelle
    ui.update_analysis_display(test_analysis)
    
    ui.run()

if __name__ == "__main__":
    main() 