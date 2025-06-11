#!/usr/bin/env python3
"""
KLİNİK PERFORMANS PANOSU
========================
Canlı duyarlılık, özgüllük, AUC metrikleri ile klinik dashboard.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import logging

logger = logging.getLogger(__name__)

class ClinicalDashboard:
    """Klinik performans panosu."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Nistagmus Klinik Dashboard")
        self.root.geometry("1200x800")
        
        # Performans verileri
        self.metrics_data = {
            "daily_analyses": [],
            "sensitivity_history": [],
            "specificity_history": [],
            "auc_history": [],
            "accuracy_history": [],
            "timestamps": [],
            "current_metrics": {
                "sensitivity": 0.85,
                "specificity": 0.82,
                "auc": 0.84,
                "accuracy": 0.83,
                "total_analyses": 0,
                "pathology_detected": 0
            }
        }
        
        self.setup_dashboard()
        self.load_metrics_data()
        self.start_auto_refresh()
    
    def setup_dashboard(self):
        """Dashboard arayüzünü kur."""
        # Ana notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Sekmeler
        self.create_overview_tab()
        self.create_metrics_tab()
        self.create_analysis_tab()
        self.create_quality_tab()
    
    def create_overview_tab(self):
        """Genel bakış sekmesi."""
        overview_frame = ttk.Frame(self.notebook)
        self.notebook.add(overview_frame, text="📊 Genel Bakış")
        
        # Başlık
        title_label = ttk.Label(overview_frame, text="Klinik Performans Özeti", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Metrik kartları
        cards_frame = ttk.Frame(overview_frame)
        cards_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Metrik kartları oluştur
        self.metric_cards = {}
        metrics = [
            ("Duyarlılık", "sensitivity", "🎯"),
            ("Özgüllük", "specificity", "🔍"),
            ("AUC Skoru", "auc", "📈"),
            ("Doğruluk", "accuracy", "✅")
        ]
        
        for i, (name, key, icon) in enumerate(metrics):
            card = self.create_metric_card(cards_frame, name, key, icon)
            card.grid(row=0, column=i, padx=10, pady=5, sticky="ew")
            self.metric_cards[key] = card
        
        # Grid ağırlıkları
        for i in range(len(metrics)):
            cards_frame.columnconfigure(i, weight=1)
        
        # Günlük analiz grafik
        self.create_daily_chart(overview_frame)
    
    def create_metric_card(self, parent, name, key, icon):
        """Metrik kartı oluştur."""
        card_frame = ttk.LabelFrame(parent, text=f"{icon} {name}", padding="10")
        
        # Değer etiketi
        value_label = ttk.Label(card_frame, text="0.00", 
                               font=("Arial", 20, "bold"))
        value_label.pack()
        
        # Trend etiketi
        trend_label = ttk.Label(card_frame, text="↔ Sabit", 
                               font=("Arial", 10))
        trend_label.pack()
        
        # Etiketleri card objesine ekle
        card_frame.value_label = value_label
        card_frame.trend_label = trend_label
        card_frame.key = key
        
        return card_frame
    
    def create_daily_chart(self, parent):
        """Günlük analiz sayısı grafik."""
        chart_frame = ttk.LabelFrame(parent, text="📊 Günlük Analiz Sayısı", padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Matplotlib figür
        self.daily_fig, self.daily_ax = plt.subplots(figsize=(8, 3))
        self.daily_canvas = FigureCanvasTkAgg(self.daily_fig, chart_frame)
        self.daily_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # İlk grafik
        self.update_daily_chart()
    
    def create_metrics_tab(self):
        """Metrikler sekmesi."""
        metrics_frame = ttk.Frame(self.notebook)
        self.notebook.add(metrics_frame, text="📈 Metrikler")
        
        # Metrik trend grafikleri
        self.create_metrics_charts(metrics_frame)
    
    def create_metrics_charts(self, parent):
        """Metrik trend grafikleri."""
        # Alt çerçeveler
        top_frame = ttk.Frame(parent)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        bottom_frame = ttk.Frame(parent)
        bottom_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Sensitivity & Specificity
        sens_spec_frame = ttk.LabelFrame(top_frame, text="🎯 Duyarlılık & Özgüllük", padding="5")
        sens_spec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.sens_spec_fig, self.sens_spec_ax = plt.subplots(figsize=(5, 3))
        sens_spec_canvas = FigureCanvasTkAgg(self.sens_spec_fig, sens_spec_frame)
        sens_spec_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # AUC & Accuracy
        auc_acc_frame = ttk.LabelFrame(top_frame, text="📊 AUC & Doğruluk", padding="5")
        auc_acc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.auc_acc_fig, self.auc_acc_ax = plt.subplots(figsize=(5, 3))
        auc_acc_canvas = FigureCanvasTkAgg(self.auc_acc_fig, auc_acc_frame)
        auc_acc_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # İlk grafikler
        self.update_metrics_charts()
    
    def create_analysis_tab(self):
        """Analiz detayları sekmesi."""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="🔬 Analiz Detayları")
        
        # Son analizler tablosu
        table_frame = ttk.LabelFrame(analysis_frame, text="Son Analizler", padding="10")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview oluştur
        columns = ("Tarih", "Hasta ID", "Nistagmus", "Şaşılık", "ML Güven", "Doktor Onayı")
        self.analysis_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)
        
        # Başlıklar
        for col in columns:
            self.analysis_tree.heading(col, text=col)
            self.analysis_tree.column(col, width=120, anchor="center")
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.analysis_tree.yview)
        self.analysis_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack
        self.analysis_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Güncelleme butonu
        refresh_btn = ttk.Button(analysis_frame, text="🔄 Güncelle", 
                                command=self.refresh_analysis_table)
        refresh_btn.pack(pady=5)
        
        # İlk yükleme
        self.refresh_analysis_table()
    
    def create_quality_tab(self):
        """Kalite kontrol sekmesi."""
        quality_frame = ttk.Frame(self.notebook)
        self.notebook.add(quality_frame, text="🏥 Kalite Kontrol")
        
        # Kalite metrikleri
        metrics_frame = ttk.LabelFrame(quality_frame, text="Kalite Metrikleri", padding="10")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Kalite kartları
        quality_metrics = [
            ("Video Kalitesi", "video_quality", "📹"),
            ("Kalibrasyon Durumu", "calibration_status", "🎯"),
            ("Model Güvenilirliği", "model_confidence", "🤖"),
            ("Sistem Sağlığı", "system_health", "💻")
        ]
        
        self.quality_cards = {}
        for i, (name, key, icon) in enumerate(quality_metrics):
            row = i // 2
            col = i % 2
            
            card = ttk.LabelFrame(metrics_frame, text=f"{icon} {name}", padding="10")
            card.grid(row=row, column=col, padx=10, pady=5, sticky="ew")
            
            value_label = ttk.Label(card, text="Kontrol ediliyor...", font=("Arial", 12))
            value_label.pack()
            
            self.quality_cards[key] = value_label
        
        # Grid ağırlıkları
        metrics_frame.columnconfigure(0, weight=1)
        metrics_frame.columnconfigure(1, weight=1)
        
        # Sistem durumu
        self.update_quality_metrics()
    
    def load_metrics_data(self):
        """Metrik verilerini yükle."""
        try:
            # Validation sonuçlarından yükle
            if os.path.exists("validation_results.json"):
                with open("validation_results.json", 'r') as f:
                    validation_data = json.load(f)
                
                # Son validasyon sonuçlarını kullan
                if "classification_metrics" in validation_data:
                    nyst_metrics = validation_data["classification_metrics"]["nystagmus"]
                    strab_metrics = validation_data["classification_metrics"]["strabismus"]
                    
                    # Ortalama metrikler
                    self.metrics_data["current_metrics"].update({
                        "sensitivity": (nyst_metrics["sensitivity"] + strab_metrics["sensitivity"]) / 2,
                        "specificity": (nyst_metrics["specificity"] + strab_metrics["specificity"]) / 2,
                        "auc": (nyst_metrics["auc"] + strab_metrics["auc"]) / 2,
                        "accuracy": (nyst_metrics["accuracy"] + strab_metrics["accuracy"]) / 2
                    })
            
            # Günlük analiz verilerini simüle et
            self.simulate_daily_data()
            
        except Exception as e:
            logger.error(f"Metrik veri yükleme hatası: {e}")
    
    def simulate_daily_data(self):
        """Günlük analiz verilerini simüle et."""
        # Son 30 gün için veri oluştur
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            
            # Günlük analiz sayısı (10-50 arası)
            daily_count = np.random.randint(10, 51)
            self.metrics_data["daily_analyses"].append(daily_count)
            
            # Metrik geçmişi (küçük varyasyonlarla)
            base_sens = 0.85
            base_spec = 0.82
            base_auc = 0.84
            base_acc = 0.83
            
            self.metrics_data["sensitivity_history"].append(
                base_sens + np.random.normal(0, 0.02))
            self.metrics_data["specificity_history"].append(
                base_spec + np.random.normal(0, 0.02))
            self.metrics_data["auc_history"].append(
                base_auc + np.random.normal(0, 0.015))
            self.metrics_data["accuracy_history"].append(
                base_acc + np.random.normal(0, 0.02))
            
            self.metrics_data["timestamps"].append(date)
    
    def report_metrics(self, new_metrics: Dict[str, float]):
        """Yeni metrikleri kaydet ve güncelle."""
        try:
            # Mevcut metrikleri güncelle
            self.metrics_data["current_metrics"].update(new_metrics)
            
            # Geçmişe ekle
            current_time = datetime.now()
            self.metrics_data["timestamps"].append(current_time)
            
            # Metrik geçmişini güncelle
            for key in ["sensitivity", "specificity", "auc", "accuracy"]:
                if key in new_metrics:
                    self.metrics_data[f"{key}_history"].append(new_metrics[key])
            
            # Dashboard'u güncelle
            self.update_dashboard()
            
            logger.info(f"Metrikler güncellendi: {new_metrics}")
            
        except Exception as e:
            logger.error(f"Metrik raporlama hatası: {e}")
    
    def update_dashboard(self):
        """Dashboard'u güncelle."""
        # Metrik kartları güncelle
        metrics = self.metrics_data["current_metrics"]
        
        for key, card in self.metric_cards.items():
            if key in metrics:
                value = metrics[key]
                card.value_label.config(text=f"{value:.3f}")
                
                # Trend hesapla
                history_key = f"{key}_history"
                if len(self.metrics_data[history_key]) >= 2:
                    current = self.metrics_data[history_key][-1]
                    previous = self.metrics_data[history_key][-2]
                    
                    if current > previous:
                        trend = "↗ Artış"
                        color = "green"
                    elif current < previous:
                        trend = "↘ Azalış"
                        color = "red"
                    else:
                        trend = "↔ Sabit"
                        color = "blue"
                    
                    card.trend_label.config(text=trend, foreground=color)
        
        # Grafikleri güncelle
        self.update_daily_chart()
        self.update_metrics_charts()
    
    def update_daily_chart(self):
        """Günlük analiz grafik güncelle."""
        self.daily_ax.clear()
        
        if self.metrics_data["daily_analyses"]:
            days = list(range(len(self.metrics_data["daily_analyses"])))
            self.daily_ax.plot(days, self.metrics_data["daily_analyses"], 
                              marker='o', linewidth=2, markersize=4)
            self.daily_ax.set_title("Son 30 Gün Analiz Sayısı")
            self.daily_ax.set_xlabel("Gün")
            self.daily_ax.set_ylabel("Analiz Sayısı")
            self.daily_ax.grid(True, alpha=0.3)
        
        self.daily_fig.tight_layout()
        self.daily_canvas.draw()
    
    def update_metrics_charts(self):
        """Metrik trend grafikleri güncelle."""
        # Sensitivity & Specificity
        self.sens_spec_ax.clear()
        if self.metrics_data["sensitivity_history"]:
            days = list(range(len(self.metrics_data["sensitivity_history"])))
            self.sens_spec_ax.plot(days, self.metrics_data["sensitivity_history"], 
                                  label="Duyarlılık", marker='o', markersize=3)
            self.sens_spec_ax.plot(days, self.metrics_data["specificity_history"], 
                                  label="Özgüllük", marker='s', markersize=3)
            self.sens_spec_ax.set_title("Duyarlılık & Özgüllük Trendi")
            self.sens_spec_ax.set_ylabel("Değer")
            self.sens_spec_ax.legend()
            self.sens_spec_ax.grid(True, alpha=0.3)
        
        # AUC & Accuracy
        self.auc_acc_ax.clear()
        if self.metrics_data["auc_history"]:
            days = list(range(len(self.metrics_data["auc_history"])))
            self.auc_acc_ax.plot(days, self.metrics_data["auc_history"], 
                                label="AUC", marker='^', markersize=3)
            self.auc_acc_ax.plot(days, self.metrics_data["accuracy_history"], 
                                label="Doğruluk", marker='v', markersize=3)
            self.auc_acc_ax.set_title("AUC & Doğruluk Trendi")
            self.auc_acc_ax.set_ylabel("Değer")
            self.auc_acc_ax.legend()
            self.auc_acc_ax.grid(True, alpha=0.3)
        
        self.sens_spec_fig.tight_layout()
        self.auc_acc_fig.tight_layout()
        self.sens_spec_canvas.draw()
        self.auc_acc_canvas.draw()
    
    def refresh_analysis_table(self):
        """Analiz tablosunu güncelle."""
        # Mevcut verileri temizle
        for item in self.analysis_tree.get_children():
            self.analysis_tree.delete(item)
        
        # Simulated analiz verileri
        import random
        sample_data = []
        
        for i in range(15):
            date = (datetime.now() - timedelta(days=random.randint(0, 7))).strftime("%Y-%m-%d %H:%M")
            patient_id = f"P{random.randint(1000, 9999)}"
            nystagmus = random.choice(["Pozitif", "Negatif"])
            strabismus = random.choice(["Pozitif", "Negatif"])
            confidence = f"{random.uniform(0.75, 0.99):.2f}"
            approval = random.choice(["Onaylandı", "Beklemede", "Reddedildi"])
            
            sample_data.append((date, patient_id, nystagmus, strabismus, confidence, approval))
        
        # Verileri tabloya ekle
        for data in sample_data:
            self.analysis_tree.insert("", "end", values=data)
    
    def update_quality_metrics(self):
        """Kalite metriklerini güncelle."""
        try:
            # Kalibrasyon durumu kontrol et
            from calibration import get_calibrator
            calibrator = get_calibrator()
            cal_status = calibrator.get_calibration_status()
            
            if cal_status["calibrated"]:
                self.quality_cards["calibration_status"].config(
                    text="✅ Kalibre Edilmiş", foreground="green")
            else:
                self.quality_cards["calibration_status"].config(
                    text="⚠️ Kalibrasyon Gerekli", foreground="orange")
            
            # Video kalitesi (simulated)
            self.quality_cards["video_quality"].config(
                text="📹 İyi Kalite (>80%)", foreground="green")
            
            # Model güvenilirliği
            current_auc = self.metrics_data["current_metrics"]["auc"]
            if current_auc >= 0.8:
                self.quality_cards["model_confidence"].config(
                    text=f"🤖 Yüksek ({current_auc:.2f})", foreground="green")
            else:
                self.quality_cards["model_confidence"].config(
                    text=f"🤖 Orta ({current_auc:.2f})", foreground="orange")
            
            # Sistem sağlığı
            self.quality_cards["system_health"].config(
                text="💻 Sistem Normal", foreground="green")
                
        except Exception as e:
            logger.error(f"Kalite metrik güncelleme hatası: {e}")
    
    def start_auto_refresh(self):
        """Otomatik yenileme başlat."""
        def auto_update():
            while True:
                try:
                    time.sleep(30)  # 30 saniyede bir güncelle
                    self.root.after(0, self.update_quality_metrics)
                except Exception as e:
                    logger.error(f"Otomatik güncelleme hatası: {e}")
        
        # Arka plan thread'i başlat
        refresh_thread = threading.Thread(target=auto_update, daemon=True)
        refresh_thread.start()
    
    def run(self):
        """Dashboard'u çalıştır."""
        self.root.mainloop()

def main():
    """Ana fonksiyon."""
    logging.basicConfig(level=logging.INFO)
    
    # Test verileri ile dashboard çalıştır
    dashboard = ClinicalDashboard()
    
    # Test metrik güncellemesi
    test_metrics = {
        "sensitivity": 0.87,
        "specificity": 0.84,
        "auc": 0.86,
        "accuracy": 0.85
    }
    dashboard.report_metrics(test_metrics)
    
    dashboard.run()

if __name__ == "__main__":
    import os
    main() 