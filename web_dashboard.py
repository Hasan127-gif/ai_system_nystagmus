#!/usr/bin/env python3
"""
WEB TABANLI KLİNİK DASHBOARD
===========================
Flask ile Tkinter olmadan çalışan web arayüzü.
"""

import json
import threading
import webbrowser
from datetime import datetime
from typing import Dict, Any

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

class WebDashboard:
    """Web tabanlı klinik dashboard."""
    
    def __init__(self, port=5000):
        self.port = port
        self.app = None
        self.metrics_data = {
            "sensitivity": 0.85,
            "specificity": 0.82,
            "auc": 0.84,
            "accuracy": 0.83,
            "last_updated": datetime.now().isoformat()
        }
        
        if FLASK_AVAILABLE:
            self.setup_flask_app()
    
    def setup_flask_app(self):
        """Flask uygulamasını kurar."""
        self.app = Flask(__name__)
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_html())
        
        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(self.get_current_metrics())
        
        @self.app.route('/api/update_metrics', methods=['POST'])
        def update_metrics():
            new_metrics = request.json
            self.metrics_data.update(new_metrics)
            self.metrics_data["last_updated"] = datetime.now().isoformat()
            return jsonify({"status": "success"})
    
    def get_current_metrics(self):
        """Güncel metrikleri döndürür."""
        try:
            from logger import get_clinical_logger
            
            clinical_logger = get_clinical_logger()
            audit_report = clinical_logger.generate_audit_report()
            
            self.metrics_data["total_analyses"] = audit_report["summary"]["total_analyses"]
            self.metrics_data["approved_analyses"] = audit_report["summary"]["approved_analyses"]
            self.metrics_data["pending_analyses"] = audit_report["summary"]["pending_analyses"]
            
        except Exception as e:
            print(f"Metrik yükleme hatası: {e}")
        
        return self.metrics_data
    
    def get_dashboard_html(self):
        """Dashboard HTML template."""
        return '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🏥 Klinik Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-value { font-size: 2em; font-weight: bold; color: #27ae60; }
        .metric-label { font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }
        .status-section { background: white; padding: 20px; border-radius: 8px; margin-top: 20px; }
        .status-item { padding: 10px; margin: 5px 0; border-left: 4px solid #27ae60; background: #ecf0f1; }
        .refresh-btn { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        .refresh-btn:hover { background: #2980b9; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏥 Klinik Performans Dashboard</h1>
        <p>📅 Son Güncelleme: <span id="last-updated"></span></p>
        <button class="refresh-btn" onclick="refreshMetrics()">🔄 Yenile</button>
    </div>
    
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-value" id="sensitivity">-</div>
            <div class="metric-label">🎯 Duyarlılık</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="specificity">-</div>
            <div class="metric-label">🔍 Özgüllük</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="auc">-</div>
            <div class="metric-label">📈 AUC Skoru</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" id="accuracy">-</div>
            <div class="metric-label">✅ Doğruluk</div>
        </div>
    </div>
    
    <div class="status-section">
        <h2>🔧 Sistem Durumu</h2>
        <div class="status-item" id="calibration-status">🎯 Kalibrasyon: Kontrol ediliyor...</div>
        <div class="status-item" id="privacy-status">🔐 Gizlilik: Kontrol ediliyor...</div>
        <div class="status-item" id="logger-status">📝 Kayıt: Kontrol ediliyor...</div>
        <div class="status-item" id="analysis-stats">📊 Analiz: Yükleniyor...</div>
    </div>
    
    <script>
        function refreshMetrics() {
            fetch('/api/metrics')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('sensitivity').textContent = (data.sensitivity * 100).toFixed(1) + '%';
                    document.getElementById('specificity').textContent = (data.specificity * 100).toFixed(1) + '%';
                    document.getElementById('auc').textContent = data.auc.toFixed(3);
                    document.getElementById('accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
                    document.getElementById('last-updated').textContent = data.last_updated.substring(0, 19);
                    
                    // Analiz istatistikleri
                    const totalAnalyses = data.total_analyses || 0;
                    const approved = data.approved_analyses || 0;
                    const pending = data.pending_analyses || 0;
                    document.getElementById('analysis-stats').innerHTML = 
                        `📊 Analiz: ${totalAnalyses} toplam, ${approved} onaylı, ${pending} bekleyen`;
                })
                .catch(error => console.error('Metrik yükleme hatası:', error));
        }
        
        // Sayfa yüklendiğinde ve her 30 saniyede bir yenile
        refreshMetrics();
        setInterval(refreshMetrics, 30000);
    </script>
</body>
</html>
        '''
    
    def run(self, auto_open=True):
        """Web dashboard'u başlatır."""
        if not FLASK_AVAILABLE:
            print("❌ Flask modülü bulunamadı!")
            print("   Kurulum: pip install flask")
            return False
        
        if self.app:
            print(f"🌐 Web Dashboard başlatılıyor: http://localhost:{self.port}")
            
            if auto_open:
                # Tarayıcıyı otomatik aç
                threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{self.port}")).start()
            
            try:
                self.app.run(host='localhost', port=self.port, debug=False, use_reloader=False)
                return True
            except Exception as e:
                print(f"❌ Web dashboard hatası: {e}")
                return False
        else:
            print("❌ Flask app kurulamadı!")
            return False
    
    def report_metrics(self, new_metrics: Dict[str, float]):
        """Yeni metrikleri günceller."""
        self.metrics_data.update(new_metrics)
        self.metrics_data["last_updated"] = datetime.now().isoformat()
        print(f"✅ Web dashboard metrikleri güncellendi: {new_metrics}")

def main():
    """Ana fonksiyon - Web dashboard'u başlatır."""
    print("🌐 WEB TABANLI KLİNİK DASHBOARD")
    print("=" * 35)
    
    if not FLASK_AVAILABLE:
        print("\n❌ Flask modülü gerekli!")
        print("Kurulum: pip install flask")
        print("\nAlternatif olarak CLI dashboard kullanın:")
        print("python dashboard_cli.py --interactive")
        return
    
    dashboard = WebDashboard(port=5000)
    
    # Test metriklerini yükle
    test_metrics = {
        "sensitivity": 0.87,
        "specificity": 0.84,
        "auc": 0.86,
        "accuracy": 0.85
    }
    dashboard.report_metrics(test_metrics)
    
    print("Web tarayıcıda otomatik açılacak...")
    print("Kapatmak için Ctrl+C tuşlayın")
    
    dashboard.run(auto_open=True)

if __name__ == "__main__":
    main() 