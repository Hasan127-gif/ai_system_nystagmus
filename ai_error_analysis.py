import numpy as np
import pandas as pd
import joblib
import os
import time
import threading
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from error_manager import ErrorManager, ErrorSeverity, ErrorCategory, ErrorEvent

class AIErrorAnalyzer:
    """
    Yapay zeka tabanlı hata analiz sistemi.
    
    Özellikleri:
    1. Hata örüntülerinin tespiti
    2. Olası hataların tahmini
    3. Anomali tespiti
    4. Hata kök neden analizi
    5. Önleyici öneriler üretme
    """
    
    def __init__(self, error_manager, model_dir="models"):
        """
        Yapay zeka hata analiz modelini başlatır.
        
        Args:
            error_manager: Hata yönetim sistemi nesnesi
            model_dir: Model dosyalarının kaydedileceği dizin
        """
        self.error_manager = error_manager
        self.model_dir = model_dir
        
        # Model dizinini oluştur
        os.makedirs(model_dir, exist_ok=True)
        
        # Veri yapıları
        self.error_data = []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Modeller
        self.pattern_model = None  # Hata örüntülerini tespit eden model
        self.prediction_model = None  # Olası hataları tahmin eden model
        self.anomaly_model = None  # Anormal hataları tespit eden model
        
        # Etiketleyiciler
        self._init_encoders()
        
        # Modelleri yükle veya oluştur
        self._load_or_create_models()
        
        # Analiz zaman aralığı (saat)
        self.analysis_interval = 1
        
        # Analiz thread'i
        self.analysis_thread = None
        self.running = False
        
        # Error Manager'a gözlemci olarak kaydol
        self.error_manager.add_observer(self._on_error_event)
    
    def _init_encoders(self):
        """Label encoder'ları başlat"""
        self.label_encoders["category"] = LabelEncoder()
        self.label_encoders["severity"] = LabelEncoder()
        
        # Tüm kategoriler ve ciddiyet seviyelerini tanımla
        self.label_encoders["category"].fit([c.name for c in ErrorCategory])
        self.label_encoders["severity"].fit([s.name for s in ErrorSeverity])
    
    def _load_or_create_models(self):
        """Mevcut modelleri yükle veya yenilerini oluştur"""
        pattern_model_path = os.path.join(self.model_dir, "pattern_model.pkl")
        prediction_model_path = os.path.join(self.model_dir, "prediction_model.pkl")
        anomaly_model_path = os.path.join(self.model_dir, "anomaly_model.pkl")
        
        try:
            # Modelleri yüklemeyi dene
            if os.path.exists(pattern_model_path):
                self.pattern_model = joblib.load(pattern_model_path)
                print("Hata örüntü modeli yüklendi.")
            
            if os.path.exists(prediction_model_path):
                self.prediction_model = joblib.load(prediction_model_path)
                print("Hata tahmin modeli yüklendi.")
            
            if os.path.exists(anomaly_model_path):
                self.anomaly_model = joblib.load(anomaly_model_path)
                print("Anomali tespit modeli yüklendi.")
        
        except Exception as e:
            print(f"Model yükleme hatası: {str(e)}")
            
            # Hata durumunda yeni modeller oluştur
            self._create_default_models()
    
    def _create_default_models(self):
        """Varsayılan modelleri oluştur"""
        # Hata örüntü modeli (Random Forest)
        self.pattern_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Hata tahmin modeli (Random Forest)
        self.prediction_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Anomali tespit modeli (Isolation Forest)
        self.anomaly_model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        
        print("Varsayılan modeller oluşturuldu.")
    
    def _on_error_event(self, event):
        """Hata olaylarını dinle ve veriyi topla"""
        if isinstance(event, ErrorEvent):
            # Olaydan veriyi çıkar
            error_info = {
                "timestamp": event.timestamp,
                "category": event.category.name,
                "severity": event.severity.name,
                "message": event.message,
                "hour": datetime.fromtimestamp(event.timestamp).hour,
                "minute": datetime.fromtimestamp(event.timestamp).minute,
                "second": datetime.fromtimestamp(event.timestamp).second,
                "day_of_week": datetime.fromtimestamp(event.timestamp).weekday(),
            }
            
            # Detayları ekle
            if event.details:
                for key, value in event.details.items():
                    # Sadece sayısal veya kategorik değerleri ekle
                    if isinstance(value, (int, float, str, bool)):
                        error_info[f"detail_{key}"] = value
            
            self.error_data.append(error_info)
            
            # Yeterli veri toplandıysa anomali tespiti yap
            if len(self.error_data) >= 10:
                threading.Thread(target=self._check_for_anomalies).start()
    
    def start_analysis(self):
        """Periyodik analiz döngüsünü başlat"""
        if self.analysis_thread is None or not self.analysis_thread.is_alive():
            self.running = True
            self.analysis_thread = threading.Thread(target=self._analysis_loop)
            self.analysis_thread.daemon = True
            self.analysis_thread.start()
            print(f"Hata analizi her {self.analysis_interval} saatte bir çalışacak şekilde başlatıldı.")
    
    def stop_analysis(self):
        """Analiz döngüsünü durdur"""
        self.running = False
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=1.0)
            print("Hata analiz döngüsü durduruldu.")
    
    def _analysis_loop(self):
        """Periyodik analiz döngüsü"""
        while self.running:
            # Analiz yap
            self.analyze_errors()
            
            # Modelleri eğit
            if len(self.error_data) >= 20:
                self.train_models()
            
            # Analiz aralığı kadar bekle (saat * 3600 saniye)
            sleep_time = self.analysis_interval * 3600
            time.sleep(sleep_time)
    
    def analyze_errors(self):
        """Hata verilerini analiz et ve raporla"""
        if not self.error_data:
            print("Analiz için hata verisi bulunamadı.")
            return
        
        try:
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(self.error_data)
            
            # Zaman bazlı analizler
            self._analyze_time_patterns(df)
            
            # Kategori dağılımı
            self._analyze_category_distribution(df)
            
            # İlişki analizi
            self._analyze_correlations(df)
            
            # Tahmin
            if len(df) >= 30:
                self._predict_future_errors(df)
        
        except Exception as e:
            print(f"Analiz hatası: {str(e)}")
    
    def _analyze_time_patterns(self, df):
        """Zaman bazlı örüntüleri analiz et"""
        if 'timestamp' not in df.columns:
            return
        
        # Timestamp'ten datetime objesi oluştur
        df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
        
        # Son 24 saatteki hata sayısı
        now = datetime.now()
        last_day = df[df['datetime'] > (now - timedelta(days=1))]
        
        if not last_day.empty:
            print(f"Son 24 saatte {len(last_day)} hata oluştu.")
            
            # Saatlik dağılım
            hourly = last_day.groupby(last_day['datetime'].dt.hour).size()
            peak_hour = hourly.idxmax() if not hourly.empty else None
            
            if peak_hour is not None:
                print(f"En çok hata {peak_hour}:00 saatinde oluştu ({hourly[peak_hour]} hata).")
            
            # Ciddiyet seviyesine göre dağılım
            if 'severity' in last_day.columns:
                severity_counts = last_day['severity'].value_counts()
                print("Ciddiyet dağılımı:")
                for sev, count in severity_counts.items():
                    print(f"  - {sev}: {count} hata")
    
    def _analyze_category_distribution(self, df):
        """Hata kategorilerinin dağılımını analiz et"""
        if 'category' not in df.columns:
            return
        
        category_counts = df['category'].value_counts()
        print("\nKategori dağılımı:")
        for cat, count in category_counts.items():
            print(f"  - {cat}: {count} hata ({count/len(df)*100:.1f}%)")
        
        # En problemli kategoriyi bul
        if not category_counts.empty:
            problematic_category = category_counts.idxmax()
            print(f"\nEn problemli kategori: {problematic_category} ({category_counts[problematic_category]} hata)")
    
    def _analyze_correlations(self, df):
        """Hatalar arasındaki ilişkileri analiz et"""
        # Sayısal ve kategorik sütunları ayır
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Korelasyon matrisi
            corr_matrix = df[numeric_cols].corr()
            
            # Yüksek korelasyonlu özellikleri bul
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.7:
                        high_corr_pairs.append(
                            (numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j])
                        )
            
            if high_corr_pairs:
                print("\nYüksek korelasyonlu özellikler:")
                for col1, col2, corr in high_corr_pairs:
                    print(f"  - {col1} ve {col2}: {corr:.2f}")
    
    def _predict_future_errors(self, df):
        """Gelecekteki olası hataları tahmin et"""
        if self.prediction_model is None or len(df) < 30:
            return
        
        try:
            # Veriyi hazırla
            X, _ = self._prepare_data(df)
            
            if X is None or X.shape[0] == 0:
                return
            
            # Son 5 hatayı al
            last_errors = X[-5:]
            
            # Tahmin yap
            if isinstance(self.prediction_model, RandomForestClassifier):
                # Sınıf olasılıklarını al
                proba = self.prediction_model.predict_proba(last_errors)
                
                # Yüksek olasılıklı sınıfları bul
                high_prob_classes = []
                for i, probabilities in enumerate(proba):
                    high_probs = [(self.prediction_model.classes_[j], p) 
                                for j, p in enumerate(probabilities) if p > 0.3]
                    high_prob_classes.extend(high_probs)
                
                if high_prob_classes:
                    print("\nOlası gelecek hatalar:")
                    for cls, prob in sorted(high_prob_classes, key=lambda x: x[1], reverse=True)[:3]:
                        try:
                            category_idx = int(cls)
                            category_name = self.label_encoders["category"].inverse_transform([category_idx])[0]
                            print(f"  - {category_name} kategorisinde hata olasılığı: {prob:.2f}")
                        except:
                            print(f"  - Kategori {cls} için hata olasılığı: {prob:.2f}")
        
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
    
    def _check_for_anomalies(self):
        """Anormal hataları tespit et"""
        if self.anomaly_model is None or len(self.error_data) < 10:
            return
        
        try:
            # Son 10 hatayı al
            recent_errors = self.error_data[-10:]
            df_recent = pd.DataFrame(recent_errors)
            
            # Veriyi hazırla
            X, _ = self._prepare_data(df_recent)
            
            if X is None or X.shape[0] == 0:
                return
            
            # Anomali tahmini
            predictions = self.anomaly_model.predict(X)
            
            # -1 değerleri anomali
            anomalies = [i for i, pred in enumerate(predictions) if pred == -1]
            
            if anomalies:
                for idx in anomalies:
                    error = recent_errors[idx]
                    print(f"\nANOMALİ TESPİT EDİLDİ: {error['category']} - {error['message']}")
                    
                    # Anormal hata bildirimi yap
                    self.error_manager.report_error(
                        message=f"Anomali tespit edildi: {error['message']}",
                        category=ErrorCategory[error['category']],
                        severity=ErrorSeverity.WARNING,
                        details={
                            "original_error": error['message'],
                            "original_category": error['category'],
                            "original_severity": error['severity'],
                            "detected_by": "AI Anomaly Detection"
                        },
                        suggestions=[
                            "Bu hata beklenmeyen bir örüntüde oluştu",
                            "Sistem günlüklerini kontrol edin",
                            "Yakın zamandaki sistem değişikliklerini gözden geçirin"
                        ]
                    )
        
        except Exception as e:
            print(f"Anomali tespiti hatası: {str(e)}")
    
    def _prepare_data(self, df):
        """Veriyi model için hazırla"""
        # Kategorik değişkenleri kodla
        X_encoded = df.copy()
        
        for col, encoder in self.label_encoders.items():
            if col in X_encoded.columns:
                X_encoded[col] = encoder.transform(X_encoded[col])
        
        # Metin sütunlarını çıkar
        text_columns = X_encoded.select_dtypes(include=['object']).columns
        X_encoded = X_encoded.drop(columns=text_columns)
        
        # Eksik değerleri doldur
        X_encoded = X_encoded.fillna(0)
        
        # Timestamp ve datetime sütunlarını çıkar
        if 'timestamp' in X_encoded.columns:
            X_encoded = X_encoded.drop(columns=['timestamp'])
        if 'datetime' in X_encoded.columns:
            X_encoded = X_encoded.drop(columns=['datetime'])
        
        # Hedef değişkeni ayır (eğer varsa)
        y = None
        if 'category' in X_encoded.columns:
            y = X_encoded['category']
            X_encoded = X_encoded.drop(columns=['category'])
        
        # Veri yoksa None döndür
        if X_encoded.empty:
            return None, None
        
        # Veriyi ölçeklendir
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        return X_scaled, y
    
    def train_models(self):
        """Toplanan verilerle modelleri eğit"""
        if not self.error_data or len(self.error_data) < 20:
            print("Eğitim için yeterli veri yok (en az 20 hata gerekli).")
            return
        
        try:
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(self.error_data)
            
            # Veriyi hazırla
            X, y = self._prepare_data(df)
            
            if X is None or y is None:
                print("Eğitim için uygun veri bulunamadı.")
                return
            
            # Eğitim-test ayrımı
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            
            # Örüntü modelini eğit
            if self.pattern_model is not None:
                self.pattern_model.fit(X_train, y_train)
                
                # Değerlendirme
                y_pred = self.pattern_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                print(f"\nÖrüntü modeli eğitildi:")
                print(f"  - Doğruluk: {accuracy:.2f}")
                print(f"  - Kesinlik: {precision:.2f}")
                print(f"  - Duyarlılık: {recall:.2f}")
                print(f"  - F1 Skoru: {f1:.2f}")
            
            # Tahmin modelini eğit (aynı veri kullanılabilir)
            if self.prediction_model is not None:
                self.prediction_model.fit(X_train, y_train)
            
            # Anomali modelini eğit
            if self.anomaly_model is not None:
                self.anomaly_model.fit(X)
            
            # Modelleri kaydet
            self._save_models()
            
            print("Modeller başarıyla eğitildi ve kaydedildi.")
        
        except Exception as e:
            print(f"Model eğitimi hatası: {str(e)}")
    
    def _save_models(self):
        """Eğitilmiş modelleri kaydet"""
        try:
            # Model dizinini kontrol et
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Modelleri kaydet
            if self.pattern_model is not None:
                joblib.dump(self.pattern_model, os.path.join(self.model_dir, "pattern_model.pkl"))
            
            if self.prediction_model is not None:
                joblib.dump(self.prediction_model, os.path.join(self.model_dir, "prediction_model.pkl"))
            
            if self.anomaly_model is not None:
                joblib.dump(self.anomaly_model, os.path.join(self.model_dir, "anomaly_model.pkl"))
            
            # Ölçekleyiciyi kaydet
            joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
            
            # Etiketleyicileri kaydet
            for name, encoder in self.label_encoders.items():
                joblib.dump(encoder, os.path.join(self.model_dir, f"encoder_{name}.pkl"))
        
        except Exception as e:
            print(f"Model kaydetme hatası: {str(e)}")
    
    def get_recommendations(self):
        """Mevcut hata verilerine göre sistem iyileştirme önerileri üret"""
        if not self.error_data or len(self.error_data) < 10:
            return ["Öneri üretmek için yeterli veri bulunmuyor."]
        
        recommendations = []
        
        try:
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(self.error_data)
            
            # Kategori bazlı öneriler
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                
                # En sık karşılaşılan kategori
                if not category_counts.empty:
                    top_category = category_counts.idxmax()
                    
                    if top_category == 'CAMERA':
                        recommendations.append("Kamera donanımını yükseltmeyi düşünün")
                        recommendations.append("Kamera sürücülerinin güncel olduğundan emin olun")
                    
                    elif top_category == 'TRACKING':
                        recommendations.append("Göz izleme algoritmasını güncelleyin")
                        recommendations.append("Daha iyi ışık koşulları sağlayın")
                        recommendations.append("Gözlük kullanan kullanıcılar için özel kalibrasyon yapın")
                    
                    elif top_category == 'CALIBRATION':
                        recommendations.append("Kalibrasyon sürecini uzatın")
                        recommendations.append("Daha fazla kalibrasyon noktası ekleyin")
                        recommendations.append("Kalibrasyon sırasında kullanıcı talimatlarını iyileştirin")
                    
                    elif top_category == 'SYSTEM':
                        recommendations.append("Sistem kaynaklarını optimize edin")
                        recommendations.append("Donanım gereksinimlerini artırın")
                        recommendations.append("Performans optimizasyonu yapın")
                    
                    elif top_category == 'DATA':
                        recommendations.append("Veri depolama sisteminizi gözden geçirin")
                        recommendations.append("Veriler için yedekleme stratejisi oluşturun")
            
            # Ciddiyet bazlı öneriler
            if 'severity' in df.columns:
                severe_errors = df[df['severity'].isin(['ERROR', 'CRITICAL'])]
                
                if len(severe_errors) > len(df) * 0.3:  # %30'dan fazla ciddi hata
                    recommendations.append("Sistem genelinde kapsamlı bir hata analizi yapın")
                    recommendations.append("Kritik bileşenlerin güvenilirliğini artırın")
            
            # Zaman bazlı öneriler
            if 'timestamp' in df.columns:
                df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
                
                # Son 1 saatteki hata oranı
                now = datetime.now()
                recent_errors = df[df['datetime'] > (now - timedelta(hours=1))]
                
                if len(recent_errors) > 5:  # Son 1 saatte 5'ten fazla hata
                    recommendations.append("Sistem şu anda anormal davranıyor olabilir, diagnostik yapın")
            
            # Genel öneriler
            recommendations.append("Düzenli bakım ve kontrol planı oluşturun")
            recommendations.append("Kullanıcıları sistem hatalarına karşı eğitin")
            
            # Yapay zeka sisteminden özel öneriler
            if self.pattern_model is not None and len(self.error_data) >= 30:
                recommendations.append("Yapay zeka modelini yeni veri ile düzenli eğiterek hata tespitini geliştirin")
        
        except Exception as e:
            print(f"Öneri üretme hatası: {str(e)}")
            recommendations.append("Öneri oluşturma sırasında bir hata oluştu.")
        
        return recommendations

    def get_error_summary(self):
        """Hata verilerinin bir özetini oluştur"""
        if not self.error_data:
            return "Özet için veri bulunmuyor."
        
        try:
            # Veriyi DataFrame'e dönüştür
            df = pd.DataFrame(self.error_data)
            
            summary = []
            summary.append(f"Toplam Hata Sayısı: {len(df)}")
            
            # Kategori dağılımı
            if 'category' in df.columns:
                summary.append("\nKategori Dağılımı:")
                for cat, count in df['category'].value_counts().items():
                    summary.append(f"  - {cat}: {count} ({count/len(df)*100:.1f}%)")
            
            # Ciddiyet dağılımı
            if 'severity' in df.columns:
                summary.append("\nCiddiyet Dağılımı:")
                for sev, count in df['severity'].value_counts().items():
                    summary.append(f"  - {sev}: {count} ({count/len(df)*100:.1f}%)")
            
            # Zaman dağılımı
            if 'timestamp' in df.columns:
                df['datetime'] = df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))
                
                now = datetime.now()
                last_hour = df[df['datetime'] > (now - timedelta(hours=1))]
                last_day = df[df['datetime'] > (now - timedelta(days=1))]
                last_week = df[df['datetime'] > (now - timedelta(days=7))]
                
                summary.append("\nZaman Dağılımı:")
                summary.append(f"  - Son 1 saat: {len(last_hour)} hata")
                summary.append(f"  - Son 24 saat: {len(last_day)} hata")
                summary.append(f"  - Son 7 gün: {len(last_week)} hata")
            
            # En sık görülen hatalar
            if 'message' in df.columns:
                summary.append("\nEn Sık Görülen Hatalar:")
                for msg, count in df['message'].value_counts().head(5).items():
                    summary.append(f"  - {msg}: {count} kez")
            
            return "\n".join(summary)
        
        except Exception as e:
            return f"Özet oluşturma hatası: {str(e)}" 