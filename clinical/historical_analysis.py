"""
Göz İzleme Sistemi - Tarihsel Veri Analizi

Bu modül, hasta göz izleme verilerinin tarihsel analizini 
ve karşılaştırmasını yapmak için sınıflar ve fonksiyonlar sağlar.
"""

import os
import json
import datetime
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64

from clinical.patient_management import PatientInfo, ClinicalCase, PatientManager

logger = logging.getLogger('eye_tracker.clinical.historical')

class EyeTrackingMetrics:
    """Göz izleme metrikleri sınıfı"""
    
    # Metrik türleri ve birimleri
    METRIC_TYPES = {
        "saccade_velocity": {"label": "Sakkad Hızı", "unit": "°/s", "normal_range": (30, 500)},
        "fixation_duration": {"label": "Fiksasyon Süresi", "unit": "ms", "normal_range": (200, 500)},
        "pupil_diameter": {"label": "Pupil Çapı", "unit": "mm", "normal_range": (2, 8)},
        "blink_rate": {"label": "Kırpma Hızı", "unit": "blink/min", "normal_range": (8, 21)},
        "pursuit_accuracy": {"label": "Düzgün İzleme Doğruluğu", "unit": "%", "normal_range": (85, 100)},
        "divergence_amplitude": {"label": "Diverjans Genliği", "unit": "°", "normal_range": (4, 10)},
        "convergence_amplitude": {"label": "Konverjans Genliği", "unit": "°", "normal_range": (25, 40)},
        "fixation_stability": {"label": "Fiksasyon Stabilitesi", "unit": "°", "normal_range": (0.1, 1)},
    }
    
    @classmethod
    def get_label(cls, metric_key: str) -> str:
        """Metrik anahtarına göre etiket döndür"""
        return cls.METRIC_TYPES.get(metric_key, {}).get("label", metric_key)
    
    @classmethod
    def get_unit(cls, metric_key: str) -> str:
        """Metrik anahtarına göre birim döndür"""
        return cls.METRIC_TYPES.get(metric_key, {}).get("unit", "")
    
    @classmethod
    def get_normal_range(cls, metric_key: str) -> Tuple[float, float]:
        """Metrik anahtarına göre normal değer aralığını döndür"""
        return cls.METRIC_TYPES.get(metric_key, {}).get("normal_range", (0, 0))
    
    @classmethod
    def is_in_normal_range(cls, metric_key: str, value: float) -> bool:
        """Değerin normal aralıkta olup olmadığını kontrol et"""
        if metric_key not in cls.METRIC_TYPES:
            return True
        
        low, high = cls.get_normal_range(metric_key)
        return low <= value <= high
    
    @classmethod
    def calculate_percentile(cls, metric_key: str, value: float) -> float:
        """
        Değerin normal aralık içindeki yüzdelik dilimini hesapla
        
        Args:
            metric_key: Metrik anahtarı
            value: Metrik değeri
            
        Returns:
            Yüzdelik dilim (0-100)
        """
        if metric_key not in cls.METRIC_TYPES:
            return 50.0
        
        low, high = cls.get_normal_range(metric_key)
        range_size = high - low
        
        if range_size <= 0:
            return 50.0
        
        if value < low:
            return 0.0
        elif value > high:
            return 100.0
        else:
            return ((value - low) / range_size) * 100.0


class HistoricalDataAnalyzer:
    """Tarihsel veri analiz sınıfı"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "clinical/analysis"):
        """
        Args:
            data_dir: Veri dizini
            output_dir: Çıktı dizini
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Dizinleri oluştur
        os.makedirs(output_dir, exist_ok=True)
        
        # Hasta yöneticisi
        self.patient_manager = PatientManager()
    
    def _extract_session_data(self, session: Dict) -> Dict[str, Any]:
        """
        Seans verisinden göz izleme metriklerini çıkar
        
        Args:
            session: Seans verisi
            
        Returns:
            Göz izleme metrikleri sözlüğü
        """
        metrics = {}
        
        # Tarihi dönüştür
        if 'date' in session:
            try:
                metrics['date'] = datetime.datetime.fromisoformat(session['date'])
            except (ValueError, TypeError):
                metrics['date'] = datetime.datetime.now()
        else:
            metrics['date'] = datetime.datetime.now()
        
        # Göz izleme verilerini çıkar
        if 'eye_tracking_data' in session:
            for key, value in session['eye_tracking_data'].items():
                if isinstance(value, (int, float)):
                    metrics[key] = value
        
        return metrics
    
    def get_patient_historical_data(self, patient_id: str, case_id: Optional[str] = None,
                                  metric_keys: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Hastanın tarihsel verilerini getir
        
        Args:
            patient_id: Hasta ID'si
            case_id: Vaka ID'si (None: tüm vakalar)
            metric_keys: Alınacak metrik anahtarları (None: tüm metrikler)
            
        Returns:
            Tarihsel veri listesi
        """
        all_sessions = []
        
        # Vaka(ları) al
        cases = []
        if case_id:
            case = self.patient_manager.get_case(case_id)
            if case and case.patient_id == patient_id:
                cases.append(case)
        else:
            cases = self.patient_manager.get_patient_cases(patient_id)
        
        # Vakalardan seansları çıkar
        for case in cases:
            for session in case.sessions:
                session_data = self._extract_session_data(session)
                
                # Vaka bilgilerini ekle
                session_data['case_id'] = case.id
                session_data['case_type'] = case.case_type
                
                # İstenen metrikleri filtrele
                if metric_keys:
                    session_data = {k: v for k, v in session_data.items() 
                                   if k in metric_keys or k in ['date', 'case_id', 'case_type']}
                
                all_sessions.append(session_data)
        
        # Tarihe göre sırala
        all_sessions.sort(key=lambda x: x['date'])
        
        return all_sessions
    
    def calculate_metric_statistics(self, sessions: List[Dict[str, Any]], 
                                  metric_key: str) -> Dict[str, Any]:
        """
        Bir metrik için istatistikleri hesapla
        
        Args:
            sessions: Seans listesi
            metric_key: Metrik anahtarı
            
        Returns:
            İstatistik sözlüğü
        """
        # Metriği çıkar
        values = [s[metric_key] for s in sessions if metric_key in s and isinstance(s[metric_key], (int, float))]
        
        if not values:
            return {
                'count': 0,
                'min': 0,
                'max': 0,
                'mean': 0,
                'median': 0,
                'std': 0,
                'trend': 'stable',
                'normal_range': EyeTrackingMetrics.get_normal_range(metric_key)
            }
        
        # Temel istatistikler
        stats = {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'normal_range': EyeTrackingMetrics.get_normal_range(metric_key)
        }
        
        # Eğilim analizi
        if len(values) >= 3:
            # Basit lineer regresyon ile eğilim hesapla
            x = np.arange(len(values))
            y = np.array(values)
            
            slope, _ = np.polyfit(x, y, 1)
            
            # Eğilimi belirle
            threshold = 0.05 * np.mean(values)  # Ortalama değerin %5'i
            
            if abs(slope) < threshold:
                stats['trend'] = 'stable'
            elif slope > 0:
                stats['trend'] = 'increasing'
            else:
                stats['trend'] = 'decreasing'
        else:
            stats['trend'] = 'insufficient_data'
        
        return stats
    
    def analyze_patient_metrics(self, patient_id: str, case_id: Optional[str] = None,
                               metric_keys: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Hasta metriklerini analiz et
        
        Args:
            patient_id: Hasta ID'si
            case_id: Vaka ID'si (None: tüm vakalar)
            metric_keys: Analiz edilecek metrik anahtarları (None: tüm metrikler)
            
        Returns:
            Metrik analizlerinin sözlüğü
        """
        # Tarihsel verileri al
        sessions = self.get_patient_historical_data(patient_id, case_id)
        
        if not sessions:
            logger.warning(f"Analiz için veri bulunamadı: {patient_id}")
            return {}
        
        # Metrik anahtarlarını belirle
        if not metric_keys:
            # Tüm mevcut metrikleri bul
            all_keys = set()
            for session in sessions:
                all_keys.update(session.keys())
            
            # Sadece göz izleme metriklerini filtrele
            metric_keys = [k for k in all_keys if k in EyeTrackingMetrics.METRIC_TYPES]
        
        # Her metrik için istatistikleri hesapla
        analysis = {}
        for key in metric_keys:
            analysis[key] = self.calculate_metric_statistics(sessions, key)
            analysis[key]['label'] = EyeTrackingMetrics.get_label(key)
            analysis[key]['unit'] = EyeTrackingMetrics.get_unit(key)
        
        return analysis
    
    def generate_trend_chart(self, patient_id: str, metric_key: str, 
                           case_id: Optional[str] = None, width: int = 10, height: int = 6) -> Optional[Figure]:
        """
        Bir metrik için eğilim grafiği oluştur
        
        Args:
            patient_id: Hasta ID'si
            metric_key: Metrik anahtarı
            case_id: Vaka ID'si (None: tüm vakalar)
            width: Grafik genişliği (inç)
            height: Grafik yüksekliği (inç)
            
        Returns:
            Matplotlib Figure nesnesi
        """
        # Tarihsel verileri al
        sessions = self.get_patient_historical_data(patient_id, case_id)
        
        if not sessions:
            logger.warning(f"Grafik için veri bulunamadı: {patient_id}, {metric_key}")
            return None
        
        # Metriği içeren seansları filtrele
        filtered_sessions = [s for s in sessions if metric_key in s and isinstance(s[metric_key], (int, float))]
        
        if not filtered_sessions:
            logger.warning(f"Grafik için metrik verisi bulunamadı: {metric_key}")
            return None
        
        # Veri noktalarını hazırla
        dates = [s['date'] for s in filtered_sessions]
        values = [s[metric_key] for s in filtered_sessions]
        
        # Hasta bilgisini al
        patient = self.patient_manager.get_patient(patient_id)
        patient_name = patient.full_name if patient else "Bilinmeyen Hasta"
        
        # Normal aralığı al
        normal_low, normal_high = EyeTrackingMetrics.get_normal_range(metric_key)
        
        # Matplotlib ile grafiği oluştur
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Veri çizgisi
        ax.plot(dates, values, marker='o', linestyle='-', color='#4285F4', label='Ölçüm')
        
        # Normal aralık
        if normal_low > 0 or normal_high > 0:
            ax.axhspan(normal_low, normal_high, alpha=0.2, color='green', label='Normal Aralık')
        
        # Trend line (lineer regresyon)
        if len(dates) >= 3:
            x = np.arange(len(dates))
            y = np.array(values)
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            ax.plot(dates, p(x), "r--", alpha=0.8, label='Eğilim')
        
        # Grafik başlığı ve etiketleri
        metric_label = EyeTrackingMetrics.get_label(metric_key)
        metric_unit = EyeTrackingMetrics.get_unit(metric_key)
        
        ax.set_title(f"{patient_name} - {metric_label} Değişimi", fontsize=14)
        ax.set_xlabel("Tarih", fontsize=12)
        ax.set_ylabel(f"{metric_label} ({metric_unit})", fontsize=12)
        
        # Izgara çizgileri
        ax.grid(True, alpha=0.3)
        
        # Lejant
        ax.legend()
        
        # Tarih formatını ayarla
        fig.autofmt_xdate()
        
        return fig
    
    def generate_comparison_chart(self, patient_ids: List[str], metric_key: str,
                                width: int = 12, height: int = 7) -> Optional[Figure]:
        """
        Birden fazla hasta için karşılaştırma grafiği oluştur
        
        Args:
            patient_ids: Hasta ID'leri listesi
            metric_key: Metrik anahtarı
            width: Grafik genişliği (inç)
            height: Grafik yüksekliği (inç)
            
        Returns:
            Matplotlib Figure nesnesi
        """
        if not patient_ids:
            logger.warning("Karşılaştırma için hasta ID'leri belirtilmedi")
            return None
        
        # Matplotlib ile grafiği oluştur
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Her hasta için veri oluştur
        metric_label = EyeTrackingMetrics.get_label(metric_key)
        metric_unit = EyeTrackingMetrics.get_unit(metric_key)
        normal_low, normal_high = EyeTrackingMetrics.get_normal_range(metric_key)
        
        # Renk paleti
        colors = ['#4285F4', '#EA4335', '#FBBC05', '#34A853', '#8334A2']
        
        for i, pid in enumerate(patient_ids):
            # Hasta verisini al
            sessions = self.get_patient_historical_data(pid)
            
            # Metriği içeren seansları filtrele
            filtered_sessions = [s for s in sessions if metric_key in s and isinstance(s[metric_key], (int, float))]
            
            if not filtered_sessions:
                continue
            
            # Hasta bilgisini al
            patient = self.patient_manager.get_patient(pid)
            patient_name = patient.full_name if patient else f"Hasta {i+1}"
            
            # Veri noktalarını hazırla
            dates = [s['date'] for s in filtered_sessions]
            values = [s[metric_key] for s in filtered_sessions]
            
            # Rengi seç
            color = colors[i % len(colors)]
            
            # Veri çizgisi
            ax.plot(dates, values, marker='o', linestyle='-', color=color, label=patient_name)
        
        # Normal aralık
        if normal_low > 0 or normal_high > 0:
            ax.axhspan(normal_low, normal_high, alpha=0.2, color='green', label='Normal Aralık')
        
        # Grafik başlığı ve etiketleri
        ax.set_title(f"{metric_label} Hasta Karşılaştırması", fontsize=14)
        ax.set_xlabel("Tarih", fontsize=12)
        ax.set_ylabel(f"{metric_label} ({metric_unit})", fontsize=12)
        
        # Izgara çizgileri
        ax.grid(True, alpha=0.3)
        
        # Lejant
        ax.legend()
        
        # Tarih formatını ayarla
        fig.autofmt_xdate()
        
        return fig
    
    def save_chart(self, fig: Figure, filename: str) -> str:
        """
        Grafiği dosyaya kaydet
        
        Args:
            fig: Matplotlib Figure nesnesi
            filename: Dosya adı
            
        Returns:
            Dosya yolu
        """
        if not fig:
            return ""
        
        # Dosya yolunu oluştur
        file_path = os.path.join(self.output_dir, filename)
        
        # Grafiği kaydet
        fig.savefig(file_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return file_path
    
    def get_chart_as_base64(self, fig: Figure) -> str:
        """
        Grafiği base64 kodlu resme dönüştür
        
        Args:
            fig: Matplotlib Figure nesnesi
            
        Returns:
            Base64 kodlu resim
        """
        if not fig:
            return ""
        
        # Resmi belleğe kaydet
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        
        # Base64'e dönüştür
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_str}"


class ComparativeAnalysis:
    """Karşılaştırmalı analiz sınıfı"""
    
    def __init__(self):
        """Karşılaştırmalı analiz için başlangıç ayarları"""
        self.analyzer = HistoricalDataAnalyzer()
        self.patient_manager = PatientManager()
    
    def compare_session_data(self, session1: Dict[str, Any], 
                           session2: Dict[str, Any]) -> Dict[str, Any]:
        """
        İki seans verisini karşılaştır
        
        Args:
            session1: İlk seans verisi
            session2: İkinci seans verisi
            
        Returns:
            Karşılaştırma sonuçları
        """
        comparison = {
            'date1': session1.get('date', ''),
            'date2': session2.get('date', ''),
            'metrics': {}
        }
        
        # Ortak metrikleri bul
        common_metrics = set(session1.keys()) & set(session2.keys())
        
        # Karşılaştırılabilir metrikleri filtrele
        comparable_metrics = [m for m in common_metrics if m in EyeTrackingMetrics.METRIC_TYPES]
        
        # Her metrik için karşılaştır
        for metric in comparable_metrics:
            value1 = session1[metric]
            value2 = session2[metric]
            
            if not isinstance(value1, (int, float)) or not isinstance(value2, (int, float)):
                continue
            
            # Değişim hesapla
            change = value2 - value1
            pct_change = (change / value1) * 100 if value1 != 0 else 0
            
            # Normal aralığını kontrol et
            normal_range = EyeTrackingMetrics.get_normal_range(metric)
            in_range1 = EyeTrackingMetrics.is_in_normal_range(metric, value1)
            in_range2 = EyeTrackingMetrics.is_in_normal_range(metric, value2)
            
            # Değişim yönünü değerlendir
            if metric == "saccade_velocity":
                improved = change > 0  # Sakkad hızı artışı genelde iyidir
            elif metric == "fixation_duration":
                improved = change < 0  # Fiksasyon süresi azalışı genelde iyidir
            elif metric == "blink_rate":
                improved = abs(value2 - 15) < abs(value1 - 15)  # 15'e yakınlaşmak iyidir
            elif metric == "pupil_diameter":
                improved = abs(change) < 0.5  # Stabil pupil çapı iyidir
            elif metric == "pursuit_accuracy":
                improved = change > 0  # Doğruluk artışı iyidir
            else:
                # Varsayılan olarak aralığa yakınlaşma iyidir
                mid_range = (normal_range[0] + normal_range[1]) / 2
                improved = abs(value2 - mid_range) < abs(value1 - mid_range)
            
            # Karşılaştırma sonucunu kaydet
            comparison['metrics'][metric] = {
                'label': EyeTrackingMetrics.get_label(metric),
                'unit': EyeTrackingMetrics.get_unit(metric),
                'value1': value1,
                'value2': value2,
                'change': change,
                'pct_change': pct_change,
                'in_range1': in_range1,
                'in_range2': in_range2,
                'improved': improved
            }
        
        return comparison
    
    def compare_case_endpoints(self, case_id: str) -> Dict[str, Any]:
        """
        Vaka başlangıç ve son durumunu karşılaştır
        
        Args:
            case_id: Vaka ID'si
            
        Returns:
            Karşılaştırma sonuçları
        """
        # Vakayı al
        case = self.patient_manager.get_case(case_id)
        if not case or not case.sessions or len(case.sessions) < 2:
            logger.warning(f"Karşılaştırma için yeterli seans bulunamadı: {case_id}")
            return {}
        
        # İlk ve son seansları al
        sessions = sorted(case.sessions, key=lambda x: x.get('date', ''))
        first_session = self.analyzer._extract_session_data(sessions[0])
        last_session = self.analyzer._extract_session_data(sessions[-1])
        
        # Karşılaştır
        comparison = self.compare_session_data(first_session, last_session)
        
        # Vaka bilgilerini ekle
        comparison['case_id'] = case_id
        comparison['case_type'] = case.case_type
        comparison['diagnosis'] = case.diagnosis
        comparison['duration'] = (last_session['date'] - first_session['date']).days
        comparison['session_count'] = len(case.sessions)
        
        # Hasta bilgilerini ekle
        patient = self.patient_manager.get_patient(case.patient_id)
        if patient:
            comparison['patient_id'] = patient.id
            comparison['patient_name'] = patient.full_name
            comparison['patient_age'] = patient.age
            comparison['patient_gender'] = patient.gender
        
        # Genel ilerleme değerlendirmesi
        if comparison['metrics']:
            improved_count = sum(1 for m in comparison['metrics'].values() if m['improved'])
            total_metrics = len(comparison['metrics'])
            improvement_ratio = improved_count / total_metrics if total_metrics > 0 else 0
            
            if improvement_ratio >= 0.7:
                comparison['overall_progress'] = "Çok İyi"
            elif improvement_ratio >= 0.5:
                comparison['overall_progress'] = "İyi"
            elif improvement_ratio >= 0.3:
                comparison['overall_progress'] = "Orta"
            else:
                comparison['overall_progress'] = "Minimal"
        else:
            comparison['overall_progress'] = "Belirlenemedi"
        
        return comparison
    
    def compare_patients(self, patient_ids: List[str], 
                       metric_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Hastaları karşılaştır
        
        Args:
            patient_ids: Hasta ID'leri listesi
            metric_keys: Karşılaştırılacak metrik anahtarları (None: tüm metrikler)
            
        Returns:
            Karşılaştırma sonuçları
        """
        if not patient_ids or len(patient_ids) < 2:
            logger.warning("Karşılaştırma için en az 2 hasta gerekli")
            return {}
        
        # Metrik anahtarlarını belirle
        if not metric_keys:
            metric_keys = list(EyeTrackingMetrics.METRIC_TYPES.keys())
        
        comparison = {
            'patient_count': len(patient_ids),
            'metrics': {},
            'patients': {}
        }
        
        # Her hasta için bilgi topla
        for pid in patient_ids:
            patient = self.patient_manager.get_patient(pid)
            if not patient:
                continue
            
            # Hasta analizini yap
            analysis = self.analyzer.analyze_patient_metrics(pid, metric_keys=metric_keys)
            
            # Hasta bilgilerini ekle
            comparison['patients'][pid] = {
                'id': pid,
                'name': patient.full_name,
                'age': patient.age,
                'gender': patient.gender,
                'analysis': analysis
            }
        
        # Her metrik için karşılaştırma yap
        for metric in metric_keys:
            # Metrik bilgilerini hazırla
            comparison['metrics'][metric] = {
                'label': EyeTrackingMetrics.get_label(metric),
                'unit': EyeTrackingMetrics.get_unit(metric),
                'normal_range': EyeTrackingMetrics.get_normal_range(metric),
                'values': {}
            }
            
            # Her hasta için metrik değerlerini topla
            for pid in patient_ids:
                if pid in comparison['patients'] and metric in comparison['patients'][pid]['analysis']:
                    patient_analysis = comparison['patients'][pid]['analysis'][metric]
                    
                    # Metrik değerlerini ekle
                    comparison['metrics'][metric]['values'][pid] = {
                        'min': patient_analysis['min'],
                        'max': patient_analysis['max'],
                        'mean': patient_analysis['mean'],
                        'trend': patient_analysis['trend']
                    }
        
        return comparison 