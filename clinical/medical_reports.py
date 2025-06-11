"""
Göz İzleme Sistemi - Medikal Raporlama

Bu modül, göz izleme sisteminden elde edilen verileri kullanarak
standart medikal raporlar oluşturmak için sınıflar sağlar.
PDF ve DICOM uyumlu çıktılar üretebilir.
"""

import os
import json
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import base64
from pathlib import Path

from clinical.patient_management import PatientInfo, ClinicalCase, PatientManager

logger = logging.getLogger('eye_tracker.clinical.reports')

class ReportTemplate:
    """Rapor şablonu sınıfı"""
    
    def __init__(self, template_name: str, template_path: Optional[str] = None):
        """
        Args:
            template_name: Şablon adı
            template_path: Şablon dosya yolu (None: varsayılan şablonlar)
        """
        self.name = template_name
        
        if template_path:
            self.path = template_path
        else:
            # Varsayılan şablon dizini
            self.path = os.path.join("clinical/templates", f"{template_name}.html")
    
    def load(self) -> str:
        """
        Şablonu yükle
        
        Returns:
            Şablon içeriği
        """
        try:
            if os.path.exists(self.path):
                with open(self.path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                # Varsayılan şablona geri dön
                return self._get_default_template()
        except Exception as e:
            logger.error(f"Şablon yükleme hatası: {e}")
            return self._get_default_template()
    
    def _get_default_template(self) -> str:
        """
        Varsayılan şablonu döndür
        
        Returns:
            Varsayılan şablon içeriği
        """
        if self.name == "clinical_report":
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Klinik Rapor</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { border-bottom: 1px solid #ccc; padding-bottom: 10px; }
                    .patient-info { margin: 20px 0; }
                    .diagnosis { margin: 20px 0; }
                    .data-section { margin: 20px 0; }
                    .footer { margin-top: 50px; border-top: 1px solid #ccc; padding-top: 10px; }
                    table { width: 100%; border-collapse: collapse; }
                    table, th, td { border: 1px solid #ddd; }
                    th, td { padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .chart-container { margin: 20px 0; text-align: center; }
                    @page { size: A4; margin: 2cm; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Göz İzleme Klinik Raporu</h1>
                    <p>Rapor Tarihi: {{report_date}}</p>
                    <p>Rapor No: {{report_id}}</p>
                </div>
                
                <div class="patient-info">
                    <h2>Hasta Bilgileri</h2>
                    <p><strong>Ad Soyad:</strong> {{patient_name}}</p>
                    <p><strong>Doğum Tarihi:</strong> {{birth_date}}</p>
                    <p><strong>Yaş:</strong> {{age}}</p>
                    <p><strong>Cinsiyet:</strong> {{gender}}</p>
                    <p><strong>Hasta No:</strong> {{patient_id}}</p>
                </div>
                
                <div class="diagnosis">
                    <h2>Tanı ve Değerlendirme</h2>
                    <p><strong>Tanı:</strong> {{diagnosis}}</p>
                    <p><strong>Vaka Tipi:</strong> {{case_type}}</p>
                    <p><strong>Başlangıç Tarihi:</strong> {{start_date}}</p>
                    <p><strong>Durum:</strong> {{status}}</p>
                    <p><strong>Tedavi Planı:</strong> {{treatment_plan}}</p>
                </div>
                
                <div class="data-section">
                    <h2>Göz İzleme Verileri</h2>
                    {{eye_tracking_data}}
                </div>
                
                <div class="chart-container">
                    <h3>Göz Hareketleri Grafiği</h3>
                    <img src="{{chart_image}}" alt="Göz Hareketleri Grafiği" width="500">
                </div>
                
                <div class="footer">
                    <p><strong>Hekim:</strong> {{clinician_name}}</p>
                    <p><strong>İmza:</strong> ________________________</p>
                    <p><small>Bu rapor {{generation_time}} tarihinde otomatik olarak oluşturulmuştur.</small></p>
                </div>
            </body>
            </html>
            """
        elif self.name == "progress_report":
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>İlerleme Raporu</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { border-bottom: 1px solid #ccc; padding-bottom: 10px; }
                    .patient-info { margin: 20px 0; }
                    .progress { margin: 20px 0; }
                    .data-section { margin: 20px 0; }
                    .footer { margin-top: 50px; border-top: 1px solid #ccc; padding-top: 10px; }
                    table { width: 100%; border-collapse: collapse; }
                    table, th, td { border: 1px solid #ddd; }
                    th, td { padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .chart-container { margin: 20px 0; text-align: center; }
                    @page { size: A4; margin: 2cm; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Göz İzleme İlerleme Raporu</h1>
                    <p>Rapor Tarihi: {{report_date}}</p>
                    <p>Rapor No: {{report_id}}</p>
                </div>
                
                <div class="patient-info">
                    <h2>Hasta Bilgileri</h2>
                    <p><strong>Ad Soyad:</strong> {{patient_name}}</p>
                    <p><strong>Hasta No:</strong> {{patient_id}}</p>
                </div>
                
                <div class="progress">
                    <h2>İlerleme Özeti</h2>
                    <p><strong>Değerlendirme Dönemi:</strong> {{period_start}} - {{period_end}}</p>
                    <p><strong>Seans Sayısı:</strong> {{session_count}}</p>
                    <p><strong>Genel Durum:</strong> {{overall_status}}</p>
                </div>
                
                <div class="data-section">
                    <h2>Ölçüm Karşılaştırması</h2>
                    <table>
                        <tr>
                            <th>Metrik</th>
                            <th>İlk Ölçüm</th>
                            <th>Son Ölçüm</th>
                            <th>Değişim</th>
                        </tr>
                        {{comparison_rows}}
                    </table>
                </div>
                
                <div class="chart-container">
                    <h3>İlerleme Grafiği</h3>
                    <img src="{{progress_chart}}" alt="İlerleme Grafiği" width="500">
                </div>
                
                <div class="data-section">
                    <h2>Notlar ve Öneriler</h2>
                    <p>{{notes}}</p>
                </div>
                
                <div class="footer">
                    <p><strong>Hekim:</strong> {{clinician_name}}</p>
                    <p><strong>İmza:</strong> ________________________</p>
                    <p><small>Bu rapor {{generation_time}} tarihinde otomatik olarak oluşturulmuştur.</small></p>
                </div>
            </body>
            </html>
            """
        else:
            # Diğer şablonlar için basit bir HTML
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{{title}}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                </style>
            </head>
            <body>
                <h1>{{title}}</h1>
                <div>{{content}}</div>
            </body>
            </html>
            """


class MedicalReportGenerator:
    """Medikal rapor oluşturucu sınıf"""
    
    def __init__(self, reports_dir: str = "clinical/reports", templates_dir: str = "clinical/templates"):
        """
        Args:
            reports_dir: Raporların kaydedileceği dizin
            templates_dir: Şablon dizini
        """
        self.reports_dir = reports_dir
        self.templates_dir = templates_dir
        
        # Dizinleri oluştur
        os.makedirs(reports_dir, exist_ok=True)
        os.makedirs(templates_dir, exist_ok=True)
        
        # Hasta yöneticisi
        self.patient_manager = PatientManager()
    
    def _generate_report_id(self) -> str:
        """
        Benzersiz rapor ID'si oluştur
        
        Returns:
            Rapor ID'si
        """
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        random_suffix = os.urandom(2).hex()
        return f"RPT-{timestamp}-{random_suffix}"
    
    def _format_date(self, date_str: str) -> str:
        """
        Tarih biçimini formatla
        
        Args:
            date_str: Tarih yazısı (YYYY-MM-DD)
            
        Returns:
            Formatlanmış tarih (DD.MM.YYYY)
        """
        try:
            dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            return dt.strftime("%d.%m.%Y")
        except:
            return date_str
    
    def generate_clinical_report(self, case_id: str, include_eye_data: bool = True,
                                include_chart: bool = True, clinician_name: str = "") -> str:
        """
        Klinik rapor oluştur
        
        Args:
            case_id: Vaka ID'si
            include_eye_data: Göz izleme verilerini dahil et
            include_chart: Grafiği dahil et
            clinician_name: Hekim adı
            
        Returns:
            Rapor dosya yolu
        """
        # Vaka bilgilerini al
        case = self.patient_manager.get_case(case_id)
        if not case:
            logger.error(f"Rapor oluşturmak için vaka bulunamadı: {case_id}")
            return ""
        
        # Hasta bilgilerini al
        patient = self.patient_manager.get_patient(case.patient_id)
        if not patient:
            logger.error(f"Rapor oluşturmak için hasta bulunamadı: {case.patient_id}")
            return ""
        
        # Rapor ID'si oluştur
        report_id = self._generate_report_id()
        
        # Şablonu yükle
        template = ReportTemplate("clinical_report")
        template_content = template.load()
        
        # Göz izleme verileri
        eye_tracking_data = "<p>Göz izleme verileri mevcut değil.</p>"
        if include_eye_data and case.sessions:
            latest_session = case.get_latest_session()
            if latest_session and 'eye_tracking_data' in latest_session:
                # Göz izleme verilerini tablo olarak formatla
                data_points = latest_session['eye_tracking_data']
                eye_tracking_data = "<table><tr><th>Metrik</th><th>Değer</th><th>Birim</th><th>Referans Aralığı</th></tr>"
                
                metrics = {
                    "saccade_velocity": ("Sakkad Hızı", "°/s", "30-500"),
                    "fixation_duration": ("Fiksasyon Süresi", "ms", "200-500"),
                    "pupil_diameter": ("Pupil Çapı", "mm", "2-8"),
                    "blink_rate": ("Kırpma Hızı", "blink/min", "8-21")
                }
                
                for key, (label, unit, ref_range) in metrics.items():
                    if key in data_points:
                        value = data_points[key]
                        eye_tracking_data += f"<tr><td>{label}</td><td>{value}</td><td>{unit}</td><td>{ref_range}</td></tr>"
                
                eye_tracking_data += "</table>"
        
        # Grafik oluşturma (gerçekte bu matplotlib gibi bir kütüphane ile yapılır)
        chart_image = ""
        if include_chart and case.sessions:
            # Burada normalde matplotlib vb. ile grafik oluşturulur
            # Örnek için placeholder imajı kullanıyoruz
            chart_image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        
        # Rapor verilerini hazırla
        report_data = {
            "report_date": datetime.datetime.now().strftime("%d.%m.%Y"),
            "report_id": report_id,
            "patient_name": patient.full_name,
            "birth_date": self._format_date(patient.birth_date),
            "age": patient.age,
            "gender": patient.gender,
            "patient_id": patient.id,
            "diagnosis": case.diagnosis,
            "case_type": case.case_type,
            "start_date": self._format_date(case.start_date),
            "status": case.status,
            "treatment_plan": case.treatment_plan,
            "eye_tracking_data": eye_tracking_data,
            "chart_image": chart_image,
            "clinician_name": clinician_name,
            "generation_time": datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        }
        
        # Şablonu doldur
        for key, value in report_data.items():
            template_content = template_content.replace("{{" + key + "}}", str(value))
        
        # Raporu kaydet
        output_file = os.path.join(self.reports_dir, f"{report_id}.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"Klinik rapor oluşturuldu: {output_file}")
        
        return output_file
    
    def generate_progress_report(self, patient_id: str, case_id: Optional[str] = None, 
                               period_days: int = 30, clinician_name: str = "") -> str:
        """
        İlerleme raporu oluştur
        
        Args:
            patient_id: Hasta ID'si
            case_id: Vaka ID'si (None: en son aktif vaka)
            period_days: Değerlendirme periyodu (gün)
            clinician_name: Hekim adı
            
        Returns:
            Rapor dosya yolu
        """
        # Hasta bilgilerini al
        patient = self.patient_manager.get_patient(patient_id)
        if not patient:
            logger.error(f"Rapor oluşturmak için hasta bulunamadı: {patient_id}")
            return ""
        
        # Vaka bilgilerini al
        case = None
        if case_id:
            case = self.patient_manager.get_case(case_id)
        else:
            # En son aktif vakayı al
            cases = self.patient_manager.get_patient_cases(patient_id)
            active_cases = [c for c in cases if c.status in ["Açık", "Takipte"]]
            if active_cases:
                case = active_cases[0]
        
        if not case:
            logger.error(f"Rapor oluşturmak için vaka bulunamadı: {patient_id}")
            return ""
        
        # Rapor ID'si oluştur
        report_id = self._generate_report_id()
        
        # Şablonu yükle
        template = ReportTemplate("progress_report")
        template_content = template.load()
        
        # Periyot tarihlerini hesapla
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=period_days)
        
        # Seansları filtrele
        sessions = []
        for session in case.sessions:
            try:
                session_date = datetime.datetime.fromisoformat(session.get('date', ''))
                if start_date <= session_date <= end_date:
                    sessions.append(session)
            except (ValueError, TypeError):
                pass
        
        # İlk ve son seansları al
        first_session = None
        last_session = None
        if sessions:
            # Tarihe göre sırala
            sorted_sessions = sorted(sessions, key=lambda x: x.get('date', ''))
            first_session = sorted_sessions[0]
            last_session = sorted_sessions[-1]
        
        # Ölçüm karşılaştırma tablosunu oluştur
        comparison_rows = ""
        overall_status = "Veri yetersiz"
        
        if first_session and last_session and 'eye_tracking_data' in first_session and 'eye_tracking_data' in last_session:
            first_data = first_session['eye_tracking_data']
            last_data = last_session['eye_tracking_data']
            
            metrics = {
                "saccade_velocity": ("Sakkad Hızı", "°/s"),
                "fixation_duration": ("Fiksasyon Süresi", "ms"),
                "pupil_diameter": ("Pupil Çapı", "mm"),
                "blink_rate": ("Kırpma Hızı", "blink/min")
            }
            
            # İyileşme durumunu izle
            improvement_count = 0
            total_metrics = 0
            
            for key, (label, unit) in metrics.items():
                if key in first_data and key in last_data:
                    first_value = first_data[key]
                    last_value = last_data[key]
                    
                    # Değişim hesapla
                    if isinstance(first_value, (int, float)) and isinstance(last_value, (int, float)):
                        change = last_value - first_value
                        percent_change = (change / first_value) * 100 if first_value != 0 else 0
                        
                        # Pozitif değişim varsay (bu mantığı vaka tipine göre özelleştirmek gerekir)
                        is_improvement = (key == "saccade_velocity" and change > 0) or \
                                         (key == "fixation_duration" and change < 0) or \
                                         (key == "pupil_diameter" and abs(change) < 0.5) or \
                                         (key == "blink_rate" and change > 0)
                        
                        if is_improvement:
                            improvement_count += 1
                        
                        total_metrics += 1
                        
                        # Değişim işareti
                        change_str = f"+{change:.2f}" if change > 0 else f"{change:.2f}"
                        change_str += f" ({percent_change:.1f}%)"
                        
                        comparison_rows += f"<tr><td>{label}</td><td>{first_value} {unit}</td>" \
                                          f"<td>{last_value} {unit}</td><td>{change_str}</td></tr>"
            
            # Genel durumu hesapla
            if total_metrics > 0:
                improvement_ratio = improvement_count / total_metrics
                if improvement_ratio >= 0.7:
                    overall_status = "İyi İlerleme"
                elif improvement_ratio >= 0.4:
                    overall_status = "Orta İlerleme"
                else:
                    overall_status = "Minimal İlerleme"
        
        # İlerleme grafiği
        progress_chart = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        
        # Rapor verilerini hazırla
        report_data = {
            "report_date": datetime.datetime.now().strftime("%d.%m.%Y"),
            "report_id": report_id,
            "patient_name": patient.full_name,
            "patient_id": patient.id,
            "period_start": start_date.strftime("%d.%m.%Y"),
            "period_end": end_date.strftime("%d.%m.%Y"),
            "session_count": len(sessions),
            "overall_status": overall_status,
            "comparison_rows": comparison_rows,
            "progress_chart": progress_chart,
            "notes": case.notes,
            "clinician_name": clinician_name,
            "generation_time": datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
        }
        
        # Şablonu doldur
        for key, value in report_data.items():
            template_content = template_content.replace("{{" + key + "}}", str(value))
        
        # Raporu kaydet
        output_file = os.path.join(self.reports_dir, f"{report_id}.html")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"İlerleme raporu oluşturuldu: {output_file}")
        
        return output_file
    
    def convert_to_pdf(self, html_file: str) -> str:
        """
        HTML raporu PDF'e dönüştür
        
        Args:
            html_file: HTML rapor dosya yolu
            
        Returns:
            PDF dosya yolu
        """
        try:
            import weasyprint
            
            pdf_file = html_file.replace(".html", ".pdf")
            
            # HTML'i PDF'e dönüştür
            weasyprint.HTML(filename=html_file).write_pdf(pdf_file)
            
            logger.info(f"PDF rapor oluşturuldu: {pdf_file}")
            return pdf_file
            
        except ImportError:
            logger.warning("PDF dönüşümü için WeasyPrint kütüphanesi yüklü değil.")
            logger.warning("Kurulum: pip install weasyprint")
            return ""
        except Exception as e:
            logger.error(f"PDF dönüşüm hatası: {e}")
            return ""
    
    def prepare_dicom_report(self, html_file: str) -> str:
        """
        HTML raporu DICOM formatına hazırla (Encapsulated PDF olarak)
        
        Args:
            html_file: HTML rapor dosya yolu
            
        Returns:
            DICOM dosya yolu
        """
        # Önce PDF'e dönüştür
        pdf_file = self.convert_to_pdf(html_file)
        if not pdf_file:
            logger.error("DICOM rapor için önce PDF oluşturulmalı")
            return ""
        
        try:
            import pydicom
            from pydicom.dataset import Dataset, FileMetaDataset
            from pydicom.uid import generate_uid
            
            # PDF dosyasını oku
            with open(pdf_file, 'rb') as f:
                pdf_content = f.read()
            
            # DICOM dataset oluştur
            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.104.1'  # Encapsulated PDF
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
            
            ds = Dataset()
            ds.file_meta = file_meta
            
            # Zorunlu DICOM alanları
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
            ds.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
            ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
            ds.ContentTime = datetime.datetime.now().strftime("%H%M%S")
            ds.AccessionNumber = ''
            ds.Modality = 'OT'  # Other
            ds.PatientName = "^".join(Path(html_file).stem.split("_")[:2])
            ds.PatientID = Path(html_file).stem.split("_")[0]
            ds.PatientBirthDate = ''
            ds.PatientSex = ''
            ds.StudyInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.StudyID = ''
            ds.SeriesNumber = 1
            ds.InstanceNumber = 1
            ds.MIMETypeOfEncapsulatedDocument = 'application/pdf'
            ds.EncapsulatedDocument = pdf_content
            
            # DICOM dosyasını kaydet
            dicom_file = pdf_file.replace(".pdf", ".dcm")
            ds.save_as(dicom_file)
            
            logger.info(f"DICOM rapor oluşturuldu: {dicom_file}")
            return dicom_file
            
        except ImportError:
            logger.warning("DICOM dönüşümü için pydicom kütüphanesi yüklü değil.")
            logger.warning("Kurulum: pip install pydicom")
            return ""
        except Exception as e:
            logger.error(f"DICOM dönüşüm hatası: {e}")
            return "" 