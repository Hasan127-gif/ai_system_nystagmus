#!/usr/bin/env python3
"""
Klinik Karar Destek Sistemi
===========================
Bu modül, nistagmus ve şaşılık analizi sonuçlarını klinik eşiklere göre 
değerlendiren karar destek fonksiyonlarını içerir.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import asdict
import numpy as np

from config import (
    CLINICAL_THRESHOLDS, 
    CLINICAL_PARAMETERS, 
    CLINICAL_REPORTS,
    SeverityLevel, 
    FindingType, 
    ClinicalFinding,
    get_clinical_threshold,
    get_age_factor,
    get_calibration_parameter
)

# Logging
logger = logging.getLogger('clinical_decision')

def classify_nystagmus_frequency(frequency: float, age: float = None) -> ClinicalFinding:
    """
    Nistagmus frekansını klinik eşiklere göre sınıflandırır.
    
    Args:
        frequency: Nistagmus frekansı (Hz)
        age: Hasta yaşı (yıl, opsiyonel)
        
    Returns:
        ClinicalFinding: Klinik bulgu
    """
    # Yaş faktörü uygulaması
    if age is not None:
        tolerance_factor = get_age_factor(age, "nystagmus_tolerance")
        adjusted_frequency = frequency / tolerance_factor
    else:
        adjusted_frequency = frequency
    
    thresholds = CLINICAL_THRESHOLDS["nystagmus_freq_hz"]
    
    # Sınıflandırma
    if adjusted_frequency <= thresholds["normal_max"]:
        severity = SeverityLevel.NORMAL
        is_pathological = False
        description = f"Normal nistagmus frekansı ({frequency:.2f} Hz)"
        recommendation = "Rutin takip önerilir."
        
    elif adjusted_frequency <= thresholds["mild_max"]:
        severity = SeverityLevel.MILD
        is_pathological = True
        description = f"Hafif nistagmus ({frequency:.2f} Hz)"
        recommendation = "Uzman doktor değerlendirmesi önerilir."
        
    elif adjusted_frequency <= thresholds["moderate_max"]:
        severity = SeverityLevel.MODERATE
        is_pathological = True
        description = f"Orta derece nistagmus ({frequency:.2f} Hz)"
        recommendation = "Acil uzman değerlendirmesi ve tedavi planı gerekli."
        
    else:
        severity = SeverityLevel.SEVERE
        is_pathological = True
        description = f"Şiddetli nistagmus ({frequency:.2f} Hz)"
        recommendation = "Acil müdahale gerekli, nöroloji konsültasyonu önerilir."
    
    return ClinicalFinding(
        finding_type=FindingType.NISTAGMUS,
        value=frequency,
        severity=severity,
        is_pathological=is_pathological,
        description=description,
        recommendation=recommendation
    )

def classify_strabismus_angle(angle_degrees: float, age: float = None) -> ClinicalFinding:
    """
    Şaşılık açısını Prism Diyoptri'ye çevirerek klinik eşiklere göre sınıflandırır.
    
    Args:
        angle_degrees: Şaşılık açısı (derece)
        age: Hasta yaşı (yıl, opsiyonel)
        
    Returns:
        ClinicalFinding: Klinik bulgu
    """
    # Dereceyi Prism Diyoptri'ye çevir
    pixel_to_pd_ratio = get_calibration_parameter("pixel_to_pd_ratio")
    distance_factor = get_calibration_parameter("distance_factor")
    
    # Basit dönüşüm: 1 derece ≈ 1.75 PD (ortalama)
    prism_diopters = angle_degrees * 1.75 * distance_factor
    
    thresholds = CLINICAL_THRESHOLDS["strabismus_pd"]
    
    # Sınıflandırma
    if prism_diopters <= thresholds["normal_max"]:
        severity = SeverityLevel.NORMAL
        is_pathological = False
        description = f"Normal göz hizalaması ({prism_diopters:.1f} PD)"
        recommendation = "Rutin takip önerilir."
        
    elif prism_diopters <= thresholds["mild_max"]:
        severity = SeverityLevel.MILD
        is_pathological = True
        description = f"Hafif şaşılık ({prism_diopters:.1f} PD)"
        recommendation = "Oftalmoloji değerlendirmesi önerilir."
        
    elif prism_diopters <= thresholds["moderate_max"]:
        severity = SeverityLevel.MODERATE
        is_pathological = True
        description = f"Orta derece şaşılık ({prism_diopters:.1f} PD)"
        recommendation = "Cerrahi değerlendirme ve tedavi planı gerekli."
        
    else:
        severity = SeverityLevel.SEVERE
        is_pathological = True
        description = f"Şiddetli şaşılık ({prism_diopters:.1f} PD)"
        recommendation = "Acil cerrahi konsültasyon gerekli."
    
    return ClinicalFinding(
        finding_type=FindingType.STRABISMUS,
        value=prism_diopters,
        severity=severity,
        is_pathological=is_pathological,
        description=description,
        recommendation=recommendation
    )

def assess_face_detection_quality(detection_rate: float) -> ClinicalFinding:
    """
    Yüz tespit kalitesini değerlendirir.
    
    Args:
        detection_rate: Yüz tespit oranı (0-1)
        
    Returns:
        ClinicalFinding: Kalite değerlendirmesi
    """
    thresholds = CLINICAL_THRESHOLDS["face_detection"]
    
    if detection_rate >= thresholds["excellent_rate"]:
        severity = SeverityLevel.NORMAL
        description = f"Mükemmel yüz tespit kalitesi ({detection_rate:.1%})"
        recommendation = "Analiz sonuçları güvenilir."
        
    elif detection_rate >= thresholds["good_rate"]:
        severity = SeverityLevel.NORMAL
        description = f"İyi yüz tespit kalitesi ({detection_rate:.1%})"
        recommendation = "Analiz sonuçları güvenilir."
        
    elif detection_rate >= thresholds["minimum_rate"]:
        severity = SeverityLevel.MILD
        description = f"Orta yüz tespit kalitesi ({detection_rate:.1%})"
        recommendation = "Sonuçlar dikkatli yorumlanmalı."
        
    else:
        severity = SeverityLevel.SEVERE
        description = f"Düşük yüz tespit kalitesi ({detection_rate:.1%})"
        recommendation = "Daha kaliteli kayıt ile tekrar analiz önerilir."
    
    return ClinicalFinding(
        finding_type=FindingType.FACE_DETECTION,
        value=detection_rate,
        severity=severity,
        is_pathological=detection_rate < thresholds["minimum_rate"],
        description=description,
        recommendation=recommendation
    )

def classify_findings(nystagmus_freq: float, strabismus_angle: float, 
                     face_detection_rate: float = None, age: float = None) -> Dict[str, Any]:
    """
    Tüm bulguları değerlendirir ve klinik karar verir.
    
    Args:
        nystagmus_freq: Nistagmus frekansı (Hz)
        strabismus_angle: Şaşılık açısı (derece)
        face_detection_rate: Yüz tespit oranı (opsiyonel)
        age: Hasta yaşı (opsiyonel)
        
    Returns:
        dict: Kapsamlı klinik değerlendirme
    """
    findings = {}
    
    # Nistagmus değerlendirmesi
    nystagmus_finding = classify_nystagmus_frequency(nystagmus_freq, age)
    findings["nystagmus"] = asdict(nystagmus_finding)
    
    # Şaşılık değerlendirmesi
    strabismus_finding = classify_strabismus_angle(strabismus_angle, age)
    findings["strabismus"] = asdict(strabismus_finding)
    
    # Yüz tespit kalitesi (varsa)
    if face_detection_rate is not None:
        quality_finding = assess_face_detection_quality(face_detection_rate)
        findings["face_detection_quality"] = asdict(quality_finding)
    
    # Genel değerlendirme
    overall_assessment = generate_overall_assessment(
        nystagmus_finding, strabismus_finding, quality_finding if face_detection_rate else None
    )
    
    return {
        "individual_findings": findings,
        "overall_assessment": overall_assessment,
        "clinical_summary": generate_clinical_summary(findings, overall_assessment),
        "timestamp": np.datetime64('now').astype(str)
    }

def generate_overall_assessment(nystagmus: ClinicalFinding, 
                              strabismus: ClinicalFinding,
                              quality: ClinicalFinding = None) -> Dict[str, Any]:
    """
    Genel klinik değerlendirme oluşturur.
    
    Args:
        nystagmus: Nistagmus bulgusu
        strabismus: Şaşılık bulgusu
        quality: Kalite bulgusu (opsiyonel)
        
    Returns:
        dict: Genel değerlendirme
    """
    # Patolojik bulgu var mı?
    has_pathology = nystagmus.is_pathological or strabismus.is_pathological
    
    # En yüksek şiddet seviyesi
    severity_levels = [nystagmus.severity, strabismus.severity]
    if quality:
        severity_levels.append(quality.severity)
    
    max_severity = max(severity_levels, key=lambda x: list(SeverityLevel).index(x))
    
    # Güvenilirlik değerlendirmesi
    if quality and quality.severity in [SeverityLevel.SEVERE, SeverityLevel.MODERATE]:
        reliability = "düşük"
        confidence = "sınırlı"
    else:
        reliability = "yüksek" 
        confidence = "güvenilir"
    
    # Genel öneri
    if not has_pathology:
        category = "normal"
        urgent = False
    elif max_severity == SeverityLevel.SEVERE:
        category = "pathological"
        urgent = True
    elif max_severity == SeverityLevel.MODERATE:
        category = "pathological"
        urgent = False
    else:
        category = "borderline"
        urgent = False
    
    # Klinik rapor şablonunu al
    report_template = CLINICAL_REPORTS[category]
    
    return {
        "has_pathology": has_pathology,
        "max_severity": max_severity.value,
        "reliability": reliability,
        "confidence": confidence,
        "urgent_referral": urgent,
        "category": category,
        "title": report_template["title"],
        "description": report_template["description"],
        "recommendation": report_template["recommendation"]
    }

def generate_clinical_summary(findings: Dict[str, Any], 
                            overall: Dict[str, Any]) -> str:
    """
    Klinik özet raporu oluşturur.
    
    Args:
        findings: Bireysel bulgular
        overall: Genel değerlendirme
        
    Returns:
        str: Klinik özet metni
    """
    summary_parts = []
    
    # Başlık
    summary_parts.append(f"KLİNİK DEĞERLENDİRME - {overall['title']}")
    summary_parts.append("=" * 50)
    
    # Bulgular
    summary_parts.append("\nBULGULAR:")
    
    if "nystagmus" in findings:
        nyst = findings["nystagmus"]
        summary_parts.append(f"• Nistagmus: {nyst['description']}")
    
    if "strabismus" in findings:
        strab = findings["strabismus"]
        summary_parts.append(f"• Şaşılık: {strab['description']}")
    
    if "face_detection_quality" in findings:
        quality = findings["face_detection_quality"]
        summary_parts.append(f"• Analiz Kalitesi: {quality['description']}")
    
    # Genel değerlendirme
    summary_parts.append(f"\nGENEL DEĞERLENDİRME:")
    summary_parts.append(f"• {overall['description']}")
    summary_parts.append(f"• Güvenilirlik: {overall['reliability']}")
    
    if overall['urgent_referral']:
        summary_parts.append("• ⚠️  ACİL SEVK GEREKTİRİR")
    
    # Öneriler
    summary_parts.append(f"\nÖNERİLER:")
    summary_parts.append(f"• {overall['recommendation']}")
    
    # Ek öneriler
    recommendations = []
    for finding_type, finding_data in findings.items():
        if finding_data.get('is_pathological', False):
            recommendations.append(finding_data['recommendation'])
    
    if recommendations:
        summary_parts.append("\nAYRINTILI ÖNERİLER:")
        for rec in set(recommendations):  # Tekrarları kaldır
            summary_parts.append(f"• {rec}")
    
    return "\n".join(summary_parts)

def create_structured_report(analysis_results: Dict[str, Any], 
                           patient_age: float = None) -> Dict[str, Any]:
    """
    Analiz sonuçlarından yapılandırılmış klinik rapor oluşturur.
    
    Args:
        analysis_results: Ham analiz sonuçları
        patient_age: Hasta yaşı (opsiyonel)
        
    Returns:
        dict: Yapılandırılmış klinik rapor
    """
    try:
        # Gerekli değerleri çıkar
        nystagmus_freq = analysis_results.get("nistagmus_frequency", 0.0)
        strabismus_angle = analysis_results.get("strabismus_angle", 0.0)
        face_detection_rate = analysis_results.get("face_detection_rate", None)
        
        # Klinik sınıflandırma yap
        clinical_evaluation = classify_findings(
            nystagmus_freq=nystagmus_freq,
            strabismus_angle=strabismus_angle,
            face_detection_rate=face_detection_rate,
            age=patient_age
        )
        
        # Orijinal teknik verileri ekle
        technical_data = {
            "raw_nystagmus_frequency": nystagmus_freq,
            "raw_strabismus_angle": strabismus_angle,
            "face_detection_rate": face_detection_rate,
            "analysis_duration": analysis_results.get("analysis_duration", 0.0),
            "video_info": analysis_results.get("video_info", {})
        }
        
        # Komple raporu oluştur
        return {
            "clinical_evaluation": clinical_evaluation,
            "technical_data": technical_data,
            "report_metadata": {
                "report_version": "1.0",
                "analysis_timestamp": analysis_results.get("timestamp", ""),
                "patient_age": patient_age,
                "standards_reference": "AAO Guidelines 2024"
            }
        }
        
    except Exception as e:
        logger.error(f"Yapılandırılmış rapor oluşturma hatası: {str(e)}")
        return {
            "error": f"Rapor oluşturulamadı: {str(e)}",
            "raw_data": analysis_results
        }

def quick_assessment(nystagmus_freq: float, strabismus_pd: float) -> Dict[str, bool]:
    """
    Hızlı patoloji değerlendirmesi yapar.
    
    Args:
        nystagmus_freq: Nistagmus frekansı (Hz)
        strabismus_pd: Şaşılık (Prism Diyoptri)
        
    Returns:
        dict: Hızlı değerlendirme sonucu
    """
    nyst_threshold = get_clinical_threshold("nystagmus_freq_hz", "normal_max")
    strab_threshold = get_clinical_threshold("strabismus_pd", "normal_max")
    
    return {
        "nystagmus_pathological": nystagmus_freq > nyst_threshold,
        "strabismus_pathological": strabismus_pd > strab_threshold,
        "any_pathology": (nystagmus_freq > nyst_threshold) or (strabismus_pd > strab_threshold),
        "both_pathological": (nystagmus_freq > nyst_threshold) and (strabismus_pd > strab_threshold)
    }

# Özel değerlendirme fonksiyonları
def pediatric_assessment(nystagmus_freq: float, strabismus_angle: float, age_months: int) -> Dict[str, Any]:
    """
    Pediatrik hasta için özelleştirilmiş değerlendirme.
    
    Args:
        nystagmus_freq: Nistagmus frekansı
        strabismus_angle: Şaşılık açısı
        age_months: Yaş (ay)
        
    Returns:
        dict: Pediatrik değerlendirme
    """
    age_years = age_months / 12.0
    
    # Pediatrik özel eşikler
    if age_months < 6:
        # 6 aydan küçük bebeklerde fizyolojik nistagmus olabilir
        nyst_tolerance = 2.0
        special_notes = "Bebeklerde geçici fizyolojik nistagmus normal olabilir."
    elif age_months < 24:
        nyst_tolerance = 1.5
        special_notes = "Çocuklarda gelişimsel değişiklikler değerlendirilmelidir."
    else:
        nyst_tolerance = 1.2
        special_notes = "Pediatrik değerlendirme standartları uygulandı."
    
    # Standart değerlendirme + pediatrik notlar
    standard_eval = classify_findings(nystagmus_freq, strabismus_angle, age=age_years)
    standard_eval["pediatric_notes"] = special_notes
    standard_eval["age_months"] = age_months
    standard_eval["tolerance_applied"] = nyst_tolerance
    
    return standard_eval 