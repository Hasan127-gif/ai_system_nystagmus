#!/usr/bin/env python3
"""
Klinik Konfigürasyon ve Eşikler
===============================
Bu modül, nistagmus ve şaşılık analizi için klinik eşikleri ve 
değerlendirme parametrelerini içerir.
"""

import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Logging
logger = logging.getLogger('clinical_config')

# Klinik Eşikler - Uluslararası Standartlara Göre
CLINICAL_THRESHOLDS = {
    # Nistagmus Frekans Eşikleri (Hz)
    "nystagmus_freq_hz": {
        "normal_max": 0.5,          # ≤0.5 Hz normal kabul edilir
        "mild_min": 0.5,            # 0.5-2.0 Hz hafif nistagmus
        "mild_max": 2.0,
        "moderate_min": 2.0,        # 2.0-5.0 Hz orta derece nistagmus
        "moderate_max": 5.0,
        "severe_min": 5.0           # ≥5.0 Hz şiddetli nistagmus
    },
    
    # Şaşılık Eşikleri (Prism Diyoptri - PD)
    "strabismus_pd": {
        "normal_max": 1.0,          # ≤1 PD normal kabul edilir
        "mild_min": 1.0,            # 1-5 PD hafif şaşılık
        "mild_max": 5.0,
        "moderate_min": 5.0,        # 5-15 PD orta derece şaşılık
        "moderate_max": 15.0,
        "severe_min": 15.0          # ≥15 PD şiddetli şaşılık
    },
    
    # Yüz Tespit Kalitesi Eşikleri
    "face_detection": {
        "minimum_rate": 0.3,        # Minimum %30 yüz tespit oranı
        "good_rate": 0.7,           # %70 üstü iyi kalite
        "excellent_rate": 0.9       # %90 üstü mükemmel kalite
    },
    
    # Video Kalitesi Eşikleri
    "video_quality": {
        "min_duration": 2.0,        # Minimum 2 saniye
        "min_fps": 15.0,            # Minimum 15 FPS
        "min_resolution": 240,      # Minimum 240p
        "min_data_points": 5        # Minimum 5 veri noktası
    }
}

# Klinik Sınıflandırma Seviyeleri
class SeverityLevel(Enum):
    NORMAL = "normal"
    MILD = "hafif"
    MODERATE = "orta"
    SEVERE = "şiddetli"
    UNKNOWN = "belirsiz"

class FindingType(Enum):
    NISTAGMUS = "nistagmus"
    STRABISMUS = "şaşılık"
    FACE_DETECTION = "yüz_tespit"
    VIDEO_QUALITY = "video_kalite"

@dataclass
class ClinicalFinding:
    """Klinik bulgu sınıfı"""
    finding_type: FindingType
    value: float
    severity: SeverityLevel
    is_pathological: bool
    description: str
    recommendation: str

# Klinik Değerlendirme Parametreleri
CLINICAL_PARAMETERS = {
    # Yaş gruplarına göre değerlendirme faktörleri
    "age_factors": {
        "infant": {"min_age": 0, "max_age": 2, "nystagmus_tolerance": 1.2},      # Bebeklerde %20 tolerans
        "child": {"min_age": 2, "max_age": 12, "nystagmus_tolerance": 1.1},     # Çocuklarda %10 tolerans
        "adult": {"min_age": 12, "max_age": 65, "nystagmus_tolerance": 1.0},    # Yetişkinlerde standart
        "elderly": {"min_age": 65, "max_age": 100, "nystagmus_tolerance": 1.1}  # Yaşlılarda %10 tolerans
    },
    
    # Kalibrasyon parametreleri
    "calibration": {
        "pixel_to_pd_ratio": 0.1,      # Piksel -> Prism Diyoptri dönüşüm
        "distance_factor": 1.0,         # Mesafe düzeltme faktörü
        "camera_angle_correction": 1.0   # Kamera açısı düzeltmesi
    },
    
    # Güvenilirlik eşikleri
    "reliability": {
        "min_confidence": 0.6,          # Minimum güven skoru
        "min_analysis_duration": 1.0,   # Minimum analiz süresi
        "max_noise_level": 0.3          # Maksimum gürültü seviyesi
    }
}

# Klinik Raporlama Şablonları
CLINICAL_REPORTS = {
    "normal": {
        "title": "Normal Bulgular",
        "description": "Analiz sonuçları normal sınırlar içerisindedir.",
        "recommendation": "Rutin takip önerilir."
    },
    
    "pathological": {
        "title": "Patolojik Bulgular",
        "description": "Analiz sonuçları patolojik bulgular göstermektedir.",
        "recommendation": "Uzman doktor değerlendirmesi önerilir."
    },
    
    "borderline": {
        "title": "Sınırda Bulgular",
        "description": "Analiz sonuçları sınırda değerlerde bulunmuştur.",
        "recommendation": "Yakın takip ve tekrar değerlendirme önerilir."
    },
    
    "insufficient": {
        "title": "Yetersiz Veri",
        "description": "Analiz için yeterli veri elde edilememiştir.",
        "recommendation": "Daha kaliteli kayıt ile tekrar analiz önerilir."
    }
}

# Uluslararası Standartlar ve Referanslar
INTERNATIONAL_STANDARDS = {
    "references": [
        "American Academy of Ophthalmology Guidelines",
        "European Strabismus Association Standards",
        "International Nystagmus Classification",
        "WHO Vision Impairment Guidelines"
    ],
    
    "validation_studies": [
        "Leigh & Zee Nystagmus Classification (2015)",
        "Wright & Spiegel Strabismus Measurements (2018)",
        "PEDIG Clinical Protocols (2020)"
    ]
}

def get_clinical_threshold(parameter: str, sub_parameter: str = None) -> float:
    """
    Klinik eşik değerini döndürür.
    
    Args:
        parameter: Ana parametre adı
        sub_parameter: Alt parametre adı
        
    Returns:
        float: Eşik değeri
    """
    try:
        if sub_parameter:
            return CLINICAL_THRESHOLDS[parameter][sub_parameter]
        else:
            return CLINICAL_THRESHOLDS[parameter]
    except KeyError:
        logger.warning(f"Bilinmeyen klinik eşik: {parameter}.{sub_parameter}")
        return 0.0

def get_age_factor(age: float, parameter: str = "nystagmus_tolerance") -> float:
    """
    Yaş grubuna göre düzeltme faktörünü döndürür.
    
    Args:
        age: Yaş (yıl)
        parameter: Düzeltme parametresi
        
    Returns:
        float: Düzeltme faktörü
    """
    for age_group, params in CLINICAL_PARAMETERS["age_factors"].items():
        if params["min_age"] <= age < params["max_age"]:
            return params.get(parameter, 1.0)
    
    return 1.0  # Varsayılan faktör

def update_clinical_threshold(parameter: str, sub_parameter: str, value: float):
    """
    Klinik eşik değerini günceller.
    
    Args:
        parameter: Ana parametre adı
        sub_parameter: Alt parametre adı
        value: Yeni değer
    """
    try:
        CLINICAL_THRESHOLDS[parameter][sub_parameter] = value
        logger.info(f"Klinik eşik güncellendi: {parameter}.{sub_parameter} = {value}")
    except KeyError:
        logger.error(f"Geçersiz klinik eşik parametresi: {parameter}.{sub_parameter}")

def get_calibration_parameter(parameter: str) -> float:
    """
    Kalibrasyon parametresini döndürür.
    
    Args:
        parameter: Parametre adı
        
    Returns:
        float: Parametre değeri
    """
    return CLINICAL_PARAMETERS["calibration"].get(parameter, 1.0)

def update_calibration_parameter(parameter: str, value: float):
    """
    Kalibrasyon parametresini günceller.
    
    Args:
        parameter: Parametre adı
        value: Yeni değer
    """
    CLINICAL_PARAMETERS["calibration"][parameter] = value
    logger.info(f"Kalibrasyon parametresi güncellendi: {parameter} = {value}")

# Konfigürasyon doğrulama fonksiyonu
def validate_clinical_config() -> bool:
    """
    Klinik konfigürasyonun geçerliliğini kontrol eder.
    
    Returns:
        bool: Konfigürasyon geçerli mi?
    """
    try:
        # Eşiklerin mantıklı olup olmadığını kontrol et
        nyst_thresholds = CLINICAL_THRESHOLDS["nystagmus_freq_hz"]
        if not (nyst_thresholds["normal_max"] <= nyst_thresholds["mild_min"] <= 
                nyst_thresholds["mild_max"] <= nyst_thresholds["moderate_min"]):
            logger.error("Nistagmus eşikleri mantıklı değil")
            return False
        
        strab_thresholds = CLINICAL_THRESHOLDS["strabismus_pd"]
        if not (strab_thresholds["normal_max"] <= strab_thresholds["mild_min"] <= 
                strab_thresholds["mild_max"] <= strab_thresholds["moderate_min"]):
            logger.error("Şaşılık eşikleri mantıklı değil")
            return False
        
        logger.info("Klinik konfigürasyon doğrulaması başarılı")
        return True
        
    except Exception as e:
        logger.error(f"Klinik konfigürasyon doğrulama hatası: {str(e)}")
        return False

# Başlangıç doğrulaması
if __name__ == "__main__":
    validate_clinical_config() 