"""
Göz İzleme Sistemi - Klinik Hasta ve Vaka Yönetimi

Bu modül, göz izleme sisteminde klinik kullanım için hasta profilleri
ve vaka yönetimi işlevlerini sağlar.
"""

import os
import json
import uuid
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

logger = logging.getLogger('eye_tracker.clinical.patient')

@dataclass
class PatientInfo:
    """Hasta bilgilerini tutan veri sınıfı"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    first_name: str = ""
    last_name: str = ""
    birth_date: str = ""  # YYYY-MM-DD formatında
    gender: str = ""
    contact_number: str = ""
    email: str = ""
    address: str = ""
    insurance_id: str = ""
    medical_history: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    notes: str = ""
    
    @property
    def full_name(self) -> str:
        """Hastanın tam adını döndürür"""
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self) -> int:
        """Hastanın yaşını hesaplar"""
        if not self.birth_date:
            return 0
        
        try:
            birth_date = datetime.datetime.strptime(self.birth_date, "%Y-%m-%d")
            today = datetime.datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except ValueError:
            logger.error(f"Geçersiz doğum tarihi formatı: {self.birth_date}")
            return 0

@dataclass
class ClinicalCase:
    """Klinik vaka bilgilerini tutan veri sınıfı"""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = ""
    case_type: str = ""  # Vaka tipi (nistagmus, strabismus, vb.)
    diagnosis: str = ""
    start_date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d"))
    end_date: Optional[str] = None
    status: str = "Açık"  # Açık, Kapalı, Takipte, vb.
    clinician_id: str = ""
    sessions: List[Dict[str, Any]] = field(default_factory=list)
    treatment_plan: str = ""
    notes: str = ""
    
    def add_session(self, session_data: Dict[str, Any]) -> None:
        """
        Vakaya yeni bir seans ekler
        
        Args:
            session_data: Seans verileri
        """
        if 'date' not in session_data:
            session_data['date'] = datetime.datetime.now().isoformat()
        
        if 'id' not in session_data:
            session_data['id'] = str(uuid.uuid4())
            
        self.sessions.append(session_data)
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        ID'ye göre seans verisini alır
        
        Args:
            session_id: Seans ID'si
            
        Returns:
            Seans verisi veya None
        """
        for session in self.sessions:
            if session.get('id') == session_id:
                return session
        return None
    
    def get_latest_session(self) -> Optional[Dict[str, Any]]:
        """
        En son seansı alır
        
        Returns:
            En son seans verisi veya None
        """
        if not self.sessions:
            return None
            
        # Tarihe göre sırala ve son seansı döndür
        return sorted(self.sessions, key=lambda x: x.get('date', ''), reverse=True)[0]
    
    def close_case(self, notes: str = "") -> None:
        """
        Vakayı kapat
        
        Args:
            notes: Kapanış notları
        """
        self.status = "Kapalı"
        self.end_date = datetime.datetime.now().strftime("%Y-%m-%d")
        if notes:
            self.notes += f"\n--- Kapanış Notları ({self.end_date}) ---\n{notes}"


class PatientManager:
    """Hasta kayıtlarını yöneten sınıf"""
    
    def __init__(self, data_dir: str = "clinical/data"):
        """
        Args:
            data_dir: Veri dizini
        """
        self.data_dir = data_dir
        self.patients_file = os.path.join(data_dir, "patients.json")
        self.cases_file = os.path.join(data_dir, "cases.json")
        
        # Veri dizinini oluştur
        os.makedirs(data_dir, exist_ok=True)
        
        # Verileri yükle
        self.patients: Dict[str, PatientInfo] = {}
        self.cases: Dict[str, ClinicalCase] = {}
        self._load_data()
    
    def _load_data(self) -> None:
        """Verileri JSON dosyalarından yükle"""
        # Hasta verilerini yükle
        if os.path.exists(self.patients_file):
            try:
                with open(self.patients_file, 'r', encoding='utf-8') as f:
                    patients_data = json.load(f)
                    
                for p_data in patients_data:
                    patient = PatientInfo(**p_data)
                    self.patients[patient.id] = patient
                    
                logger.info(f"{len(self.patients)} hasta kaydı yüklendi.")
            except Exception as e:
                logger.error(f"Hasta verilerini yükleme hatası: {e}")
        
        # Vaka verilerini yükle
        if os.path.exists(self.cases_file):
            try:
                with open(self.cases_file, 'r', encoding='utf-8') as f:
                    cases_data = json.load(f)
                    
                for c_data in cases_data:
                    case = ClinicalCase(**c_data)
                    self.cases[case.id] = case
                    
                logger.info(f"{len(self.cases)} vaka kaydı yüklendi.")
            except Exception as e:
                logger.error(f"Vaka verilerini yükleme hatası: {e}")
    
    def _save_data(self) -> None:
        """Verileri JSON dosyalarına kaydet"""
        # Hasta verilerini kaydet
        try:
            with open(self.patients_file, 'w', encoding='utf-8') as f:
                patients_data = [asdict(p) for p in self.patients.values()]
                json.dump(patients_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Hasta verilerini kaydetme hatası: {e}")
        
        # Vaka verilerini kaydet
        try:
            with open(self.cases_file, 'w', encoding='utf-8') as f:
                cases_data = [asdict(c) for c in self.cases.values()]
                json.dump(cases_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Vaka verilerini kaydetme hatası: {e}")
    
    def add_patient(self, patient_info: Dict[str, Any]) -> PatientInfo:
        """
        Yeni hasta ekle
        
        Args:
            patient_info: Hasta bilgileri
            
        Returns:
            Oluşturulan hasta nesnesi
        """
        patient = PatientInfo(**patient_info)
        self.patients[patient.id] = patient
        self._save_data()
        logger.info(f"Yeni hasta eklendi: {patient.full_name} (ID: {patient.id})")
        
        return patient
    
    def update_patient(self, patient_id: str, updated_info: Dict[str, Any]) -> Optional[PatientInfo]:
        """
        Hasta bilgilerini güncelle
        
        Args:
            patient_id: Hasta ID'si
            updated_info: Güncellenecek bilgiler
            
        Returns:
            Güncellenmiş hasta nesnesi veya None
        """
        if patient_id not in self.patients:
            logger.warning(f"Güncellenmek istenen hasta bulunamadı: {patient_id}")
            return None
        
        # ID değiştirilmesin
        if 'id' in updated_info:
            del updated_info['id']
            
        # Mevcut bilgileri güncelle
        patient = self.patients[patient_id]
        
        for key, value in updated_info.items():
            if hasattr(patient, key):
                setattr(patient, key, value)
        
        self._save_data()
        logger.info(f"Hasta bilgileri güncellendi: {patient.full_name} (ID: {patient.id})")
        
        return patient
    
    def get_patient(self, patient_id: str) -> Optional[PatientInfo]:
        """
        ID'ye göre hasta al
        
        Args:
            patient_id: Hasta ID'si
            
        Returns:
            Hasta nesnesi veya None
        """
        return self.patients.get(patient_id)
    
    def get_all_patients(self) -> List[PatientInfo]:
        """
        Tüm hastaları al
        
        Returns:
            Hasta listesi
        """
        return list(self.patients.values())
    
    def delete_patient(self, patient_id: str) -> bool:
        """
        Hasta kaydını sil
        
        Args:
            patient_id: Hasta ID'si
            
        Returns:
            Başarı durumu
        """
        if patient_id not in self.patients:
            logger.warning(f"Silinmek istenen hasta bulunamadı: {patient_id}")
            return False
        
        # Hastanın tüm vakalarını sil
        cases_to_delete = [c_id for c_id, case in self.cases.items() 
                          if case.patient_id == patient_id]
        
        for case_id in cases_to_delete:
            del self.cases[case_id]
        
        # Hastayı sil
        patient = self.patients[patient_id]
        del self.patients[patient_id]
        
        self._save_data()
        logger.info(f"Hasta silindi: {patient.full_name} (ID: {patient.id}) "
                  f"ve ilişkili {len(cases_to_delete)} vaka kaldırıldı.")
        
        return True
    
    def search_patients(self, query: str) -> List[PatientInfo]:
        """
        Hastalarda arama yap
        
        Args:
            query: Arama sorgusu
            
        Returns:
            Eşleşen hasta listesi
        """
        query = query.lower()
        results = []
        
        for patient in self.patients.values():
            if (query in patient.first_name.lower() or
                query in patient.last_name.lower() or
                query in patient.full_name.lower() or
                query in patient.id.lower()):
                results.append(patient)
                
        return results
    
    def add_case(self, case_info: Dict[str, Any]) -> Optional[ClinicalCase]:
        """
        Yeni vaka ekle
        
        Args:
            case_info: Vaka bilgileri
            
        Returns:
            Oluşturulan vaka nesnesi veya None
        """
        # Hastanın var olup olmadığını kontrol et
        patient_id = case_info.get('patient_id')
        if patient_id not in self.patients:
            logger.warning(f"Vaka eklenmek istenen hasta bulunamadı: {patient_id}")
            return None
        
        case = ClinicalCase(**case_info)
        self.cases[case.id] = case
        self._save_data()
        
        patient = self.patients[patient_id]
        logger.info(f"Yeni vaka eklendi: {case.case_type} "
                  f"(Hasta: {patient.full_name}, ID: {case.id})")
        
        return case
    
    def update_case(self, case_id: str, updated_info: Dict[str, Any]) -> Optional[ClinicalCase]:
        """
        Vaka bilgilerini güncelle
        
        Args:
            case_id: Vaka ID'si
            updated_info: Güncellenecek bilgiler
            
        Returns:
            Güncellenmiş vaka nesnesi veya None
        """
        if case_id not in self.cases:
            logger.warning(f"Güncellenmek istenen vaka bulunamadı: {case_id}")
            return None
        
        # ID değiştirilmesin
        if 'id' in updated_info:
            del updated_info['id']
            
        # Mevcut bilgileri güncelle
        case = self.cases[case_id]
        
        for key, value in updated_info.items():
            if hasattr(case, key):
                setattr(case, key, value)
        
        self._save_data()
        logger.info(f"Vaka bilgileri güncellendi: {case.case_type} (ID: {case.id})")
        
        return case
    
    def get_case(self, case_id: str) -> Optional[ClinicalCase]:
        """
        ID'ye göre vaka al
        
        Args:
            case_id: Vaka ID'si
            
        Returns:
            Vaka nesnesi veya None
        """
        return self.cases.get(case_id)
    
    def get_patient_cases(self, patient_id: str) -> List[ClinicalCase]:
        """
        Hastaya ait vakaları al
        
        Args:
            patient_id: Hasta ID'si
            
        Returns:
            Vaka listesi
        """
        return [case for case in self.cases.values() 
               if case.patient_id == patient_id]
    
    def add_session_to_case(self, case_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Vakaya yeni seans ekle
        
        Args:
            case_id: Vaka ID'si
            session_data: Seans verileri
            
        Returns:
            Başarı durumu
        """
        if case_id not in self.cases:
            logger.warning(f"Seans eklenecek vaka bulunamadı: {case_id}")
            return False
        
        case = self.cases[case_id]
        case.add_session(session_data)
        self._save_data()
        
        logger.info(f"Vakaya yeni seans eklendi: {case.case_type} (ID: {case.id})")
        return True
    
    def get_active_cases(self) -> List[ClinicalCase]:
        """
        Aktif vakaları al
        
        Returns:
            Aktif vaka listesi
        """
        return [case for case in self.cases.values() 
               if case.status == "Açık" or case.status == "Takipte"] 