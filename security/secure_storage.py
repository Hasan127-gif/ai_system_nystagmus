"""
Göz İzleme Sistemi - Güvenli Veri Depolama Modülü

Bu modül, güvenli veri depolama işlevlerini sağlar ve mevcut DataStorage sınıfını
güvenlik özellikleriyle genişletir.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Mevcut veri depolama sınıfını içe aktar (yol projeye göre ayarlanmalı)
try:
    from ..model_and_storage import DataStorage
except ImportError:
    # Alternatif içe aktarma yolu
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from model_and_storage import DataStorage
    except ImportError:
        # Sınıfın gerçeklenemediği durumda basit bir taklit sınıf kullan
        class DataStorage:
            def __init__(self, output_directory: str = "output"):
                self.output_directory = output_directory
            
            def save_data(self, data: Dict[str, Any]) -> None:
                pass
                
            def release(self) -> None:
                pass

# Veri güvenliği modülünü içe aktar
from security.data_security import VeriSifrelemeServisi, ErisimKontrolSistemi, VeriAnonimlestirme

# Logger yapılandırması
logger = logging.getLogger('eye_tracker.secure_storage')

class GuvenliVeriDepolama:
    """Veri depolama sınıfı için güvenlik katmanı sağlar"""
    
    def __init__(self, output_directory: str = "secure_output"):
        """
        Args:
            output_directory: Çıktı dosyaları dizini
        """
        # Temel veri depolama sınıfı
        self.data_storage = DataStorage(output_directory)
        
        # Güvenlik bileşenleri
        self.sifrele = VeriSifrelemeServisi()
        self.erisim_kontrol = ErisimKontrolSistemi()
        self.anonimlestir = VeriAnonimlestirme()
        
        # Güvenlik yapılandırması
        self.sifrele_etkin = True
        self.erisim_kontrolu_etkin = True
        self.anonimlestirme_etkin = False
        self.aktif_kullanici_id = None
        
        # Şifreleme anahtarını yükle veya oluştur
        if not self.sifrele.load_master_key():
            self.sifrele.save_master_key()
    
    def aktif_kullanici_ayarla(self, kullanici_id: str, rol: str) -> bool:
        """
        Aktif kullanıcıyı ve rolünü ayarla
        
        Args:
            kullanici_id: Kullanıcı kimliği
            rol: Kullanıcı rolü
            
        Returns:
            İşlem başarı durumu
        """
        self.aktif_kullanici_id = kullanici_id
        return self.erisim_kontrol.rol_ata(kullanici_id, rol)
    
    def veri_kaydet(self, veri: Dict[str, Any], hasta_id: Optional[str] = None, 
                    veri_turu: str = "hasta_verileri") -> bool:
        """
        Veriyi güvenli bir şekilde kaydet
        
        Args:
            veri: Kaydedilecek veri
            hasta_id: Hasta kimliği (varsa)
            veri_turu: Veri türü
            
        Returns:
            İşlem başarı durumu
        """
        try:
            # Erişim kontrolü
            if self.erisim_kontrolu_etkin and self.aktif_kullanici_id:
                if not self.erisim_kontrol.yetki_kontrol(
                    self.aktif_kullanici_id, veri_turu, "yazma", hasta_id):
                    logger.warning(f"Erişim reddedildi: {self.aktif_kullanici_id} kullanıcısı {veri_turu} yazma yetkisine sahip değil")
                    return False
            
            islenecek_veri = veri.copy()
            
            # Anonimleştirme gerekli mi?
            if self.anonimlestirme_etkin:
                islenecek_veri = self.anonimlestir.kimlik_alanlari_maskele(islenecek_veri)
            
            # Hasta ID ve veri türünü kaydet
            islenecek_veri["_meta"] = {
                "veri_turu": veri_turu,
                "hasta_id": hasta_id,
                "sifrelenmis": self.sifrele_etkin
            }
            
            # Şifreleme gerekli mi?
            if self.sifrele_etkin:
                # Meta verileri koru
                meta = islenecek_veri.pop("_meta", {})
                
                # Verileri şifrele
                sifrelenmis_veri = self.sifrele.sifrele(islenecek_veri)
                
                if sifrelenmis_veri:
                    # Şifreli veriyi ve meta verileri birleştir
                    kayit = {
                        "_meta": meta,
                        "_encrypted_data": sifrelenmis_veri
                    }
                    # Veriyi kaydet
                    self.data_storage.save_data(kayit)
                    return True
                else:
                    logger.error("Veri şifreleme başarısız")
                    return False
            else:
                # Veriyi şifrelemeden kaydet
                self.data_storage.save_data(islenecek_veri)
                return True
                
        except Exception as e:
            logger.error(f"Güvenli veri kaydetme hatası: {e}")
            return False
    
    def veri_oku(self, veri_id: str, veri_turu: str = "hasta_verileri", 
                hasta_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Veriyi güvenli bir şekilde oku
        
        Args:
            veri_id: Veri kimliği
            veri_turu: Veri türü
            hasta_id: Hasta kimliği (varsa)
            
        Returns:
            Okunan veri veya None (hata durumunda)
        """
        try:
            # Erişim kontrolü
            if self.erisim_kontrolu_etkin and self.aktif_kullanici_id:
                if not self.erisim_kontrol.yetki_kontrol(
                    self.aktif_kullanici_id, veri_turu, "okuma", hasta_id):
                    logger.warning(f"Erişim reddedildi: {self.aktif_kullanici_id} kullanıcısı {veri_turu} okuma yetkisine sahip değil")
                    return None
            
            # Burada veri deposundan veri okuma işlemi yapılmalı
            # DataStorage sınıfının bu işlevi yoksa eklenmeli
            
            # Örnek veri okuma (gerçek uygulamada değiştirilmeli)
            veri_dosyasi = Path(self.data_storage.output_directory) / f"{veri_id}.json"
            
            if not veri_dosyasi.exists():
                logger.warning(f"Veri bulunamadı: {veri_id}")
                return None
            
            with open(veri_dosyasi, 'r') as f:
                veri = json.load(f)
            
            # Şifreli mi kontrol et
            if "_encrypted_data" in veri:
                sifrelenmis_veri = veri["_encrypted_data"]
                meta = veri.get("_meta", {})
                
                # Şifreyi çöz
                cozulmus_veri = self.sifrele.sifreyi_coz(sifrelenmis_veri, "dict")
                
                if cozulmus_veri:
                    # Meta verileri ekle
                    cozulmus_veri["_meta"] = meta
                    return cozulmus_veri
                else:
                    logger.error(f"Veri şifre çözme başarısız: {veri_id}")
                    return None
            else:
                # Şifresiz veri
                return veri
                
        except Exception as e:
            logger.error(f"Güvenli veri okuma hatası: {e}")
            return None
    
    def arastirma_verisi_olustur(self, hasta_verileri: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Araştırma amaçlı anonimleştirilmiş veri seti oluştur
        
        Args:
            hasta_verileri: Ham hasta verileri listesi
            
        Returns:
            Anonimleştirilmiş veri seti
        """
        if not self.aktif_kullanici_id:
            logger.warning("Aktif kullanıcı kimliği belirtilmemiş")
            return []
            
        # Erişim kontrolü
        if self.erisim_kontrolu_etkin:
            if not self.erisim_kontrol.yetki_kontrol(
                self.aktif_kullanici_id, "hasta_verileri", "anonimleştirme"):
                logger.warning(f"Erişim reddedildi: {self.aktif_kullanici_id} kullanıcısı anonimleştirme yetkisine sahip değil")
                return []
        
        # Verileri anonimleştir
        return self.anonimlestir.arastirma_verisi_olustur(hasta_verileri)
    
    def kapat(self) -> None:
        """Kaynakları serbest bırak"""
        self.data_storage.release()

# Kullanım örneği
if __name__ == "__main__":
    # Logger yapılandırması
    logging.basicConfig(level=logging.INFO)
    
    # Güvenli veri depolama örneği
    secure_storage = GuvenliVeriDepolama()
    
    # Aktif kullanıcı ayarla
    secure_storage.aktif_kullanici_ayarla("doktor123", "doktor")
    
    # Örnek hasta verisi
    hasta_verisi = {
        "hasta_id": "12345",
        "ad": "Mehmet",
        "soyad": "Yılmaz",
        "tc_kimlik": "12345678901",
        "dogum_tarihi": "1980-05-15",
        "olcum_sonuclari": {
            "sakkad_hizi": 450.2,
            "fiksasyon_suresi": 285.3,
            "pupil_capi": 4.2
        },
        "teshis": "Nistagmus şüphesi",
        "olcum_zamani": "2023-07-22T14:35:22"
    }
    
    # Veriyi güvenli şekilde kaydet
    sonuc = secure_storage.veri_kaydet(hasta_verisi, hasta_id="12345")
    print(f"Veri kaydetme sonucu: {'Başarılı' if sonuc else 'Başarısız'}")
    
    # Kaynakları serbest bırak
    secure_storage.kapat() 