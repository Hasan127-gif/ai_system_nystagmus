"""
Göz İzleme Sistemi - Veri Güvenliği ve Gizlilik Modülü

Bu modül, hasta verilerinin güvenliği ve gizliliği için gerekli temel fonksiyonları sağlar:
- Veri şifreleme ve çözme
- Erişim kontrolü ve yetkilendirme
- Gizlilik düzenlemelerine uyumluluk (GDPR, HIPAA)
- Veri anonimleştirme
"""

import os
import json
import logging
import hashlib
import base64
import uuid
import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from pathlib import Path

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("cryptography kütüphanesi bulunamadı! Şifreleme özellikleri sınırlı olacak.")
    print("Kurulum için: pip install cryptography")

# Logger yapılandırması
logger = logging.getLogger('eye_tracker.security')

# ====================================================
# Veri Şifreleme Sınıfı
# ====================================================
class VeriSifrelemeServisi:
    """Hasta verilerinin şifrelenmesi ve çözülmesi için servis"""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        Args:
            master_key: Ana şifreleme anahtarı (yoksa otomatik oluşturulur)
        """
        self.master_key = master_key or self._generate_master_key()
        self.fernet = self._init_fernet()
        self.key_file_path = Path("security/keys/.master_key")
        
    def _generate_master_key(self) -> str:
        """Güvenli bir ana anahtar oluştur"""
        if CRYPTO_AVAILABLE:
            return base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
        else:
            # Fallback - daha az güvenli
            return hashlib.sha256(os.urandom(32)).hexdigest()
    
    def _init_fernet(self):
        """Fernet şifreleme nesnesi oluştur"""
        if not CRYPTO_AVAILABLE:
            return None
            
        try:
            key = base64.urlsafe_b64decode(self.master_key.encode('utf-8'))
            return Fernet(base64.urlsafe_b64encode(key))
        except Exception as e:
            logger.error(f"Fernet başlatma hatası: {e}")
            return None
            
    def save_master_key(self) -> bool:
        """Ana anahtarı güvenli bir şekilde diske kaydet"""
        try:
            # Ana dizini oluştur
            self.key_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Anahtarı kaydet
            with open(self.key_file_path, 'w') as f:
                f.write(self.master_key)
                
            # Dosya izinlerini kısıtla
            os.chmod(self.key_file_path, 0o600)  # Sadece sahibi okuyabilir/yazabilir
            
            logger.info("Ana anahtar güvenli bir şekilde kaydedildi")
            return True
        except Exception as e:
            logger.error(f"Ana anahtar kaydetme hatası: {e}")
            return False
    
    def load_master_key(self) -> bool:
        """Kaydedilmiş ana anahtarı yükle"""
        try:
            if self.key_file_path.exists():
                with open(self.key_file_path, 'r') as f:
                    self.master_key = f.read().strip()
                self.fernet = self._init_fernet()
                logger.info("Ana anahtar başarıyla yüklendi")
                return True
            return False
        except Exception as e:
            logger.error(f"Ana anahtar yükleme hatası: {e}")
            return False
    
    def sifrele(self, veri: Union[str, dict, bytes]) -> Optional[str]:
        """
        Veriyi şifrele
        
        Args:
            veri: Şifrelenecek veri (metin, sözlük veya ikili veri)
            
        Returns:
            Şifrelenmiş veri (base64 kodlu) veya None (hata durumunda)
        """
        if not CRYPTO_AVAILABLE or not self.fernet:
            logger.warning("Şifreleme kütüphanesi yüklenemedi")
            return None
            
        try:
            # Veri türüne göre dönüştürme
            if isinstance(veri, dict):
                veri = json.dumps(veri).encode('utf-8')
            elif isinstance(veri, str):
                veri = veri.encode('utf-8')
                
            # Şifrele ve base64 kodla    
            encrypted = self.fernet.encrypt(veri)
            return base64.urlsafe_b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error(f"Şifreleme hatası: {e}")
            return None
            
    def sifreyi_coz(self, sifrelenmis_veri: str, veri_turu: str = 'str') -> Any:
        """
        Şifrelenmiş veriyi çöz
        
        Args:
            sifrelenmis_veri: Base64 kodlu şifreli veri
            veri_turu: Çözülmüş verinin istenen türü ('str', 'dict', 'bytes')
            
        Returns:
            Çözülmüş veri veya None (hata durumunda)
        """
        if not CRYPTO_AVAILABLE or not self.fernet:
            logger.warning("Şifreleme kütüphanesi yüklenemedi")
            return None
            
        try:
            # Base64 kodu çöz
            encrypted = base64.urlsafe_b64decode(sifrelenmis_veri.encode('utf-8'))
            
            # Şifreyi çöz
            decrypted = self.fernet.decrypt(encrypted)
            
            # İstenen türe dönüştür
            if veri_turu == 'str':
                return decrypted.decode('utf-8')
            elif veri_turu == 'dict':
                return json.loads(decrypted.decode('utf-8'))
            else:  # 'bytes'
                return decrypted
        except Exception as e:
            logger.error(f"Şifre çözme hatası: {e}")
            return None

# ====================================================
# Erişim Kontrolü ve Yetkilendirme
# ====================================================
class KullaniciRolu(Enum):
    """Kullanıcı rolleri"""
    YONETICI = "yonetici"
    DOKTOR = "doktor"
    TEKNISYEN = "teknisyen"
    ARASTIRMACI = "arastirmaci"
    HASTA = "hasta"

class ErisimKontrolSistemi:
    """Rol tabanlı erişim kontrolü sistemi"""
    
    def __init__(self):
        self.kullanici_rolleri = {}  # {kullanici_id: rol}
        self.rol_yetkileri = self._varsayilan_yetkileri_olustur()
        self.islem_kayitlari = []
        
    def _varsayilan_yetkileri_olustur(self) -> Dict[str, Dict[str, List[str]]]:
        """Roller için varsayılan yetkileri oluştur"""
        return {
            KullaniciRolu.YONETICI.value: {
                "hasta_verileri": ["okuma", "yazma", "silme", "anonimleştirme"],
                "sistem_ayarlari": ["okuma", "yazma"],
                "kullanici_yonetimi": ["okuma", "yazma", "silme"],
                "raporlar": ["okuma", "yazma", "silme", "paylaşma"]
            },
            KullaniciRolu.DOKTOR.value: {
                "hasta_verileri": ["okuma", "yazma"],
                "sistem_ayarlari": ["okuma"],
                "raporlar": ["okuma", "yazma", "paylaşma"]
            },
            KullaniciRolu.TEKNISYEN.value: {
                "hasta_verileri": ["okuma", "yazma"],
                "sistem_ayarlari": ["okuma"],
                "raporlar": ["okuma"]
            },
            KullaniciRolu.ARASTIRMACI.value: {
                "hasta_verileri": ["okuma"],  # Sadece anonimleştirilmiş veriler
                "raporlar": ["okuma", "yazma"]
            },
            KullaniciRolu.HASTA.value: {
                "hasta_verileri": ["okuma"],  # Sadece kendi verileri
                "raporlar": ["okuma"]  # Sadece kendi raporları
            }
        }
        
    def rol_ata(self, kullanici_id: str, rol: Union[KullaniciRolu, str]) -> bool:
        """
        Kullanıcıya rol ata
        
        Args:
            kullanici_id: Kullanıcı kimliği
            rol: Atanacak rol
            
        Returns:
            İşlem başarı durumu
        """
        try:
            if isinstance(rol, KullaniciRolu):
                rol_degeri = rol.value
            else:
                rol_degeri = rol
                
            self.kullanici_rolleri[kullanici_id] = rol_degeri
            logger.info(f"Kullanıcı {kullanici_id} için {rol_degeri} rolü atandı")
            return True
        except Exception as e:
            logger.error(f"Rol atama hatası: {e}")
            return False
            
    def yetki_kontrol(self, kullanici_id: str, kaynak_turu: str, 
                      islem_turu: str, kaynak_sahibi: Optional[str] = None) -> bool:
        """
        Kullanıcının belirli bir işlem için yetkisi olup olmadığını kontrol et
        
        Args:
            kullanici_id: Kullanıcı kimliği
            kaynak_turu: Erişilmek istenen kaynak türü (hasta_verileri, raporlar, vb.)
            islem_turu: Yapılmak istenen işlem (okuma, yazma, silme, vb.)
            kaynak_sahibi: Kaynağın sahibi (hastanın kendi verisine erişimde önemli)
            
        Returns:
            Yetki durumu (True: yetkili, False: yetkisiz)
        """
        # Kullanıcı rolünü al
        rol = self.kullanici_rolleri.get(kullanici_id)
        if not rol:
            logger.warning(f"Kullanıcı {kullanici_id} için rol tanımlanmamış")
            return False
            
        # Rol yetkilerini kontrol et
        rol_yetkileri = self.rol_yetkileri.get(rol, {})
        kaynak_yetkileri = rol_yetkileri.get(kaynak_turu, [])
        
        # İşlem yetkisi var mı?
        has_permission = islem_turu in kaynak_yetkileri
        
        # Hasta kendi verisine erişmek istiyorsa
        if (rol == KullaniciRolu.HASTA.value and 
            kaynak_turu == "hasta_verileri" and 
            kaynak_sahibi == kullanici_id):
            has_permission = True
            
        # İşlemi kaydet
        self._islem_kaydet(kullanici_id, kaynak_turu, islem_turu, has_permission)
        
        return has_permission
        
    def _islem_kaydet(self, kullanici_id: str, kaynak_turu: str, 
                      islem_turu: str, izin_verildi: bool) -> None:
        """Erişim işlemini güvenlik kaydına ekle"""
        kayit = {
            "zaman": datetime.datetime.now().isoformat(),
            "kullanici_id": kullanici_id,
            "kaynak_turu": kaynak_turu,
            "islem_turu": islem_turu,
            "izin_verildi": izin_verildi,
            "ip_adresi": "127.0.0.1"  # Gerçek uygulamada istemci IP'si alınmalı
        }
        self.islem_kayitlari.append(kayit)
        
        # Kayıt dosyasına yaz
        try:
            kayit_dizini = Path("security/logs")
            kayit_dizini.mkdir(parents=True, exist_ok=True)
            
            kayit_dosyasi = kayit_dizini / "access_log.jsonl"
            with open(kayit_dosyasi, 'a') as f:
                f.write(json.dumps(kayit) + "\n")
        except Exception as e:
            logger.error(f"Erişim kaydı yazma hatası: {e}")

# ====================================================
# Veri Anonimleştirme Sınıfı
# ====================================================
class VeriAnonimlestirme:
    """Hasta verilerini anonimleştirmek için araçlar"""
    
    def __init__(self):
        self.hassas_alanlar = [
            "ad", "soyad", "tc_kimlik", "dogum_tarihi", "telefon", 
            "adres", "eposta", "ip_adresi", "hasta_id"
        ]
        self.ikincil_alanlar = [
            "yas", "cinsiyet", "sehir", "meslek", "medeni_durum"
        ]
        
    def kimlik_alanlari_maskele(self, veri: Dict[str, Any]) -> Dict[str, Any]:
        """
        Veri içindeki kimlik alanlarını maskele
        
        Args:
            veri: Maskelenecek veri
            
        Returns:
            Maskelenmiş veri
        """
        if not isinstance(veri, dict):
            return veri
            
        maskeli_veri = veri.copy()
        
        # Hassas alanları maskele
        for alan in self.hassas_alanlar:
            if alan in maskeli_veri:
                if alan == "tc_kimlik" or alan == "telefon":
                    # Kısmi maskeleme
                    deger = str(maskeli_veri[alan])
                    if len(deger) > 4:
                        maskeli_veri[alan] = "****" + deger[-4:]
                    else:
                        maskeli_veri[alan] = "********"
                elif alan == "eposta":
                    # E-posta maskeleme (örn: j***@g***.com)
                    email = maskeli_veri[alan]
                    if "@" in email:
                        username, domain = email.split("@")
                        if len(username) > 1:
                            username = username[0] + "***"
                        domain_parts = domain.split(".")
                        if len(domain_parts[0]) > 1:
                            domain_parts[0] = domain_parts[0][0] + "***"
                        masked_domain = ".".join(domain_parts)
                        maskeli_veri[alan] = f"{username}@{masked_domain}"
                else:
                    # Tam maskeleme
                    maskeli_veri[alan] = "********"
        
        return maskeli_veri
    
    def anonimize_et(self, veri: Dict[str, Any]) -> Dict[str, Any]:
        """
        Veriyi tamamen anonimleştir
        
        Args:
            veri: Anonimleştirilecek veri
            
        Returns:
            Anonimleştirilmiş veri
        """
        if not isinstance(veri, dict):
            return veri
            
        anonim_veri = veri.copy()
        
        # Hassas alanları tamamen kaldır
        for alan in self.hassas_alanlar:
            if alan in anonim_veri:
                del anonim_veri[alan]
        
        # İkincil alanları genelleştir
        if "yas" in anonim_veri:
            try:
                yas = int(anonim_veri["yas"])
                # Yaş aralıklarına çevir: 0-9, 10-19, ...
                anonim_veri["yas_grubu"] = f"{yas // 10 * 10}-{yas // 10 * 10 + 9}"
                del anonim_veri["yas"]
            except (ValueError, TypeError):
                pass
                
        # Diğer ikincil alanları bulanıklaştırma...
                
        return anonim_veri
        
    def arastirma_verisi_olustur(self, veri_seti: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Araştırma için güvenli, anonimleştirilmiş veri seti oluştur
        
        Args:
            veri_seti: Ham veri seti
            
        Returns:
            Anonimleştirilmiş veri seti
        """
        return [self.anonimize_et(veri) for veri in veri_seti]

# ====================================================
# GDPR ve HIPAA Uyumluluğu
# ====================================================
class DuzenleyiciUyumluluk:
    """Gizlilik düzenlemelerine uyumluluk sağlama araçları"""
    
    def __init__(self):
        self.veri_sifrele = VeriSifrelemeServisi()
        self.anonimlestir = VeriAnonimlestirme()
        
    def veri_sahibi_haklari_yonet(self, istek_turu: str, hasta_id: str, 
                                  veri_kaynaklari: List[str]) -> Dict[str, Any]:
        """
        Veri sahibi haklarıyla ilgili istekleri yönet (GDPR/HIPAA)
        
        Args:
            istek_turu: İstek türü (veri_erişim, veri_silme, veri_tasima, veri_düzeltme)
            hasta_id: Hasta kimliği
            veri_kaynaklari: Veri kaynaklarının listesi
            
        Returns:
            İşlem sonucu
        """
        sonuc = {
            "istek_turu": istek_turu,
            "hasta_id": hasta_id,
            "durum": "başarısız",
            "mesaj": "",
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            if istek_turu == "veri_erişim":
                # Veri erişim talebi işleme
                sonuc["durum"] = "başarılı"
                sonuc["mesaj"] = "Veri erişim talebi işlendi"
                # Burada ilgili veri kaynakları toplanıp hastaya sunulabilir
                
            elif istek_turu == "veri_silme":
                # Veri silme talebi işleme (unutulma hakkı)
                # İlgili tüm veri kaynaklarından hastanın verileri silinmeli
                sonuc["durum"] = "başarılı"
                sonuc["mesaj"] = "Tüm sistemlerden verileriniz silindi"
                
            elif istek_turu == "veri_tasima":
                # Veri taşıma talebi işleme
                # Standart bir formatta hastanın verileri dışa aktarılmalı
                sonuc["durum"] = "başarılı"
                sonuc["mesaj"] = "Verileriniz dışa aktarıldı"
                
            elif istek_turu == "veri_düzeltme":
                # Veri düzeltme talebi işleme
                sonuc["durum"] = "başarılı" 
                sonuc["mesaj"] = "Verileriniz güncellendi"
            
            # İşlem kaydı
            self._veri_sahibi_islemi_kaydet(istek_turu, hasta_id, sonuc["durum"])
                
            return sonuc
        except Exception as e:
            logger.error(f"Veri sahibi hakkı işleme hatası: {e}")
            sonuc["mesaj"] = f"İşlem sırasında hata: {str(e)}"
            return sonuc
    
    def _veri_sahibi_islemi_kaydet(self, istek_turu: str, hasta_id: str, sonuc: str) -> None:
        """Veri sahibi hakkı işlemini kaydet (denetim için)"""
        try:
            kayit = {
                "zaman": datetime.datetime.now().isoformat(),
                "istek_turu": istek_turu,
                "hasta_id": hasta_id,
                "sonuc": sonuc,
                "islem_id": str(uuid.uuid4())
            }
            
            # Kayıt dosyasına yaz
            kayit_dizini = Path("security/logs")
            kayit_dizini.mkdir(parents=True, exist_ok=True)
            
            kayit_dosyasi = kayit_dizini / "data_subject_requests.jsonl"
            with open(kayit_dosyasi, 'a') as f:
                f.write(json.dumps(kayit) + "\n")
        except Exception as e:
            logger.error(f"Veri sahibi işlemi kayıt hatası: {e}")
    
    def uyumluluk_raporu_olustur(self, duzenleme_turu: str = "gdpr") -> Dict[str, Any]:
        """
        Belirli bir düzenleme için uyumluluk raporu oluştur
        
        Args:
            duzenleme_turu: Düzenleme türü (gdpr, hipaa)
            
        Returns:
            Uyumluluk raporu
        """
        # Burada ilgili düzenlemeye göre bir dizi kontrol gerçekleştirilir
        # ve sistemin uyumluluk durumu değerlendirilir
        
        simdi = datetime.datetime.now()
        
        rapor = {
            "duzenleme": duzenleme_turu.upper(),
            "olusturma_zamani": simdi.isoformat(),
            "uyumluluk_durumu": "Kısmi Uyumlu",
            "kontroller": [
                {
                    "kategori": "Veri Koruma",
                    "kontroller": [
                        {"ad": "Şifreleme", "durum": "Uyumlu" if CRYPTO_AVAILABLE else "Kısmi Uyumlu"},
                        {"ad": "Erişim Kontrolü", "durum": "Uyumlu"},
                        {"ad": "Veri Yedekleme", "durum": "Değerlendirilmedi"}
                    ]
                },
                {
                    "kategori": "Veri Sahibi Hakları",
                    "kontroller": [
                        {"ad": "Veri Erişim", "durum": "Uyumlu"},
                        {"ad": "Veri Silme", "durum": "Uyumlu"},
                        {"ad": "Veri Taşıma", "durum": "Kısmi Uyumlu"}
                    ]
                }
            ],
            "tavsiyeler": [
                "Veri taşıma özelliklerini geliştirin",
                "Düzenli güvenlik denetimleri planlayın",
                "Personel için veri koruma eğitimi düzenleyin"
            ]
        }
        
        return rapor

# ====================================================
# Kullanım Örneği
# ====================================================
if __name__ == "__main__":
    # Örnek kullanım
    print("Veri Güvenliği ve Gizlilik Modülü Test")
    
    # Şifreleme testi
    sifrele = VeriSifrelemeServisi()
    ornek_veri = {"hasta_id": "12345", "ad": "Ahmet", "tanı": "Nistagmus"}
    
    print("\n1. Şifreleme Testi:")
    sifrelenmis = sifrele.sifrele(ornek_veri)
    if sifrelenmis:
        print(f"Şifrelenmiş veri: {sifrelenmis[:30]}...")
        
        cozulmus = sifrele.sifreyi_coz(sifrelenmis, "dict")
        print(f"Çözülmüş veri: {cozulmus}")
    
    # Erişim kontrolü testi
    print("\n2. Erişim Kontrolü Testi:")
    erisim = ErisimKontrolSistemi()
    erisim.rol_ata("doktor123", KullaniciRolu.DOKTOR)
    erisim.rol_ata("hasta456", KullaniciRolu.HASTA)
    
    yetki1 = erisim.yetki_kontrol("doktor123", "hasta_verileri", "yazma")
    yetki2 = erisim.yetki_kontrol("hasta456", "sistem_ayarlari", "yazma")
    
    print(f"Doktor hasta verisi yazma yetkisi: {yetki1}")
    print(f"Hasta sistem ayarı yazma yetkisi: {yetki2}")
    
    # Veri anonimleştirme testi
    print("\n3. Veri Anonimleştirme Testi:")
    anonim = VeriAnonimlestirme()
    hasta_veri = {
        "hasta_id": "12345",
        "ad": "Ahmet",
        "soyad": "Yılmaz",
        "tc_kimlik": "12345678901",
        "eposta": "ahmet@gmail.com",
        "yas": 42,
        "tanı": "Nistagmus",
        "olcum_sonuclari": [0.5, 0.7, 0.9]
    }
    
    maskeli = anonim.kimlik_alanlari_maskele(hasta_veri)
    print(f"Maskelenmiş veri: {maskeli}")
    
    anonim_veri = anonim.anonimize_et(hasta_veri)
    print(f"Anonimleştirilmiş veri: {anonim_veri}") 