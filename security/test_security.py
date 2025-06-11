"""
Veri Güvenliği ve Gizlilik Modüllerini Test Etme

Bu script, Tkinter olmadan veri güvenliği ve 
gizlilik bileşenlerini test eder.
"""

import os
import json
import logging
import sys
from pathlib import Path

# Log yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('security_test')

# Ana dizini ayarla
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

try:
    from security.data_security import VeriSifrelemeServisi, ErisimKontrolSistemi, VeriAnonimlestirme, KullaniciRolu
    from security.secure_storage import GuvenliVeriDepolama
except ImportError as e:
    logger.error(f"Modül içe aktarma hatası: {e}")
    sys.exit(1)

def test_sifreleme():
    """Şifreleme özelliklerini test et"""
    print("\n=== Şifreleme Testi ===")
    
    # Şifreleme servisi oluştur
    sifrele = VeriSifrelemeServisi()
    
    # Test verileri
    test_string = "Bu bir test metnidir!"
    test_dict = {"hasta_id": "12345", "ad": "Ahmet", "tanı": "Nistagmus"}
    test_bytes = b"Binary test verisi"
    
    # Metin şifreleme
    print("\nMetin şifreleme testi:")
    encrypted_string = sifrele.sifrele(test_string)
    if encrypted_string:
        print(f"  Orijinal: {test_string}")
        print(f"  Şifreli: {encrypted_string[:30]}...")
        
        # Şifre çözme
        decrypted_string = sifrele.sifreyi_coz(encrypted_string)
        print(f"  Çözülmüş: {decrypted_string}")
        print(f"  Başarılı mı: {'Evet' if decrypted_string == test_string else 'Hayır'}")
    
    # Sözlük şifreleme
    print("\nSözlük şifreleme testi:")
    encrypted_dict = sifrele.sifrele(test_dict)
    if encrypted_dict:
        print(f"  Orijinal: {test_dict}")
        print(f"  Şifreli: {encrypted_dict[:30]}...")
        
        # Şifre çözme
        decrypted_dict = sifrele.sifreyi_coz(encrypted_dict, "dict")
        print(f"  Çözülmüş: {decrypted_dict}")
        print(f"  Başarılı mı: {'Evet' if decrypted_dict == test_dict else 'Hayır'}")
    
    # Binary şifreleme
    print("\nBinary şifreleme testi:")
    encrypted_bytes = sifrele.sifrele(test_bytes)
    if encrypted_bytes:
        print(f"  Orijinal: {test_bytes}")
        print(f"  Şifreli: {encrypted_bytes[:30]}...")
        
        # Şifre çözme
        decrypted_bytes = sifrele.sifreyi_coz(encrypted_bytes, "bytes")
        print(f"  Çözülmüş: {decrypted_bytes}")
        print(f"  Başarılı mı: {'Evet' if decrypted_bytes == test_bytes else 'Hayır'}")
    
    # Anahtar kaydetme/yükleme
    print("\nAnahtar yönetimi testi:")
    key_saved = sifrele.save_master_key()
    print(f"  Anahtar kaydedildi mi: {'Evet' if key_saved else 'Hayır'}")
    
    # Yeni şifreleme servisi oluştur
    new_sifrele = VeriSifrelemeServisi()
    key_loaded = new_sifrele.load_master_key()
    print(f"  Anahtar yüklendi mi: {'Evet' if key_loaded else 'Hayır'}")
    
    # Önceki şifreli veriyi yeni servis ile çöz
    if key_loaded and encrypted_string:
        decrypted = new_sifrele.sifreyi_coz(encrypted_string)
        print(f"  Yeni servis ile çözüldü mü: {'Evet' if decrypted == test_string else 'Hayır'}")

def test_erisim_kontrolu():
    """Erişim kontrolü özelliklerini test et"""
    print("\n=== Erişim Kontrolü Testi ===")
    
    # Erişim kontrol sistemi oluştur
    erisim = ErisimKontrolSistemi()
    
    # Kullanıcı ve roller
    print("\nKullanıcı rolleri oluşturma:")
    erisim.rol_ata("doktor123", KullaniciRolu.DOKTOR)
    erisim.rol_ata("yonetici456", KullaniciRolu.YONETICI)
    erisim.rol_ata("hasta789", KullaniciRolu.HASTA)
    erisim.rol_ata("arastirmaci321", KullaniciRolu.ARASTIRMACI)
    
    # Rol yetkilerini yazdır
    print("\nRol yetkileri:")
    for rol, yetkiler in erisim.rol_yetkileri.items():
        print(f"  {rol}:")
        for kaynak, islemler in yetkiler.items():
            print(f"    - {kaynak}: {', '.join(islemler)}")
    
    # Yetki kontrolleri
    print("\nYetki kontrolleri:")
    test_cases = [
        ("doktor123", "hasta_verileri", "okuma", None),
        ("doktor123", "hasta_verileri", "silme", None),
        ("yonetici456", "sistem_ayarlari", "yazma", None),
        ("hasta789", "hasta_verileri", "okuma", "hasta789"),
        ("hasta789", "hasta_verileri", "okuma", "başka_hasta"),
        ("arastirmaci321", "hasta_verileri", "yazma", None)
    ]
    
    for kullanici, kaynak, islem, sahip in test_cases:
        yetki = erisim.yetki_kontrol(kullanici, kaynak, islem, sahip)
        print(f"  {kullanici} → {kaynak}/{islem} (Sahip: {sahip or 'Yok'}): {'İZİNLİ' if yetki else 'REDDEDİLDİ'}")
    
    # Log kayıtlarını kontrol et
    log_dosyasi = Path("security/logs/access_log.jsonl")
    if log_dosyasi.exists():
        print(f"\nErişim kayıtları ({log_dosyasi}):")
        with open(log_dosyasi, 'r') as f:
            kayitlar = f.readlines()
            for i, kayit in enumerate(kayitlar[-5:]):  # Son 5 kaydı göster
                kayit_verisi = json.loads(kayit)
                print(f"  {i+1}. {kayit_verisi['zaman']}: {kayit_verisi['kullanici_id']} → {kayit_verisi['kaynak_turu']}/{kayit_verisi['islem_turu']} ({'İZİNLİ' if kayit_verisi['izin_verildi'] else 'REDDEDİLDİ'})")

def test_veri_anonimize():
    """Veri anonimleştirme özelliklerini test et"""
    print("\n=== Veri Anonimleştirme Testi ===")
    
    # Anonimleştirme servisi oluştur
    anonim = VeriAnonimlestirme()
    
    # Test verileri
    hasta_verisi = {
        "hasta_id": "12345",
        "ad": "Mehmet",
        "soyad": "Yılmaz",
        "tc_kimlik": "12345678901",
        "eposta": "mehmet@gmail.com",
        "telefon": "05551234567",
        "adres": "Atatürk Cad. No: 123 İstanbul",
        "dogum_tarihi": "1980-05-15",
        "yas": 43,
        "cinsiyet": "Erkek",
        "tani": "Nistagmus",
        "olcum_sonuclari": {
            "sakkad_hizi": 450.2,
            "fiksasyon_suresi": 285.3,
            "pupil_capi": 4.2
        }
    }
    
    # Maskeleme testi
    print("\nMaskeleme testi:")
    masked_data = anonim.kimlik_alanlari_maskele(hasta_verisi)
    print("  Maskelenmiş veri:")
    for key, value in masked_data.items():
        print(f"    {key}: {value}")
    
    # Tam anonimleştirme testi
    print("\nTam anonimleştirme testi:")
    anon_data = anonim.anonimize_et(hasta_verisi)
    print("  Anonimleştirilmiş veri:")
    for key, value in anon_data.items():
        print(f"    {key}: {value}")
    
    # Araştırma veri seti oluşturma
    print("\nAraştırma veri seti testi:")
    veri_seti = [hasta_verisi.copy() for _ in range(3)]
    veri_seti[1]["ad"] = "Ayşe"
    veri_seti[1]["soyad"] = "Kaya"
    veri_seti[1]["cinsiyet"] = "Kadın"
    veri_seti[1]["yas"] = 35
    
    veri_seti[2]["ad"] = "Ali"
    veri_seti[2]["soyad"] = "Demir"
    veri_seti[2]["yas"] = 58
    
    anon_set = anonim.arastirma_verisi_olustur(veri_seti)
    print(f"  Anonimleştirilmiş veri seti ({len(anon_set)} kayıt):")
    for i, kayit in enumerate(anon_set):
        print(f"    Kayıt {i+1}:")
        for key, value in kayit.items():
            if key not in ("olcum_sonuclari"):  # Ayrıntılı olmayan alanları göster
                print(f"      {key}: {value}")

def test_guvenli_depolama():
    """Güvenli veri depolama özelliklerini test et"""
    print("\n=== Güvenli Veri Depolama Testi ===")
    
    # Güvenli depolama oluştur
    secure_storage = GuvenliVeriDepolama("security/test_output")
    
    # Kullanıcı rolü ayarla
    print("\nKullanıcı rolü ayarlama:")
    secure_storage.aktif_kullanici_ayarla("doktor123", "doktor")
    print(f"  Aktif kullanıcı: {secure_storage.aktif_kullanici_id}")
    
    # Test verisi
    test_data = {
        "hasta_id": "12345",
        "ad": "Hasan",
        "soyad": "Sevgi",
        "olcum_zamani": "2023-08-22T15:30:00",
        "olcum_sonuclari": {
            "sakkad_hizi": [450.2, 455.3, 448.1],
            "fiksasyon_suresi": [285.3, 290.1, 283.5],
            "pupil_capi": [4.2, 4.3, 4.1]
        },
        "teshis": "Sağlıklı"
    }
    
    # Veri kaydetme testi
    print("\nVeri kaydetme testi:")
    result = secure_storage.veri_kaydet(test_data, hasta_id="12345")
    print(f"  Sonuç: {'Başarılı' if result else 'Başarısız'}")
    
    # Farklı güvenlik ayarlarıyla test
    print("\nFarklı güvenlik ayarları testleri:")
    
    # Şifreleme olmadan
    secure_storage.sifrele_etkin = False
    result = secure_storage.veri_kaydet(test_data, hasta_id="12345", veri_turu="test_sifrelemesiz")
    print(f"  Şifreleme olmadan: {'Başarılı' if result else 'Başarısız'}")
    
    # Anonimleştirme ile
    secure_storage.sifrele_etkin = True
    secure_storage.anonimlestirme_etkin = True
    result = secure_storage.veri_kaydet(test_data, hasta_id="12345", veri_turu="test_anonim")
    print(f"  Anonimleştirme ile: {'Başarılı' if result else 'Başarısız'}")
    
    # Erişim kontrolü testleri
    print("\nErişim kontrolü testleri:")
    
    # Doktor rolüyle hasta verisi okuma (izinli)
    secure_storage.anonimlestirme_etkin = False
    yetki = secure_storage.erisim_kontrol.yetki_kontrol("doktor123", "hasta_verileri", "okuma")
    print(f"  Doktor → hasta verileri okuma: {'İZİNLİ' if yetki else 'REDDEDİLDİ'}")
    
    # Teknisyen rolüyle hasta verisi
    secure_storage.aktif_kullanici_ayarla("teknisyen456", "teknisyen")
    yetki = secure_storage.erisim_kontrol.yetki_kontrol("teknisyen456", "hasta_verileri", "silme")
    print(f"  Teknisyen → hasta verileri silme: {'İZİNLİ' if yetki else 'REDDEDİLDİ'}")
    
    # Hasta rolüyle kendi verisi
    secure_storage.aktif_kullanici_ayarla("hasta12345", "hasta")
    yetki = secure_storage.erisim_kontrol.yetki_kontrol("hasta12345", "hasta_verileri", "okuma", "hasta12345")
    print(f"  Hasta → kendi verileri okuma: {'İZİNLİ' if yetki else 'REDDEDİLDİ'}")
    
    # Başka hastanın verisi
    yetki = secure_storage.erisim_kontrol.yetki_kontrol("hasta12345", "hasta_verileri", "okuma", "baska_hasta")
    print(f"  Hasta → başka hasta verisi okuma: {'İZİNLİ' if yetki else 'REDDEDİLDİ'}")
    
    # Kaynakları serbest bırak
    secure_storage.kapat()

if __name__ == "__main__":
    print("Veri Güvenliği ve Gizlilik Bileşenleri Testi")
    print("=" * 60)
    
    # Ana dizin kontrolü
    print(f"Çalışma dizini: {os.getcwd()}")
    print(f"Script dizini: {script_dir}")
    
    # Dizin yapısı
    Path("security/logs").mkdir(parents=True, exist_ok=True)
    Path("security/keys").mkdir(parents=True, exist_ok=True)
    Path("security/test_output").mkdir(parents=True, exist_ok=True)
    
    # Testleri çalıştır
    test_sifreleme()
    test_erisim_kontrolu()
    test_veri_anonimize()
    test_guvenli_depolama()
    
    print("\nTüm testler tamamlandı!") 