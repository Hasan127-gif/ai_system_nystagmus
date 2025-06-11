"""
Göz İzleme Sistemi - Güvenlik Ayarları Kullanıcı Arayüzü

Bu modül, veri güvenliği ve gizlilik ayarlarını yönetmek için
kullanıcı arayüzü bileşenlerini sağlar.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import logging
from typing import Callable, Dict, List, Any, Optional
import os
from pathlib import Path
import json

# Güvenlik modüllerini içe aktar
try:
    from security.data_security import KullaniciRolu, VeriSifrelemeServisi, ErisimKontrolSistemi
    from security.secure_storage import GuvenliVeriDepolama
except ImportError:
    # Alternatif içe aktarma yolu (göreceli import durumlarında)
    try:
        from data_security import KullaniciRolu, VeriSifrelemeServisi, ErisimKontrolSistemi
        from secure_storage import GuvenliVeriDepolama
    except ImportError:
        messagebox.showerror("Hata", "Güvenlik modülleri yüklenemedi!")

# Logger yapılandırması
logger = logging.getLogger('eye_tracker.security_ui')

class GuvenlikAyarlariUI(ttk.Frame):
    """Güvenlik ayarları için kullanıcı arayüzü"""
    
    def __init__(self, parent, secure_storage: Optional[GuvenliVeriDepolama] = None):
        """
        Args:
            parent: Üst pencere/çerçeve
            secure_storage: Güvenli veri depolama nesnesi
        """
        super().__init__(parent)
        self.parent = parent
        self.secure_storage = secure_storage or GuvenliVeriDepolama()
        
        # Değişkenler
        self.kullanici_id_var = tk.StringVar(value="")
        self.sifreleme_var = tk.BooleanVar(value=True)
        self.erisim_kontrol_var = tk.BooleanVar(value=True)
        self.anonimlestirme_var = tk.BooleanVar(value=False)
        self.secili_rol = tk.StringVar(value=KullaniciRolu.DOKTOR.value)
        
        # Arayüz oluştur
        self._arayuz_olustur()
        
        # Başlangıç durumunu ayarla
        self._durum_guncelle()
    
    def _arayuz_olustur(self):
        """Kullanıcı arayüzünü oluştur"""
        # Ana çerçeve
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        
        # Sekme kontrolü
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        # ===== Genel Güvenlik Sekmesi =====
        self.genel_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.genel_frame, text="Genel Güvenlik")
        
        # Kullanıcı Kimliği
        ttk.Label(self.genel_frame, text="Kullanıcı Kimliği:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(self.genel_frame, textvariable=self.kullanici_id_var).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        
        # Kullanıcı Rolü
        ttk.Label(self.genel_frame, text="Kullanıcı Rolü:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        roller = [rol.value for rol in KullaniciRolu]
        ttk.Combobox(self.genel_frame, textvariable=self.secili_rol, values=roller, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Giriş butonu
        ttk.Button(self.genel_frame, text="Giriş Yap", command=self._giris_yap).grid(
            row=2, column=0, columnspan=2, pady=10)
        
        # Güvenlik seçenekleri
        ttk.Separator(self.genel_frame, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Checkbutton(self.genel_frame, text="Veri Şifreleme", variable=self.sifreleme_var, 
                        command=self._guvenligi_guncelle).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(self.genel_frame, text="Erişim Kontrolü", variable=self.erisim_kontrol_var, 
                        command=self._guvenligi_guncelle).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        ttk.Checkbutton(self.genel_frame, text="Veri Anonimleştirme", variable=self.anonimlestirme_var, 
                        command=self._guvenligi_guncelle).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        
        # Güvenlik durumu
        ttk.Separator(self.genel_frame, orient="horizontal").grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=10)
        
        self.durum_label = ttk.Label(self.genel_frame, text="", foreground="blue")
        self.durum_label.grid(row=8, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # ===== Erişim Kontrolü Sekmesi =====
        self.erisim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.erisim_frame, text="Erişim Kontrolü")
        
        # Kullanıcı rolleri tablosu
        ttk.Label(self.erisim_frame, text="Kullanıcı Rolleri ve Yetkileri:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Treeview
        self.rol_tree = ttk.Treeview(self.erisim_frame, columns=("rol", "hasta_verileri", "sistem_ayarlari", "raporlar"),
                                    show="headings", height=5)
        self.rol_tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Başlıklar
        self.rol_tree.heading("rol", text="Rol")
        self.rol_tree.heading("hasta_verileri", text="Hasta Verileri")
        self.rol_tree.heading("sistem_ayarlari", text="Sistem Ayarları")
        self.rol_tree.heading("raporlar", text="Raporlar")
        
        # Genişlikler
        self.rol_tree.column("rol", width=100)
        self.rol_tree.column("hasta_verileri", width=150)
        self.rol_tree.column("sistem_ayarlari", width=150)
        self.rol_tree.column("raporlar", width=150)
        
        # Ağaç doldur
        self._rol_agacini_doldur()
        
        # Kaydırma çubuğu
        scrollbar = ttk.Scrollbar(self.erisim_frame, orient="vertical", command=self.rol_tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.rol_tree.configure(yscrollcommand=scrollbar.set)
        
        # Erişim kayıtları
        ttk.Label(self.erisim_frame, text="Erişim Kayıtları:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5)
        
        self.log_text = tk.Text(self.erisim_frame, height=8, width=50, wrap="word")
        self.log_text.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.log_text.config(state="disabled")
        
        # Kaydırma çubuğu
        log_scrollbar = ttk.Scrollbar(self.erisim_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.grid(row=3, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # ===== Şifreleme Sekmesi =====
        self.sifreleme_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sifreleme_frame, text="Şifreleme")
        
        # Anahtar bilgileri
        ttk.Label(self.sifreleme_frame, text="Şifreleme Anahtarı Yönetimi").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        ttk.Button(self.sifreleme_frame, text="Yeni Anahtar Oluştur", 
                  command=self._yeni_anahtar_olustur).grid(
            row=1, column=0, padx=5, pady=5)
        
        ttk.Button(self.sifreleme_frame, text="Anahtarı Yedekle", 
                  command=self._anahtari_yedekle).grid(
            row=1, column=1, padx=5, pady=5)
        
        # Anahtar durumu
        self.anahtar_durum_label = ttk.Label(self.sifreleme_frame, text="")
        self.anahtar_durum_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Test alanı
        ttk.Separator(self.sifreleme_frame, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=10)
        
        ttk.Label(self.sifreleme_frame, text="Şifreleme Testi:").grid(
            row=4, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        self.test_text = tk.Text(self.sifreleme_frame, height=3, width=40)
        self.test_text.grid(row=5, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.test_text.insert("1.0", "Test metni buraya yazın...")
        
        test_button_frame = ttk.Frame(self.sifreleme_frame)
        test_button_frame.grid(row=6, column=0, columnspan=2, sticky="ew")
        
        ttk.Button(test_button_frame, text="Şifrele", command=self._test_sifrele).grid(
            row=0, column=0, padx=5, pady=5)
        
        ttk.Button(test_button_frame, text="Şifreyi Çöz", command=self._test_sifreyi_coz).grid(
            row=0, column=1, padx=5, pady=5)
        
        self.test_sonuc = tk.Text(self.sifreleme_frame, height=5, width=40)
        self.test_sonuc.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.test_sonuc.config(state="disabled")
        
        # ===== Anonimleştirme Sekmesi =====
        self.anonim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.anonim_frame, text="Anonimleştirme")
        
        ttk.Label(self.anonim_frame, text="Anonimleştirme Ayarları").grid(
            row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        # Hassas alanlar
        ttk.Label(self.anonim_frame, text="Hassas Alanlar:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5)
        
        self.hassas_listbox = tk.Listbox(self.anonim_frame, height=5, selectmode=tk.MULTIPLE)
        self.hassas_listbox.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        
        # Listbox doldur
        hassas_alanlar = [
            "ad", "soyad", "tc_kimlik", "dogum_tarihi", "telefon", 
            "adres", "eposta", "ip_adresi", "hasta_id"
        ]
        for alan in hassas_alanlar:
            self.hassas_listbox.insert(tk.END, alan)
            
        # Örnek veri anonimleştirme
        ttk.Separator(self.anonim_frame, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=10)
            
        ttk.Label(self.anonim_frame, text="Örnek Veri Anonimleştirme:").grid(
            row=3, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        örnek_veri = """
        {
            "hasta_id": "12345", 
            "ad": "Ahmet", 
            "soyad": "Yılmaz",
            "tc_kimlik": "12345678901",
            "eposta": "ahmet@gmail.com",
            "yas": 42,
            "tani": "Nistagmus",
            "olcum_sonuclari": [0.5, 0.7, 0.9]
        }
        """
        
        self.ornek_veri_text = tk.Text(self.anonim_frame, height=8, width=40)
        self.ornek_veri_text.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.ornek_veri_text.insert("1.0", örnek_veri)
        
        ttk.Button(self.anonim_frame, text="Maskeleme Uygula", 
                  command=self._ornek_maskeleme).grid(
            row=5, column=0, padx=5, pady=5)
        
        ttk.Button(self.anonim_frame, text="Tam Anonimleştir", 
                  command=self._ornek_anonimize).grid(
            row=5, column=1, padx=5, pady=5)
        
        self.anonim_sonuc = tk.Text(self.anonim_frame, height=8, width=40)
        self.anonim_sonuc.grid(row=6, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        self.anonim_sonuc.config(state="disabled")
        
        # Sekmelerin boyutlarını ayarla
        for frame in [self.genel_frame, self.erisim_frame, self.sifreleme_frame, self.anonim_frame]:
            for i in range(10):
                frame.rowconfigure(i, weight=1)
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
        
        # Erişim kaydı dosyasını kontrol et
        self._erisim_kayitlarini_goster()
        
        # Şifreleme anahtarı durumunu kontrol et
        self._anahtar_durumunu_guncelle()
    
    def _durum_guncelle(self):
        """Güvenlik durumunu güncelle"""
        # Kullanıcı girişi yapıldı mı?
        if self.secure_storage.aktif_kullanici_id:
            rol = self.secure_storage.erisim_kontrol.kullanici_rolleri.get(
                self.secure_storage.aktif_kullanici_id, "bilinmiyor")
            self.durum_label.config(
                text=f"Aktif Kullanıcı: {self.secure_storage.aktif_kullanici_id} ({rol})",
                foreground="green")
        else:
            self.durum_label.config(
                text="Kullanıcı girişi yapılmadı",
                foreground="red")
        
        # Güvenlik ayarları
        sifrele = "Etkin" if self.secure_storage.sifrele_etkin else "Devre Dışı"
        erisim = "Etkin" if self.secure_storage.erisim_kontrolu_etkin else "Devre Dışı"
        anonim = "Etkin" if self.secure_storage.anonimlestirme_etkin else "Devre Dışı"
        
        durum_metni = f"\nŞifreleme: {sifrele}, Erişim Kontrolü: {erisim}, Anonimleştirme: {anonim}"
        self.durum_label.config(text=self.durum_label.cget("text") + durum_metni)
    
    def _giris_yap(self):
        """Kullanıcı girişi yap"""
        kullanici_id = self.kullanici_id_var.get()
        rol = self.secili_rol.get()
        
        if not kullanici_id:
            messagebox.showerror("Hata", "Kullanıcı kimliği gerekli!")
            return
        
        # Giriş yap
        if self.secure_storage.aktif_kullanici_ayarla(kullanici_id, rol):
            messagebox.showinfo("Başarılı", f"{kullanici_id} kullanıcısı için {rol} rolüyle giriş yapıldı")
            self._durum_guncelle()
        else:
            messagebox.showerror("Hata", "Kullanıcı girişi başarısız")
    
    def _guvenligi_guncelle(self):
        """Güvenlik ayarlarını güncelle"""
        if not self.secure_storage:
            return
            
        self.secure_storage.sifrele_etkin = self.sifreleme_var.get()
        self.secure_storage.erisim_kontrolu_etkin = self.erisim_kontrol_var.get()
        self.secure_storage.anonimlestirme_etkin = self.anonimlestirme_var.get()
        
        self._durum_guncelle()
    
    def _rol_agacini_doldur(self):
        """Rol ağacını doldur"""
        # Ağacı temizle
        for item in self.rol_tree.get_children():
            self.rol_tree.delete(item)
            
        # Roller ve yetkilerini ekle
        for rol_adi, yetkiler in self.secure_storage.erisim_kontrol.rol_yetkileri.items():
            hasta_verileri = ", ".join(yetkiler.get("hasta_verileri", []))
            sistem_ayarlari = ", ".join(yetkiler.get("sistem_ayarlari", []))
            raporlar = ", ".join(yetkiler.get("raporlar", []))
            
            self.rol_tree.insert("", "end", values=(rol_adi, hasta_verileri, sistem_ayarlari, raporlar))
    
    def _erisim_kayitlarini_goster(self):
        """Erişim kayıtlarını göster"""
        # Log dosyasını oku
        log_dosyasi = Path("security/logs/access_log.jsonl")
        
        if not log_dosyasi.exists():
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", tk.END)
            self.log_text.insert("1.0", "Erişim kaydı bulunamadı.")
            self.log_text.config(state="disabled")
            return
            
        try:
            self.log_text.config(state="normal")
            self.log_text.delete("1.0", tk.END)
            
            with open(log_dosyasi, 'r') as f:
                kayitlar = f.readlines()
            
            # Son 10 kaydı göster
            for kayit in kayitlar[-10:]:
                try:
                    kayit_verisi = json.loads(kayit)
                    kullanici = kayit_verisi.get("kullanici_id", "")
                    kaynak = kayit_verisi.get("kaynak_turu", "")
                    islem = kayit_verisi.get("islem_turu", "")
                    izin = kayit_verisi.get("izin_verildi", False)
                    zaman = kayit_verisi.get("zaman", "")
                    
                    durum = "İZİN VERİLDİ" if izin else "REDDEDİLDİ"
                    renk = "green" if izin else "red"
                    
                    log_metni = f"{zaman}: {kullanici} → {kaynak}/{islem} ({durum})\n"
                    self.log_text.insert(tk.END, log_metni)
                    
                    # Duruma göre renk
                    son_satir = self.log_text.index(tk.END + "-1c").split(".")[0]
                    self.log_text.tag_add(durum, f"{son_satir}.0", f"{son_satir}.end")
                    self.log_text.tag_config(durum, foreground=renk)
                    
                except json.JSONDecodeError:
                    pass
                    
            self.log_text.config(state="disabled")
        except Exception as e:
            logger.error(f"Erişim kayıtları okuma hatası: {e}")
            self.log_text.insert(tk.END, f"Hata: {str(e)}")
            self.log_text.config(state="disabled")
    
    def _anahtar_durumunu_guncelle(self):
        """Şifreleme anahtarı durumunu güncelle"""
        key_file = Path("security/keys/.master_key")
        
        if key_file.exists():
            self.anahtar_durum_label.config(
                text="Şifreleme anahtarı mevcut", 
                foreground="green")
        else:
            self.anahtar_durum_label.config(
                text="Şifreleme anahtarı bulunamadı!", 
                foreground="red")
    
    def _yeni_anahtar_olustur(self):
        """Yeni şifreleme anahtarı oluştur"""
        # Onay al
        onay = messagebox.askyesno(
            "Onay", 
            "Yeni anahtar oluşturmak, mevcut şifreli verilerin çözülememesine neden olabilir. Devam etmek istiyor musunuz?")
        
        if not onay:
            return
            
        # Yeni anahtar oluştur
        self.secure_storage.sifrele = VeriSifrelemeServisi()
        if self.secure_storage.sifrele.save_master_key():
            messagebox.showinfo("Başarılı", "Yeni şifreleme anahtarı oluşturuldu")
            self._anahtar_durumunu_guncelle()
        else:
            messagebox.showerror("Hata", "Anahtar oluşturma başarısız")
    
    def _anahtari_yedekle(self):
        """Şifreleme anahtarını yedekle"""
        if not Path("security/keys/.master_key").exists():
            messagebox.showerror("Hata", "Yedeklenecek anahtar bulunamadı!")
            return
            
        # Yedekleme konumu sor
        yedek_dizini = simpledialog.askstring(
            "Yedekleme", 
            "Anahtarın yedekleneceği dizini girin:",
            initialvalue=str(Path.home()))
            
        if not yedek_dizini:
            return
            
        try:
            # Yedekle
            yedek_yolu = Path(yedek_dizini) / "eye_tracker_key_backup.key"
            
            with open("security/keys/.master_key", 'r') as src:
                anahtar = src.read()
                
            with open(yedek_yolu, 'w') as dst:
                dst.write(anahtar)
                
            os.chmod(yedek_yolu, 0o600)  # Sadece sahibi okuyabilir/yazabilir
            
            messagebox.showinfo("Başarılı", f"Anahtar şu konuma yedeklendi:\n{yedek_yolu}")
            
        except Exception as e:
            logger.error(f"Anahtar yedekleme hatası: {e}")
            messagebox.showerror("Hata", f"Yedekleme başarısız: {str(e)}")
    
    def _test_sifrele(self):
        """Test metnini şifrele"""
        test_metni = self.test_text.get("1.0", tk.END).strip()
        
        if not test_metni:
            messagebox.showerror("Hata", "Şifrelenecek metin gerekli!")
            return
            
        # Şifrele
        sifrelenmis = self.secure_storage.sifrele.sifrele(test_metni)
        
        if sifrelenmis:
            self.test_sonuc.config(state="normal")
            self.test_sonuc.delete("1.0", tk.END)
            self.test_sonuc.insert("1.0", sifrelenmis)
            self.test_sonuc.config(state="disabled")
        else:
            messagebox.showerror("Hata", "Şifreleme başarısız!")
    
    def _test_sifreyi_coz(self):
        """Test metninin şifresini çöz"""
        sifrelenmis = self.test_sonuc.get("1.0", tk.END).strip()
        
        if not sifrelenmis:
            messagebox.showerror("Hata", "Çözülecek şifreli metin gerekli!")
            return
            
        # Şifreyi çöz
        cozulmus = self.secure_storage.sifrele.sifreyi_coz(sifrelenmis)
        
        if cozulmus:
            self.test_text.delete("1.0", tk.END)
            self.test_text.insert("1.0", cozulmus)
        else:
            messagebox.showerror("Hata", "Şifre çözme başarısız!")
    
    def _ornek_maskeleme(self):
        """Örnek veriyi maskele"""
        try:
            # JSON verisini ayrıştır
            metin = self.ornek_veri_text.get("1.0", tk.END)
            veri = json.loads(metin)
            
            # Maskele
            maskeli = self.secure_storage.anonimlestir.kimlik_alanlari_maskele(veri)
            
            # Görüntüle
            self.anonim_sonuc.config(state="normal")
            self.anonim_sonuc.delete("1.0", tk.END)
            self.anonim_sonuc.insert("1.0", json.dumps(maskeli, indent=4))
            self.anonim_sonuc.config(state="disabled")
            
        except json.JSONDecodeError:
            messagebox.showerror("Hata", "Geçersiz JSON veri formatı!")
        except Exception as e:
            messagebox.showerror("Hata", f"Maskeleme hatası: {str(e)}")
    
    def _ornek_anonimize(self):
        """Örnek veriyi tam anonimleştir"""
        try:
            # JSON verisini ayrıştır
            metin = self.ornek_veri_text.get("1.0", tk.END)
            veri = json.loads(metin)
            
            # Anonimleştir
            anonim = self.secure_storage.anonimlestir.anonimize_et(veri)
            
            # Görüntüle
            self.anonim_sonuc.config(state="normal")
            self.anonim_sonuc.delete("1.0", tk.END)
            self.anonim_sonuc.insert("1.0", json.dumps(anonim, indent=4))
            self.anonim_sonuc.config(state="disabled")
            
        except json.JSONDecodeError:
            messagebox.showerror("Hata", "Geçersiz JSON veri formatı!")
        except Exception as e:
            messagebox.showerror("Hata", f"Anonimleştirme hatası: {str(e)}")

# Bağımsız test için
if __name__ == "__main__":
    # Ana pencere
    root = tk.Tk()
    root.title("Göz İzleme Sistemi - Güvenlik Ayarları")
    root.geometry("600x500")
    
    # Güvenlik UI örneği
    security_ui = GuvenlikAyarlariUI(root)
    security_ui.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Ana döngü
    root.mainloop() 