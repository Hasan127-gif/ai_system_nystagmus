"""
Göz İzleme Sistemi - Güvenlik Entegrasyonu

Bu modül, veri güvenliği ve gizlilik bileşenlerini ana uygulamaya
entegre etmek için gerekli arayüzleri sağlar.
"""

import tkinter as tk
from tkinter import ttk, messagebox
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable

# Güvenlik modüllerini içe aktar
try:
    from security.data_security import VeriSifrelemeServisi, ErisimKontrolSistemi, VeriAnonimlestirme
    from security.secure_storage import GuvenliVeriDepolama
    from security.security_ui import GuvenlikAyarlariUI
except ImportError:
    # Alternatif içe aktarma yolu (göreceli import durumlarında)
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.dirname(__file__)))
        from data_security import VeriSifrelemeServisi, ErisimKontrolSistemi, VeriAnonimlestirme
        from secure_storage import GuvenliVeriDepolama
        from security_ui import GuvenlikAyarlariUI
    except ImportError:
        print("Güvenlik modülleri yüklenemedi!")

# Logger yapılandırması
logger = logging.getLogger('eye_tracker.security_integration')

class GuvenlikEntegrasyonu:
    """
    Ana uygulamaya güvenlik özelliklerini entegre eder
    """
    
    def __init__(self):
        """Güvenlik entegrasyonunu başlat"""
        # Güvenli veri depolama
        self.secure_storage = GuvenliVeriDepolama()
        
        # Varsayılan kullanıcı kimliği
        self.secure_storage.aktif_kullanici_ayarla("varsayilan_kullanici", "doktor")
        
        # UI bileşenleri
        self.ui_frame = None
        self.menu_items = []
        
        # Oturum durumu
        self.oturum_acik = False
        
        # Olay işleyicileri
        self.on_logout_handlers = []
    
    def ana_menusunu_olustur(self, menu_bar: tk.Menu) -> tk.Menu:
        """
        Ana menüye güvenlik menüsü ekle
        
        Args:
            menu_bar: Ana menü çubuğu
            
        Returns:
            Oluşturulan güvenlik menüsü
        """
        # Güvenlik menüsü
        guvenlik_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="Güvenlik", menu=guvenlik_menu)
        
        # Menü öğeleri
        guvenlik_menu.add_command(label="Giriş Yap...", command=self._giris_dialog_goster)
        guvenlik_menu.add_command(label="Çıkış Yap", command=self._cikis_yap)
        guvenlik_menu.add_separator()
        guvenlik_menu.add_command(label="Güvenlik Ayarları...", command=self._guvenlik_ayarlarini_goster)
        guvenlik_menu.add_command(label="Erişim Kayıtları...", command=self._erisim_kayitlarini_goster)
        guvenlik_menu.add_separator()
        guvenlik_menu.add_command(label="Veri Maskeleme/Anonimleştirme...", command=self._veri_anonimize_goster)
        
        # Oluşturulan menüyü döndür
        return guvenlik_menu
    
    def ayarlar_sekmesi_ekle(self, notebook: ttk.Notebook) -> ttk.Frame:
        """
        Ayarlar not defterine güvenlik sekmesi ekle
        
        Args:
            notebook: Sekme kontrolü
            
        Returns:
            Oluşturulan sekme çerçevesi
        """
        # Güvenlik ayarları çerçevesi
        self.ui_frame = GuvenlikAyarlariUI(notebook, self.secure_storage)
        notebook.add(self.ui_frame, text="Güvenlik")
        
        return self.ui_frame
    
    def oturum_ac(self, kullanici_id: str, rol: str) -> bool:
        """
        Kullanıcı oturumu aç
        
        Args:
            kullanici_id: Kullanıcı kimliği
            rol: Kullanıcı rolü
            
        Returns:
            İşlem başarı durumu
        """
        sonuc = self.secure_storage.aktif_kullanici_ayarla(kullanici_id, rol)
        
        if sonuc:
            self.oturum_acik = True
            logger.info(f"Oturum açıldı: {kullanici_id} ({rol})")
        else:
            logger.warning(f"Oturum açma başarısız: {kullanici_id}")
        
        return sonuc
    
    def oturum_kapat(self) -> None:
        """Kullanıcı oturumunu kapat"""
        if self.oturum_acik:
            # Çıkış olayı işleyicilerini çağır
            for handler in self.on_logout_handlers:
                try:
                    handler()
                except Exception as e:
                    logger.error(f"Oturum kapatma işleyici hatası: {e}")
            
            # Oturum durumunu güncelle
            self.oturum_acik = False
            
            # Güvenlik ayarlarını sıfırla
            self.secure_storage.aktif_kullanici_id = None
            
            logger.info("Oturum kapatıldı")
    
    def on_logout(self, handler: Callable[[], None]) -> None:
        """
        Oturum kapatma olayı için işleyici ekle
        
        Args:
            handler: Çağrılacak fonksiyon
        """
        self.on_logout_handlers.append(handler)
    
    def veriyi_guvenli_kaydet(self, veri: Dict[str, Any], hasta_id: Optional[str] = None) -> bool:
        """
        Veriyi güvenli bir şekilde kaydet
        
        Args:
            veri: Kaydedilecek veri
            hasta_id: Hasta kimliği (varsa)
            
        Returns:
            İşlem başarı durumu
        """
        return self.secure_storage.veri_kaydet(veri, hasta_id)
    
    def veriyi_guvenli_oku(self, veri_id: str, hasta_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Veriyi güvenli bir şekilde oku
        
        Args:
            veri_id: Veri kimliği
            hasta_id: Hasta kimliği (varsa)
            
        Returns:
            Okunan veri veya None (hata durumunda)
        """
        return self.secure_storage.veri_oku(veri_id, hasta_id=hasta_id)
    
    def hasta_verilerini_anonimize_et(self, hasta_verileri: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hasta verilerini anonimleştir
        
        Args:
            hasta_verileri: Ham hasta verileri
            
        Returns:
            Anonimleştirilmiş veriler
        """
        return self.secure_storage.anonimlestir.anonimize_et(hasta_verileri)
    
    def _giris_dialog_goster(self) -> None:
        """Giriş diyaloğunu göster"""
        # Yeni pencere
        giris_pencere = tk.Toplevel()
        giris_pencere.title("Kullanıcı Girişi")
        giris_pencere.geometry("300x200")
        giris_pencere.resizable(False, False)
        giris_pencere.grab_set()  # Modal diyalog
        
        # Ortala
        giris_pencere.update_idletasks()
        width = giris_pencere.winfo_width()
        height = giris_pencere.winfo_height()
        x = (giris_pencere.winfo_screenwidth() // 2) - (width // 2)
        y = (giris_pencere.winfo_screenheight() // 2) - (height // 2)
        giris_pencere.geometry(f"{width}x{height}+{x}+{y}")
        
        # İçerik çerçevesi
        frame = ttk.Frame(giris_pencere, padding=20)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Değişkenler
        kullanici_id = tk.StringVar()
        rol = tk.StringVar(value="doktor")
        
        # Kullanıcı kimliği
        ttk.Label(frame, text="Kullanıcı Kimliği:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(frame, textvariable=kullanici_id, width=20).grid(row=0, column=1, pady=5)
        
        # Rol seçimi
        ttk.Label(frame, text="Rol:").grid(row=1, column=0, sticky=tk.W, pady=5)
        roller = ["yonetici", "doktor", "teknisyen", "arastirmaci", "hasta"]
        ttk.Combobox(frame, textvariable=rol, values=roller, state="readonly").grid(row=1, column=1, pady=5)
        
        # Giriş işlevi
        def giris_yap():
            if not kullanici_id.get():
                messagebox.showerror("Hata", "Kullanıcı kimliği gerekli!", parent=giris_pencere)
                return
                
            if self.oturum_ac(kullanici_id.get(), rol.get()):
                messagebox.showinfo("Başarılı", 
                                  f"{kullanici_id.get()} kullanıcısı için {rol.get()} rolüyle giriş yapıldı",
                                  parent=giris_pencere)
                giris_pencere.destroy()
            else:
                messagebox.showerror("Hata", "Giriş başarısız!", parent=giris_pencere)
        
        # Butonlar
        buton_frame = ttk.Frame(frame)
        buton_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(buton_frame, text="Giriş", command=giris_yap).grid(row=0, column=0, padx=10)
        ttk.Button(buton_frame, text="İptal", command=giris_pencere.destroy).grid(row=0, column=1, padx=10)
    
    def _cikis_yap(self) -> None:
        """Kullanıcı oturumunu kapat"""
        if not self.oturum_acik:
            messagebox.showinfo("Bilgi", "Zaten oturum açık değil")
            return
            
        # Onay al
        onay = messagebox.askyesno("Onay", "Oturumu kapatmak istiyor musunuz?")
        if onay:
            self.oturum_kapat()
            messagebox.showinfo("Başarılı", "Oturum kapatıldı")
    
    def _guvenlik_ayarlarini_goster(self) -> None:
        """Güvenlik ayarları penceresini göster"""
        # Yeni pencere
        ayarlar_pencere = tk.Toplevel()
        ayarlar_pencere.title("Güvenlik Ayarları")
        ayarlar_pencere.geometry("650x500")
        
        # Güvenlik ayarları UI
        ayarlar_ui = GuvenlikAyarlariUI(ayarlar_pencere, self.secure_storage)
        ayarlar_ui.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _erisim_kayitlarini_goster(self) -> None:
        """Erişim kayıtları penceresini göster"""
        # Yeni pencere
        log_pencere = tk.Toplevel()
        log_pencere.title("Erişim Kayıtları")
        log_pencere.geometry("600x400")
        
        # İçerik çerçevesi
        frame = ttk.Frame(log_pencere, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Kayıt görüntüleme
        log_text = tk.Text(frame, wrap="word")
        log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Kaydırma çubuğu
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        log_text.configure(yscrollcommand=scrollbar.set)
        
        # Kayıtları yükle
        log_dosyasi = Path("security/logs/access_log.jsonl")
        
        if not log_dosyasi.exists():
            log_text.insert(tk.END, "Erişim kaydı bulunamadı.")
            return
            
        try:
            with open(log_dosyasi, 'r') as f:
                kayitlar = f.readlines()
                
            # Tüm kayıtları göster
            import json
            
            for i, kayit in enumerate(kayitlar):
                try:
                    kayit_verisi = json.loads(kayit)
                    kullanici = kayit_verisi.get("kullanici_id", "")
                    kaynak = kayit_verisi.get("kaynak_turu", "")
                    islem = kayit_verisi.get("islem_turu", "")
                    izin = kayit_verisi.get("izin_verildi", False)
                    zaman = kayit_verisi.get("zaman", "")
                    ip = kayit_verisi.get("ip_adresi", "")
                    
                    durum = "İZİN VERİLDİ" if izin else "REDDEDİLDİ"
                    renk = "green" if izin else "red"
                    
                    log_metni = f"{i+1}. {zaman}: {kullanici} ({ip}) → {kaynak}/{islem} ({durum})\n"
                    log_text.insert(tk.END, log_metni)
                    
                    # Duruma göre renk
                    son_satir = log_text.index(tk.END + "-1c").split(".")[0]
                    log_text.tag_add(f"durum_{i}", f"{son_satir}.0", f"{son_satir}.end")
                    log_text.tag_config(f"durum_{i}", foreground=renk)
                    
                except json.JSONDecodeError:
                    pass
                    
        except Exception as e:
            logger.error(f"Erişim kayıtları okuma hatası: {e}")
            log_text.insert(tk.END, f"Hata: {str(e)}")
    
    def _veri_anonimize_goster(self) -> None:
        """Veri anonimleştirme arayüzünü göster"""
        # Yeni pencere
        anonim_pencere = tk.Toplevel()
        anonim_pencere.title("Veri Maskeleme/Anonimleştirme")
        anonim_pencere.geometry("600x500")
        
        # İçerik çerçevesi
        frame = ttk.Frame(anonim_pencere, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Başlık
        ttk.Label(frame, text="Veri Maskeleme ve Anonimleştirme", font=("Helvetica", 12, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=10)
        
        # Sol panel - Kontroller
        sol_panel = ttk.LabelFrame(frame, text="Ayarlar")
        sol_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        # Hassas alanlar
        ttk.Label(sol_panel, text="Hassas Alanlar:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        hassas_listbox = tk.Listbox(sol_panel, height=8, selectmode=tk.MULTIPLE)
        hassas_listbox.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Listbox doldur
        hassas_alanlar = [
            "ad", "soyad", "tc_kimlik", "dogum_tarihi", "telefon", 
            "adres", "eposta", "ip_adresi", "hasta_id"
        ]
        for alan in hassas_alanlar:
            hassas_listbox.insert(tk.END, alan)
        
        # İşlem butonları
        ttk.Button(sol_panel, text="Maskeleme Uygula", command=lambda: _maskele()).grid(
            row=2, column=0, sticky="ew", padx=5, pady=5)
            
        ttk.Button(sol_panel, text="Tam Anonimleştir", command=lambda: _anonimize()).grid(
            row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Sağ panel - Veri girişi/çıkışı
        sag_panel = ttk.Frame(frame)
        sag_panel.grid(row=1, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        # Girdi
        ttk.Label(sag_panel, text="İşlenecek Veri (JSON):").grid(row=0, column=0, sticky="w", pady=5)
        
        girdi_text = tk.Text(sag_panel, height=10, width=40)
        girdi_text.grid(row=1, column=0, sticky="nsew", pady=5)
        
        # Örnek veri
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
        girdi_text.insert("1.0", örnek_veri)
        
        # Çıktı
        ttk.Label(sag_panel, text="İşlenmiş Veri:").grid(row=2, column=0, sticky="w", pady=5)
        
        cikti_text = tk.Text(sag_panel, height=10, width=40)
        cikti_text.grid(row=3, column=0, sticky="nsew", pady=5)
        
        # İşleme fonksiyonları
        def _maskele():
            try:
                import json
                # JSON verisini ayrıştır
                metin = girdi_text.get("1.0", tk.END)
                veri = json.loads(metin)
                
                # Seçilen hassas alanları al
                secilen_alanlar = [hassas_alanlar[i] for i in hassas_listbox.curselection()]
                
                # Seçilen alanları maskele
                maskeli_veri = veri.copy()
                for alan in secilen_alanlar:
                    if alan in maskeli_veri:
                        if alan == "tc_kimlik" or alan == "telefon":
                            # Kısmi maskeleme
                            deger = str(maskeli_veri[alan])
                            if len(deger) > 4:
                                maskeli_veri[alan] = "****" + deger[-4:]
                            else:
                                maskeli_veri[alan] = "********"
                        elif alan == "eposta":
                            # E-posta maskeleme
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
                
                # Görüntüle
                cikti_text.delete("1.0", tk.END)
                cikti_text.insert("1.0", json.dumps(maskeli_veri, indent=4))
                
            except json.JSONDecodeError:
                messagebox.showerror("Hata", "Geçersiz JSON veri formatı!", parent=anonim_pencere)
            except Exception as e:
                messagebox.showerror("Hata", f"Maskeleme hatası: {str(e)}", parent=anonim_pencere)
        
        def _anonimize():
            try:
                import json
                # JSON verisini ayrıştır
                metin = girdi_text.get("1.0", tk.END)
                veri = json.loads(metin)
                
                # Anonimleştir
                anonim_veri = self.secure_storage.anonimlestir.anonimize_et(veri)
                
                # Görüntüle
                cikti_text.delete("1.0", tk.END)
                cikti_text.insert("1.0", json.dumps(anonim_veri, indent=4))
                
            except json.JSONDecodeError:
                messagebox.showerror("Hata", "Geçersiz JSON veri formatı!", parent=anonim_pencere)
            except Exception as e:
                messagebox.showerror("Hata", f"Anonimleştirme hatası: {str(e)}", parent=anonim_pencere)
        
        # Yerleşimi ayarla
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(1, weight=1)
        
        sol_panel.columnconfigure(0, weight=1)
        sol_panel.rowconfigure(1, weight=1)
        
        sag_panel.columnconfigure(0, weight=1)
        sag_panel.rowconfigure(1, weight=1)
        sag_panel.rowconfigure(3, weight=1)

# Kullanım örneği
if __name__ == "__main__":
    # Ana pencere
    root = tk.Tk()
    root.title("Göz İzleme Sistemi")
    root.geometry("800x600")
    
    # Menü çubuğu
    menu_bar = tk.Menu(root)
    root.config(menu=menu_bar)
    
    # Dosya menüsü
    dosya_menu = tk.Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Dosya", menu=dosya_menu)
    dosya_menu.add_command(label="Çıkış", command=root.quit)
    
    # Güvenlik entegrasyonu
    guvenlik = GuvenlikEntegrasyonu()
    guvenlik.ana_menusunu_olustur(menu_bar)
    
    # Ana çerçeve
    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Sekme kontrolü
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # Ana sekme
    ana_frame = ttk.Frame(notebook)
    notebook.add(ana_frame, text="Ana Sayfa")
    
    # Güvenlik sekmesi ekle
    guvenlik.ayarlar_sekmesi_ekle(notebook)
    
    # Ana döngü
    root.mainloop() 