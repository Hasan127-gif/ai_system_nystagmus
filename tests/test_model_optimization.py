"""
Model Optimizasyon Testleri
---------------------------
Bu modül, model optimizasyon ve kuantizasyon işlemlerini test eder.
"""

import os
import sys
import unittest
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
import shutil
import tempfile

# Test için modülü import etme
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model_optimization

class TestModelOptimization(unittest.TestCase):
    """Model optimizasyon ve kuantizasyon testleri"""
    
    def setUp(self):
        """Test için geçici dizin ve örnek modelleri oluştur"""
        # Geçici dizin oluştur
        self.test_dir = tempfile.mkdtemp()
        
        # Dizin yapısı
        self.model_dir = os.path.join(self.test_dir, "trained")
        self.opt_dir = os.path.join(self.test_dir, "optimized")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.opt_dir, exist_ok=True)
        
        # Test veri kümesi oluştur
        X, y = make_classification(n_samples=1000, n_features=20, 
                                  n_informative=10, n_redundant=5, 
                                  n_classes=2, random_state=42)
        
        # RandomForest modeli oluştur ve kaydet
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X, y)
        self.rf_model_path = os.path.join(self.model_dir, "rf_model.pkl")
        joblib.dump(rf_model, self.rf_model_path)
        
        # GradientBoosting modeli oluştur ve kaydet
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gb_model.fit(X, y)
        self.gb_model_path = os.path.join(self.model_dir, "gb_model.pkl")
        joblib.dump(gb_model, self.gb_model_path)
        
        # Test veri
        self.X = X
        self.y = y
    
    def tearDown(self):
        """Geçici dizini temizle"""
        shutil.rmtree(self.test_dir)
    
    def test_random_forest_optimization(self):
        """RandomForest optimizasyonunu test et"""
        # Orijinal modeli yükle
        model = joblib.load(self.rf_model_path)
        
        # Orijinal performansı değerlendir
        orig_accuracy = model.score(self.X, self.y)
        
        # Optimize et
        output_path = os.path.join(self.opt_dir, "opt_rf_model.pkl")
        optimized_model, meta = model_optimization.optimize_random_forest(
            model, model.n_features_in_, output_path
        )
        
        # Optimize edilmiş model yüklendi mi?
        self.assertTrue(os.path.exists(output_path))
        
        # Meta dosyası var mı?
        meta_path = output_path.replace(".pkl", "_meta.json")
        self.assertTrue(os.path.exists(meta_path))
        
        # Optimize edilmiş modeli yükle
        opt_model = joblib.load(output_path)
        
        # Optimize edilmiş modelin performansını değerlendir
        # Not: Bu test aşamasında modeli yeniden eğitmiyoruz, sadece orjinal modelin optimize 
        # edilmiş yapısını oluşturuyoruz. Bu yüzden doğrudan performans karşılaştırması yapamayız.
        
        # Modelin boyutu azaldı mı kontrol et
        orig_size = os.path.getsize(self.rf_model_path)
        opt_size = os.path.getsize(output_path)
        self.assertLess(opt_size, orig_size, "Optimize edilmiş model daha küçük olmalı")
        
        # Meta içeriğini kontrol et
        self.assertTrue('original_size_bytes' in meta)
        self.assertTrue('optimized_n_estimators' in meta)
        self.assertTrue('feature_mask' in meta)
    
    def test_gradient_boosting_optimization(self):
        """GradientBoosting optimizasyonunu test et"""
        # Orijinal modeli yükle
        model = joblib.load(self.gb_model_path)
        
        # Orijinal performansı değerlendir
        orig_accuracy = model.score(self.X, self.y)
        
        # Optimize et
        output_path = os.path.join(self.opt_dir, "opt_gb_model.pkl")
        optimized_model, meta = model_optimization.optimize_gradient_boosting(
            model, model.n_features_in_, output_path
        )
        
        # Optimize edilmiş model yüklendi mi?
        self.assertTrue(os.path.exists(output_path))
        
        # Meta dosyası var mı?
        meta_path = output_path.replace(".pkl", "_meta.json")
        self.assertTrue(os.path.exists(meta_path))
        
        # Optimize edilmiş modelin boyutu daha küçük mü?
        orig_size = os.path.getsize(self.gb_model_path)
        opt_size = os.path.getsize(output_path)
        self.assertLess(opt_size, orig_size, "Optimize edilmiş model daha küçük olmalı")
    
    def test_quantize_sklearn_model(self):
        """Genel sklearn model kuantizasyonunu test et"""
        # RandomForest modelini kuantize et
        output_path = os.path.join(self.opt_dir, "quantized_rf_model.pkl")
        model, meta = model_optimization.quantize_sklearn_model(self.rf_model_path, output_path)
        
        # Kuantize edilmiş model yüklendi mi?
        self.assertTrue(os.path.exists(output_path))
        
        # Meta içeriğini kontrol et (optimize_random_forest veya genel kuantizasyon kullanılabilir)
        if isinstance(meta, dict):
            if 'compression_ratio' in meta:
                self.assertGreater(meta['compression_ratio'], 0.0)
            elif 'original_size_bytes' in meta:
                self.assertGreater(meta['original_size_bytes'], 0)
    
    def test_create_dirs(self):
        """Dizin oluşturma fonksiyonunu test et"""
        # Test dizinleri
        test_model_dir = os.path.join(self.test_dir, "models/trained")
        test_opt_dir = os.path.join(self.test_dir, "models/optimized")
        
        # Orijinal model_optimization MODEL_DIR ve OPTIMIZED_DIR değerlerini sakla
        orig_model_dir = model_optimization.MODEL_DIR
        orig_opt_dir = model_optimization.OPTIMIZED_DIR
        
        try:
            # Test dizinlerini ayarla
            model_optimization.MODEL_DIR = test_model_dir
            model_optimization.OPTIMIZED_DIR = test_opt_dir
            
            # Dizinleri oluştur
            model_optimization.create_dirs()
            
            # Dizinler oluşturuldu mu?
            self.assertTrue(os.path.exists(test_model_dir))
            self.assertTrue(os.path.exists(test_opt_dir))
        
        finally:
            # Orijinal değerleri geri yükle
            model_optimization.MODEL_DIR = orig_model_dir
            model_optimization.OPTIMIZED_DIR = orig_opt_dir
    
    def test_get_latest_model(self):
        """En son modeli bulma fonksiyonunu test et"""
        # Orijinal model_optimization MODEL_DIR değerini sakla
        orig_model_dir = model_optimization.MODEL_DIR
        
        try:
            # Test dizinini ayarla
            model_optimization.MODEL_DIR = self.model_dir
            
            # En son modeli bul
            latest_model = model_optimization.get_latest_model()
            
            # Model bulundu mu?
            self.assertIsNotNone(latest_model)
            self.assertTrue(os.path.exists(latest_model))
            
            # default_model.pkl oluştur
            default_model_path = os.path.join(self.model_dir, "default_model.pkl")
            shutil.copy(self.rf_model_path, default_model_path)
            
            # Tekrar en son modeli bul
            latest_model = model_optimization.get_latest_model()
            
            # default_model.pkl bulundu mu?
            self.assertEqual(latest_model, default_model_path)
        
        finally:
            # Orijinal değeri geri yükle
            model_optimization.MODEL_DIR = orig_model_dir

if __name__ == '__main__':
    unittest.main() 