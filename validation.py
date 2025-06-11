#!/usr/bin/env python3
"""
SİSTEM VALİDASYON MODÜLü
========================
MAE hesaplama, confusion matrix ve sistem doğruluğu testi.
"""

import numpy as np
import logging
import time
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class NystagmusValidator:
    """Nistagmus sistemi validasyon sınıfı."""
    
    def __init__(self):
        self.test_results = []
        self.thresholds = {
            "mae_frequency": 0.2,  # Hz
            "mae_angle": 0.5,      # Derece
            "min_sensitivity": 0.8,
            "min_specificity": 0.8
        }
    
    def validate_predictions(self, ground_truth: List[Dict], predictions: List[Dict]) -> Dict[str, Any]:
        """
        Tahminleri ground truth ile karşılaştır.
        
        Args:
            ground_truth: [{"nystagmus_freq": float, "strabismus_angle": float, "has_nystagmus": bool, "has_strabismus": bool}, ...]
            predictions: [{"nystagmus_freq": float, "strabismus_angle": float, "has_nystagmus": bool, "has_strabismus": bool}, ...]
            
        Returns:
            Dict: Validasyon sonuçları
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Ground truth ve prediction sayıları eşleşmiyor")
        
        # Regresyon metrikleri (MAE)
        freq_errors = []
        angle_errors = []
        
        # Sınıflandırma için confusion matrix verileri
        nystagmus_true_labels = []
        nystagmus_pred_labels = []
        strabismus_true_labels = []
        strabismus_pred_labels = []
        
        for gt, pred in zip(ground_truth, predictions):
            # Regresyon hataları
            freq_error = abs(gt["nystagmus_freq"] - pred["nystagmus_freq"])
            angle_error = abs(gt["strabismus_angle"] - pred["strabismus_angle"])
            
            freq_errors.append(freq_error)
            angle_errors.append(angle_error)
            
            # Sınıflandırma labels
            nystagmus_true_labels.append(gt["has_nystagmus"])
            nystagmus_pred_labels.append(pred["has_nystagmus"])
            strabismus_true_labels.append(gt["has_strabismus"])
            strabismus_pred_labels.append(pred["has_strabismus"])
        
        # MAE hesaplama
        mae_frequency = np.mean(freq_errors)
        mae_angle = np.mean(angle_errors)
        
        # Confusion matrix hesaplama
        nystagmus_metrics = self.compute_confusion_matrix(nystagmus_true_labels, nystagmus_pred_labels)
        strabismus_metrics = self.compute_confusion_matrix(strabismus_true_labels, strabismus_pred_labels)
        
        # Validasyon sonuçları
        results = {
            "regression_metrics": {
                "mae_frequency": mae_frequency,
                "mae_angle": mae_angle,
                "frequency_errors": freq_errors,
                "angle_errors": angle_errors,
                "mae_frequency_passed": mae_frequency <= self.thresholds["mae_frequency"],
                "mae_angle_passed": mae_angle <= self.thresholds["mae_angle"]
            },
            "classification_metrics": {
                "nystagmus": nystagmus_metrics,
                "strabismus": strabismus_metrics
            },
            "overall_assessment": {
                "regression_passed": (mae_frequency <= self.thresholds["mae_frequency"] and 
                                    mae_angle <= self.thresholds["mae_angle"]),
                "classification_passed": (nystagmus_metrics["sensitivity"] >= self.thresholds["min_sensitivity"] and
                                         nystagmus_metrics["specificity"] >= self.thresholds["min_specificity"] and
                                         strabismus_metrics["sensitivity"] >= self.thresholds["min_sensitivity"] and
                                         strabismus_metrics["specificity"] >= self.thresholds["min_specificity"]),
                "test_count": len(ground_truth)
            }
        }
        
        # Genel başarı
        results["overall_assessment"]["validation_passed"] = (
            results["overall_assessment"]["regression_passed"] and
            results["overall_assessment"]["classification_passed"]
        )
        
        logger.info(f"Validasyon tamamlandı: {len(ground_truth)} test, "
                   f"MAE_freq={mae_frequency:.3f}, MAE_angle={mae_angle:.3f}")
        
        return results
    
    def compute_confusion_matrix(self, true_labels: List[bool], pred_labels: List[bool]) -> Dict[str, float]:
        """
        Confusion matrix hesapla ve metrikleri döndür.
        
        Args:
            true_labels: Gerçek etiketler
            pred_labels: Tahmin edilen etiketler
            
        Returns:
            Dict: Confusion matrix metrikleri
        """
        true_labels = np.array(true_labels, dtype=bool)
        pred_labels = np.array(pred_labels, dtype=bool)
        
        # Confusion matrix elemanları
        tp = np.sum((true_labels == True) & (pred_labels == True))   # True Positive
        tn = np.sum((true_labels == False) & (pred_labels == False)) # True Negative
        fp = np.sum((true_labels == False) & (pred_labels == True))  # False Positive
        fn = np.sum((true_labels == True) & (pred_labels == False))  # False Negative
        
        # Metrikler
        total = len(true_labels)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Recall, True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # F1 Score
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        # AUC yaklaşımı (basit versiyon)
        auc = (sensitivity + specificity) / 2.0
        
        return {
            "confusion_matrix": {
                "tp": int(tp), "tn": int(tn), 
                "fp": int(fp), "fn": int(fn)
            },
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "accuracy": float(accuracy),
            "f1_score": float(f1_score),
            "auc": float(auc),
            "passed_sensitivity": sensitivity >= self.thresholds["min_sensitivity"],
            "passed_specificity": specificity >= self.thresholds["min_specificity"]
        }
    
    def generate_test_videos(self, test_count: int = 20) -> List[Dict]:
        """
        Test videoları ve ground truth oluştur.
        
        Args:
            test_count: Oluşturulacak test sayısı
            
        Returns:
            List: Test verisi [{"video_path": str, "ground_truth": Dict}, ...]
        """
        try:
            from improved_test_video_generator import MedicalGradeVideoGenerator
            generator = MedicalGradeVideoGenerator()
            
            test_data = []
            
            for i in range(test_count):
                # Rastgele parametreler
                nystagmus_freq = np.random.uniform(0.0, 8.0)  # 0-8 Hz
                strabismus_angle = np.random.uniform(0.0, 15.0)  # 0-15 derece
                
                # Patoloji eşikleri
                has_nystagmus = nystagmus_freq >= 2.0  # 2 Hz üstü patolojik
                has_strabismus = strabismus_angle >= 3.0  # 3 derece üstü patolojik
                
                # Video oluştur
                video_path = f"validation_test_{i:03d}.mp4"
                success = generator.create_realistic_nystagmus_video(
                    video_path, 
                    duration=3.0,
                    nystagmus_freq=nystagmus_freq,
                    strabismus_angle=strabismus_angle
                )
                
                if success:
                    test_data.append({
                        "video_path": video_path,
                        "ground_truth": {
                            "nystagmus_freq": nystagmus_freq,
                            "strabismus_angle": strabismus_angle,
                            "has_nystagmus": has_nystagmus,
                            "has_strabismus": has_strabismus
                        }
                    })
            
            logger.info(f"Test videoları oluşturuldu: {len(test_data)} video")
            return test_data
            
        except Exception as e:
            logger.error(f"Test video oluşturma hatası: {e}")
            return []
    
    def run_validation_test(self, test_count: int = 20, use_ml: bool = True) -> Dict[str, Any]:
        """
        Tam validasyon testi çalıştır.
        
        Args:
            test_count: Test video sayısı
            use_ml: ML sınıflandırma kullanılsın mı
            
        Returns:
            Dict: Validasyon sonuçları
        """
        logger.info(f"Validasyon testi başlatılıyor: {test_count} video, ML={'AÇIK' if use_ml else 'KAPALI'}")
        start_time = time.time()
        
        try:
            # Test videoları oluştur
            test_data = self.generate_test_videos(test_count)
            if not test_data:
                return {"error": "Test videoları oluşturulamadı"}
            
            # Tahminleri toplama
            ground_truths = []
            predictions = []
            
            for i, test_item in enumerate(test_data):
                video_path = test_item["video_path"]
                gt = test_item["ground_truth"]
                
                logger.info(f"Analiz ediliyor ({i+1}/{len(test_data)}): {video_path}")
                
                # Video analizi
                if use_ml:
                    from analysis_utils import analyze_video_with_ml
                    results = analyze_video_with_ml(video_path, use_ml_classification=True)
                    
                    # ML sonuçlarını kullan
                    if "ml_analysis" in results:
                        ml_data = results["ml_analysis"]
                        prediction = {
                            "nystagmus_freq": ml_data["regression"]["nystagmus_frequency"],
                            "strabismus_angle": ml_data["regression"]["strabismus_angle"],
                            "has_nystagmus": ml_data["predictions"]["nystagmus_detected"],
                            "has_strabismus": ml_data["predictions"]["strabismus_detected"]
                        }
                    else:
                        # ML başarısızsa geleneksel sonuçları kullan
                        prediction = {
                            "nystagmus_freq": results.get("nistagmus_frequency", 0.0),
                            "strabismus_angle": results.get("strabismus_angle", 0.0),
                            "has_nystagmus": results.get("nistagmus_frequency", 0.0) >= 2.0,
                            "has_strabismus": results.get("strabismus_angle", 0.0) >= 3.0
                        }
                else:
                    # Sadece geleneksel analiz
                    from analysis_utils import analyze_video_file
                    results = analyze_video_file(video_path)
                    
                    if "error" not in results:
                        prediction = {
                            "nystagmus_freq": results["nistagmus_frequency"],
                            "strabismus_angle": results["strabismus_angle"],
                            "has_nystagmus": results["nistagmus_frequency"] >= 2.0,
                            "has_strabismus": results["strabismus_angle"] >= 3.0
                        }
                    else:
                        # Hatalı analiz
                        prediction = {
                            "nystagmus_freq": 0.0,
                            "strabismus_angle": 0.0,
                            "has_nystagmus": False,
                            "has_strabismus": False
                        }
                
                ground_truths.append(gt)
                predictions.append(prediction)
            
            # Validasyon hesaplama
            validation_results = self.validate_predictions(ground_truths, predictions)
            
            # Test süresi
            total_duration = time.time() - start_time
            validation_results["test_duration"] = total_duration
            validation_results["test_info"] = {
                "test_count": len(test_data),
                "ml_enabled": use_ml,
                "avg_time_per_video": total_duration / len(test_data) if test_data else 0
            }
            
            # Test dosyalarını temizle
            self.cleanup_test_files([item["video_path"] for item in test_data])
            
            # Sonuçları kaydet
            self.save_validation_results(validation_results)
            
            logger.info(f"Validasyon testi tamamlandı: {total_duration:.2f}s")
            return validation_results
            
        except Exception as e:
            logger.error(f"Validasyon testi hatası: {e}")
            return {"error": f"Validasyon hatası: {str(e)}"}
    
    def cleanup_test_files(self, file_paths: List[str]):
        """Test dosyalarını temizle."""
        import os
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Test dosyası silinemiyor {file_path}: {e}")
    
    def save_validation_results(self, results: Dict[str, Any], filename: str = "validation_results.json"):
        """Validasyon sonuçlarını kaydet."""
        try:
            # Numpy tiplerini JSON'a uygun hale getir
            results_serializable = self._make_json_serializable(results)
            
            with open(filename, 'w') as f:
                json.dump(results_serializable, f, indent=2)
            
            logger.info(f"Validasyon sonuçları kaydedildi: {filename}")
            
        except Exception as e:
            logger.error(f"Sonuç kaydetme hatası: {e}")
    
    def _make_json_serializable(self, obj):
        """Nesneyi JSON serileştirilebilir hale getir."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Validasyon raporu oluştur."""
        if "error" in results:
            return f"VALIDASYON HATASI: {results['error']}"
        
        reg_metrics = results["regression_metrics"]
        nyst_metrics = results["classification_metrics"]["nystagmus"]
        strab_metrics = results["classification_metrics"]["strabismus"]
        overall = results["overall_assessment"]
        
        report = "NISTAGMUS SİSTEMİ VALİDASYON RAPORU\n"
        report += "=" * 50 + "\n\n"
        
        # Genel durum
        status_icon = "✅" if overall["validation_passed"] else "❌"
        report += f"{status_icon} GENEL DURUM: {'BAŞARILI' if overall['validation_passed'] else 'BAŞARISIZ'}\n"
        report += f"Test Sayısı: {overall['test_count']}\n"
        report += f"Test Süresi: {results.get('test_duration', 0):.2f}s\n\n"
        
        # Regresyon metrikleri
        report += "REGRESYON METRİKLERİ:\n"
        report += "-" * 25 + "\n"
        freq_icon = "✅" if reg_metrics["mae_frequency_passed"] else "❌"
        angle_icon = "✅" if reg_metrics["mae_angle_passed"] else "❌"
        report += f"{freq_icon} MAE Frekans: {reg_metrics['mae_frequency']:.3f} Hz (eşik: ≤{self.thresholds['mae_frequency']} Hz)\n"
        report += f"{angle_icon} MAE Açı: {reg_metrics['mae_angle']:.3f}° (eşik: ≤{self.thresholds['mae_angle']}°)\n\n"
        
        # Sınıflandırma metrikleri
        report += "SINIFLANDIRMA METRİKLERİ:\n"
        report += "-" * 30 + "\n"
        
        # Nistagmus
        nyst_sens_icon = "✅" if nyst_metrics["passed_sensitivity"] else "❌"
        nyst_spec_icon = "✅" if nyst_metrics["passed_specificity"] else "❌"
        report += f"Nistagmus:\n"
        report += f"  {nyst_sens_icon} Sensitivity: {nyst_metrics['sensitivity']:.3f} (eşik: ≥{self.thresholds['min_sensitivity']})\n"
        report += f"  {nyst_spec_icon} Specificity: {nyst_metrics['specificity']:.3f} (eşik: ≥{self.thresholds['min_specificity']})\n"
        report += f"  Accuracy: {nyst_metrics['accuracy']:.3f}\n"
        report += f"  F1-Score: {nyst_metrics['f1_score']:.3f}\n"
        report += f"  AUC: {nyst_metrics['auc']:.3f}\n\n"
        
        # Şaşılık
        strab_sens_icon = "✅" if strab_metrics["passed_sensitivity"] else "❌"
        strab_spec_icon = "✅" if strab_metrics["passed_specificity"] else "❌"
        report += f"Şaşılık:\n"
        report += f"  {strab_sens_icon} Sensitivity: {strab_metrics['sensitivity']:.3f} (eşik: ≥{self.thresholds['min_sensitivity']})\n"
        report += f"  {strab_spec_icon} Specificity: {strab_metrics['specificity']:.3f} (eşik: ≥{self.thresholds['min_specificity']})\n"
        report += f"  Accuracy: {strab_metrics['accuracy']:.3f}\n"
        report += f"  F1-Score: {strab_metrics['f1_score']:.3f}\n"
        report += f"  AUC: {strab_metrics['auc']:.3f}\n\n"
        
        # Confusion Matrix
        report += "CONFUSION MATRIX:\n"
        report += "-" * 20 + "\n"
        report += f"Nistagmus - TP:{nyst_metrics['confusion_matrix']['tp']}, "
        report += f"TN:{nyst_metrics['confusion_matrix']['tn']}, "
        report += f"FP:{nyst_metrics['confusion_matrix']['fp']}, "
        report += f"FN:{nyst_metrics['confusion_matrix']['fn']}\n"
        report += f"Şaşılık - TP:{strab_metrics['confusion_matrix']['tp']}, "
        report += f"TN:{strab_metrics['confusion_matrix']['tn']}, "
        report += f"FP:{strab_metrics['confusion_matrix']['fp']}, "
        report += f"FN:{strab_metrics['confusion_matrix']['fn']}\n"
        
        return report

def run_quick_validation(test_count: int = 10, use_ml: bool = True) -> Dict[str, Any]:
    """Hızlı validasyon testi çalıştır."""
    validator = NystagmusValidator()
    return validator.run_validation_test(test_count, use_ml)

def main():
    """Test amaçlı ana fonksiyon."""
    logging.basicConfig(level=logging.INFO)
    
    print("🔬 SİSTEM VALİDASYON TESTİ")
    print("=" * 40)
    
    # Hızlı test
    results = run_quick_validation(test_count=5, use_ml=True)
    
    if "error" not in results:
        validator = NystagmusValidator()
        report = validator.generate_validation_report(results)
        print(report)
    else:
        print(f"❌ Test hatası: {results['error']}")

if __name__ == "__main__":
    main() 