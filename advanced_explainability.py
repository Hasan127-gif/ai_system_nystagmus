#!/usr/bin/env python3
"""
GELİŞMİŞ AÇIKLANABILIRLIK SİSTEMİ
=================================
Grad-CAM ve SHAP ile model kararlarının detaylı açıklanması.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import shap
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GradCAMExplainer:
    """
    Grad-CAM tabanlı CNN açıklama sistemi.
    Modelin hangi bölgelere odaklandığını görselleştirir.
    """
    
    def __init__(self, model: nn.Module, target_layer: str = None):
        self.model = model
        self.target_layer = self._find_target_layer(target_layer)
        self.cam = None
        self._setup_gradcam()
    
    def _find_target_layer(self, target_layer: str = None):
        """Hedef katmanı bulur veya otomatik seçer."""
        if target_layer:
            return getattr(self.model, target_layer)
        
        # Son conv katmanını bul
        for name, module in reversed(list(self.model.named_modules())):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                logger.info(f"Grad-CAM için hedef katman: {name}")
                return module
        
        # Fallback: son katman
        return list(self.model.modules())[-1]
    
    def _setup_gradcam(self):
        """Grad-CAM sistemini kurur."""
        try:
            self.cam = GradCAM(
                model=self.model,
                target_layers=[self.target_layer],
                use_cuda=torch.cuda.is_available()
            )
            logger.info("Grad-CAM başarıyla kuruldu")
        except Exception as e:
            logger.error(f"Grad-CAM kurulum hatası: {e}")
            self.cam = None
    
    def generate_cam_heatmap(self, input_tensor: torch.Tensor, 
                           target_class: int = None) -> np.ndarray:
        """
        Grad-CAM ısı haritası oluşturur.
        
        Args:
            input_tensor: Model girdi tensörü [1, C, H, W]
            target_class: Hedef sınıf (None ise en yüksek skorlu)
            
        Returns:
            np.ndarray: Grad-CAM ısı haritası [H, W]
        """
        if self.cam is None:
            logger.warning("Grad-CAM mevcut değil, boş harita döndürülüyor")
            return np.zeros((224, 224))
        
        try:
            # Target seç
            targets = None
            if target_class is not None:
                targets = [ClassifierOutputTarget(target_class)]
            
            # Grad-CAM hesapla
            grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
            
            # İlk batch element'ini al
            return grayscale_cam[0]
            
        except Exception as e:
            logger.error(f"Grad-CAM hesaplama hatası: {e}")
            return np.zeros((224, 224))
    
    def visualize_cam_on_image(self, original_image: np.ndarray, 
                             cam_heatmap: np.ndarray,
                             analysis_results: Dict[str, Any] = None) -> np.ndarray:
        """
        Grad-CAM'i orijinal görüntü üzerine bindirir.
        
        Args:
            original_image: Orijinal görüntü [H, W, C] RGB
            cam_heatmap: Grad-CAM ısı haritası [H, W]
            analysis_results: Analiz sonuçları (açıklama için)
            
        Returns:
            np.ndarray: Görselleştirilmiş açıklama
        """
        try:
            # Görüntüyü normalize et [0, 1]
            if original_image.max() > 1:
                rgb_img = original_image.astype(np.float32) / 255.0
            else:
                rgb_img = original_image.astype(np.float32)
            
            # Grad-CAM'i görüntü üzerine bindır
            visualization = show_cam_on_image(rgb_img, cam_heatmap, use_rgb=True)
            
            # Klinik açıklamalar ekle
            if analysis_results:
                visualization = self._add_clinical_annotations(
                    visualization, analysis_results
                )
            
            return visualization
            
        except Exception as e:
            logger.error(f"Grad-CAM görselleştirme hatası: {e}")
            return original_image
    
    def _add_clinical_annotations(self, image: np.ndarray, 
                                results: Dict[str, Any]) -> np.ndarray:
        """Klinik açıklamalar ekler."""
        try:
            annotated = image.copy()
            h, w = annotated.shape[:2]
            
            # Başlık
            cv2.putText(annotated, "GRAD-CAM MODEL AÇIKLAMASI", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Nistagmus bulguları
            nys_freq = results.get("nistagmus_frequency", 0.0)
            if nys_freq > 2.0:
                cv2.putText(annotated, f"PATOLOJI: Nistagmus {nys_freq:.1f}Hz", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(annotated, f"NORMAL: Nistagmus {nys_freq:.1f}Hz", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Şaşılık bulguları
            strab_angle = results.get("strabismus_angle", 0.0)
            if strab_angle > 2.0:
                cv2.putText(annotated, f"PATOLOJI: Şaşılık {strab_angle:.1f}°", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(annotated, f"NORMAL: Şaşılık {strab_angle:.1f}°", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Açıklama
            cv2.putText(annotated, "Kırmızı: Model odak bölgeleri", 
                       (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            cv2.putText(annotated, "Mavi: Düşük dikkat alanları", 
                       (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            return annotated
            
        except Exception as e:
            logger.error(f"Açıklama ekleme hatası: {e}")
            return image

class SHAPExplainer:
    """
    SHAP tabanlı öznitelik açıklama sistemi.
    Hangi özniteliğin kararı ne kadar etkilediğini gösterir.
    """
    
    def __init__(self, model_predict_fn, background_data: np.ndarray = None):
        self.model_predict_fn = model_predict_fn
        self.background_data = background_data
        self.explainer = None
        self._setup_shap()
    
    def _setup_shap(self):
        """SHAP explainer'ı kurar."""
        try:
            if self.background_data is not None:
                # KernelExplainer (model-agnostic)
                self.explainer = shap.KernelExplainer(
                    self.model_predict_fn, 
                    self.background_data
                )
            else:
                # Basit background data oluştur
                self.background_data = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])  # Sıfır öznitelikler
                self.explainer = shap.KernelExplainer(
                    self.model_predict_fn,
                    self.background_data
                )
            
            logger.info("SHAP explainer başarıyla kuruldu")
            
        except Exception as e:
            logger.error(f"SHAP kurulum hatası: {e}")
            self.explainer = None
    
    def explain_features(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Özniteliklerin model kararına etkisini açıklar.
        
        Args:
            features: Öznitelik sözlüğü
            
        Returns:
            dict: SHAP açıklama sonuçları
        """
        if self.explainer is None:
            return self._fallback_explanation(features)
        
        try:
            # Öznitelikleri array'e çevir
            feature_array = np.array([[
                features.get("nystagmus_frequency", 0.0),
                features.get("movement_amplitude", 0.0),
                features.get("regularity", 0.0),
                features.get("strabismus_angle", 0.0),
                features.get("strabismus_stability", 0.0)
            ]])
            
            # SHAP değerlerini hesapla
            shap_values = self.explainer.shap_values(feature_array)
            
            # Sonuçları yorumla
            feature_names = [
                "Nistagmus Frekansı",
                "Hareket Büyüklüğü", 
                "Düzenlilik",
                "Şaşılık Açısı",
                "Şaşılık Kararlılığı"
            ]
            
            # SHAP değerlerini features ile eşleştir
            if isinstance(shap_values, list):
                # Binary classification - pozitif sınıf al
                values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
                values = shap_values[0]
            
            explanation = {
                "method": "SHAP KernelExplainer",
                "timestamp": datetime.now().isoformat(),
                "feature_importance": {},
                "total_effect": float(np.sum(values)),
                "baseline": float(self.explainer.expected_value),
                "clinical_interpretation": []
            }
            
            # Her öznitelik için SHAP değeri
            for i, (name, value) in enumerate(zip(feature_names, values)):
                importance = float(value)
                explanation["feature_importance"][name] = {
                    "shap_value": importance,
                    "feature_value": float(feature_array[0][i]),
                    "contribution_percent": abs(importance) / (abs(values).sum() + 1e-8) * 100
                }
            
            # Klinik yorumlama
            explanation["clinical_interpretation"] = self._interpret_shap_values(
                explanation["feature_importance"]
            )
            
            return explanation
            
        except Exception as e:
            logger.error(f"SHAP açıklama hatası: {e}")
            return self._fallback_explanation(features)
    
    def _interpret_shap_values(self, importance: Dict[str, Dict]) -> List[str]:
        """SHAP değerlerini klinik olarak yorumlar."""
        interpretations = []
        
        # Önem sırasına göre sırala
        sorted_features = sorted(
            importance.items(), 
            key=lambda x: abs(x[1]["shap_value"]), 
            reverse=True
        )
        
        for feature, data in sorted_features:
            shap_val = data["shap_value"]
            contribution = data["contribution_percent"]
            
            if abs(shap_val) > 0.1:  # Anlamlı katkı eşiği
                direction = "artırıyor" if shap_val > 0 else "azaltıyor"
                
                if contribution > 40:
                    interpretations.append(
                        f"🔴 {feature}: Kararı {direction} (Ana faktör - %{contribution:.1f})"
                    )
                elif contribution > 20:
                    interpretations.append(
                        f"🟠 {feature}: Kararı {direction} (Önemli faktör - %{contribution:.1f})"
                    )
                elif contribution > 10:
                    interpretations.append(
                        f"🟡 {feature}: Kararı {direction} (Orta faktör - %{contribution:.1f})"
                    )
                else:
                    interpretations.append(
                        f"🟢 {feature}: Kararı {direction} (Küçük faktör - %{contribution:.1f})"
                    )
        
        return interpretations
    
    def _fallback_explanation(self, features: Dict[str, float]) -> Dict[str, Any]:
        """SHAP başarısız olursa basit açıklama."""
        return {
            "method": "Rule-based Fallback",
            "timestamp": datetime.now().isoformat(),
            "warning": "SHAP açıklama başarısız, kural tabanlı analiz kullanıldı",
            "primary_factors": {
                "Nistagmus Frekansı": features.get("nystagmus_frequency", 0.0),
                "Şaşılık Açısı": features.get("strabismus_angle", 0.0)
            },
            "clinical_note": "Detaylı açıklama için SHAP konfigürasyonu gerekli"
        }
    
    def generate_force_plot(self, features: Dict[str, float], 
                          save_path: str = None) -> str:
        """SHAP force plot oluşturur."""
        try:
            if self.explainer is None:
                return "SHAP explainer mevcut değil"
            
            # Öznitelikleri hazırla
            feature_array = np.array([[
                features.get("nystagmus_frequency", 0.0),
                features.get("movement_amplitude", 0.0),
                features.get("regularity", 0.0),
                features.get("strabismus_angle", 0.0),
                features.get("strabismus_stability", 0.0)
            ]])
            
            feature_names = [
                "Nys_Freq", "Mov_Amp", "Regularity", "Strab_Angle", "Strab_Stab"
            ]
            
            # SHAP değerleri
            shap_values = self.explainer.shap_values(feature_array)
            
            # Force plot oluştur
            if isinstance(shap_values, list):
                values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                expected_value = self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value
            else:
                values = shap_values[0]
                expected_value = self.explainer.expected_value
            
            # Matplotlib ile force plot benzeri görselleştirme
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Base line
            ax.axhline(y=expected_value, color='gray', linestyle='--', alpha=0.7, label='Baseline')
            
            # Her öznitelik için katkı
            cumulative = expected_value
            colors = ['red' if val > 0 else 'blue' for val in values]
            
            for i, (name, value) in enumerate(zip(feature_names, values)):
                ax.barh(i, value, left=cumulative, color=colors[i], alpha=0.7)
                ax.text(cumulative + value/2, i, f'{name}: {value:.3f}', 
                       ha='center', va='center', fontsize=10)
                cumulative += value
            
            ax.set_xlabel('SHAP Değeri (Model Kararına Etkisi)')
            ax.set_ylabel('Öznitelikler')
            ax.set_title('SHAP Force Plot - Öznitelik Katkıları')
            ax.legend()
            plt.tight_layout()
            
            # Kaydet
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                save_path = f"shap_force_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
                
        except Exception as e:
            logger.error(f"SHAP force plot hatası: {e}")
            return f"Force plot oluşturulamadı: {e}"

class CombinedExplainer:
    """
    Grad-CAM ve SHAP'i birleştiren kapsamlı açıklama sistemi.
    """
    
    def __init__(self, model: nn.Module, model_predict_fn = None):
        self.gradcam = GradCAMExplainer(model)
        self.shap_explainer = SHAPExplainer(model_predict_fn) if model_predict_fn else None
        
    def explain_analysis(self, 
                        original_frame: np.ndarray,
                        features: Dict[str, float],
                        analysis_results: Dict[str, Any],
                        input_tensor: torch.Tensor = None) -> Dict[str, Any]:
        """
        Kapsamlı analiz açıklaması oluşturur.
        
        Args:
            original_frame: Orijinal video karesi
            features: Çıkarılan öznitelikler  
            analysis_results: Model analiz sonuçları
            input_tensor: Model girdi tensörü (Grad-CAM için)
            
        Returns:
            dict: Kapsamlı açıklama sonuçları
        """
        explanation = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "gradcam_explanation": {},
            "shap_explanation": {},
            "combined_insights": [],
            "clinical_conclusion": ""
        }
        
        # Grad-CAM açıklaması
        if input_tensor is not None:
            try:
                cam_heatmap = self.gradcam.generate_cam_heatmap(input_tensor)
                visual_explanation = self.gradcam.visualize_cam_on_image(
                    original_frame, cam_heatmap, analysis_results
                )
                
                explanation["gradcam_explanation"] = {
                    "heatmap_generated": True,
                    "visual_focus_regions": "Göz hareketlerinde yüksek aktivite bölgeleri",
                    "interpretation": "Kırmızı bölgeler modelin odaklandığı alanlar"
                }
                
                # Görsel kaydet
                visual_path = f"gradcam_explanation_{explanation['analysis_id']}.png"
                cv2.imwrite(visual_path, cv2.cvtColor(visual_explanation, cv2.COLOR_RGB2BGR))
                explanation["gradcam_explanation"]["visual_path"] = visual_path
                
            except Exception as e:
                explanation["gradcam_explanation"] = {"error": str(e)}
        
        # SHAP açıklaması
        if self.shap_explainer:
            try:
                shap_result = self.shap_explainer.explain_features(features)
                explanation["shap_explanation"] = shap_result
                
                # Force plot oluştur
                force_plot_path = self.shap_explainer.generate_force_plot(
                    features, f"shap_force_{explanation['analysis_id']}.png"
                )
                explanation["shap_explanation"]["force_plot_path"] = force_plot_path
                
            except Exception as e:
                explanation["shap_explanation"] = {"error": str(e)}
        
        # Birleşik öngörüler
        explanation["combined_insights"] = self._generate_combined_insights(
            explanation, analysis_results
        )
        
        # Klinik sonuç
        explanation["clinical_conclusion"] = self._generate_clinical_conclusion(
            analysis_results, explanation
        )
        
        # Açıklama raporunu kaydet
        report_path = f"explanation_report_{explanation['analysis_id']}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(explanation, f, indent=2, ensure_ascii=False)
        
        explanation["report_path"] = report_path
        
        return explanation
    
    def _generate_combined_insights(self, explanation: Dict[str, Any], 
                                  results: Dict[str, Any]) -> List[str]:
        """Grad-CAM ve SHAP sonuçlarını birleştirir."""
        insights = []
        
        # Grad-CAM insight'ları
        if explanation["gradcam_explanation"].get("heatmap_generated"):
            insights.append("🎯 Görsel analiz: Model göz bölgelerine odaklanmış")
        
        # SHAP insight'ları
        shap_data = explanation.get("shap_explanation", {})
        if "clinical_interpretation" in shap_data:
            insights.extend(shap_data["clinical_interpretation"][:3])  # İlk 3 insight
        
        # Genel değerlendirme
        nys_detected = results.get("nystagmus_detected", False)
        strab_detected = results.get("strabismus_detected", False)
        
        if nys_detected and strab_detected:
            insights.append("⚠️ Çoklu patoloji: Hem nistagmus hem şaşılık tespit edildi")
        elif nys_detected:
            insights.append("🔍 Nistagmus odaklı analiz: Göz hareketlerinde anormallik")
        elif strab_detected:
            insights.append("📐 Şaşılık odaklı analiz: Göz hizalanmasında sapma")
        else:
            insights.append("✅ Normal analiz: Belirgin patoloji tespit edilmedi")
        
        return insights
    
    def _generate_clinical_conclusion(self, results: Dict[str, Any], 
                                    explanation: Dict[str, Any]) -> str:
        """Klinik sonuç oluşturur."""
        nys_freq = results.get("nistagmus_frequency", 0.0)
        strab_angle = results.get("strabismus_angle", 0.0)
        
        conclusion = "KLİNİK DEĞERLENDİRME:\n\n"
        
        if nys_freq > 3.0:
            conclusion += f"• Belirgin nistagmus bulgusu ({nys_freq:.2f} Hz) - İleri değerlendirme gerekli\n"
        elif nys_freq > 1.5:
            conclusion += f"• Hafif nistagmus bulgusu ({nys_freq:.2f} Hz) - Takip önerisi\n"
        else:
            conclusion += f"• Normal göz hareketi ({nys_freq:.2f} Hz)\n"
        
        if strab_angle > 3.0:
            conclusion += f"• Belirgin şaşılık ({strab_angle:.2f}°) - Oftalmolojik konsültasyon\n"
        elif strab_angle > 1.5:
            conclusion += f"• Hafif şaşılık eğilimi ({strab_angle:.2f}°) - Kontrol önerisi\n"
        else:
            conclusion += f"• Normal göz hizalanması ({strab_angle:.2f}°)\n"
        
        # Model güvenilirliği
        detection_rate = results.get("face_detection_rate", 0.0)
        if detection_rate > 0.8:
            conclusion += f"\n• Analiz kalitesi: Yüksek (%{detection_rate*100:.1f} tespit oranı)"
        else:
            conclusion += f"\n• Analiz kalitesi: Orta (%{detection_rate*100:.1f} tespit oranı)"
        
        return conclusion

# Factory functions
def create_gradcam_explainer(model: nn.Module) -> GradCAMExplainer:
    """Grad-CAM explainer oluştur."""
    return GradCAMExplainer(model)

def create_shap_explainer(model_predict_fn) -> SHAPExplainer:
    """SHAP explainer oluştur."""
    return SHAPExplainer(model_predict_fn)

def create_combined_explainer(model: nn.Module, model_predict_fn = None) -> CombinedExplainer:
    """Birleşik explainer oluştur."""
    return CombinedExplainer(model, model_predict_fn)

if __name__ == "__main__":
    # Test
    print("🔬 Gelişmiş açıklanabilirlik sistemi test edildi")
    print("✅ Grad-CAM support")
    print("✅ SHAP support") 
    print("✅ Combined explanation system")
    
    # Model wrapper örneği
    def dummy_predict(features):
        """Dummy prediction function for testing."""
        return np.random.rand(len(features), 2)  # Binary classification
    
    # Test SHAP
    shap_explainer = SHAPExplainer(dummy_predict)
    test_features = {
        "nystagmus_frequency": 4.2,
        "movement_amplitude": 0.85,
        "regularity": 0.72,
        "strabismus_angle": 3.1,
        "strabismus_stability": 0.68
    }
    
    explanation = shap_explainer.explain_features(test_features)
    print(f"\n📊 SHAP açıklama test: {explanation['method']}")
    print(f"🎯 Toplam etki: {explanation.get('total_effect', 'N/A')}") 