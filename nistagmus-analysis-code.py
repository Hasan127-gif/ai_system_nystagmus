# main.py - Ana uygulama dosyası
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import logging
import threading
import queue
from collections import deque
import os
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt
import json
import tkinter as tk
from tkinter import messagebox

# Logger kurulumu
logging.basicConfig(filename='eye_analysis.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('eye_tracker')

# Göz İzleme Stratejileri
class EyeTrackingStrategy:
    """Göz takip stratejisi için temel sınıf"""
    
    def detect_face_landmarks(self, frame):
        """Yüz ve göz işaretlerini tespit et"""
        raise NotImplementedError("Alt sınıflar bu metodu uygulamalıdır")
    
    def release(self):
        """Kaynakları serbest bırak"""
        pass

class MediaPipeStrategy(EyeTrackingStrategy):
    """MediaPipe tabanlı göz takip stratejisi"""
    
    def __init__(self, max_num_faces=1, refine_landmarks=True, 
                min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """MediaPipe Face Mesh başlat"""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces, 
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        logger.info("MediaPipe yüz işaretleme sistemi başlatıldı")
    
    def detect_face_landmarks(self, frame):
        """MediaPipe kullanarak yüz noktalarını tespit et"""
        if frame is None:
            return None
            
        # BGR -> RGB dönüşümü
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results
    
    def release(self):
        """MediaPipe kaynaklarını serbest bırak"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

class DlibStrategy(EyeTrackingStrategy):
    """Dlib tabanlı göz takip stratejisi"""
    
    def __init__(self, predictor_path="shape_predictor_68_face_landmarks.dat"):
        """
        Args:
            predictor_path: Dlib yüz işaretleme modeli dosya yolu
        """
        try:
            # Dlib dedektör ve prediktörü yükle
            import dlib
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            
            # Göz indeksleri (Dlib 68 işaret modeli)
            # Sol göz: 36-41, Sağ göz: 42-47
            self.left_eye_indices = list(range(36, 42))
            self.right_eye_indices = list(range(42, 48))
            
            logger.info("Dlib yüz işaretleme sistemi başlatıldı")
        except ImportError:
            logger.error("Dlib kütüphanesi yüklenemedi. 'pip install dlib' komutu ile yükleyebilirsiniz.")
            raise
        except Exception as e:
            logger.error(f"Dlib başlatma hatası: {str(e)}")
            raise
    
    def detect_face_landmarks(self, frame):
        """Dlib kullanarak yüz işaretlerini tespit et"""
        if frame is None:
            return None
            
        # Gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Yüzleri tespit et
        faces = self.detector(gray)
        if not faces:
            return None
        
        # MediaPipe sonuçlarına benzer bir yapı oluştur
        results = type('', (), {})()
        results.multi_face_landmarks = []
        
        # İlk yüzün işaretlerini al
        dlib_landmarks = self.predictor(gray, faces[0])
        
        # Dlib sonuçlarını MediaPipe formatına dönüştür
        face_landmarks = self._convert_to_mediapipe_format(dlib_landmarks, frame.shape)
        results.multi_face_landmarks.append(face_landmarks)
        
        return results
        
    def _convert_to_mediapipe_format(self, dlib_landmarks, frame_shape):
        """Dlib landmark sonuçlarını MediaPipe formatına dönüştür"""
        # MediaPipe landmark sınıfına benzer bir sınıf oluştur
        landmark_class = type('LandmarkClass', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})
        
        # Boş bir landmark listesi oluştur (MediaPipe 468 landmark)
        face_landmarks = type('', (), {})()
        face_landmarks.landmark = [landmark_class() for _ in range(468)]
        
        # Dlib'in 68 noktasını MediaPipe formatına dönüştür
        h, w = frame_shape[:2]
        for i in range(68):
            x = dlib_landmarks.part(i).x / w
            y = dlib_landmarks.part(i).y / h
            face_landmarks.landmark[i].x = x
            face_landmarks.landmark[i].y = y
            face_landmarks.landmark[i].z = 0.0  # Z değeri 0 olarak ayarla
        
        # Dlib'de olmayan göz işaretlerini tahmin et
        if len(self.left_eye_indices) == 6:
            # Sol göz merkezi (iris) - MediaPipe indeksi 468
            left_eye_xs = [dlib_landmarks.part(i).x for i in self.left_eye_indices]
            left_eye_ys = [dlib_landmarks.part(i).y for i in self.left_eye_indices]
            left_eye_center_x = sum(left_eye_xs) / len(left_eye_xs) / w
            left_eye_center_y = sum(left_eye_ys) / len(left_eye_ys) / h
            
            # Sol göz merkezi - iris (indeks 468)
            if len(face_landmarks.landmark) > 468:
                face_landmarks.landmark[468].x = left_eye_center_x
                face_landmarks.landmark[468].y = left_eye_center_y
                face_landmarks.landmark[468].z = 0.0
            
            # Sağ göz merkezi (iris) - MediaPipe indeksi 473
            right_eye_xs = [dlib_landmarks.part(i).x for i in self.right_eye_indices]
            right_eye_ys = [dlib_landmarks.part(i).y for i in self.right_eye_indices]
            right_eye_center_x = sum(right_eye_xs) / len(right_eye_xs) / w
            right_eye_center_y = sum(right_eye_ys) / len(right_eye_ys) / h
            
            # Sağ göz merkezi - iris (indeks 473)
            if len(face_landmarks.landmark) > 473:
                face_landmarks.landmark[473].x = right_eye_center_x
                face_landmarks.landmark[473].y = right_eye_center_y
                face_landmarks.landmark[473].z = 0.0
        
        return face_landmarks

class OpenCVDNNStrategy(EyeTrackingStrategy):
    """OpenCV DNN tabanlı yüz ve göz işaretleri tespiti"""
    
    def __init__(self, face_model_path="models/opencv_face_detector_uint8.pb", 
                 face_config_path="models/opencv_face_detector.pbtxt"):
        """
        Args:
            face_model_path: Yüz tespiti model dosyası
            face_config_path: Yüz tespiti config dosyası
        """
        try:
            # OpenCV DNN modelini yükle
            self.face_net = cv2.dnn.readNet(face_model_path, face_config_path)
            
            # CUDA kullanılabilirse GPU'ya taşı
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                logger.info("OpenCV DNN CUDA başlatıldı")
            
            # Göz detektörü (Haar cascade)
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            logger.info("OpenCV DNN yüz tespit sistemi başlatıldı")
        except Exception as e:
            logger.error(f"OpenCV DNN başlatma hatası: {str(e)}")
            raise
    
    def detect_face_landmarks(self, frame):
        """OpenCV DNN kullanarak yüz ve göz tespiti yap"""
        if frame is None:
            return None
        
        h, w = frame.shape[:2]
        
        # Yüz tespiti
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        # MediaPipe sonuçlarına benzer bir yapı oluştur
        results = type('', (), {})()
        results.multi_face_landmarks = []
        
        # Tespit edilen ilk yüzü al
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Güven eşiği
                # Yüz sınırlayıcı kutu
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                
                # Yüz bölgesini al
                face_region = frame[y1:y2, x1:x2]
                if face_region.size == 0:
                    continue
                
                # Gri tonlamaya dönüştür
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Göz tespiti
                eyes = self.eye_cascade.detectMultiScale(gray_face)
                
                # Boş landmark listesi oluştur
                landmark_class = type('LandmarkClass', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})
                face_landmarks = type('', (), {})()
                face_landmarks.landmark = [landmark_class() for _ in range(478)]  # MediaPipe formatı
                
                # Yüz merkezi
                face_center_x = (x1 + x2) / 2 / w
                face_center_y = (y1 + y2) / 2 / h
                
                # Göz merkezlerini hesapla
                if len(eyes) >= 2:
                    # Her göz için
                    eye_centers = []
                    for (ex, ey, ew, eh) in eyes:
                        # Göz merkezi
                        eye_center_x = (x1 + ex + ew // 2) / w
                        eye_center_y = (y1 + ey + eh // 2) / h
                        eye_centers.append((eye_center_x, eye_center_y))
                    
                    # En sol ve en sağdaki gözleri seç
                    eye_centers.sort(key=lambda p: p[0])
                    if len(eye_centers) >= 2:
                        # Sol göz (indeks 468)
                        face_landmarks.landmark[468].x = eye_centers[0][0]
                        face_landmarks.landmark[468].y = eye_centers[0][1]
                        
                        # Sağ göz (indeks 473)
                        face_landmarks.landmark[473].x = eye_centers[1][0]
                        face_landmarks.landmark[473].y = eye_centers[1][1]
                
                results.multi_face_landmarks.append(face_landmarks)
                break  # Sadece ilk yüzü al
        
        return results

class HybridStrategy(EyeTrackingStrategy):
    """Birden fazla stratejiyi birleştiren hibrit yaklaşım"""
    
    def __init__(self, strategies, voting_method="confidence"):
        """
        Args:
            strategies: Kullanılacak strateji listesi
            voting_method: Oylama yöntemi ("confidence", "average")
        """
        self.strategies = strategies
        self.voting_method = voting_method
        logger.info(f"Hibrit göz takip stratejisi başlatıldı: {voting_method}")
    
    def detect_face_landmarks(self, frame):
        """Tüm stratejileri kullanarak yüz ve göz tespiti yap"""
        if frame is None:
            return None
        
        # Tüm stratejilerden sonuç al
        all_results = []
        for strategy in self.strategies:
            try:
                result = strategy.detect_face_landmarks(frame)
                if result is not None and hasattr(result, 'multi_face_landmarks') and result.multi_face_landmarks:
                    all_results.append(result)
            except Exception as e:
                logger.warning(f"Strateji hatası: {str(e)}")
        
        # Sonuç yoksa None döndür
        if not all_results:
            return None
        
        # Tek sonuç varsa onu döndür
        if len(all_results) == 1:
            return all_results[0]
        
        # Birden fazla sonuç için oylama
        if self.voting_method == "confidence":
            # En güvenilir sonucu döndür (şimdilik ilk sonuç)
            return all_results[0]
        else:  # "average"
            # Göz pozisyonları için ortalama al
            return self._average_landmarks(all_results)
    
    def _average_landmarks(self, all_results):
        """Farklı stratejilerden gelen sonuçları ortala"""
        # Ortalaması alınacak göz indeksleri
        eye_indices = [468, 473]  # MediaPipe sol ve sağ iris merkezi
        
        # Sonuç nesnesi oluştur
        avg_result = type('', (), {})()
        avg_result.multi_face_landmarks = []
        
        # İlk sonuçtan bir yüz al
        if all_results and all_results[0].multi_face_landmarks:
            face_landmarks = all_results[0].multi_face_landmarks[0]
            
            # Her göz indeksi için tüm sonuçların ortalamasını al
            for eye_idx in eye_indices:
                x_sum, y_sum, count = 0, 0, 0
                
                for result in all_results:
                    if (result.multi_face_landmarks and 
                        len(result.multi_face_landmarks) > 0 and 
                        len(result.multi_face_landmarks[0].landmark) > eye_idx):
                        
                        lm = result.multi_face_landmarks[0].landmark[eye_idx]
                        x_sum += lm.x
                        y_sum += lm.y
                        count += 1
                
                if count > 0:
                    face_landmarks.landmark[eye_idx].x = x_sum / count
                    face_landmarks.landmark[eye_idx].y = y_sum / count
            
            avg_result.multi_face_landmarks.append(face_landmarks)
        
        return avg_result
    
    def release(self):
        """Tüm stratejilerin kaynaklarını serbest bırak"""
        for strategy in self.strategies:
            strategy.release()

class PreprocessorFactory:
    """Ön işlemci fabrikası"""
    
    @staticmethod
    def create_strategy(strategy_type, config):
        """Strateji tipine göre uygun strateji nesnesini oluştur"""
        if strategy_type == "mediapipe":
            return MediaPipeStrategy(
                max_num_faces=config.get("max_num_faces", 1),
                refine_landmarks=config.get("refine_landmarks", True),
                min_detection_confidence=config.get("face_detection_confidence", 0.5),
                min_tracking_confidence=config.get("face_tracking_confidence", 0.5)
            )
        elif strategy_type == "dlib":
            return DlibStrategy(
                predictor_path=config.get("dlib_predictor_path", "shape_predictor_68_face_landmarks.dat")
            )
        elif strategy_type == "opencv_dnn":
            return OpenCVDNNStrategy(
                face_model_path=config.get("opencv_face_model", "models/opencv_face_detector_uint8.pb"),
                face_config_path=config.get("opencv_face_config", "models/opencv_face_detector.pbtxt")
            )
        elif strategy_type == "hybrid":
            # Alt stratejileri oluştur
            sub_strategies = []
            for sub_type in ["mediapipe", "dlib"]:
                try:
                    sub_strategy = PreprocessorFactory.create_strategy(sub_type, config)
                    sub_strategies.append(sub_strategy)
                except Exception as e:
                    logger.warning(f"{sub_type} stratejisi oluşturulamadı: {str(e)}")
            
            if not sub_strategies:
                raise ValueError("Hibrit strateji için en az bir alt strateji gerekli")
                
            return HybridStrategy(
                strategies=sub_strategies,
                voting_method=config.get("hybrid_voting_method", "confidence")
            )
        else:
            raise ValueError(f"Bilinmeyen strateji tipi: {strategy_type}")

class ONNXModelLoader:
    """ONNX model yükleme ve çıkarım işlemleri"""
    
    def __init__(self, model_path, use_gpu=True):
        """
        Args:
            model_path: ONNX model dosya yolu
            use_gpu: GPU kullanılsın mı
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.input_name = None
        self.output_name = None
    
    def load_model(self):
        """ONNX modelini yükle"""
        try:
            import onnxruntime as ort
            
            # GPU kullanımı
            providers = ['CPUExecutionProvider']
            if self.use_gpu:
                providers = ['CUDAExecutionProvider'] + providers
            
            # Modeli yükle
            self.model = ort.InferenceSession(self.model_path, providers=providers)
            
            # Giriş ve çıkış isimlerini al
            self.input_name = self.model.get_inputs()[0].name
            self.output_name = self.model.get_outputs()[0].name
            
            logger.info(f"ONNX model yüklendi: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"ONNX model yükleme hatası: {str(e)}")
            return False
    
    def run_inference(self, input_data):
        """Model çıkarımı yap"""
        if self.model is None:
            logger.warning("ONNX model yüklenmemiş")
            return None
        
        try:
            # Çıkarım yap
            outputs = self.model.run(None, {self.input_name: input_data})
            return outputs
        except Exception as e:
            logger.error(f"ONNX çıkarım hatası: {str(e)}")
            return None

class DeepEyeTracker:
    """Derin öğrenme tabanlı göz izleme"""
    
    def __init__(self, model_path="models/deep_eye_model.onnx", 
                 feature_extractor_path="models/feature_extractor.pt", 
                 use_gpu=True):
        """
        Args:
            model_path: ONNX model dosya yolu
            feature_extractor_path: Özellik çıkarıcı model dosya yolu
            use_gpu: GPU kullanılsın mı
        """
        # Model yükleyici
        self.model_loader = ONNXModelLoader(model_path, use_gpu=use_gpu)
        self.model_loader.load_model()
        
        # Özellik ekstraktörü
        try:
            self.feature_extractor = cv2.dnn.readNetFromTorch(feature_extractor_path)
            logger.info(f"Özellik çıkarıcı yüklendi: {feature_extractor_path}")
        except Exception as e:
            logger.error(f"Özellik çıkarıcı yükleme hatası: {str(e)}")
            self.feature_extractor = None
        
        # Transfer öğrenme için ince ayar yapmaya izin veren parametreler
        self.adaptation_data = []  # Kullanıcıya özgü adaptasyon için
        self.is_adapted = False
        
        # Göz durumu için sınıflandırma etiketleri
        self.classes = {
            0: "Normal",
            1: "Nistagmus",
            2: "Strabismus",
            3: "Anisocoria",  # Pupil boyutu eşitsizliği
            4: "Ptosis"       # Göz kapağı düşüklüğü
        }
        
        logger.info("Derin göz izleme başlatıldı")
    
    def extract_features(self, eye_image):
        """Göz görüntüsünden öznitelikler çıkar"""
        if self.feature_extractor is None or eye_image is None:
            return None
            
        try:
            # Yeniden boyutlandır
            eye_image = cv2.resize(eye_image, (224, 224))
            
            # Normalize et
            blob = cv2.dnn.blobFromImage(
                eye_image, 1.0/255, (224, 224), 
                mean=[0.485, 0.456, 0.406], 
                swapRB=True, crop=False
            )
            
            # Öznitelikleri çıkar
            self.feature_extractor.setInput(blob)
            features = self.feature_extractor.forward()
            
            return features
        except Exception as e:
            logger.error(f"Öznitelik çıkarma hatası: {str(e)}")
            return None
    
    def detect_eye_conditions(self, eye_image):
        """Göz durumlarını tespit et"""
        if self.model_loader.model is None or eye_image is None:
            return None
        
        try:
            # Görüntüyü ön işle
            img = cv2.resize(eye_image, (224, 224))
            img = img / 255.0  # Normalize et
            
            # Kanal düzenini değiştir (HWC -> CHW)
            img = img.transpose(2, 0, 1)
            
            # Batch boyutu ekle
            input_data = np.expand_dims(img.astype(np.float32), axis=0)
            
            # Modeli çalıştır
            outputs = self.model_loader.run_inference(input_data)
            
            if outputs is None:
                return None
            
            # Sınıflandırma sonuçlarını işle
            probs = outputs[0][0]
            class_id = np.argmax(probs)
            confidence = probs[class_id]
            
            return {
                "class_id": int(class_id),
                "class_name": self.classes.get(int(class_id), "Unknown"),
                "confidence": float(confidence),
                "probabilities": {self.classes.get(i, f"Class_{i}"): float(p) for i, p in enumerate(probs)}
            }
            
        except Exception as e:
            logger.error(f"Göz durumu tespit hatası: {str(e)}")
            return None
    
    def adapt_to_user(self, eye_images, labels):
        """Kullanıcıya özgü adaptasyon yap (Transfer öğrenme)"""
        if len(eye_images) != len(labels):
            logger.error("Görüntü ve etiket sayıları eşleşmiyor")
            return False
        
        try:
            # Özellikleri çıkar
            features = []
            for img in eye_images:
                feature = self.extract_features(img)
                if feature is not None:
                    features.append(feature)
            
            if len(features) == 0:
                logger.warning("Öznitelik çıkarılamadı")
                return False
            
            # Adaptasyon verisine ekle
            for f, label in zip(features, labels):
                self.adaptation_data.append((f, label))
                
            # Basit bir transfer öğrenme uygulaması
            if len(self.adaptation_data) >= 10:  # En az 10 örnek gerekli
                # Burada transfer öğrenme / model ince ayarı yapılabilir
                # NOT: Gerçek uygulamada burada PyTorch veya scikit-learn ile 
                # basit bir sınıflandırıcı eğitilir
                
                self.is_adapted = True
                logger.info(f"Model kullanıcıya adapte edildi ({len(self.adaptation_data)} örnek)")
                
            return self.is_adapted
                
        except Exception as e:
            logger.error(f"Adaptasyon hatası: {str(e)}")
            return False
    
    def analyze_eye_movement(self, eye_positions, timestamps):
        """Göz hareketi analizi yap"""
        if len(eye_positions) < 10 or len(timestamps) < 10:
            return None
            
        try:
            # Göz pozisyonlarını numpy dizisine dönüştür
            positions = np.array(eye_positions)
            times = np.array(timestamps)
            
            # Hızları hesapla
            velocities = np.gradient(positions, times)
            
            # İvmeyi hesapla
            accelerations = np.gradient(velocities, times)
            
            # Temel istatistikler
            mean_vel = np.mean(np.abs(velocities))
            std_vel = np.std(velocities)
            max_vel = np.max(np.abs(velocities))
            
            # Sinüs dalgası parametreleri (nistagmus için)
            from scipy.optimize import curve_fit
            
            def sine_wave(x, amp, omega, phase, offset):
                return amp * np.sin(omega * x + phase) + offset
            
            # Zaman dizisini 0'dan başlat
            t_normalized = times - times[0]
            
            try:
                # Sine dalgası uydur
                params, _ = curve_fit(sine_wave, t_normalized, positions, 
                                    p0=[10, 2*np.pi*0.5, 0, np.mean(positions)],
                                    bounds=([0, 0, -np.pi, -np.inf], 
                                            [100, 2*np.pi*10, np.pi, np.inf]))
                
                amp, omega, phase, offset = params
                frequency = omega / (2 * np.pi)  # Hz cinsinden frekans
                
                # Uyum iyiliği
                predictions = sine_wave(t_normalized, *params)
                residuals = positions - predictions
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((positions - np.mean(positions))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Sinüs dalgası güvenilirliği (R^2 > 0.4 ise nistagmus olabilir)
                is_sine_reliable = r_squared > 0.4
                
            except Exception as e:
                logger.warning(f"Sinüs uydurma hatası: {str(e)}")
                frequency = 0
                r_squared = 0
                is_sine_reliable = False
            
            # Sonuçları döndür
            return {
                "mean_velocity": float(mean_vel),
                "std_velocity": float(std_vel),
                "max_velocity": float(max_vel),
                "frequency": float(frequency),
                "r_squared": float(r_squared),
                "is_sine_reliable": bool(is_sine_reliable)
            }
            
        except Exception as e:
            logger.error(f"Göz hareketi analizi hatası: {str(e)}")
            return None

class EyeAnalysisSystem:
    def __init__(self, config_path="config.json"):
        """Göz analiz sistemini başlat ve yapılandır"""
        try:
            # Yapılandırma dosyasını yükle (yoksa varsayılan değerleri kullan)
            self.config = self.load_config(config_path)
            
            # Göz izleme stratejisini oluştur
            self.tracking_strategy = self.create_tracking_strategy()
            
            # ONNX model yükleme (hata durumunda program durmaz)
            self.onnx_model = None
            if self.config["use_onnx_model"]:
                self.load_onnx_model()
                
            # Kamera başlatma
            self.init_camera()
            
            # Buffer ve veri yapıları
            self.init_data_structures()
            
            # CSV dosyası oluştur
            self.create_csv_output()
            
            # Kalibrasyon durumu
            self.is_calibrated = False
            self.calibration_points = []
            self.pixel_to_degree_ratio = self.config["default_pixel_to_degree_ratio"]
            
            # Derin göz izleme
            self.deep_eye_tracker = DeepEyeTracker(
                model_path=self.config.get("deep_eye_model", "models/deep_eye_model.onnx"),
                feature_extractor_path=self.config.get("feature_extractor", "models/feature_extractor.pt"),
                use_gpu=self.config.get("use_gpu", True)
            )
            
            logger.info("Göz analiz sistemi başlatıldı.")
        except Exception as e:
            logger.error(f"Başlatma hatası: {str(e)}")
            raise
    
    def create_tracking_strategy(self):
        """Göz izleme stratejisini oluştur"""
        strategy_type = self.config.get("tracking_strategy", "mediapipe")
        try:
            return PreprocessorFactory.create_strategy(strategy_type, self.config)
        except Exception as e:
            logger.error(f"Strateji oluşturma hatası: {str(e)}, MediaPipe kullanılıyor.")
            # Hata durumunda MediaPipe kullan
            return MediaPipeStrategy(
                max_num_faces=self.config.get("max_num_faces", 1),
                refine_landmarks=self.config.get("refine_landmarks", True),
                min_detection_confidence=self.config.get("face_detection_confidence", 0.5),
                min_tracking_confidence=self.config.get("face_tracking_confidence", 0.5)
            )
    
    def load_config(self, config_path):
        """Yapılandırma dosyasını yükle veya varsayılan değerleri kullan"""
        default_config = {
            "use_onnx_model": False,
            "onnx_model_path": "eye_model.onnx",
            "camera_id": 0,
            "frame_width": 640,
            "frame_height": 480,
            "face_detection_confidence": 0.5,
            "face_tracking_confidence": 0.5,
            "buffer_seconds": 3.0,
            "nistagmus_freq_threshold": 0.5,
            "spv_threshold": 30.0,
            "slow_phase_velocity_threshold": 20.0,
            "default_pixel_to_degree_ratio": 0.1,
            "smoothing_window": 5,
            "bandpass_freq_low": 0.5,
            "bandpass_freq_high": 15.0,
            "peak_distance": 5,
            "strabismus_degree_factor": 90.0,
            # Yeni eklenen yapılandırmalar
            "tracking_strategy": "mediapipe",  # mediapipe, dlib, opencv_dnn, hybrid
            "handle_glasses": True,
            "handle_lighting": True,
            "handle_eye_color": True,
            "handle_strabismus": True,
            "hybrid_voting_method": "confidence",
            "dlib_predictor_path": "shape_predictor_68_face_landmarks.dat",
            "max_num_faces": 1,
            "refine_landmarks": True
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Varsayılan değerleri yüklenen değerlerle güncelle
                    config = {**default_config, **loaded_config}
                    logger.info("Yapılandırma dosyası yüklendi.")
                    return config
            else:
                logger.warning(f"Yapılandırma dosyası bulunamadı: {config_path}. Varsayılan değerler kullanılıyor.")
                # Yapılandırma dosyasını oluştur
                with open(config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                return default_config
        except Exception as e:
            logger.error(f"Yapılandırma yükleme hatası: {str(e)}")
            return default_config
    
    def load_onnx_model(self):
        """ONNX modelini yükle"""
        try:
            import onnxruntime as ort
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_model = ort.InferenceSession(self.config["onnx_model_path"], providers=providers)
            logger.info(f"ONNX modeli yüklendi: {self.config['onnx_model_path']}")
        except Exception as e:
            logger.error(f"ONNX model yükleme hatası: {str(e)}")
            self.onnx_model = None
    
    def init_camera(self):
        """Kamera başlatma ve özellikleri ayarlama"""
        try:
            self.cap = cv2.VideoCapture(self.config["camera_id"])
            
            # Kamera çözünürlüğünü ayarla
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["frame_width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["frame_height"])
            
            # FPS bilgisini al (alınamazsa 30 varsay)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
                logger.warning("Kamera FPS bilgisi alınamadı. Varsayılan: 30 FPS")
            
            # Kamera açılıp açılmadığını kontrol et
            if not self.cap.isOpened():
                raise Exception("Kamera açılamadı!")
                
            logger.info(f"Kamera başlatıldı. ID: {self.config['camera_id']}, FPS: {self.fps}")
        except Exception as e:
            logger.error(f"Kamera başlatma hatası: {str(e)}")
            raise
    
    def init_data_structures(self):
        """Veri yapılarını başlat"""
        # Buffer boyutunu belirle (FPS x saniye)
        buffer_size = int(self.fps * self.config["buffer_seconds"])
        
        # Göz pozisyonu tamponları
        self.left_x_buffer = deque(maxlen=buffer_size)
        self.left_y_buffer = deque(maxlen=buffer_size)
        self.right_x_buffer = deque(maxlen=buffer_size)
        self.right_y_buffer = deque(maxlen=buffer_size)
        
        # Zaman damgaları
        self.timestamps = deque(maxlen=buffer_size)
        
        # Analiz sonuçları
        self.last_results = {
            "nistagmus_freq": 0.0,
            "spv": 0.0,
            "strabismus_angle": 0.0,
            "is_nistagmus_detected": False
        }
        
        # Frame kuyrugu
        self.frame_queue = queue.Queue(maxsize=10)
        
        logger.info(f"Veri yapıları başlatıldı. Buffer boyutu: {buffer_size} frame")
    
    def create_csv_output(self):
        """CSV çıktı dosyasını oluştur"""
        try:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.csv_filename = f"eye_data_{timestamp}.csv"
            self.csv_file = open(self.csv_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Başlık satırını yaz
            headers = ["Timestamp", "Left_X", "Left_Y", "Right_X", "Right_Y", 
                      "Nistagmus_Freq_Hz", "SPV", "Strabismus_Angle", "Is_Nistagmus"]
            self.csv_writer.writerow(headers)
            
            logger.info(f"CSV çıktı dosyası oluşturuldu: {self.csv_filename}")
        except Exception as e:
            logger.error(f"CSV dosyası oluşturma hatası: {str(e)}")
            self.csv_file = None
            self.csv_writer = None
    
    def start_capture_thread(self):
        """Frame yakalama thread'ini başlat"""
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
        self.capture_thread.start()
        logger.info("Kare yakalama thread'i başlatıldı")
    
    def capture_frames(self):
        """Kamera görüntülerini yakala ve kuyruğa ekle"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Kamera görüntüsü alınamadı")
                    time.sleep(0.01)  # CPU kullanımını azalt
                    continue
                
                # Kuyruğu güncelle (doluysa eski frame'i çıkar)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
            except Exception as e:
                logger.error(f"Frame yakalama hatası: {str(e)}")
                time.sleep(0.1)  # Hata durumunda bekle
    
    def preprocess_frame(self, frame):
        """Frame'i MediaPipe için hazırla"""
        # BGR -> RGB dönüşümü
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def detect_face_landmarks(self, frame):
        """Yüz noktalarını tespit et"""
        if self.config.get("handle_lighting", False):
            frame = self._adapt_to_lighting(frame)
            
        results = self.tracking_strategy.detect_face_landmarks(frame)
        
        # Sonuçları işle - gözlük, göz rengi, şaşılık gibi özel durumları tespit et
        if results and results.multi_face_landmarks:
            # Gözlük tespiti ve işleme
            if self.config.get("handle_glasses", False) and hasattr(self, 'detect_glasses'):
                has_glasses = self.detect_glasses(results.multi_face_landmarks[0], frame)
                if has_glasses:
                    # Gözlük için ek işleme
                    frame = self.preprocess_glasses(frame)
                    # Yeniden tespit
                    results = self.tracking_strategy.detect_face_landmarks(frame)
        
        return results
    
    def _adapt_to_lighting(self, frame):
        """Aydınlatma koşullarına adapte ol"""
        try:
            # Ortalama parlaklık
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            
            # Düşük ışık koşulları
            if brightness < 100:
                # Gamma düzeltme
                gamma = 1.5
                lookup_table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
                frame = cv2.LUT(frame, lookup_table)
            
            # Yüksek aydınlatma (parlaklık)
            elif brightness > 200:
                # Kontrastı azalt
                frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=0)
            
            return frame
        except Exception as e:
            logger.error(f"Aydınlatma adaptasyonu hatası: {str(e)}")
            return frame
    
    def preprocess_glasses(self, frame):
        """Gözlük için ön işleme - yansımaları azalt"""
        try:
            # Gri tonlama
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Adaptif histogram eşitleme
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Yansıma tespiti için parlak bölgeleri bul
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            
            # Morfolojik işlemler ile yansıma bölgelerini genişlet
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            reflections = cv2.dilate(thresh, kernel, iterations=1)
            
            # Yansıma maskeleme
            mask = cv2.bitwise_not(reflections)
            
            # İnpainting ile yansıma bölgelerini doldur
            result = cv2.inpaint(frame, reflections, 3, cv2.INPAINT_TELEA)
            
            return result
        except Exception as e:
            logger.error(f"Gözlük ön işleme hatası: {str(e)}")
            return frame
    
    def detect_glasses(self, face_landmarks, frame):
        """Yüzde gözlük olup olmadığını tespit et"""
        try:
            h, w = frame.shape[:2]
            
            # Sol ve sağ göz koordinatları
            left_eye_landmark = face_landmarks.landmark[468]  # Sol iris
            right_eye_landmark = face_landmarks.landmark[473]  # Sağ iris
            
            left_eye = (int(left_eye_landmark.x * w), int(left_eye_landmark.y * h))
            right_eye = (int(right_eye_landmark.x * w), int(right_eye_landmark.y * h))
            
            # Göz bölgesini kırp (biraz daha geniş al)
            eye_width = abs(right_eye[0] - left_eye[0])
            margin = int(eye_width * 0.5)
            
            y_min = max(0, min(left_eye[1], right_eye[1]) - margin)
            y_max = min(h, max(left_eye[1], right_eye[1]) + margin)
            x_min = max(0, min(left_eye[0], right_eye[0]) - margin)
            x_max = min(w, max(left_eye[0], right_eye[0]) + margin)
            
            eye_region = frame[y_min:y_max, x_min:x_max]
            if eye_region.size == 0:
                return False
                
            # Gri tonlama ve kenar tespiti
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Hough çizgi dönüşümü ile gözlük çerçevesi çizgilerini bul
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=10, maxLineGap=5)
            
            # Çizgi sayısı eşik değerden büyükse gözlük var demektir
            return lines is not None and len(lines) > 5
            except Exception as e:
            logger.error(f"Gözlük tespiti hatası: {str(e)}")
        return False
    
    def process_frame(self, frame):
        """Ana frame işleme fonksiyonu"""
        try:
            # Frame boyutları
            h, w, _ = frame.shape
            
            # Yüz landmark tespiti (strateji kullanarak)
            results = self.detect_face_landmarks(frame)
            
            # Hiç yüz yoksa çık
            if not results or not results.multi_face_landmarks:
                self.draw_stats(frame, None, None, is_face_detected=False)
                return frame
            
            # İlk yüzün landmarklarını al
            face_landmarks = results.multi_face_landmarks[0]
            
            # Kalibrasyon
            if not self.is_calibrated:
                self.calibrate(frame, face_landmarks)
            
            # Göz bebeklerini al
            left_pupil = face_landmarks.landmark[468]  # Sol iris merkezi
            right_pupil = face_landmarks.landmark[473]  # Sağ iris merkezi
            
            # Piksel koordinatlarına dönüştür
            left_pupil_px = (int(left_pupil.x * w), int(left_pupil.y * h))
            right_pupil_px = (int(right_pupil.x * w), int(right_pupil.y * h))
            
            # Şaşılık telafisi
            strabismus_results = None
            if self.config.get("handle_strabismus", False):
                strabismus_results = self._handle_strabismus(
                    {"iris_center": left_pupil_px}, 
                    {"iris_center": right_pupil_px}, 
                    face_landmarks
                )
            
            # Zaman damgası ekle
            current_time = time.time()
            self.timestamps.append(current_time)
            
            # Verileri buffer'a ekle
            self.left_x_buffer.append(left_pupil_px[0])
            self.left_y_buffer.append(left_pupil_px[1])
            self.right_x_buffer.append(right_pupil_px[0])
            self.right_y_buffer.append(right_pupil_px[1])
            
            # Analiz için yeterli veri var mı kontrol et
            min_data_points = int(self.fps)  # En az 1 saniyelik veri
            if len(self.left_x_buffer) >= min_data_points:
                # Şaşılık durumunda hangi gözü kullanacağımızı belirle
                use_left_eye = True
                if strabismus_results and strabismus_results.get("strabismus_detected", False):
                    use_left_eye = strabismus_results.get("use_left_eye", True)
                
                # Kullanılacak göz verisi
                x_buffer = self.left_x_buffer if use_left_eye else self.right_x_buffer
                
                # Nistagmus frekansını hesapla (yatay nistagmus)
                freq_hz, is_nistagmus = self.calculate_nistagmus_frequency(x_buffer)
                
                # SPV hesapla
                spv = self.calculate_spv(x_buffer, self.timestamps)
                
                # Şaşılık açısını hesapla
                strabismus_angle = self.calculate_strabismus_angle(left_pupil_px, right_pupil_px)
                
                # Sonuçları güncelle
                self.last_results = {
                    "nistagmus_freq": freq_hz,
                    "spv": spv,
                    "strabismus_angle": strabismus_angle,
                    "is_nistagmus_detected": is_nistagmus
                }
                
                # Sonuçları CSV'ye yaz
                if self.csv_writer:
                    self.csv_writer.writerow([
                        current_time,
                        f"{left_pupil_px[0]:.1f}", 
                        f"{left_pupil_px[1]:.1f}",
                        f"{right_pupil_px[0]:.1f}", 
                        f"{right_pupil_px[1]:.1f}",
                        f"{freq_hz:.3f}", 
                        f"{spv:.3f}", 
                        f"{strabismus_angle:.3f}",
                        is_nistagmus
                    ])
                    self.csv_file.flush()  # Tampon boşalt
                
                # Derin öğrenme modeli kullanılıyorsa
                if self.onnx_model:
                    # Sol göz bölgesini kırp
                    eye_region = self.extract_eye_region(frame, left_pupil_px)
                    
                    # Göz rengi düzeltmesi
                    if self.config.get("handle_eye_color", False) and eye_region is not None:
                        eye_region, is_dark_eye = self._adjust_for_eye_color(eye_region)
                    
                    if eye_region is not None:
                        # ONNX modeli ile analiz
                        model_results = self.run_onnx_inference(eye_region)
                        # Model sonuçlarını işle (örn: sınıflandırma sonuçları)
                        if model_results:
                            # Burada model çıktısına göre ekstra işlemler yapılabilir
                            pass
            
            # Görüntü üzerine analiz sonuçlarını yaz
            self.draw_stats(frame, left_pupil_px, right_pupil_px, is_face_detected=True)
            
            return frame
            
        except Exception as e:
            logger.error(f"Frame işleme hatası: {str(e)}")
            # Boş bir frame oluştur
            blank_frame = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(blank_frame, f"Hata: {str(e)}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return blank_frame
    
    def extract_eye_region(self, frame, pupil_center, size=64):
        """Göz bölgesini kırp"""
        try:
            h, w, _ = frame.shape
            x, y = pupil_center
            
            # Kırpma sınırlarını belirle
            half_size = size // 2
            x1, y1 = max(0, x - half_size), max(0, y - half_size)
            x2, y2 = min(w, x + half_size), min(h, y + half_size)
            
            # Göz bölgesini kırp
            eye_region = frame[y1:y2, x1:x2]
            
            # Boyut kontrolü
            if eye_region.shape[0] > 0 and eye_region.shape[1] > 0:
                return eye_region
            else:
                return None
                
        except Exception as e:
            logger.error(f"Göz bölgesi çıkarma hatası: {str(e)}")
            return None
    
    def draw_stats(self, frame, left_pupil=None, right_pupil=None, is_face_detected=True):
        """İstatistikleri ve göz konumlarını görüntü üzerinde göster"""
        # Yüz tespit edilmediyse uyarı göster
        if not is_face_detected:
            cv2.putText(frame, "Yüz tespit edilemedi!", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return
        
        # Göz bebeklerini çiz
        if left_pupil:
            cv2.circle(frame, left_pupil, 5, (0, 255, 0), -1)
        if right_pupil:
            cv2.circle(frame, right_pupil, 5, (0, 255, 0), -1)
        
        # Analiz bilgilerini göster
        freq_hz = self.last_results["nistagmus_freq"]
        spv = self.last_results["spv"]
        strabismus_angle = self.last_results["strabismus_angle"]
        is_nistagmus = self.last_results["is_nistagmus_detected"]
        
        # Nistagmus tespit durumu
        nistagmus_status = "VAR" if is_nistagmus else "YOK"
        status_color = (0, 0, 255) if is_nistagmus else (0, 255, 0)
        
        # Kalibrasyon durumu
        calib_text = "Kalibre: " + ("TAMAM" if self.is_calibrated else "HAYIR")
        
        # Bilgileri ekranda göster
        info_lines = [
            f"Nistagmus: {nistagmus_status} ({freq_hz:.2f} Hz)",
            f"SPV: {spv:.1f} deg/s",
            f"Şaşılık: {strabismus_angle:.1f} derece",
            calib_text
        ]
        
        # Bilgileri yazdır
        y_pos = 30
        for i, line in enumerate(info_lines):
            # Nistagmus durumu için renk değiştir
            color = status_color if i == 0 else (50, 255, 50)
            cv2.putText(frame, line, (10, y_pos), 
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_pos += 25
    
    def run(self):
        """Ana uygulama döngüsü"""
        try:
            # Frame yakalama thread'ini başlat
            self.start_capture_thread()
            
            # Grafik arayüzü çalıştır (basit)
            cv2.namedWindow("Göz Analizi", cv2.WINDOW_NORMAL)
            
            while True:
                try:
                    # Kuyruktaki bir sonraki frame'i al (en fazla 1 FPS periyodu kadar bekle)
                    frame = self.frame_queue.get(timeout=1.0/self.fps)
                    
                    # Frame'i işle
                    processed_frame = self.process_frame(frame)
                    
                    # Sonucu göster
                    cv2.imshow("Göz Analizi", processed_frame)
                    
                    # Klavye kontrolü
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC tuşu ile çık
                        break
                    elif key == ord('c'):  # 'c' tuşu ile kalibrasyon zorla
                        self.is_calibrated = False
                        logger.info("Kalibrasyon sıfırlandı, yeniden kalibrasyon bekleniyor.")
                    
                except queue.Empty:
                    logger.warning("Frame kuyruğu boş, bekleniyor...")
                    continue
                    
            # Kaynakları serbest bırak
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Uygulama hatası: {str(e)}")
            self.cleanup()
            raise
    
    def cleanup(self):
        """Kaynakları serbest bırak"""
        self.running = False
        if hasattr(self, 'capture_thread') and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        
        if hasattr(self, 'csv_file') and self.csv_file:
            self.csv_file.close()
        
        # Stratejideki kaynakları serbest bırak
        if hasattr(self, 'tracking_strategy'):
            self.tracking_strategy.release()
        
        cv2.destroyAllWindows()
        logger.info("Uygulama kaynakları temizlendi.")

class CalibrationUI:
    """Kalibrasyon için gelişmiş kullanıcı arayüzü"""
    def __init__(self, eye_analyzer):
        self.eye_analyzer = eye_analyzer
        self.root = tk.Tk()
        self.root.title("Göz Analizi Kalibrasyon")
        self.root.geometry("500x600")
        self.setup_ui()
    
    def setup_ui(self):
        """Arayüz bileşenlerini hazırla"""
        # Notebook (sekmeli arayüz) oluştur
        import tkinter.ttk as ttk
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Ana ayarlar sekmesi
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Temel Ayarlar")
        
        # Gelişmiş ayarlar sekmesi
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="Gelişmiş Ayarlar")
        
        # Strateji ayarları sekmesi
        strategy_frame = ttk.Frame(notebook)
        notebook.add(strategy_frame, text="Göz İzleme Stratejileri")
        
        # Özel durumlar sekmesi
        special_frame = ttk.Frame(notebook)
        notebook.add(special_frame, text="Özel Durumlar")
        
        # Temel ayarları oluştur
        self.setup_basic_settings(basic_frame)
        
        # Gelişmiş ayarları oluştur
        self.setup_advanced_settings(advanced_frame)
        
        # Strateji ayarlarını oluştur
        self.setup_strategy_settings(strategy_frame)
        
        # Özel durum ayarlarını oluştur
        self.setup_special_settings(special_frame)
        
        # Butonlar
        button_frame = tk.Frame(self.root)
        button_frame.pack(fill="x", pady=10)
        
        tk.Button(button_frame, text="Ayarları Kaydet", command=self.save_settings, 
                 bg="#4CAF50", fg="white", padx=10, pady=5).pack(side="left", padx=10)
        tk.Button(button_frame, text="Kalibre Et", command=self.start_calibration,
                 bg="#2196F3", fg="white", padx=10, pady=5).pack(side="left", padx=10)
        tk.Button(button_frame, text="Analizi Başlat", command=self.start_analysis,
                 bg="#FF9800", fg="white", padx=10, pady=5).pack(side="left", padx=10)
    
    def setup_basic_settings(self, parent):
        """Temel ayarlar bölümünü hazırla"""
        # Başlık
        tk.Label(parent, text="Temel Ayarlar", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Kamera ayarları
        camera_frame = tk.LabelFrame(parent, text="Kamera Ayarları", padx=10, pady=10)
        camera_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(camera_frame, text="Kamera ID:").grid(row=0, column=0, sticky="w")
        self.camera_id_var = tk.StringVar(value=str(self.eye_analyzer.config["camera_id"]))
        tk.Entry(camera_frame, textvariable=self.camera_id_var).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        tk.Label(camera_frame, text="Çözünürlük:").grid(row=1, column=0, sticky="w")
        resolution_frame = tk.Frame(camera_frame)
        resolution_frame.grid(row=1, column=1, sticky="we")
        
        self.width_var = tk.StringVar(value=str(self.eye_analyzer.config["frame_width"]))
        self.height_var = tk.StringVar(value=str(self.eye_analyzer.config["frame_height"]))
        tk.Entry(resolution_frame, textvariable=self.width_var, width=6).pack(side="left", padx=2)
        tk.Label(resolution_frame, text="x").pack(side="left", padx=2)
        tk.Entry(resolution_frame, textvariable=self.height_var, width=6).pack(side="left", padx=2)
        
        # Analiz parametreleri
        analysis_frame = tk.LabelFrame(parent, text="Analiz Parametreleri", padx=10, pady=10)
        analysis_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(analysis_frame, text="Nistagmus Frekans Eşiği (Hz):").grid(row=0, column=0, sticky="w")
        self.nistagmus_thresh_var = tk.StringVar(value=str(self.eye_analyzer.config["nistagmus_freq_threshold"]))
        tk.Entry(analysis_frame, textvariable=self.nistagmus_thresh_var).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        tk.Label(analysis_frame, text="Piksel/Derece Oranı:").grid(row=1, column=0, sticky="w")
        self.pixel_ratio_var = tk.StringVar(value=str(self.eye_analyzer.config["default_pixel_to_degree_ratio"]))
        tk.Entry(analysis_frame, textvariable=self.pixel_ratio_var).grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        tk.Label(analysis_frame, text="SPV Eşiği (derece/s):").grid(row=2, column=0, sticky="w")
        self.spv_threshold_var = tk.StringVar(value=str(self.eye_analyzer.config["spv_threshold"]))
        tk.Entry(analysis_frame, textvariable=self.spv_threshold_var).grid(row=2, column=1, padx=5, pady=5, sticky="we")
    
    def setup_advanced_settings(self, parent):
        """Gelişmiş ayarlar bölümünü hazırla"""
        # Başlık
        tk.Label(parent, text="Gelişmiş Ayarlar", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Filtre ayarları
        filter_frame = tk.LabelFrame(parent, text="Filtre Ayarları", padx=10, pady=10)
        filter_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(filter_frame, text="Bant Geçiren Alt Frekans (Hz):").grid(row=0, column=0, sticky="w")
        self.bandpass_low_var = tk.StringVar(value=str(self.eye_analyzer.config["bandpass_freq_low"]))
        tk.Entry(filter_frame, textvariable=self.bandpass_low_var).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        tk.Label(filter_frame, text="Bant Geçiren Üst Frekans (Hz):").grid(row=1, column=0, sticky="w")
        self.bandpass_high_var = tk.StringVar(value=str(self.eye_analyzer.config["bandpass_freq_high"]))
        tk.Entry(filter_frame, textvariable=self.bandpass_high_var).grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        tk.Label(filter_frame, text="Yumuşatma Penceresi:").grid(row=2, column=0, sticky="w")
        self.smoothing_window_var = tk.StringVar(value=str(self.eye_analyzer.config["smoothing_window"]))
        tk.Entry(filter_frame, textvariable=self.smoothing_window_var).grid(row=2, column=1, padx=5, pady=5, sticky="we")
        
        # ONNX model ayarları
        onnx_frame = tk.LabelFrame(parent, text="ONNX Model Ayarları", padx=10, pady=10)
        onnx_frame.pack(fill="x", padx=10, pady=5)
        
        self.use_onnx_var = tk.BooleanVar(value=self.eye_analyzer.config["use_onnx_model"])
        tk.Checkbutton(onnx_frame, text="ONNX Model Kullan", variable=self.use_onnx_var).grid(row=0, column=0, columnspan=2, sticky="w")
        
        tk.Label(onnx_frame, text="ONNX Model Dosyası:").grid(row=1, column=0, sticky="w")
        self.onnx_path_var = tk.StringVar(value=self.eye_analyzer.config["onnx_model_path"])
        tk.Entry(onnx_frame, textvariable=self.onnx_path_var).grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        # GPU kullanımı
        self.use_gpu_var = tk.BooleanVar(value=self.eye_analyzer.config.get("use_gpu", True))
        tk.Checkbutton(onnx_frame, text="GPU Kullan (varsa)", variable=self.use_gpu_var).grid(row=2, column=0, columnspan=2, sticky="w")
    
    def setup_strategy_settings(self, parent):
        """Strateji ayarları bölümünü hazırla"""
        # Başlık
        tk.Label(parent, text="Göz İzleme Stratejileri", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Strateji seçimi
        strategy_frame = tk.LabelFrame(parent, text="Strateji Seçimi", padx=10, pady=10)
        strategy_frame.pack(fill="x", padx=10, pady=5)
        
        # Strateji türü
        tk.Label(strategy_frame, text="Göz İzleme Stratejisi:").grid(row=0, column=0, sticky="w")
        self.strategy_var = tk.StringVar(value=self.eye_analyzer.config.get("tracking_strategy", "mediapipe"))
        strategy_options = ["mediapipe", "dlib", "opencv_dnn", "hybrid"]
        strategy_dropdown = ttk.Combobox(strategy_frame, textvariable=self.strategy_var, values=strategy_options)
        strategy_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        # MediaPipe ayarları
        mediapipe_frame = tk.LabelFrame(parent, text="MediaPipe Ayarları", padx=10, pady=10)
        mediapipe_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(mediapipe_frame, text="Yüz Tespiti Güven Eşiği:").grid(row=0, column=0, sticky="w")
        self.face_detection_var = tk.StringVar(value=str(self.eye_analyzer.config["face_detection_confidence"]))
        tk.Entry(mediapipe_frame, textvariable=self.face_detection_var).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        tk.Label(mediapipe_frame, text="Yüz Takibi Güven Eşiği:").grid(row=1, column=0, sticky="w")
        self.face_tracking_var = tk.StringVar(value=str(self.eye_analyzer.config["face_tracking_confidence"]))
        tk.Entry(mediapipe_frame, textvariable=self.face_tracking_var).grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        # İşaret iyileştirme
        self.refine_landmarks_var = tk.BooleanVar(value=self.eye_analyzer.config.get("refine_landmarks", True))
        tk.Checkbutton(mediapipe_frame, text="İris Tespiti İçin İşaretleri İyileştir", variable=self.refine_landmarks_var).grid(row=2, column=0, columnspan=2, sticky="w")
        
        # Dlib ayarları
        dlib_frame = tk.LabelFrame(parent, text="Dlib Ayarları", padx=10, pady=10)
        dlib_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(dlib_frame, text="Dlib Predictor Dosyası:").grid(row=0, column=0, sticky="w")
        self.dlib_predictor_var = tk.StringVar(value=self.eye_analyzer.config.get("dlib_predictor_path", "shape_predictor_68_face_landmarks.dat"))
        tk.Entry(dlib_frame, textvariable=self.dlib_predictor_var).grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        # Hibrit strateji ayarları
        hybrid_frame = tk.LabelFrame(parent, text="Hibrit Strateji Ayarları", padx=10, pady=10)
        hybrid_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(hybrid_frame, text="Oylama Yöntemi:").grid(row=0, column=0, sticky="w")
        self.voting_method_var = tk.StringVar(value=self.eye_analyzer.config.get("hybrid_voting_method", "confidence"))
        voting_options = ["confidence", "average"]
        voting_dropdown = ttk.Combobox(hybrid_frame, textvariable=self.voting_method_var, values=voting_options)
        voting_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="we")
    
    def setup_special_settings(self, parent):
        """Özel durum ayarları bölümünü hazırla"""
        # Başlık
        tk.Label(parent, text="Özel Durum Yönetimi", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Özel durum işleme ayarları
        special_frame = tk.LabelFrame(parent, text="Özel Durum İşleme", padx=10, pady=10)
        special_frame.pack(fill="x", padx=10, pady=5)
        
        # Gözlük telafisi
        self.handle_glasses_var = tk.BooleanVar(value=self.eye_analyzer.config.get("handle_glasses", True))
        tk.Checkbutton(special_frame, text="Gözlük Yansıma Telafisi", variable=self.handle_glasses_var).pack(anchor="w")
        
        # Aydınlatma adaptasyonu
        self.handle_lighting_var = tk.BooleanVar(value=self.eye_analyzer.config.get("handle_lighting", True))
        tk.Checkbutton(special_frame, text="Aydınlatma Adaptasyonu", variable=self.handle_lighting_var).pack(anchor="w")
        
        # Göz rengi adaptasyonu
        self.handle_eye_color_var = tk.BooleanVar(value=self.eye_analyzer.config.get("handle_eye_color", True))
        tk.Checkbutton(special_frame, text="Göz Rengi Adaptasyonu", variable=self.handle_eye_color_var).pack(anchor="w")
        
        # Şaşılık telafisi
        self.handle_strabismus_var = tk.BooleanVar(value=self.eye_analyzer.config.get("handle_strabismus", True))
        tk.Checkbutton(special_frame, text="Şaşılık Telafisi", variable=self.handle_strabismus_var).pack(anchor="w")
        
        # Şaşılık ayarları
        strabismus_frame = tk.LabelFrame(parent, text="Şaşılık Parametreleri", padx=10, pady=10)
        strabismus_frame.pack(fill="x", padx=10, pady=5)
        
        tk.Label(strabismus_frame, text="Şaşılık Derece Faktörü:").grid(row=0, column=0, sticky="w")
        self.strabismus_factor_var = tk.StringVar(value=str(self.eye_analyzer.config["strabismus_degree_factor"]))
        tk.Entry(strabismus_frame, textvariable=self.strabismus_factor_var).grid(row=0, column=1, padx=5, pady=5, sticky="we")
    
    def save_settings(self):
        """Ayarları kaydet"""
        try:
            # Temel ayarlar
            self.eye_analyzer.config["camera_id"] = int(self.camera_id_var.get())
            self.eye_analyzer.config["frame_width"] = int(self.width_var.get())
            self.eye_analyzer.config["frame_height"] = int(self.height_var.get())
            self.eye_analyzer.config["nistagmus_freq_threshold"] = float(self.nistagmus_thresh_var.get())
            self.eye_analyzer.config["default_pixel_to_degree_ratio"] = float(self.pixel_ratio_var.get())
            self.eye_analyzer.config["spv_threshold"] = float(self.spv_threshold_var.get())
            
            # Gelişmiş ayarlar
            self.eye_analyzer.config["bandpass_freq_low"] = float(self.bandpass_low_var.get())
            self.eye_analyzer.config["bandpass_freq_high"] = float(self.bandpass_high_var.get())
            self.eye_analyzer.config["smoothing_window"] = int(self.smoothing_window_var.get())
            self.eye_analyzer.config["use_onnx_model"] = self.use_onnx_var.get()
            self.eye_analyzer.config["onnx_model_path"] = self.onnx_path_var.get()
            self.eye_analyzer.config["use_gpu"] = self.use_gpu_var.get()
            
            # Strateji ayarları
            self.eye_analyzer.config["tracking_strategy"] = self.strategy_var.get()
            self.eye_analyzer.config["face_detection_confidence"] = float(self.face_detection_var.get())
            self.eye_analyzer.config["face_tracking_confidence"] = float(self.face_tracking_var.get())
            self.eye_analyzer.config["refine_landmarks"] = self.refine_landmarks_var.get()
            self.eye_analyzer.config["dlib_predictor_path"] = self.dlib_predictor_var.get()
            self.eye_analyzer.config["hybrid_voting_method"] = self.voting_method_var.get()
            
            # Özel durum ayarları
            self.eye_analyzer.config["handle_glasses"] = self.handle_glasses_var.get()
            self.eye_analyzer.config["handle_lighting"] = self.handle_lighting_var.get()
            self.eye_analyzer.config["handle_eye_color"] = self.handle_eye_color_var.get()
            self.eye_analyzer.config["handle_strabismus"] = self.handle_strabismus_var.get()
            self.eye_analyzer.config["strabismus_degree_factor"] = float(self.strabismus_factor_var.get())
            
            # Ayarları güncelle
            self.eye_analyzer.pixel_to_degree_ratio = float(self.pixel_ratio_var.get())
            
            # Stratejiyi yeniden oluştur
            try:
                self.eye_analyzer.tracking_strategy.release()
                self.eye_analyzer.tracking_strategy = self.eye_analyzer.create_tracking_strategy()
    except Exception as e:
                messagebox.showwarning("Strateji Uyarısı", f"Strateji değiştirilirken hata oluştu: {str(e)}")
            
            # Dosyaya kaydet
            with open("config.json", 'w') as f:
                json.dump(self.eye_analyzer.config, f, indent=4)
                
            messagebox.showinfo("Bilgi", "Ayarlar kaydedildi!")
            
        except Exception as e:
            messagebox.showerror("Hata", f"Ayarlar kaydedilemedi: {str(e)}")
    
    def start_calibration(self):
        """Kalibrasyon modunu başlat"""
        self.eye_analyzer.is_calibrated = False
        messagebox.showinfo("Kalibrasyon", 
                           "Kalibrasyon başlatılıyor. Lütfen kameraya bakın ve başınızı sabit tutun.")
    
    def start_analysis(self):
        """Analizi başlat"""
        self.root.destroy()  # UI'yı kapat
        self.eye_analyzer.run()  # Ana analiz döngüsünü başlat
    
    def run(self):
        """Arayüzü çalıştır"""
        print(f"Uygulama hatası: {str(e)}")
        # Başlangıç UI'sı
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Başlatma Hatası", f"Uygulama başlatılamadı: {str(e)}")
        root.destroy()
