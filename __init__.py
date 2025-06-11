"""
Nistagmus AI System
---------------
Bu paket, göz hareketlerinde nistagmus tespiti ve analizi için geliştirilmiş
yapay zeka sistemini içerir.

Modüller:
- detector: Nistagmus görüntü ve video analizi
- eye_tracking: Göz takip algoritmaları
- models: Nistagmus modelleri
- api: FastAPI tabanlı web API
"""

from ai_system.detector import NistagmusDetector
from ai_system.eye_tracking import EyeTracker

__version__ = "1.0.0"
__all__ = ['NistagmusDetector', 'EyeTracker'] 