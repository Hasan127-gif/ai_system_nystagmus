#!/usr/bin/env python3

import json
import requests
import os

def test_api_upload():
    """API video upload functionality test"""
    print('🧪 API Video Upload Test başlatılıyor...')

    try:
        # Test video yolu
        video_path = 'test_clinical_video.mp4'
        
        if os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'analysis_type': 'combined', 
                    'include_explainability': 'true',
                    'patient_id': 'TEST_PATIENT_001'
                }
                
                response = requests.post('http://localhost:8000/analyze', files=files, data=data, timeout=30)
                
                print(f'Status Code: {response.status_code}')
                
                if response.status_code == 200:
                    result = response.json()
                    print('✅ API Test Başarılı!')
                    print(f'Analysis ID: {result.get("analysis_id", "N/A")}')
                    print(f'Processing Time: {result.get("processing_time_ms", 0):.2f}ms')
                    print(f'Confidence: {result.get("confidence_score", 0):.2f}')
                    print(f'Results Keys: {list(result.get("results", {}).keys())}')
                    return True
                else:
                    print(f'❌ API Test Başarısız: {response.status_code}')
                    print(f'Error: {response.text[:200]}')
                    return False
        else:
            print('❌ Test video bulunamadı')
            return False
            
    except Exception as e:
        print(f'❌ API Test Hatası: {e}')
        return False

if __name__ == "__main__":
    test_api_upload() 