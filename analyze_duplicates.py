#!/usr/bin/env python3
"""
Kod duplikasyonu ve fazlalık analizi
"""

import ast
import os
from collections import defaultdict

def analyze_duplicates():
    duplicate_functions = defaultdict(list)
    all_functions = set()
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and not file.startswith('analyze_'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        tree = ast.parse(f.read())
                        
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            func_name = node.name
                            if func_name in all_functions:
                                duplicate_functions[func_name].append(os.path.join(root, file))
                            else:
                                all_functions.add(func_name)
                                duplicate_functions[func_name].append(os.path.join(root, file))
                except:
                    pass
    
    # Sadece gerçek duplikasyonları göster
    real_duplicates = {k: v for k, v in duplicate_functions.items() if len(v) > 1}
    
    print('🔍 FUNCTION DUPLICATE ANALİZİ')
    print('=' * 50)
    
    if real_duplicates:
        for func_name, files in real_duplicates.items():
            if not func_name.startswith('_'):  # Private fonksiyonları atla
                print(f'⚠️ {func_name}: {len(files)} dosyada')
                for f in files[:3]:  # İlk 3 dosyayı göster
                    print(f'   - {f}')
                if len(files) > 3:
                    print(f'   ... ve {len(files)-3} dosya daha')
                print()
    else:
        print('✅ Önemli duplikasyon bulunamadı')

def analyze_imports():
    import_counts = defaultdict(int)
    
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    if 'import numpy' in content:
                        import_counts['numpy'] += 1
                    if 'import cv2' in content:
                        import_counts['cv2'] += 1
                    if 'import torch' in content:
                        import_counts['torch'] += 1
                    if 'import mediapipe' in content:
                        import_counts['mediapipe'] += 1
                except:
                    pass
    
    print('\n📦 IMPORT FREQUENCY ANALİZİ')
    print('=' * 50)
    for lib, count in sorted(import_counts.items(), key=lambda x: x[1], reverse=True):
        print(f'{lib}: {count} dosyada kullanılıyor')

def analyze_file_sizes():
    print('\n📊 DOSYA BOYUT ANALİZİ')
    print('=' * 50)
    
    large_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                    if lines > 300:
                        large_files.append((filepath, lines))
                except:
                    pass
    
    large_files.sort(key=lambda x: x[1], reverse=True)
    
    print('🔴 Büyük dosyalar (>300 satır):')
    for filepath, lines in large_files[:10]:
        print(f'   {filepath}: {lines} satır')

if __name__ == "__main__":
    analyze_duplicates()
    analyze_imports() 
    analyze_file_sizes() 