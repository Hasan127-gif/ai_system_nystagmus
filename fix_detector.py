#!/usr/bin/env python3
"""
Nistagmus detektör kodu düzeltme scripti
Bu script, detector.py dosyasındaki girinti ve sözdizimi hatalarını düzeltir
"""

import os
import re
import sys
import logging
import traceback
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("detector_fixer")

def fix_detector_code():
    """
    detector.py dosyasındaki girinti ve sözdizimi hatalarını düzeltir
    """
    try:
        # Dosya yolları
        script_dir = os.path.dirname(os.path.abspath(__file__))
        detector_path = os.path.join(script_dir, "detector.py")
        backup_path = os.path.join(script_dir, "detector.py.backup")
        
        # Dosya mevcut mu kontrol et
        if not os.path.exists(detector_path):
            logger.error(f"Detector dosyası bulunamadı: {detector_path}")
            return False
        
        # Yedek oluştur
        logger.info(f"Yedek oluşturuluyor: {backup_path}")
        shutil.copy2(detector_path, backup_path)
        
        # Dosyayı oku
        logger.info("Dosya içeriği okunuyor...")
        with open(detector_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Sorunları düzelt
        logger.info("Sorunlar düzeltiliyor...")
        
        # 1. `analyze_image` metodu düzeltmesi
        logger.info("'analyze_image' metodunu düzeltme")
        lines = fix_analyze_image_method(lines)
        
        # 2. `analyze_video` metodunun içindeki `return` ve `if` ifadelerini düzeltme
        logger.info("'analyze_video' metodunu düzeltme")
        lines = fix_analyze_video_method(lines)
        
        # 3. `_extract_features_for_model` metodunu düzeltme  
        logger.info("'_extract_features_for_model' metodunu düzeltme")
        lines = fix_extract_features_method(lines)
        
        # Dosyayı yaz
        logger.info("Değişiklikler kaydediliyor...")
        with open(detector_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
            
        logger.info(f"Detector.py dosyası başarıyla düzeltildi. Orijinal dosya {backup_path} konumuna yedeklendi.")
        return True
        
    except Exception as e:
        logger.error(f"Hata: {str(e)}")
        traceback.print_exc()
        return False

def fix_analyze_image_method(lines):
    """
    'analyze_image' metodundaki girinti hatalarını düzeltir.
    
    Args:
        lines: Dosya satırları
        
    Returns:
        Düzeltilmiş satırlar
    """
    # 'analyze_image' bloğunun başlangıç ve bitiş indekslerini bul
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if 'def analyze_image' in line:
            start_idx = i
        elif start_idx != -1 and 'def _extract_eye_landmarks' in line:
            end_idx = i
            break
    
    if start_idx == -1 or end_idx == -1:
        logger.warning("analyze_image metodu bulunamadı")
        return lines
    
    # İlgili satırları düzgün girintilerle yeniden yaz
    analyze_image_lines = lines[start_idx:end_idx]
    fixed_lines = []
    
    # İlk satır (method tanımı)
    fixed_lines.append(analyze_image_lines[0])
    
    # Docstring ve diğer satırlar
    in_docstring = False
    indent_level = 1  # En üst seviye girinti (4 boşluk)
    
    for line in analyze_image_lines[1:]:
        stripped = line.strip()
        
        # Docstring kontrolü
        if stripped.startswith('"""') and not in_docstring:
            in_docstring = True
            fixed_lines.append(line)
            continue
        elif stripped.endswith('"""') and in_docstring:
            in_docstring = False
            fixed_lines.append(line)
            continue
        
        if in_docstring:
            fixed_lines.append(line)
            continue
        
        # İç içe blokları tespit et
        if stripped.startswith('if ') or stripped.startswith('try:') or stripped.startswith('else:') or stripped.startswith('elif '):
            # Yeni bir blok başlatıyor
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        elif stripped.endswith(':'):
            # Yeni bir blok başlatıyor
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        elif stripped == 'else:':
            # else bloğu - indent_level değişmez
            indent_level -= 1
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        elif stripped == 'except Exception as e:':
            # except bloğu - indent_level değişmez
            indent_level -= 1
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        else:
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
    
    # Düzeltilmiş satırları orijinal listeye yerleştir
    fixed_method = fixed_lines
    lines[start_idx:end_idx] = fixed_method
    
    return lines

def fix_analyze_video_method(lines):
    """
    'analyze_video' metodundaki girinti hatalarını düzeltir.
    
    Args:
        lines: Dosya satırları
        
    Returns:
        Düzeltilmiş satırlar
    """
    # Sorunlu satır 999: "if face_detected_frames == 0:" düzeltmesi
    for i, line in enumerate(lines):
        if "if face_detected_frames == 0:" in line:
            # Sonraki satırda "return" var ve girintisi yanlış
            if i+1 < len(lines) and "return {" in lines[i+1]:
                # Girinti düzeltmesi
                lines[i+1] = lines[i+1].lstrip() 
                indent = re.match(r'^(\s*)', line).group(1)
                lines[i+1] = indent + "    " + lines[i+1].lstrip()
                
                # Sonraki birkaç satırın girintisi muhtemelen yanlış, onları da düzelt
                j = i + 2
                while j < len(lines) and "}" not in lines[j-1]:
                    lines[j] = indent + "        " + lines[j].lstrip()
                    j += 1
                # Kapanış parantezinin girintisini de düzelt
                if j < len(lines):
                    lines[j] = indent + "    " + lines[j].lstrip()
    
    # Try/except çiftlerini düzelt
    try_except_mappings = []
    
    # Try bloklarını bul
    for i, line in enumerate(lines):
        if "try:" in line and "except" not in line:
            # Bu bir try bloğunun başlangıcı
            try_start = i
            indent = re.match(r'^(\s*)', line).group(1)
            
            # En yakın except bloğunu bul
            for j in range(i+1, len(lines)):
                if indent + "except" in lines[j]:
                    try_except_mappings.append((try_start, j))
                    break
    
    # Tüm try/except çiftlerini düzelt
    for try_start, except_start in try_except_mappings:
        # Try bloğunun girintisini al
        try_indent = re.match(r'^(\s*)', lines[try_start]).group(1)
        
        # Except bloğunun başındaki girintiyi düzelt
        lines[except_start] = try_indent + "except Exception as e:" + "\n"
        
        # Except bloğu içindeki satırların girintisini düzelt
        for i in range(except_start+1, len(lines)):
            if lines[i].strip() and not lines[i].startswith(try_indent):
                # Bu satır except bloğunda ve girinti düzeltilmeli
                lines[i] = try_indent + "    " + lines[i].lstrip()
            elif lines[i].startswith(try_indent) and not lines[i].startswith(try_indent + "    "):
                # Bu satır except bloğunun dışında, bu bloğun sonuna geldik
                break
    
    return lines

def fix_extract_features_method(lines):
    """
    '_extract_features_for_model' metodundaki girinti hatalarını düzeltir.
    
    Args:
        lines: Dosya satırları
        
    Returns:
        Düzeltilmiş satırlar
    """
    # '_extract_features_for_model' bloğunun başlangıç ve bitiş indekslerini bul
    start_idx = -1
    end_idx = -1
    
    for i, line in enumerate(lines):
        if 'def _extract_features_for_model' in line:
            start_idx = i
        elif start_idx != -1 and 'def ' in line and i > start_idx + 10:  # sonraki metodu bul
            end_idx = i
            break
    
    if start_idx == -1:
        logger.warning("_extract_features_for_model metodu bulunamadı")
        return lines
        
    if end_idx == -1:
        end_idx = len(lines)  # Dosya sonuna kadar
    
    # İlgili satırları düzeltilmiş girintilerle yaz
    method_lines = lines[start_idx:end_idx]
    fixed_lines = []
    
    # İlk satır (method tanımı)
    fixed_lines.append(method_lines[0])
    
    # Docstring ve diğer satırlar
    in_docstring = False
    indent_level = 1  # En üst seviye girinti (4 boşluk)
    
    for line in method_lines[1:]:
        stripped = line.strip()
        
        # Docstring kontrolü
        if stripped.startswith('"""') and not in_docstring:
            in_docstring = True
            fixed_lines.append(line)
            continue
        elif stripped.endswith('"""') and in_docstring:
            in_docstring = False
            fixed_lines.append(line)
            continue
        
        if in_docstring:
            fixed_lines.append(line)
            continue
        
        # İç içe blokları tespit et
        if stripped.startswith('if ') or stripped.startswith('try:') or stripped.startswith('else:') or stripped.startswith('elif '):
            # Yeni bir blok başlatıyor
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        elif stripped.endswith(':'):
            # Yeni bir blok başlatıyor
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        elif stripped == 'else:':
            # else bloğu - indent_level değişmez
            indent_level -= 1
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        elif stripped == 'except Exception as e:':
            # except bloğu - indent_level değişmez
            indent_level -= 1
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
            indent_level += 1
        else:
            fixed_line = ' ' * (indent_level * 4) + stripped + '\n'
            fixed_lines.append(fixed_line)
    
    # Düzeltilmiş satırları orijinal listeye yerleştir
    fixed_method = fixed_lines
    lines[start_idx:end_idx] = fixed_method
    
    return lines

if __name__ == "__main__":
    logger.info("Detector.py düzeltme işlemi başlatılıyor...")
    success = fix_detector_code()
    if success:
        logger.info("Düzeltme başarıyla tamamlandı.")
    else:
        logger.error("Düzeltme başarısız oldu.") 