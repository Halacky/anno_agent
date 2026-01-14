from pathlib import Path
from typing import List
import json
from datetime import datetime

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_annotation(annotation_data: dict, output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(annotation_data, f, ensure_ascii=False, indent=2)

def load_annotation(annotation_path: Path) -> dict:
    with open(annotation_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_image_files(directory: Path, extensions: List[str] = None) -> List[Path]:
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
        image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)