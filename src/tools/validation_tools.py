
from typing import Dict, List, Any
from pathlib import Path
import json

class AnnotationValidator:
    """Валидатор для проверки корректности аннотаций"""
    
    @staticmethod
    def validate_bbox(bbox: List[float], img_width: int = None, img_height: int = None) -> Dict:
        """Валидация bounding box"""
        errors = []
        
        if len(bbox) != 4:
            errors.append("BBox must have exactly 4 coordinates [x_min, y_min, x_max, y_max]")
            return {"valid": False, "errors": errors}
        
        x_min, y_min, x_max, y_max = bbox
        
        if x_min >= x_max:
            errors.append(f"x_min ({x_min}) must be less than x_max ({x_max})")
        
        if y_min >= y_max:
            errors.append(f"y_min ({y_min}) must be less than y_max ({y_max})")
        
        if img_width and img_height:
            if x_min < 0 or x_max > img_width:
                errors.append(f"BBox x coordinates out of image bounds (0, {img_width})")
            if y_min < 0 or y_max > img_height:
                errors.append(f"BBox y coordinates out of image bounds (0, {img_height})")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    @staticmethod
    def validate_annotation(annotation: Dict) -> Dict:
        """Валидация полной аннотации"""
        errors = []
        warnings = []
        
        # Проверка обязательных полей
        if "annotations" not in annotation:
            errors.append("Missing required field: 'annotations'")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        if not isinstance(annotation["annotations"], list):
            errors.append("'annotations' must be a list")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        # Проверка каждого объекта
        for idx, obj in enumerate(annotation["annotations"]):
            if "object" not in obj:
                errors.append(f"Object {idx}: missing 'object' field")
            
            if "bbox" in obj:
                bbox_validation = AnnotationValidator.validate_bbox(obj["bbox"])
                if not bbox_validation["valid"]:
                    errors.extend([f"Object {idx}: {err}" for err in bbox_validation["errors"]])
            else:
                warnings.append(f"Object {idx}: no bounding box provided")
            
            if "confidence" in obj:
                conf = obj["confidence"]
                if not (0.0 <= conf <= 1.0):
                    errors.append(f"Object {idx}: confidence must be between 0.0 and 1.0")
            else:
                warnings.append(f"Object {idx}: no confidence score provided")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    @staticmethod
    def validate_annotation_file(file_path: Path) -> Dict:
        """Валидация файла аннотации"""
        try:
            with open(file_path, 'r') as f:
                annotation = json.load(f)
            
            validation = AnnotationValidator.validate_annotation(annotation)
            validation["file"] = str(file_path)
            
            return validation
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "file": str(file_path),
                "errors": [f"Invalid JSON: {str(e)}"],
                "warnings": []
            }
        except Exception as e:
            return {
                "valid": False,
                "file": str(file_path),
                "errors": [f"Error reading file: {str(e)}"],
                "warnings": []
            }