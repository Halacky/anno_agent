from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import json
from pathlib import Path

class AnnotationValidationInput(BaseModel):
    annotation_data: str = Field(description="JSON annotation data to validate")

class AnnotationValidationTool(BaseTool):
    name = "validate_annotation"
    description = "Validate annotation data structure and completeness"
    args_schema: Type[BaseModel] = AnnotationValidationInput
    
    def _run(self, annotation_data: str) -> str:
        try:
            data = json.loads(annotation_data)
            
            if "annotations" not in data:
                return "Invalid: Missing 'annotations' field"
            
            if not isinstance(data["annotations"], list):
                return "Invalid: 'annotations' must be a list"
            
            for idx, ann in enumerate(data["annotations"]):
                if "object" not in ann:
                    return f"Invalid: Annotation {idx} missing 'object' field"
                if "bbox" in ann and len(ann["bbox"]) != 4:
                    return f"Invalid: Annotation {idx} bbox must have 4 coordinates"
            
            return "Valid annotation structure"
            
        except json.JSONDecodeError:
            return "Invalid: Not valid JSON"